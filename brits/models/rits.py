"""
RITS: Recurrent Imputation for Time Series

This module implements the RITS model which forms the building block for BRITS.
RITS uses a recurrent neural network with temporal decay to impute missing values
in time series data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class FeatureRegression(nn.Module):
    """
    Feature regression module that estimates missing values based on
    correlations with other features at the same time step.

    The key insight is that missing values in one feature can often be
    estimated from observed values of correlated features.
    """

    def __init__(self, input_size: int):
        """
        Initialize the FeatureRegression module.

        Args:
            input_size: Number of input features
        """
        super(FeatureRegression, self).__init__()
        self.input_size = input_size

        # Weight matrix for feature regression
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        # Mask to prevent self-regression (diagonal elements are 0)
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using uniform distribution."""
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feature regression.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Estimated values based on feature correlations
        """
        # Apply masked weight matrix to prevent self-regression
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h


class TemporalDecay(nn.Module):
    """
    Temporal decay module that models how the relevance of past observations
    decreases over time when there are missing values.

    The decay factor gamma is learned from the time gaps (deltas) between
    observations.
    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        """
        Initialize the TemporalDecay module.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            diag: If True, use diagonal weight matrix (feature-independent decay)
        """
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.input_size = input_size
        self.output_size = output_size

        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size, "For diagonal mode, input_size must equal output_size"
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using uniform distribution."""
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal decay.

        Args:
            d: Time gap (delta) tensor of shape (batch_size, input_size)

        Returns:
            Decay factor gamma in range (0, 1]
        """
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        # Apply exponential decay
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    """
    RITS: Recurrent Imputation for Time Series

    A unidirectional recurrent model for time series imputation that:
    1. Uses temporal decay to model the decrease in relevance of past observations
    2. Uses feature regression to estimate missing values from correlated features
    3. Combines history-based and feature-based estimates
    4. Optionally performs classification using the learned representations

    This model processes sequences in one direction only. For bidirectional
    processing, use the BRITS model which combines forward and backward RITS.
    """

    def __init__(
        self,
        input_size: int,
        rnn_hid_size: int,
        impute_weight: float = 1.0,
        label_weight: float = 0.0,
        seq_len: int = 48,
        dropout: float = 0.25,
    ):
        """
        Initialize the RITS model.

        Args:
            input_size: Number of input features
            rnn_hid_size: Hidden size of the RNN cell
            impute_weight: Weight for imputation loss
            label_weight: Weight for classification loss (0 for imputation only)
            seq_len: Length of input sequences
            dropout: Dropout probability
        """
        super(RITS, self).__init__()

        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.seq_len = seq_len

        # LSTM cell for sequential processing
        self.rnn_cell = nn.LSTMCell(input_size * 2, rnn_hid_size)

        # Temporal decay for hidden state and input
        self.temp_decay_h = TemporalDecay(
            input_size=input_size, output_size=rnn_hid_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=input_size, output_size=input_size, diag=True
        )

        # History-based regression from hidden state
        self.hist_reg = nn.Linear(rnn_hid_size, input_size)

        # Feature-based regression
        self.feat_reg = FeatureRegression(input_size)

        # Combine history and feature estimates
        self.weight_combine = nn.Linear(input_size * 2, input_size)

        self.dropout = nn.Dropout(p=dropout)

        # Classification output (optional)
        self.out = nn.Linear(rnn_hid_size, 1)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        evals: torch.Tensor = None,
        eval_masks: torch.Tensor = None,
        labels: torch.Tensor = None,
        is_train: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass of RITS.

        Args:
            values: Input tensor of shape (batch_size, seq_len, input_size)
                    with missing values filled with 0
            masks: Binary mask tensor of shape (batch_size, seq_len, input_size)
                   where 1 indicates observed values and 0 indicates missing
            deltas: Time gap tensor of shape (batch_size, seq_len, input_size)
                    indicating time since last observation for each feature
            evals: Ground truth for evaluation (optional)
            eval_masks: Mask for evaluation positions (optional)
            labels: Classification labels (optional)
            is_train: Training/validation indicator (optional)

        Returns:
            Dictionary containing:
                - loss: Total loss (imputation + classification)
                - imputations: Imputed time series
                - predictions: Classification predictions (if labels provided)
                - etc.
        """
        batch_size = values.size(0)
        device = values.device

        # Initialize hidden and cell states
        h = torch.zeros((batch_size, self.rnn_hid_size), device=device)
        c = torch.zeros((batch_size, self.rnn_hid_size), device=device)

        x_loss = 0.0
        imputations = []

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            # Apply temporal decay to hidden state
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            # History-based estimation
            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            # Complement missing values with history-based estimate
            x_c = m * x + (1 - m) * x_h

            # Feature-based estimation
            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            # Combine history and feature estimates
            alpha = torch.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=1)))
            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            # Final complemented input
            c_c = m * x + (1 - m) * c_h

            # RNN input: concatenate complemented values and mask
            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)

        # Build result dictionary
        result = {
            "imputations": imputations,
            "x_loss": x_loss * self.impute_weight,
        }

        # Classification loss (optional)
        if labels is not None and self.label_weight > 0:
            y_h = self.out(h)

            # Safely reshape labels to (batch_size, 1)
            labels_reshaped = self._reshape_to_column(labels, batch_size)

            if is_train is not None:
                is_train_reshaped = self._reshape_to_column(is_train, batch_size)
                y_loss = F.binary_cross_entropy_with_logits(y_h, labels_reshaped, reduction="none")
                y_loss = torch.sum(y_loss * is_train_reshaped) / (torch.sum(is_train_reshaped) + 1e-5)
            else:
                y_loss = F.binary_cross_entropy_with_logits(y_h, labels_reshaped)

            result["predictions"] = torch.sigmoid(y_h)
            result["y_loss"] = y_loss * self.label_weight
            result["loss"] = result["x_loss"] + result["y_loss"]
        else:
            result["loss"] = result["x_loss"]
            result["predictions"] = None

        # Add evaluation data if provided
        if evals is not None:
            result["evals"] = evals
        if eval_masks is not None:
            result["eval_masks"] = eval_masks
        if labels is not None:
            result["labels"] = self._reshape_to_column(labels, batch_size)
        if is_train is not None:
            result["is_train"] = self._reshape_to_column(is_train, batch_size)

        return result

    def _reshape_to_column(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Safely reshape a tensor to a column vector (batch_size, 1).

        Args:
            tensor: Input tensor
            batch_size: Expected batch size

        Returns:
            Tensor of shape (batch_size, 1)
        """
        if tensor.dim() == 0:
            # Scalar - expand to column
            return tensor.unsqueeze(0).unsqueeze(1).expand(batch_size, 1)
        elif tensor.dim() == 1:
            # 1D - add column dimension
            return tensor.view(-1, 1)
        elif tensor.dim() == 2 and tensor.shape[1] == 1:
            # Already correct shape
            return tensor
        else:
            # Flatten and reshape
            return tensor.reshape(-1, 1)
