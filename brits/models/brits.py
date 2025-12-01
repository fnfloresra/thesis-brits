"""
BRITS: Bidirectional Recurrent Imputation for Time Series

This module implements the BRITS model which combines forward and backward
RITS models to leverage information from both directions for improved
time series imputation.
"""

import torch
import torch.nn as nn

from brits.models.rits import RITS


class BRITS(nn.Module):
    """
    BRITS: Bidirectional Recurrent Imputation for Time Series

    Combines two RITS models (forward and backward) to impute missing values
    by leveraging temporal information from both past and future observations.

    The model:
    1. Runs a forward RITS from t=0 to t=T
    2. Runs a backward RITS from t=T to t=0
    3. Enforces consistency between forward and backward imputations
    4. Averages the imputations from both directions

    Reference:
        Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
        BRITS: Bidirectional Recurrent Imputation for Time Series.
        Advances in Neural Information Processing Systems, 31.
    """

    def __init__(
        self,
        input_size: int,
        rnn_hid_size: int,
        impute_weight: float = 1.0,
        label_weight: float = 0.0,
        consistency_weight: float = 0.1,
        seq_len: int = 48,
        dropout: float = 0.25,
    ):
        """
        Initialize the BRITS model.

        Args:
            input_size: Number of input features
            rnn_hid_size: Hidden size of the RNN cells
            impute_weight: Weight for imputation loss
            label_weight: Weight for classification loss
            consistency_weight: Weight for consistency loss between directions
            seq_len: Length of input sequences
            dropout: Dropout probability
        """
        super(BRITS, self).__init__()

        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.consistency_weight = consistency_weight
        self.seq_len = seq_len

        # Forward and backward RITS models
        self.rits_f = RITS(
            input_size=input_size,
            rnn_hid_size=rnn_hid_size,
            impute_weight=impute_weight,
            label_weight=label_weight,
            seq_len=seq_len,
            dropout=dropout,
        )

        self.rits_b = RITS(
            input_size=input_size,
            rnn_hid_size=rnn_hid_size,
            impute_weight=impute_weight,
            label_weight=label_weight,
            seq_len=seq_len,
            dropout=dropout,
        )

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
        Forward pass of BRITS.

        Args:
            values: Input tensor of shape (batch_size, seq_len, input_size)
            masks: Binary mask tensor of shape (batch_size, seq_len, input_size)
            deltas: Time gap tensor of shape (batch_size, seq_len, input_size)
            evals: Ground truth for evaluation (optional)
            eval_masks: Mask for evaluation positions (optional)
            labels: Classification labels (optional)
            is_train: Training/validation indicator (optional)

        Returns:
            Dictionary containing combined results from both directions
        """
        # Forward pass
        ret_f = self.rits_f(
            values=values,
            masks=masks,
            deltas=deltas,
            evals=evals,
            eval_masks=eval_masks,
            labels=labels,
            is_train=is_train,
        )

        # Prepare backward data (reverse sequences)
        values_b = self._reverse_tensor(values)
        masks_b = self._reverse_tensor(masks)
        deltas_b = self._compute_backward_deltas(masks)
        evals_b = self._reverse_tensor(evals) if evals is not None else None
        eval_masks_b = self._reverse_tensor(eval_masks) if eval_masks is not None else None

        # Backward pass
        ret_b = self.rits_b(
            values=values_b,
            masks=masks_b,
            deltas=deltas_b,
            evals=evals_b,
            eval_masks=eval_masks_b,
            labels=labels,
            is_train=is_train,
        )

        # Reverse backward imputations to align with forward direction
        ret_b = self._reverse_result(ret_b)

        # Merge results
        return self._merge_results(ret_f, ret_b)

    def _reverse_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse a tensor along the time dimension (dim=1)."""
        if tensor is None or tensor.dim() <= 1:
            return tensor
        return tensor.flip(dims=[1])

    def _compute_backward_deltas(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute time gaps for backward direction.

        Args:
            masks: Binary mask tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Time gaps for backward direction
        """
        # Reverse masks for backward direction
        masks_b = self._reverse_tensor(masks)

        batch_size, seq_len, input_size = masks_b.shape
        device = masks_b.device

        deltas = torch.zeros_like(masks_b)

        for t in range(seq_len):
            if t == 0:
                deltas[:, t, :] = torch.ones(batch_size, input_size, device=device)
            else:
                deltas[:, t, :] = (
                    torch.ones(batch_size, input_size, device=device)
                    + (1 - masks_b[:, t - 1, :]) * deltas[:, t - 1, :]
                )

        return deltas

    def _reverse_result(self, result: dict) -> dict:
        """Reverse time-dependent tensors in the result dictionary."""
        reversed_result = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                # Reverse 3D tensors (batch, seq, features)
                reversed_result[key] = self._reverse_tensor(value)
            else:
                reversed_result[key] = value
        return reversed_result

    def _merge_results(self, ret_f: dict, ret_b: dict) -> dict:
        """
        Merge forward and backward results.

        Combines losses, averages predictions and imputations.
        """
        # Combine losses
        loss_f = ret_f["loss"]
        loss_b = ret_b["loss"]
        loss_c = self._get_consistency_loss(ret_f["imputations"], ret_b["imputations"])

        total_loss = loss_f + loss_b + loss_c

        # Average imputations from both directions
        imputations = (ret_f["imputations"] + ret_b["imputations"]) / 2

        result = {
            "loss": total_loss,
            "loss_forward": loss_f,
            "loss_backward": loss_b,
            "loss_consistency": loss_c,
            "imputations": imputations,
            "imputations_forward": ret_f["imputations"],
            "imputations_backward": ret_b["imputations"],
        }

        # Average predictions if available
        if ret_f.get("predictions") is not None and ret_b.get("predictions") is not None:
            result["predictions"] = (ret_f["predictions"] + ret_b["predictions"]) / 2

        # Copy evaluation data from forward pass
        for key in ["evals", "eval_masks", "labels", "is_train"]:
            if key in ret_f:
                result[key] = ret_f[key]

        return result

    def _get_consistency_loss(
        self, pred_f: torch.Tensor, pred_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss between forward and backward imputations.

        This encourages the forward and backward models to produce similar
        imputations, which serves as a regularization.
        """
        loss = torch.abs(pred_f - pred_b).mean() * self.consistency_weight
        return loss

    def impute(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Impute missing values in a time series.

        This is a convenience method for inference that only returns imputations.

        Args:
            values: Input tensor with missing values filled with 0
            masks: Binary mask (1 for observed, 0 for missing)
            deltas: Time gaps (computed automatically if not provided)

        Returns:
            Imputed time series
        """
        self.eval()

        with torch.no_grad():
            if deltas is None:
                deltas = self._compute_deltas(masks)

            result = self.forward(values=values, masks=masks, deltas=deltas)

        return result["imputations"]

    def _compute_deltas(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute time gaps from masks for forward direction."""
        batch_size, seq_len, input_size = masks.shape
        device = masks.device

        deltas = torch.zeros_like(masks)

        for t in range(seq_len):
            if t == 0:
                deltas[:, t, :] = torch.ones(batch_size, input_size, device=device)
            else:
                deltas[:, t, :] = (
                    torch.ones(batch_size, input_size, device=device)
                    + (1 - masks[:, t - 1, :]) * deltas[:, t - 1, :]
                )

        return deltas
