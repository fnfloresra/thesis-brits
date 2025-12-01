"""
Utility functions for BRITS time series imputation.
"""

from typing import Dict, Union
import numpy as np
import torch


def to_device(
    data: Union[torch.Tensor, Dict, list],
    device: torch.device = None,
) -> Union[torch.Tensor, Dict, list]:
    """
    Move data to the specified device.

    Args:
        data: Tensor, dictionary of tensors, or list of tensors
        device: Target device. If None, uses CUDA if available.

    Returns:
        Data on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    else:
        return data


def compute_metrics(
    imputations: np.ndarray,
    ground_truth: np.ndarray,
    eval_masks: np.ndarray,
) -> Dict[str, float]:
    """
    Compute imputation metrics.

    Args:
        imputations: Imputed values
        ground_truth: True values
        eval_masks: Binary mask indicating which positions to evaluate

    Returns:
        Dictionary containing:
            - mae: Mean Absolute Error
            - mre: Mean Relative Error
            - rmse: Root Mean Squared Error
    """
    # Flatten and filter to evaluation positions
    mask = eval_masks.flatten() == 1
    pred = imputations.flatten()[mask]
    true = ground_truth.flatten()[mask]

    # Mean Absolute Error
    mae = np.abs(pred - true).mean()

    # Mean Relative Error
    mre = np.abs(pred - true).sum() / (np.abs(true).sum() + 1e-8)

    # Root Mean Squared Error
    rmse = np.sqrt(((pred - true) ** 2).mean())

    return {
        "mae": float(mae),
        "mre": float(mre),
        "rmse": float(rmse),
    }


def introduce_missing(
    data: np.ndarray,
    missing_rate: float = 0.2,
    random_seed: int = None,
) -> tuple:
    """
    Introduce missing values into a complete dataset.

    Args:
        data: Complete data of shape (n_samples, seq_len, n_features)
        missing_rate: Fraction of values to make missing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (values, masks, eval_data, eval_masks)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples, seq_len, n_features = data.shape

    # Create random missing mask
    missing_mask = np.random.random((n_samples, seq_len, n_features)) > missing_rate

    # Values with missing pattern applied
    values = data.copy()
    values[~missing_mask] = 0.0

    masks = missing_mask.astype(np.float32)
    eval_masks = (~missing_mask).astype(np.float32)

    return values, masks, data, eval_masks


def normalize_data(
    data: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
) -> tuple:
    """
    Normalize data to zero mean and unit variance.

    Args:
        data: Data to normalize of shape (..., n_features)
        mean: Feature means. If None, computed from data.
        std: Feature standard deviations. If None, computed from data.

    Returns:
        Tuple of (normalized_data, mean, std)
    """
    if mean is None:
        # Compute mean ignoring NaN values
        mean = np.nanmean(data, axis=tuple(range(data.ndim - 1)))

    if std is None:
        # Compute std ignoring NaN values
        std = np.nanstd(data, axis=tuple(range(data.ndim - 1)))
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero

    normalized = (data - mean) / std

    return normalized, mean, std


def denormalize_data(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Denormalize data back to original scale.

    Args:
        data: Normalized data
        mean: Feature means
        std: Feature standard deviations

    Returns:
        Denormalized data
    """
    return data * std + mean


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
