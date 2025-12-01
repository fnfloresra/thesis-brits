"""
Data loading utilities for BRITS time series imputation.

This module provides dataset classes and data loaders for training
and evaluating BRITS models on time series data with missing values.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data with missing values.

    This dataset handles:
    - Values: The observed time series data (missing values should be NaN)
    - Masks: Binary masks indicating observed (1) vs missing (0) values
    - Deltas: Time gaps since last observation for each feature
    - Labels: Optional classification labels for supervised tasks
    """

    def __init__(
        self,
        data: np.ndarray,
        masks: np.ndarray = None,
        labels: np.ndarray = None,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ):
        """
        Initialize the TimeSeriesDataset.

        Args:
            data: Time series data of shape (n_samples, seq_len, n_features)
                  Missing values should be represented as NaN
            masks: Optional binary masks of shape (n_samples, seq_len, n_features)
                   If not provided, will be computed from NaN values in data
            labels: Optional classification labels of shape (n_samples,)
            val_ratio: Ratio of samples to use for validation
            random_seed: Random seed for reproducibility
        """
        super(TimeSeriesDataset, self).__init__()

        self.data = data.astype(np.float32)
        self.n_samples, self.seq_len, self.n_features = self.data.shape

        # Compute masks from NaN values if not provided
        if masks is None:
            self.masks = (~np.isnan(self.data)).astype(np.float32)
        else:
            self.masks = masks.astype(np.float32)

        # Store original data for evaluation (before filling NaN)
        self.evals = np.nan_to_num(self.data.copy(), nan=0.0)
        self.eval_masks = self.masks.copy()

        # Fill NaN values with 0 for model input
        self.values = np.nan_to_num(self.data, nan=0.0)

        # Compute time deltas
        self.deltas = self._compute_deltas(self.masks)

        # Labels
        self.labels = labels.astype(np.float32) if labels is not None else None

        # Train/validation split
        np.random.seed(random_seed)
        indices = np.arange(self.n_samples)
        val_indices = np.random.choice(
            indices, size=int(self.n_samples * val_ratio), replace=False
        )
        self.val_indices = set(val_indices.tolist())

    def _compute_deltas(self, masks: np.ndarray) -> np.ndarray:
        """
        Compute time gaps (deltas) from observation masks.

        For each position, delta indicates how many time steps have passed
        since the last observation of that feature.
        """
        deltas = np.zeros_like(masks)

        for i in range(self.n_samples):
            for j in range(self.n_features):
                delta = 1.0
                for t in range(self.seq_len):
                    if t == 0:
                        deltas[i, t, j] = 1.0
                    else:
                        if masks[i, t - 1, j] == 1:
                            delta = 1.0
                        else:
                            delta += 1.0
                        deltas[i, t, j] = delta

        return deltas.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - values: Observed values (NaN filled with 0)
                - masks: Binary observation mask
                - deltas: Time gaps
                - evals: Ground truth values
                - eval_masks: Evaluation mask
                - label: Classification label (if available)
                - is_train: 1 if training sample, 0 if validation
        """
        sample = {
            "values": torch.from_numpy(self.values[idx]),
            "masks": torch.from_numpy(self.masks[idx]),
            "deltas": torch.from_numpy(self.deltas[idx]),
            "evals": torch.from_numpy(self.evals[idx]),
            "eval_masks": torch.from_numpy(self.eval_masks[idx]),
            "is_train": torch.tensor(0.0 if idx in self.val_indices else 1.0),
        }

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx])

        return sample


class SyntheticTimeSeriesDataset(Dataset):
    """
    Generate synthetic time series data with missing values for testing.

    This is useful for validating the model without external data.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 48,
        n_features: int = 35,
        missing_rate: float = 0.2,
        eval_rate: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Generate synthetic dataset.

        Args:
            n_samples: Number of samples
            seq_len: Sequence length
            n_features: Number of features
            missing_rate: Fraction of values to make missing
            eval_rate: Fraction of observed values to hold out for evaluation
            random_seed: Random seed
        """
        super(SyntheticTimeSeriesDataset, self).__init__()

        np.random.seed(random_seed)

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_features = n_features

        # Generate smooth time series using random walk
        self.ground_truth = self._generate_smooth_series()

        # Create missing pattern
        self.missing_mask = np.random.random((n_samples, seq_len, n_features)) > missing_rate

        # Create evaluation mask (subset of observed values)
        eval_candidates = np.where(self.missing_mask)
        n_eval = int(len(eval_candidates[0]) * eval_rate)
        eval_indices = np.random.choice(len(eval_candidates[0]), n_eval, replace=False)

        self.eval_masks = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        for idx in eval_indices:
            i, t, j = eval_candidates[0][idx], eval_candidates[1][idx], eval_candidates[2][idx]
            self.eval_masks[i, t, j] = 1.0
            self.missing_mask[i, t, j] = False  # Hide from model

        # Values with missing pattern applied
        self.values = self.ground_truth.copy()
        self.values[~self.missing_mask] = 0.0

        self.masks = self.missing_mask.astype(np.float32)
        self.evals = self.ground_truth.copy()

        # Compute deltas
        self.deltas = self._compute_deltas()

        # Generate random binary labels for classification task
        self.labels = (np.random.random(n_samples) > 0.5).astype(np.float32)

        # Train/validation split
        indices = np.arange(n_samples)
        val_indices = np.random.choice(indices, size=n_samples // 5, replace=False)
        self.val_indices = set(val_indices.tolist())

    def _generate_smooth_series(self) -> np.ndarray:
        """Generate smooth time series using random walk with momentum."""
        data = np.zeros((self.n_samples, self.seq_len, self.n_features), dtype=np.float32)

        for i in range(self.n_samples):
            for j in range(self.n_features):
                # Start with random initial value
                data[i, 0, j] = np.random.randn()
                velocity = 0.0

                for t in range(1, self.seq_len):
                    # Random walk with momentum
                    velocity = 0.8 * velocity + 0.2 * np.random.randn()
                    data[i, t, j] = data[i, t - 1, j] + 0.1 * velocity

        # Normalize each feature
        mean = data.mean(axis=(0, 1), keepdims=True)
        std = data.std(axis=(0, 1), keepdims=True) + 1e-6
        data = (data - mean) / std

        return data

    def _compute_deltas(self) -> np.ndarray:
        """Compute time deltas based on masks."""
        deltas = np.zeros((self.n_samples, self.seq_len, self.n_features), dtype=np.float32)

        for i in range(self.n_samples):
            for j in range(self.n_features):
                delta = 1.0
                for t in range(self.seq_len):
                    if t == 0:
                        deltas[i, t, j] = 1.0
                    else:
                        if self.masks[i, t - 1, j] == 1:
                            delta = 1.0
                        else:
                            delta += 1.0
                        deltas[i, t, j] = delta

        return deltas

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "values": torch.from_numpy(self.values[idx]),
            "masks": torch.from_numpy(self.masks[idx]),
            "deltas": torch.from_numpy(self.deltas[idx]),
            "evals": torch.from_numpy(self.evals[idx]),
            "eval_masks": torch.from_numpy(self.eval_masks[idx]),
            "label": torch.tensor(self.labels[idx]),
            "is_train": torch.tensor(0.0 if idx in self.val_indices else 1.0),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Stacks individual samples into batched tensors.
    """
    result = {}

    # Stack all tensors
    for key in batch[0].keys():
        result[key] = torch.stack([sample[key] for sample in batch])

    return result


def get_data_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
