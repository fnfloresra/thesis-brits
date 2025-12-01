"""
Unit tests for BRITS time series imputation models.
"""

import pytest
import torch
import numpy as np

from brits import BRITS, RITS
from brits.data import TimeSeriesDataset, SyntheticTimeSeriesDataset, get_data_loader
from brits.utils import compute_metrics, to_device


# Test configuration
BATCH_SIZE = 8
SEQ_LEN = 16
N_FEATURES = 10
RNN_HID_SIZE = 32


class TestRITS:
    """Tests for the RITS model."""

    def test_forward_pass(self):
        """Test that RITS forward pass produces correct output shapes."""
        model = RITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        values = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        masks = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        deltas = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)

        result = model(values=values, masks=masks, deltas=deltas)

        assert "imputations" in result
        assert "loss" in result
        assert result["imputations"].shape == (BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert result["loss"].dim() == 0  # Scalar loss

    def test_with_missing_values(self):
        """Test RITS with actual missing values."""
        model = RITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        values = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        masks = (torch.rand(BATCH_SIZE, SEQ_LEN, N_FEATURES) > 0.2).float()
        values = values * masks  # Zero out missing values
        deltas = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)

        result = model(values=values, masks=masks, deltas=deltas)

        assert result["imputations"].shape == (BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert not torch.isnan(result["imputations"]).any()

    def test_backward_pass(self):
        """Test that RITS can compute gradients."""
        model = RITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        values = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        masks = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        deltas = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)

        result = model(values=values, masks=masks, deltas=deltas)
        result["loss"].backward()

        # Check at least some gradients exist (not all params may be used)
        has_grad = any(
            param.grad is not None for param in model.parameters() if param.requires_grad
        )
        assert has_grad, "No gradients were computed"


class TestBRITS:
    """Tests for the BRITS model."""

    def test_forward_pass(self):
        """Test that BRITS forward pass produces correct output shapes."""
        model = BRITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        values = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        masks = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        deltas = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)

        result = model(values=values, masks=masks, deltas=deltas)

        assert "imputations" in result
        assert "loss" in result
        assert "imputations_forward" in result
        assert "imputations_backward" in result
        assert result["imputations"].shape == (BATCH_SIZE, SEQ_LEN, N_FEATURES)

    def test_bidirectional_consistency(self):
        """Test that forward and backward imputations have same shape."""
        model = BRITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        values = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        masks = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        deltas = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)

        result = model(values=values, masks=masks, deltas=deltas)

        assert result["imputations_forward"].shape == result["imputations_backward"].shape
        assert result["imputations"].shape == result["imputations_forward"].shape

    def test_impute_method(self):
        """Test the convenience impute method."""
        model = BRITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )
        model.eval()

        values = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        masks = (torch.rand(BATCH_SIZE, SEQ_LEN, N_FEATURES) > 0.2).float()
        values = values * masks

        imputations = model.impute(values=values, masks=masks)

        assert imputations.shape == values.shape
        assert not torch.isnan(imputations).any()

    def test_backward_pass(self):
        """Test that BRITS can compute gradients."""
        model = BRITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        values = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        masks = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        deltas = torch.ones(BATCH_SIZE, SEQ_LEN, N_FEATURES)

        result = model(values=values, masks=masks, deltas=deltas)
        result["loss"].backward()

        # Check at least some gradients exist (not all params may be used)
        has_grad = any(
            param.grad is not None for param in model.parameters() if param.requires_grad
        )
        assert has_grad, "No gradients were computed"


class TestDataLoader:
    """Tests for data loading utilities."""

    def test_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        dataset = SyntheticTimeSeriesDataset(
            n_samples=100,
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
            missing_rate=0.2,
        )

        assert len(dataset) == 100

        sample = dataset[0]
        assert "values" in sample
        assert "masks" in sample
        assert "deltas" in sample
        assert sample["values"].shape == (SEQ_LEN, N_FEATURES)

    def test_data_loader(self):
        """Test data loader creation and iteration."""
        dataset = SyntheticTimeSeriesDataset(
            n_samples=100,
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
        )

        loader = get_data_loader(dataset, batch_size=BATCH_SIZE)

        batch = next(iter(loader))
        assert batch["values"].shape == (BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert batch["masks"].shape == (BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert batch["deltas"].shape == (BATCH_SIZE, SEQ_LEN, N_FEATURES)

    def test_time_series_dataset(self):
        """Test TimeSeriesDataset with NumPy data."""
        data = np.random.randn(100, SEQ_LEN, N_FEATURES).astype(np.float32)
        # Introduce some NaN values
        data[np.random.rand(*data.shape) < 0.2] = np.nan

        dataset = TimeSeriesDataset(data)

        assert len(dataset) == 100
        sample = dataset[0]
        assert not torch.isnan(sample["values"]).any()
        assert sample["masks"].sum() > 0


class TestUtils:
    """Tests for utility functions."""

    def test_compute_metrics(self):
        """Test metric computation."""
        imputations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        eval_masks = np.array([1, 1, 1, 1, 1])

        metrics = compute_metrics(imputations, ground_truth, eval_masks)

        assert "mae" in metrics
        assert "mre" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0

    def test_to_device(self):
        """Test device transfer utility."""
        tensor = torch.randn(10, 10)
        data = {"a": tensor, "b": [tensor, tensor]}

        result = to_device(data, torch.device("cpu"))

        assert isinstance(result, dict)
        assert isinstance(result["b"], list)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_training_step(self):
        """Test a complete training step."""
        # Create model
        model = BRITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        # Create data
        dataset = SyntheticTimeSeriesDataset(
            n_samples=50,
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
        )
        loader = get_data_loader(dataset, batch_size=BATCH_SIZE)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training step
        model.train()
        batch = next(iter(loader))

        optimizer.zero_grad()
        result = model(
            values=batch["values"],
            masks=batch["masks"],
            deltas=batch["deltas"],
        )
        result["loss"].backward()
        optimizer.step()

        assert result["loss"].item() > 0

    def test_imputation_improves_over_epochs(self):
        """Test that imputation quality improves with training."""
        # Create model
        model = BRITS(
            input_size=N_FEATURES,
            rnn_hid_size=RNN_HID_SIZE,
            seq_len=SEQ_LEN,
        )

        # Create data
        dataset = SyntheticTimeSeriesDataset(
            n_samples=100,
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
            missing_rate=0.3,
        )
        loader = get_data_loader(dataset, batch_size=BATCH_SIZE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Track losses
        initial_loss = None
        final_loss = None

        for epoch in range(5):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                optimizer.zero_grad()
                result = model(
                    values=batch["values"],
                    masks=batch["masks"],
                    deltas=batch["deltas"],
                    evals=batch["evals"],
                    eval_masks=batch["eval_masks"],
                )
                result["loss"].backward()
                optimizer.step()

                epoch_loss += result["loss"].item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss

        # Loss should decrease
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
