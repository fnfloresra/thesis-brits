"""
Training script for BRITS time series imputation model.

Usage:
    python train.py --epochs 100 --batch_size 64 --hid_size 108

For a quick test with synthetic data:
    python train.py --synthetic --epochs 10 --batch_size 32
"""

import argparse
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim

from brits import BRITS, RITS
from brits.data import get_data_loader, SyntheticTimeSeriesDataset
from brits.utils import to_device, compute_metrics, EarlyStopping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BRITS model for time series imputation")

    # Model arguments
    parser.add_argument("--model", type=str, default="brits", choices=["brits", "rits"],
                        help="Model type (brits or rits)")
    parser.add_argument("--hid_size", type=int, default=64,
                        help="RNN hidden size")
    parser.add_argument("--impute_weight", type=float, default=1.0,
                        help="Weight for imputation loss")
    parser.add_argument("--label_weight", type=float, default=0.0,
                        help="Weight for classification loss")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Dropout probability")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")

    # Data arguments
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of synthetic samples")
    parser.add_argument("--seq_len", type=int, default=48,
                        help="Sequence length")
    parser.add_argument("--n_features", type=int, default=35,
                        help="Number of features")
    parser.add_argument("--missing_rate", type=float, default=0.2,
                        help="Missing rate for synthetic data")

    # Output arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def train_epoch(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in data_loader:
        batch = to_device(batch, device)

        optimizer.zero_grad()

        # Forward pass (both BRITS and RITS share the same interface)
        result = model(
            values=batch["values"],
            masks=batch["masks"],
            deltas=batch["deltas"],
            evals=batch["evals"],
            eval_masks=batch["eval_masks"],
            labels=batch.get("label"),
            is_train=batch.get("is_train"),
        )

        loss = result["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_imputations = []
    all_evals = []
    all_eval_masks = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = to_device(batch, device)

            # Forward pass (both BRITS and RITS share the same interface)
            result = model(
                values=batch["values"],
                masks=batch["masks"],
                deltas=batch["deltas"],
                evals=batch["evals"],
                eval_masks=batch["eval_masks"],
                labels=batch.get("label"),
                is_train=batch.get("is_train"),
            )

            total_loss += result["loss"].item()
            n_batches += 1

            all_imputations.append(result["imputations"].cpu().numpy())
            all_evals.append(batch["evals"].cpu().numpy())
            all_eval_masks.append(batch["eval_masks"].cpu().numpy())

    # Compute imputation metrics
    imputations = np.concatenate(all_imputations, axis=0)
    evals = np.concatenate(all_evals, axis=0)
    eval_masks = np.concatenate(all_eval_masks, axis=0)

    metrics = compute_metrics(imputations, evals, eval_masks)
    metrics["loss"] = total_loss / n_batches

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    if args.synthetic:
        print("Using synthetic data for testing...")
        dataset = SyntheticTimeSeriesDataset(
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            n_features=args.n_features,
            missing_rate=args.missing_rate,
            random_seed=args.seed,
        )
    else:
        raise ValueError(
            "Please provide data or use --synthetic flag for synthetic data. "
            "See the documentation for how to prepare your own dataset."
        )

    # Create data loader
    data_loader = get_data_loader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    if args.model == "brits":
        model = BRITS(
            input_size=args.n_features,
            rnn_hid_size=args.hid_size,
            impute_weight=args.impute_weight,
            label_weight=args.label_weight,
            seq_len=args.seq_len,
            dropout=args.dropout,
        )
    else:
        model = RITS(
            input_size=args.n_features,
            rnn_hid_size=args.hid_size,
            impute_weight=args.impute_weight,
            label_weight=args.label_weight,
            seq_len=args.seq_len,
            dropout=args.dropout,
        )

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}")
    print(f"Total trainable parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode="min")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    best_mae = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, data_loader, optimizer, device)

        # Evaluate
        eval_metrics = evaluate(model, data_loader, device)

        epoch_time = time.time() - start_time

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"MAE: {eval_metrics['mae']:.4f} | "
            f"MRE: {eval_metrics['mre']:.4f} | "
            f"RMSE: {eval_metrics['rmse']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if eval_metrics["mae"] < best_mae:
            best_mae = eval_metrics["mae"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mae": best_mae,
                    "args": args,
                },
                os.path.join(args.save_dir, f"best_{args.model}.pt"),
            )

        # Early stopping
        if early_stopping(eval_metrics["mae"]):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print(f"\nTraining complete. Best MAE: {best_mae:.4f}")
    print(f"Model saved to: {os.path.join(args.save_dir, f'best_{args.model}.pt')}")


if __name__ == "__main__":
    main()
