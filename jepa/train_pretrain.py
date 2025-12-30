"""
JEPA Self-Supervised Pre-training Script.

Trains the JEPA encoder to predict future latent states from past context
using VICReg loss to prevent representation collapse.

Usage:
    python jepa/train_pretrain.py --data sepsis_model_v3/tensors.npz --epochs 100
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from jepa.config import JEPAConfig, VICRegConfig, TrainingConfig
from jepa.model import SepsisJEPA, count_parameters
from jepa.loss import VICRegLoss, CollapseMonitor, compute_embedding_stats
from jepa.data import SepsisDataset, subject_aware_split, create_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="JEPA Pre-training")

    # Data
    parser.add_argument(
        "--data", type=str, default="sepsis_model_v3/tensors.npz",
        help="Path to tensor file (.npz)"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    # Model
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # VICReg
    parser.add_argument("--lambda-inv", type=float, default=1.0)
    parser.add_argument("--lambda-var", type=float, default=1.0)
    parser.add_argument("--lambda-cov", type=float, default=0.04)

    # Output
    parser.add_argument("--output-dir", type=str, default="jepa/checkpoints")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: SepsisJEPA,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: VICRegLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: Optional[float] = None,
    collapse_monitor: Optional[CollapseMonitor] = None,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: JEPA model.
        dataloader: Training dataloader.
        loss_fn: VICReg loss function.
        optimizer: Optimizer.
        device: Compute device.
        gradient_clip: Max gradient norm.
        collapse_monitor: Optional collapse monitor.

    Returns:
        Dict with training metrics.
    """
    model.train()

    total_loss = 0
    total_inv = 0
    total_var = 0
    total_cov = 0
    n_batches = 0

    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_emb, target_emb = model(batch_x)

        # Compute loss
        loss, components = loss_fn(pred_emb, target_emb)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Track metrics
        total_loss += components["total"]
        total_inv += components["invariance"]
        total_var += components["variance"]
        total_cov += components["covariance"]
        n_batches += 1

        # Monitor for collapse
        if collapse_monitor is not None:
            collapse_monitor.update(pred_emb.detach())

    return {
        "loss": total_loss / n_batches,
        "invariance": total_inv / n_batches,
        "variance": total_var / n_batches,
        "covariance": total_cov / n_batches,
    }


def validate(
    model: SepsisJEPA,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: VICRegLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model.

    Args:
        model: JEPA model.
        dataloader: Validation dataloader.
        loss_fn: VICReg loss function.
        device: Compute device.

    Returns:
        Dict with validation metrics.
    """
    model.eval()

    total_loss = 0
    total_inv = 0
    total_var = 0
    total_cov = 0
    n_batches = 0
    all_emb_stats = []

    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)

            # Forward pass
            pred_emb, target_emb = model(batch_x)

            # Compute loss
            loss, components = loss_fn(pred_emb, target_emb)

            # Track metrics
            total_loss += components["total"]
            total_inv += components["invariance"]
            total_var += components["variance"]
            total_cov += components["covariance"]
            n_batches += 1

            # Track embedding stats
            all_emb_stats.append(compute_embedding_stats(pred_emb))

    # Average embedding stats
    avg_emb_stats = {}
    for key in all_emb_stats[0].keys():
        avg_emb_stats[key] = np.mean([s[key] for s in all_emb_stats])

    return {
        "loss": total_loss / n_batches,
        "invariance": total_inv / n_batches,
        "variance": total_var / n_batches,
        "covariance": total_cov / n_batches,
        "emb_std": avg_emb_stats["per_dim_std_mean"],
    }


def save_checkpoint(
    model: SepsisJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    config: dict,
):
    """Save training checkpoint.

    Args:
        model: JEPA model.
        optimizer: Optimizer.
        epoch: Current epoch.
        metrics: Current metrics.
        path: Save path.
        config: Model config dict.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: SepsisJEPA,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, Dict[str, float]]:
    """Load training checkpoint.

    Args:
        path: Checkpoint path.
        model: Model to load into.
        optimizer: Optional optimizer to load into.

    Returns:
        Tuple of (epoch, metrics).
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["metrics"]


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"jepa_{timestamp}"

    exp_dir = output_dir / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JEPA Self-Supervised Pre-training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output: {exp_dir}")

    # Device
    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("\n[1/4] Loading data...")
    dataset = SepsisDataset.from_npz(args.data)

    # Update config based on actual data dimensions
    n_features = dataset.n_features
    n_timesteps = dataset.n_timesteps

    print(f"Data: {len(dataset)} samples, {n_timesteps} timesteps, {n_features} features")

    # Split data
    train_data, val_data, test_data = subject_aware_split(
        dataset, train_ratio=0.7, val_ratio=0.15, seed=args.seed
    )

    # Create dataloaders
    loaders = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build model
    print("\n[2/4] Building model...")
    model_config = JEPAConfig(
        n_features=n_features,
        n_timesteps=n_timesteps,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    model = SepsisJEPA(model_config)
    model = model.to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Loss function
    vicreg_config = VICRegConfig(
        lambda_inv=args.lambda_inv,
        lambda_var=args.lambda_var,
        lambda_cov=args.lambda_cov,
    )
    loss_fn = VICRegLoss(vicreg_config)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # Collapse monitor
    collapse_monitor = CollapseMonitor(warn_threshold=0.1)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_inv": [],
        "train_var": [],
        "train_cov": [],
        "val_emb_std": [],
        "lr": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Save config
    config_dict = {
        "model": vars(model_config),
        "vicreg": vars(vicreg_config),
        "training": vars(args),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Training loop
    print("\n[3/4] Training...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, loaders["train"], loss_fn, optimizer, device,
            gradient_clip=args.gradient_clip,
            collapse_monitor=collapse_monitor,
        )

        # Validate
        val_metrics = validate(model, loaders["val"], loss_fn, device)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_inv"].append(train_metrics["invariance"])
        history["train_var"].append(train_metrics["variance"])
        history["train_cov"].append(train_metrics["covariance"])
        history["val_emb_std"].append(val_metrics["emb_std"])
        history["lr"].append(current_lr)

        # Check for collapse
        collapse_report = collapse_monitor.report()
        if collapse_report["collapsing"]:
            print(f"  WARNING: Representation collapse detected! std={collapse_report['recent_std']:.4f}")

        # Print progress
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Inv: {train_metrics['invariance']:.4f} | "
            f"Var: {train_metrics['variance']:.4f} | "
            f"Cov: {train_metrics['covariance']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best model
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                exp_dir / "best_model.pt",
                config_dict,
            )
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch+1} epochs (patience={args.patience})")
                break

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                exp_dir / f"checkpoint_epoch{epoch+1}.pt",
                config_dict,
            )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_metrics,
        exp_dir / "final_model.pt",
        config_dict,
    )

    # Save history
    np.savez(
        exp_dir / "history.npz",
        **{k: np.array(v) for k, v in history.items()}
    )

    # Final evaluation on test set
    print("\n[4/4] Final evaluation...")
    test_metrics = validate(model, loaders["test"], loss_fn, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Embedding Std: {test_metrics['emb_std']:.4f}")

    # Save test metrics
    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Pre-training Complete!")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Saved to: {exp_dir}")
    print(f"\nTo visualize: python jepa/visualize.py --exp-dir {exp_dir}")


if __name__ == "__main__":
    main()
