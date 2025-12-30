"""
Visualization utilities for JEPA pre-training results.

Generates:
- Training loss curves
- VICReg component curves
- t-SNE/UMAP embeddings
- Embedding statistics

Usage:
    python jepa/visualize.py --exp-dir jepa/checkpoints/jepa_xxx
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from jepa.config import JEPAConfig
from jepa.model import SepsisJEPA
from jepa.data import SepsisDataset


def plot_training_curves(history_path: Path, output_dir: Path):
    """Plot training loss curves.

    Args:
        history_path: Path to history.npz file.
        output_dir: Output directory for figures.
    """
    history = np.load(history_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", linewidth=2)
    ax.plot(history["val_loss"], label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("VICReg Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # VICReg components
    ax = axes[0, 1]
    ax.plot(history["train_inv"], label="Invariance", linewidth=2)
    ax.plot(history["train_var"], label="Variance", linewidth=2)
    ax.plot(history["train_cov"], label="Covariance", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Component")
    ax.set_title("VICReg Components (Training)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Embedding std
    ax = axes[1, 0]
    ax.plot(history["val_emb_std"], linewidth=2, color="green")
    ax.axhline(y=0.1, color="red", linestyle="--", label="Collapse threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Embedding Std")
    ax.set_title("Embedding Variance (Collapse Monitor)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(history["lr"], linewidth=2, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved training curves to {output_dir / 'training_curves.png'}")


def compute_embeddings(
    model: SepsisJEPA,
    dataset: SepsisDataset,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple:
    """Compute embeddings for visualization.

    Args:
        model: Trained JEPA model.
        dataset: Dataset to embed.
        device: Compute device.
        max_samples: Maximum samples to embed.

    Returns:
        Tuple of (embeddings, labels).
    """
    model.eval()

    # Sample if dataset is large
    n = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n, replace=False)

    embeddings = []
    labels = []

    with torch.no_grad():
        for idx in indices:
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)

            # Get full sequence embedding
            emb = model.get_full_embedding(x)
            # Mean pool across time
            emb = emb.mean(dim=1).squeeze().cpu().numpy()

            embeddings.append(emb)
            labels.append(y.item())

    return np.array(embeddings), np.array(labels)


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    perplexity: int = 30,
):
    """Plot t-SNE visualization of embeddings.

    Args:
        embeddings: Embedding array of shape (n_samples, d_model).
        labels: Label array of shape (n_samples,).
        output_path: Output path for figure.
        perplexity: t-SNE perplexity.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn not available for t-SNE, skipping...")
        return

    print("Computing t-SNE (this may take a minute)...")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot by class
    colors = ["#2ecc71", "#e74c3c"]  # Green for control, red for sepsis
    class_names = ["Control", "Sepsis"]

    for label_val in [0, 1]:
        mask = labels == label_val
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=colors[label_val],
            label=class_names[label_val],
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("JEPA Embeddings (t-SNE)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE plot to {output_path}")


def plot_embedding_distributions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
):
    """Plot embedding dimension distributions.

    Args:
        embeddings: Embedding array.
        labels: Label array.
        output_path: Output path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Dimension variance
    ax = axes[0, 0]
    dim_std = embeddings.std(axis=0)
    ax.bar(range(len(dim_std)), dim_std)
    ax.axhline(y=np.mean(dim_std), color="red", linestyle="--", label=f"Mean: {np.mean(dim_std):.3f}")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Std Dev")
    ax.set_title("Per-Dimension Standard Deviation")
    ax.legend()

    # L2 norms by class
    ax = axes[0, 1]
    norms = np.linalg.norm(embeddings, axis=1)
    for label_val, name, color in [(0, "Control", "#2ecc71"), (1, "Sepsis", "#e74c3c")]:
        mask = labels == label_val
        ax.hist(norms[mask], bins=30, alpha=0.6, label=name, color=color)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.set_title("Embedding Norms by Class")
    ax.legend()

    # First 2 dimensions scatter
    ax = axes[1, 0]
    for label_val, name, color in [(0, "Control", "#2ecc71"), (1, "Sepsis", "#e74c3c")]:
        mask = labels == label_val
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], alpha=0.5, label=name, c=color, s=10)
    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")
    ax.set_title("First 2 Embedding Dimensions")
    ax.legend()

    # Correlation matrix (subset of dimensions)
    ax = axes[1, 1]
    n_show = min(32, embeddings.shape[1])
    corr = np.corrcoef(embeddings[:, :n_show].T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    ax.set_title(f"Dimension Correlations (first {n_show})")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved embedding distributions to {output_path}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="JEPA Visualization")
    parser.add_argument(
        "--exp-dir", type=str, required=True,
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--data", type=str, default="sepsis_model_v3/tensors.npz",
        help="Path to data for embedding visualization"
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip embedding visualization (faster)"
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("JEPA Visualization")
    print("=" * 60)
    print(f"Experiment: {exp_dir}")

    # Plot training curves
    history_path = exp_dir / "history.npz"
    if history_path.exists():
        plot_training_curves(history_path, figures_dir)
    else:
        print(f"No history.npz found at {history_path}")

    # Embedding visualization
    if not args.skip_embeddings:
        model_path = exp_dir / "best_model.pt"
        if model_path.exists() and Path(args.data).exists():
            print("\nGenerating embedding visualizations...")

            # Load model
            checkpoint = torch.load(model_path, map_location="cpu")
            config_dict = checkpoint["config"]["model"]

            # Handle both dict and object config
            if isinstance(config_dict, dict):
                model_config = JEPAConfig(**config_dict)
            else:
                model_config = config_dict

            model = SepsisJEPA(model_config)
            model.load_state_dict(checkpoint["model_state_dict"])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Load data
            dataset = SepsisDataset.from_npz(args.data)

            # Compute embeddings
            embeddings, labels = compute_embeddings(model, dataset, device)

            # Plot
            plot_embeddings_tsne(embeddings, labels, figures_dir / "embeddings_tsne.png")
            plot_embedding_distributions(embeddings, labels, figures_dir / "embedding_stats.png")
        else:
            print(f"Skipping embedding visualization (model or data not found)")

    print("\n" + "=" * 60)
    print(f"Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
