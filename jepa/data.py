"""
PyTorch Dataset and DataLoader utilities for JEPA training.

Handles loading V3/V6 tensor files and creating train/val/test splits
with subject-aware splitting to prevent data leakage.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from jepa.config import DataConfig


class SepsisDataset(Dataset):
    """PyTorch Dataset for sepsis prediction tensors.

    Loads pre-built numpy tensors and converts to PyTorch format.
    Supports both .npz (full dataset) and .npy (separate files) formats.

    Args:
        X: Feature tensor of shape (n_samples, n_timesteps, n_features).
        y: Label tensor of shape (n_samples,).
        subjects: Optional subject IDs for patient-aware operations.
        transform: Optional transform to apply to features.

    Example:
        >>> dataset = SepsisDataset.from_npz("sepsis_model_v3/tensors.npz")
        >>> x, y = dataset[0]
        >>> x.shape
        torch.Size([24, 32])
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.subjects = subjects
        self.transform = transform

        # Store metadata
        self.n_samples = len(X)
        self.n_timesteps = X.shape[1]
        self.n_features = X.shape[2]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    @classmethod
    def from_npz(cls, path: str, keys: Dict[str, str] = None) -> "SepsisDataset":
        """Load dataset from .npz file.

        Args:
            path: Path to .npz file.
            keys: Dict mapping expected keys to actual keys in file.
                  Default: {"X": "X", "y": "y", "subjects": "subjects"}

        Returns:
            SepsisDataset instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tensor file not found: {path}")

        default_keys = {"X": "X", "y": "y", "subjects": "subjects"}
        keys = keys or default_keys

        data = np.load(path)

        X = data[keys["X"]]
        y = data[keys["y"]]
        subjects = data.get(keys["subjects"], None)

        print(f"Loaded dataset from {path}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Sepsis cases: {int(y.sum())} ({y.mean()*100:.1f}%)")

        return cls(X, y, subjects)

    @classmethod
    def from_npy(
        cls,
        x_path: str,
        y_path: str,
        subjects_path: Optional[str] = None
    ) -> "SepsisDataset":
        """Load dataset from separate .npy files.

        Args:
            x_path: Path to features .npy file.
            y_path: Path to labels .npy file.
            subjects_path: Optional path to subjects .npy file.

        Returns:
            SepsisDataset instance.
        """
        X = np.load(x_path)
        y = np.load(y_path)
        subjects = np.load(subjects_path) if subjects_path else None

        print(f"Loaded dataset from .npy files")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")

        return cls(X, y, subjects)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data.

        Returns:
            Tensor of shape (2,) with weights for [negative, positive] classes.
        """
        n_pos = self.y.sum().item()
        n_neg = len(self.y) - n_pos

        # Inverse frequency weighting
        weight_pos = len(self.y) / (2 * n_pos) if n_pos > 0 else 1.0
        weight_neg = len(self.y) / (2 * n_neg) if n_neg > 0 else 1.0

        return torch.tensor([weight_neg, weight_pos])


def subject_aware_split(
    dataset: SepsisDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset ensuring no subject appears in multiple splits.

    This prevents data leakage when the same patient has multiple ICU stays.

    Args:
        dataset: SepsisDataset with subjects array.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (test gets remainder).
        seed: Random seed.

    Returns:
        Tuple of (train_subset, val_subset, test_subset).
    """
    if dataset.subjects is None:
        print("Warning: No subjects array, using random split")
        return random_split(dataset, train_ratio, val_ratio, seed)

    np.random.seed(seed)

    # Get unique subjects
    unique_subjects = np.unique(dataset.subjects)
    n_subjects = len(unique_subjects)

    # Shuffle subjects
    shuffled_subjects = np.random.permutation(unique_subjects)

    # Split subjects
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = set(shuffled_subjects[:n_train])
    val_subjects = set(shuffled_subjects[n_train:n_train + n_val])
    test_subjects = set(shuffled_subjects[n_train + n_val:])

    # Get indices for each split
    train_idx = [i for i, s in enumerate(dataset.subjects) if s in train_subjects]
    val_idx = [i for i, s in enumerate(dataset.subjects) if s in val_subjects]
    test_idx = [i for i, s in enumerate(dataset.subjects) if s in test_subjects]

    print(f"Subject-aware split:")
    print(f"  Train: {len(train_idx)} samples ({len(train_subjects)} subjects)")
    print(f"  Val:   {len(val_idx)} samples ({len(val_subjects)} subjects)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_subjects)} subjects)")

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def random_split(
    dataset: SepsisDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Random split without subject awareness.

    Args:
        dataset: SepsisDataset to split.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        Tuple of (train_subset, val_subset, test_subset).
    """
    n = len(dataset)
    indices = list(range(n))

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=seed
    )

    # Second split: val vs test
    relative_val = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=relative_val, random_state=seed
    )

    print(f"Random split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train/val/test datasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Optional test dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Dict with 'train', 'val', and optionally 'test' DataLoaders.
    """
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Important for batch norm
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders


def load_data_for_pretraining(
    data_path: str = "sepsis_model_v3/tensors.npz",
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], SepsisDataset]:
    """Convenience function to load data for JEPA pre-training.

    Args:
        data_path: Path to tensor file.
        batch_size: Training batch size.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (dataloaders dict, full dataset).
    """
    # Load dataset
    dataset = SepsisDataset.from_npz(data_path)

    # Split with subject awareness
    train_data, val_data, test_data = subject_aware_split(
        dataset, train_ratio, val_ratio, seed
    )

    # Create dataloaders
    loaders = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return loaders, dataset


class DataAugmentation:
    """Data augmentation transforms for clinical time-series.

    These augmentations help prevent overfitting during self-supervised
    pre-training by creating diverse views of the same patient trajectory.
    """

    @staticmethod
    def temporal_jitter(x: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
        """Randomly shift the time series by a few timesteps.

        Args:
            x: Input tensor of shape (timesteps, features).
            max_shift: Maximum shift in either direction.

        Returns:
            Shifted tensor (with zero padding).
        """
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return x

        result = torch.zeros_like(x)
        if shift > 0:
            result[shift:] = x[:-shift]
        else:
            result[:shift] = x[-shift:]

        return result

    @staticmethod
    def feature_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Randomly zero out some features.

        Args:
            x: Input tensor of shape (timesteps, features).
            p: Probability of dropping each feature.

        Returns:
            Tensor with some features zeroed.
        """
        mask = torch.rand(x.shape[1]) > p
        return x * mask.unsqueeze(0)

    @staticmethod
    def gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise to the input.

        Args:
            x: Input tensor.
            std: Standard deviation of noise.

        Returns:
            Noisy tensor.
        """
        return x + torch.randn_like(x) * std


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Try V3 tensors
    v3_path = Path("sepsis_model_v3/tensors.npz")
    if v3_path.exists():
        loaders, dataset = load_data_for_pretraining(str(v3_path), batch_size=32)

        print(f"\nDataset info:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Timesteps: {dataset.n_timesteps}")
        print(f"  Features: {dataset.n_features}")

        # Test batch
        for batch_x, batch_y in loaders["train"]:
            print(f"\nBatch shapes:")
            print(f"  X: {batch_x.shape}")
            print(f"  y: {batch_y.shape}")
            break

        print("\n[PASS] Data loading test passed!")
    else:
        print(f"V3 tensors not found at {v3_path}")
