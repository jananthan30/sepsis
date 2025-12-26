"""
Configuration classes for JEPA model and training.

Defines all hyperparameters as dataclasses for type safety and easy modification.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class JEPAConfig:
    """Model architecture configuration.

    Attributes:
        n_features: Number of input features per timestep (40 for V6).
        n_timesteps: Total number of timesteps in input sequence (24 hours).
        d_model: Embedding dimension for transformer.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        d_ff: Feedforward dimension in transformer (typically 4 * d_model).
        dropout: Dropout probability.
        context_len: Number of timesteps for context window.
        target_len: Number of timesteps for target window.
        predictor_hidden: Hidden dimension for predictor MLP.
    """
    # Input dimensions (from V6 model)
    n_features: int = 40  # 19 clinical + 19 masks + 2 static
    n_timesteps: int = 24  # 24 hours of observation

    # Transformer architecture
    d_model: int = 128  # Embedding dimension
    n_heads: int = 4  # Attention heads (d_model / n_heads = 32 per head)
    n_layers: int = 3  # Transformer encoder layers
    d_ff: int = 512  # Feedforward dimension (4 * d_model)
    dropout: float = 0.1

    # JEPA-specific
    context_len: int = 20  # Hours 0-19 for context
    target_len: int = 4  # Hours 20-23 for target
    predictor_hidden: int = 256  # MLP hidden dimension

    def __post_init__(self):
        """Validate configuration."""
        assert self.context_len + self.target_len == self.n_timesteps, \
            f"context_len ({self.context_len}) + target_len ({self.target_len}) must equal n_timesteps ({self.n_timesteps})"
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


@dataclass
class VICRegConfig:
    """VICReg loss configuration.

    VICReg = Variance-Invariance-Covariance Regularization
    Loss = λ_inv * invariance + λ_var * variance + λ_cov * covariance

    Attributes:
        lambda_inv: Weight for invariance term (MSE between pred and target).
        lambda_var: Weight for variance term (std > threshold per dimension).
        lambda_cov: Weight for covariance term (decorrelation).
        var_eps: Epsilon added to variance for numerical stability.
        var_gamma: Target standard deviation for variance term.
    """
    lambda_inv: float = 1.0  # Invariance weight
    lambda_var: float = 1.0  # Variance weight
    lambda_cov: float = 0.04  # Covariance weight (typically smaller)
    var_eps: float = 1e-4  # Numerical stability
    var_gamma: float = 1.0  # Target std for variance term


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        batch_size: Training batch size.
        epochs_pretrain: Maximum epochs for self-supervised pre-training.
        epochs_finetune: Maximum epochs for supervised fine-tuning.
        lr_pretrain: Learning rate for pre-training.
        lr_finetune: Learning rate for fine-tuning (lower for transfer).
        weight_decay: AdamW weight decay.
        warmup_epochs: Learning rate warmup epochs.
        patience: Early stopping patience.
        min_delta: Minimum improvement for early stopping.
        gradient_clip: Maximum gradient norm (None to disable).
        seed: Random seed for reproducibility.
    """
    batch_size: int = 64
    epochs_pretrain: int = 100
    epochs_finetune: int = 50
    lr_pretrain: float = 1e-4
    lr_finetune: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    patience: int = 15
    min_delta: float = 1e-4
    gradient_clip: Optional[float] = 1.0
    seed: int = 42


@dataclass
class DataConfig:
    """Data configuration.

    Attributes:
        data_path: Path to V6 tensors (.npz file).
        train_ratio: Fraction for training (rest split between val/test).
        val_ratio: Fraction for validation (of non-train data).
        num_workers: DataLoader workers.
        pin_memory: Pin memory for faster GPU transfer.
    """
    data_path: Path = field(default_factory=lambda: Path("sepsis_model_v6/tensors.npz"))
    train_ratio: float = 0.7
    val_ratio: float = 0.5  # 0.5 of remaining 0.3 = 0.15 val, 0.15 test
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs.

    Attributes:
        model: Model architecture configuration.
        vicreg: VICReg loss configuration.
        training: Training configuration.
        data: Data configuration.
        output_dir: Directory for checkpoints and logs.
        experiment_name: Name for this experiment run.
    """
    model: JEPAConfig = field(default_factory=JEPAConfig)
    vicreg: VICRegConfig = field(default_factory=VICRegConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: Path = field(default_factory=lambda: Path("jepa/checkpoints"))
    experiment_name: str = "jepa_v1"

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Default configurations for quick use
DEFAULT_CONFIG = ExperimentConfig()


def get_config(
    d_model: int = 128,
    n_layers: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    **kwargs
) -> ExperimentConfig:
    """Create experiment config with custom overrides.

    Args:
        d_model: Embedding dimension.
        n_layers: Number of transformer layers.
        batch_size: Training batch size.
        lr: Pre-training learning rate.
        **kwargs: Additional overrides for any config field.

    Returns:
        ExperimentConfig with specified overrides.

    Example:
        >>> config = get_config(d_model=256, batch_size=32)
        >>> config.model.d_model
        256
    """
    model_config = JEPAConfig(d_model=d_model, n_layers=n_layers, d_ff=d_model * 4)
    training_config = TrainingConfig(batch_size=batch_size, lr_pretrain=lr)

    return ExperimentConfig(
        model=model_config,
        training=training_config,
        **kwargs
    )
