"""
JEPA Model Architecture for Sepsis Prediction.

Implements:
- SepsisJEPA: Main JEPA model with Transformer encoder and MLP predictor
- SepsisClassifier: Downstream classifier using pre-trained encoder

Architecture follows the I-JEPA paper with adaptations for clinical time-series.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.config import JEPAConfig


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequences.

    Unlike sinusoidal encoding, learnable positions can adapt to the specific
    temporal patterns in clinical data (e.g., circadian rhythms, treatment schedules).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 24, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class InputProjection(nn.Module):
    """Project raw clinical features to embedding space.

    Includes layer normalization for training stability.

    Args:
        n_features: Number of input features per timestep.
        d_model: Target embedding dimension.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to embedding space.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Projected tensor of shape (batch, seq_len, d_model).
        """
        x = self.projection(x)
        x = self.layer_norm(x)
        return x


class Predictor(nn.Module):
    """MLP predictor for JEPA future state prediction.

    Takes the final context embedding and predicts the target embedding.
    Uses GELU activation following modern transformer conventions.

    Args:
        d_model: Input and output embedding dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future embedding from context embedding.

        Args:
            x: Context embedding of shape (batch, d_model).

        Returns:
            Predicted target embedding of shape (batch, d_model).
        """
        return self.layer_norm(self.mlp(x))


class SepsisJEPA(nn.Module):
    """Joint Embedding Predictive Architecture for Sepsis Prediction.

    The model learns to predict future latent representations from past context,
    forcing it to understand patient physiological dynamics.

    Architecture:
        Input (B, 24, 40) → Projection → PositionalEncoding → TransformerEncoder
            ├── Context (hours 0-19) → Encoder → z_context
            └── Target (hours 20-23) → Encoder (no_grad) → z_target

        z_20 = z_context[:, -1] → Predictor → predicted_z_24
        z_24 = z_target[:, -1] (target)

    Args:
        config: JEPAConfig with model hyperparameters.

    Example:
        >>> config = JEPAConfig()
        >>> model = SepsisJEPA(config)
        >>> x = torch.randn(32, 24, 40)  # batch of 32 patients
        >>> pred_emb, target_emb = model(x)
        >>> pred_emb.shape
        torch.Size([32, 128])
    """

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config

        # Input projection: (B, T, 40) → (B, T, d_model)
        self.input_projection = InputProjection(config.n_features, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.n_timesteps, config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,  # Input shape: (batch, seq, feature)
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        # Predictor MLP: z_context[-1] → predicted z_target[-1]
        self.predictor = Predictor(
            config.d_model, config.predictor_hidden, config.dropout
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representations.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Latent representations of shape (batch, seq_len, d_model).
        """
        # Project to embedding space
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.encoder(x)

        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for JEPA pre-training.

        Splits input into context and target windows, encodes both,
        and predicts target embedding from context.

        Args:
            x: Input tensor of shape (batch, n_timesteps, n_features).
               Expected shape: (B, 24, 40) for sepsis data.

        Returns:
            Tuple of:
                - predicted_embedding: Predicted target state (batch, d_model)
                - target_embedding: Actual target state (batch, d_model)
        """
        context_len = self.config.context_len

        # Split into context (hours 0-19) and target (hours 20-23)
        context = x[:, :context_len, :]  # (B, 20, 40)
        target = x[:, context_len:, :]   # (B, 4, 40)

        # Encode context with gradients
        z_context = self.encode(context)  # (B, 20, d_model)

        # Encode target WITHOUT gradients (target encoder in JEPA)
        with torch.no_grad():
            z_target = self.encode(target)  # (B, 4, d_model)

        # Get final embeddings
        z_context_final = z_context[:, -1, :]  # (B, d_model) - hour 19
        z_target_final = z_target[:, -1, :].detach()  # (B, d_model) - hour 23

        # Predict target from context
        predicted_embedding = self.predictor(z_context_final)  # (B, d_model)

        return predicted_embedding, z_target_final

    def get_context_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get context embeddings for downstream tasks.

        Args:
            x: Input tensor of shape (batch, n_timesteps, n_features).

        Returns:
            Context embeddings of shape (batch, context_len, d_model).
        """
        context = x[:, : self.config.context_len, :]
        return self.encode(context)

    def get_full_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for the full sequence.

        Args:
            x: Input tensor of shape (batch, n_timesteps, n_features).

        Returns:
            Full sequence embeddings of shape (batch, n_timesteps, d_model).
        """
        return self.encode(x)


class SepsisClassifier(nn.Module):
    """Downstream classifier for sepsis prediction using pre-trained JEPA encoder.

    Uses the pre-trained encoder as a feature extractor, adding a classification
    head for binary sepsis prediction.

    Args:
        encoder: Pre-trained SepsisJEPA model (or just the encoder part).
        d_model: Embedding dimension from encoder.
        freeze_encoder: Whether to freeze encoder weights during fine-tuning.
        pooling: Pooling strategy ('mean', 'last', 'cls').

    Example:
        >>> jepa = SepsisJEPA(JEPAConfig())
        >>> # ... pre-train jepa ...
        >>> classifier = SepsisClassifier(jepa, freeze_encoder=True)
        >>> x = torch.randn(32, 24, 40)
        >>> probs = classifier(x)
        >>> probs.shape
        torch.Size([32, 1])
    """

    def __init__(
        self,
        encoder: SepsisJEPA,
        d_model: int = 128,
        freeze_encoder: bool = True,
        pooling: str = "mean",
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.pooling = pooling

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for sepsis classification.

        Args:
            x: Input tensor of shape (batch, n_timesteps, n_features).

        Returns:
            Sepsis probability of shape (batch, 1).
        """
        # Get encoder embeddings
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder.get_full_embedding(x)
        else:
            embeddings = self.encoder.get_full_embedding(x)

        # Pool across time dimension
        if self.pooling == "mean":
            pooled = embeddings.mean(dim=1)  # (B, d_model)
        elif self.pooling == "last":
            pooled = embeddings[:, -1, :]  # (B, d_model)
        elif self.pooling == "cls":
            # Use first position as CLS token (if model was trained with CLS)
            pooled = embeddings[:, 0, :]  # (B, d_model)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Classification
        logits = self.classifier(pooled)  # (B, 1)

        return torch.sigmoid(logits)

    def unfreeze_encoder(self, unfreeze_layers: Optional[int] = None):
        """Unfreeze encoder for fine-tuning.

        Args:
            unfreeze_layers: Number of transformer layers to unfreeze from the top.
                           If None, unfreeze entire encoder.
        """
        self.freeze_encoder = False

        if unfreeze_layers is None:
            # Unfreeze all
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only top N layers
            # First freeze all
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Then unfreeze predictor and top layers
            for param in self.encoder.predictor.parameters():
                param.requires_grad = True

            # Unfreeze top N transformer layers
            n_layers = self.encoder.config.n_layers
            for i in range(n_layers - unfreeze_layers, n_layers):
                for param in self.encoder.encoder.layers[i].parameters():
                    param.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module) -> str:
    """Generate a summary of model architecture.

    Args:
        model: PyTorch model.

    Returns:
        Summary string with parameter counts.
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    summary = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Non-trainable parameters: {total_params - trainable_params:,}",
        "",
        "Layer-wise parameter counts:",
    ]

    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        summary.append(f"  {name}: {n_params:,}")

    return "\n".join(summary)
