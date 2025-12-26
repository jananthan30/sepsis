"""
VICReg Loss Implementation for JEPA.

VICReg (Variance-Invariance-Covariance Regularization) prevents representation
collapse in self-supervised learning without requiring negative samples.

Reference:
    Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning. ICLR 2022.
    https://arxiv.org/abs/2105.04906
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.config import VICRegConfig


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization Loss.

    The loss has three components:
    1. Invariance: MSE between predicted and target embeddings (alignment)
    2. Variance: Force embeddings to have std > gamma per dimension (prevent collapse)
    3. Covariance: Decorrelate embedding dimensions (encourage diverse features)

    Loss = λ_inv * invariance + λ_var * variance + λ_cov * covariance

    Args:
        config: VICRegConfig with loss hyperparameters.

    Example:
        >>> loss_fn = VICRegLoss(VICRegConfig())
        >>> pred = torch.randn(32, 128)  # predicted embeddings
        >>> target = torch.randn(32, 128)  # target embeddings
        >>> loss, components = loss_fn(pred, target)
        >>> print(f"Total loss: {loss.item():.4f}")
        >>> print(f"Invariance: {components['invariance']:.4f}")
    """

    def __init__(self, config: VICRegConfig = None):
        super().__init__()
        if config is None:
            config = VICRegConfig()
        self.config = config

    def invariance_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute invariance loss (MSE between embeddings).

        This encourages the predicted embedding to match the target.

        Args:
            pred: Predicted embeddings of shape (batch, d_model).
            target: Target embeddings of shape (batch, d_model).

        Returns:
            Scalar invariance loss.
        """
        return F.mse_loss(pred, target)

    def variance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute variance loss to prevent representation collapse.

        Forces each embedding dimension to have std > gamma.
        Loss = mean(max(0, gamma - std(x, dim=0)))

        This prevents all embeddings from collapsing to a single point.

        Args:
            x: Embeddings of shape (batch, d_model).

        Returns:
            Scalar variance loss.
        """
        # Compute std across batch dimension for each feature
        std = torch.sqrt(x.var(dim=0) + self.config.var_eps)

        # Hinge loss: penalize if std < gamma
        var_loss = torch.mean(F.relu(self.config.var_gamma - std))

        return var_loss

    def covariance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute covariance loss to decorrelate embedding dimensions.

        Penalizes off-diagonal elements of the covariance matrix.
        This encourages different dimensions to encode different information.

        Args:
            x: Embeddings of shape (batch, d_model).

        Returns:
            Scalar covariance loss.
        """
        batch_size, d_model = x.shape

        # Center the embeddings
        x = x - x.mean(dim=0)

        # Compute covariance matrix: (d_model, d_model)
        cov = (x.T @ x) / (batch_size - 1)

        # Zero out diagonal (we don't penalize variance of individual dimensions)
        # Only penalize off-diagonal correlations
        off_diagonal_mask = ~torch.eye(d_model, dtype=torch.bool, device=x.device)
        off_diagonal = cov[off_diagonal_mask]

        # Sum of squared off-diagonal elements, normalized by d_model
        cov_loss = (off_diagonal ** 2).sum() / d_model

        return cov_loss

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute full VICReg loss.

        Args:
            pred: Predicted embeddings of shape (batch, d_model).
            target: Target embeddings of shape (batch, d_model).

        Returns:
            Tuple of:
                - total_loss: Scalar tensor with combined loss.
                - components: Dict with individual loss components (detached floats).
        """
        # Invariance: alignment between pred and target
        inv_loss = self.invariance_loss(pred, target)

        # Variance: prevent collapse in both pred and target
        var_loss_pred = self.variance_loss(pred)
        var_loss_target = self.variance_loss(target)
        var_loss = (var_loss_pred + var_loss_target) / 2

        # Covariance: decorrelate dimensions in both pred and target
        cov_loss_pred = self.covariance_loss(pred)
        cov_loss_target = self.covariance_loss(target)
        cov_loss = (cov_loss_pred + cov_loss_target) / 2

        # Weighted sum
        total_loss = (
            self.config.lambda_inv * inv_loss
            + self.config.lambda_var * var_loss
            + self.config.lambda_cov * cov_loss
        )

        # Return components for logging (detached)
        components = {
            "invariance": inv_loss.detach().item(),
            "variance": var_loss.detach().item(),
            "covariance": cov_loss.detach().item(),
            "var_pred": var_loss_pred.detach().item(),
            "var_target": var_loss_target.detach().item(),
            "total": total_loss.detach().item(),
        }

        return total_loss, components


class CollapseMonitor:
    """Monitor for representation collapse during training.

    Tracks statistics that indicate if the model is collapsing:
    - Low embedding variance → all outputs becoming similar
    - High off-diagonal covariance → dimensions becoming redundant

    Example:
        >>> monitor = CollapseMonitor()
        >>> for batch in dataloader:
        ...     pred, target = model(batch)
        ...     monitor.update(pred)
        >>> monitor.report()
    """

    def __init__(self, warn_threshold: float = 0.1):
        """Initialize collapse monitor.

        Args:
            warn_threshold: Warn if mean std drops below this value.
        """
        self.warn_threshold = warn_threshold
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.std_history = []
        self.cov_history = []
        self.n_samples = 0

    def update(self, embeddings: torch.Tensor):
        """Update statistics with new embeddings.

        Args:
            embeddings: Embeddings of shape (batch, d_model).
        """
        with torch.no_grad():
            # Per-dimension standard deviation
            std = embeddings.std(dim=0).mean().item()
            self.std_history.append(std)

            # Off-diagonal covariance magnitude
            x = embeddings - embeddings.mean(dim=0)
            cov = (x.T @ x) / (x.shape[0] - 1)
            d_model = embeddings.shape[1]
            off_diag_mask = ~torch.eye(d_model, dtype=torch.bool, device=embeddings.device)
            off_diag_mean = cov[off_diag_mask].abs().mean().item()
            self.cov_history.append(off_diag_mean)

            self.n_samples += embeddings.shape[0]

    def is_collapsing(self) -> bool:
        """Check if embeddings are collapsing.

        Returns:
            True if collapse is detected.
        """
        if len(self.std_history) < 10:
            return False

        recent_std = sum(self.std_history[-10:]) / 10
        return recent_std < self.warn_threshold

    def report(self) -> Dict[str, float]:
        """Generate statistics report.

        Returns:
            Dict with mean std, mean off-diagonal cov, and collapse warning.
        """
        if not self.std_history:
            return {"mean_std": 0, "mean_off_diag_cov": 0, "collapsing": False}

        return {
            "mean_std": sum(self.std_history) / len(self.std_history),
            "recent_std": sum(self.std_history[-10:]) / min(10, len(self.std_history)),
            "mean_off_diag_cov": sum(self.cov_history) / len(self.cov_history),
            "collapsing": self.is_collapsing(),
            "n_samples": self.n_samples,
        }


def compute_embedding_stats(embeddings: torch.Tensor) -> Dict[str, float]:
    """Compute statistics for a batch of embeddings.

    Useful for monitoring training health.

    Args:
        embeddings: Embeddings of shape (batch, d_model).

    Returns:
        Dict with various embedding statistics.
    """
    with torch.no_grad():
        stats = {
            "mean": embeddings.mean().item(),
            "std": embeddings.std().item(),
            "min": embeddings.min().item(),
            "max": embeddings.max().item(),
            "per_dim_std_mean": embeddings.std(dim=0).mean().item(),
            "per_dim_std_min": embeddings.std(dim=0).min().item(),
            "per_dim_std_max": embeddings.std(dim=0).max().item(),
            "l2_norm_mean": embeddings.norm(dim=1).mean().item(),
        }
    return stats
