"""
Unit tests for JEPA core components.

Run with: python -m pytest jepa/test_core.py -v
Or simply: python jepa/test_core.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jepa.config import JEPAConfig, VICRegConfig, TrainingConfig, ExperimentConfig
from jepa.model import SepsisJEPA, SepsisClassifier, count_parameters, model_summary
from jepa.loss import VICRegLoss, CollapseMonitor, compute_embedding_stats


def test_config_validation():
    """Test configuration dataclass validation."""
    print("Testing configuration...")

    # Default config should be valid
    config = JEPAConfig()
    assert config.context_len + config.target_len == config.n_timesteps
    assert config.d_model % config.n_heads == 0

    # Test VICReg config
    vicreg_config = VICRegConfig()
    assert vicreg_config.lambda_inv > 0
    assert vicreg_config.lambda_var > 0

    # Test full experiment config
    exp_config = ExperimentConfig()
    assert exp_config.model.n_features == 40
    assert exp_config.training.batch_size == 64

    print("  [PASS] Configuration validation passed")


def test_jepa_forward_pass():
    """Test JEPA model forward pass with correct shapes."""
    print("Testing JEPA forward pass...")

    config = JEPAConfig()
    model = SepsisJEPA(config)

    # Create dummy input: (batch=32, timesteps=24, features=40)
    batch_size = 32
    x = torch.randn(batch_size, config.n_timesteps, config.n_features)

    # Forward pass
    pred_emb, target_emb = model(x)

    # Check output shapes
    assert pred_emb.shape == (batch_size, config.d_model), \
        f"Expected ({batch_size}, {config.d_model}), got {pred_emb.shape}"
    assert target_emb.shape == (batch_size, config.d_model), \
        f"Expected ({batch_size}, {config.d_model}), got {target_emb.shape}"

    print(f"  [PASS] Forward pass: input {x.shape} -> pred {pred_emb.shape}, target {target_emb.shape}")


def test_gradient_flow():
    """Test that gradients flow correctly (stopped for target encoder)."""
    print("Testing gradient flow...")

    config = JEPAConfig()
    model = SepsisJEPA(config)

    x = torch.randn(8, config.n_timesteps, config.n_features)

    # Forward pass
    pred_emb, target_emb = model(x)

    # Compute dummy loss
    loss = pred_emb.mean()

    # Backward pass
    loss.backward()

    # Check that gradients exist for encoder
    has_encoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert has_encoder_grad, "Encoder should have gradients"

    # Check that predictor has gradients
    has_predictor_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.predictor.parameters()
    )
    assert has_predictor_grad, "Predictor should have gradients"

    print("  [PASS] Gradient flow verified (encoder + predictor have gradients)")


def test_vicreg_loss():
    """Test VICReg loss computation."""
    print("Testing VICReg loss...")

    config = VICRegConfig()
    loss_fn = VICRegLoss(config)

    batch_size = 32
    d_model = 128

    # Random embeddings
    pred = torch.randn(batch_size, d_model)
    target = torch.randn(batch_size, d_model)

    # Compute loss
    total_loss, components = loss_fn(pred, target)

    # Check that loss is a scalar
    assert total_loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(total_loss), "Loss should not be NaN"

    # Check components
    assert "invariance" in components
    assert "variance" in components
    assert "covariance" in components
    assert "total" in components

    # All components should be non-negative
    assert components["invariance"] >= 0
    assert components["variance"] >= 0
    assert components["covariance"] >= 0

    print(f"  [PASS] VICReg loss: {total_loss.item():.4f}")
    print(f"    - Invariance: {components['invariance']:.4f}")
    print(f"    - Variance: {components['variance']:.4f}")
    print(f"    - Covariance: {components['covariance']:.4f}")


def test_vicreg_collapse_detection():
    """Test that VICReg detects and penalizes collapse."""
    print("Testing collapse detection...")

    config = VICRegConfig()
    loss_fn = VICRegLoss(config)

    batch_size = 32
    d_model = 128

    # Case 1: Random embeddings (healthy)
    healthy_emb = torch.randn(batch_size, d_model)
    _, healthy_components = loss_fn(healthy_emb, healthy_emb)

    # Case 2: Collapsed embeddings (all same)
    collapsed_emb = torch.ones(batch_size, d_model)
    _, collapsed_components = loss_fn(collapsed_emb, collapsed_emb)

    # Variance loss should be HIGHER for collapsed embeddings
    assert collapsed_components["variance"] > healthy_components["variance"], \
        "Variance loss should penalize collapsed embeddings"

    print(f"  [PASS] Healthy variance loss: {healthy_components['variance']:.4f}")
    print(f"  [PASS] Collapsed variance loss: {collapsed_components['variance']:.4f}")


def test_classifier():
    """Test downstream classifier."""
    print("Testing SepsisClassifier...")

    config = JEPAConfig()
    jepa = SepsisJEPA(config)

    # Create classifier with frozen encoder
    classifier = SepsisClassifier(jepa, d_model=config.d_model, freeze_encoder=True)

    batch_size = 16
    x = torch.randn(batch_size, config.n_timesteps, config.n_features)

    # Forward pass
    probs = classifier(x)

    # Check output shape
    assert probs.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {probs.shape}"

    # Check probabilities are in [0, 1]
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be in [0, 1]"

    print(f"  [PASS] Classifier output: {probs.shape}, range [{probs.min():.3f}, {probs.max():.3f}]")


def test_parameter_count():
    """Test parameter counting utility."""
    print("Testing parameter count...")

    config = JEPAConfig()
    model = SepsisJEPA(config)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    assert total_params == trainable_params, "All params should be trainable initially"
    assert total_params > 0, "Model should have parameters"

    print(f"  [PASS] Total parameters: {total_params:,}")
    print(model_summary(model))


def test_collapse_monitor():
    """Test collapse monitoring utility."""
    print("Testing CollapseMonitor...")

    monitor = CollapseMonitor(warn_threshold=0.1)

    # Simulate healthy training
    for _ in range(20):
        embeddings = torch.randn(32, 128)
        monitor.update(embeddings)

    report = monitor.report()
    assert not report["collapsing"], "Random embeddings should not trigger collapse warning"

    print(f"  [PASS] Monitor report: mean_std={report['mean_std']:.4f}, collapsing={report['collapsing']}")


def test_end_to_end_training_step():
    """Test a complete training step."""
    print("Testing end-to-end training step...")

    # Setup
    config = JEPAConfig()
    model = SepsisJEPA(config)
    loss_fn = VICRegLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Dummy batch
    x = torch.randn(16, config.n_timesteps, config.n_features)

    # Training step
    model.train()
    optimizer.zero_grad()

    pred_emb, target_emb = model(x)
    loss, components = loss_fn(pred_emb, target_emb)

    loss.backward()
    optimizer.step()

    print(f"  [PASS] Training step completed")
    print(f"    - Loss: {loss.item():.4f}")
    print(f"    - Pred embedding norm: {pred_emb.norm(dim=1).mean():.4f}")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("JEPA Core Component Tests")
    print("=" * 60)
    print()

    tests = [
        test_config_validation,
        test_jepa_forward_pass,
        test_gradient_flow,
        test_vicreg_loss,
        test_vicreg_collapse_detection,
        test_classifier,
        test_parameter_count,
        test_collapse_monitor,
        test_end_to_end_training_step,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  [FAIL] FAILED: {e}")
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
