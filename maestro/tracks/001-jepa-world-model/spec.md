# Specification: JEPA World Model for Sepsis Prediction

## Overview

Implement a **Joint Embedding Predictive Architecture (JEPA)** to create a "World Model" of patient physiology that learns to predict future latent states from past observations. This self-supervised approach aims to capture clinically meaningful state transitions before fine-tuning for sepsis prediction.

## Problem Statement

The current BiLSTM model directly maps clinical time-series to binary sepsis labels. While effective (AUC ~0.71), this supervised approach:

1. **Doesn't learn physiological dynamics** - Only learns features correlated with the label, not underlying state transitions
2. **Data inefficiency** - Only uses labeled examples; cannot leverage unlabeled patient trajectories
3. **Limited transfer** - Model doesn't generalize to other clinical prediction tasks

JEPA addresses these by learning to **predict latent representations of future states**, forcing the model to understand how patient physiology evolves over time.

## Goals

1. **Implement JEPA architecture** in PyTorch as a separate module (`jepa/`)
2. **Self-supervised pre-training** on patient trajectories (predict hour 24 from hours 0-20)
3. **Supervised fine-tuning** with frozen encoder + classification head
4. **A/B comparison** with existing BiLSTM in dashboard
5. **VICReg loss** to prevent representation collapse

## Non-Goals

- Full migration of existing codebase to PyTorch (keep TensorFlow models)
- Replacing the production model immediately
- Real-time inference optimization (focus on research/validation first)
- Multi-GPU training infrastructure

## Acceptance Criteria

- [ ] `jepa/model.py` - SepsisJEPA class with Transformer encoder and predictor
- [ ] `jepa/loss.py` - VICReg loss implementation
- [ ] `jepa/finetune.py` - SepsisClassifier for downstream task
- [ ] `jepa/train_pretrain.py` - Self-supervised pre-training script
- [ ] `jepa/train_finetune.py` - Supervised fine-tuning script
- [ ] `jepa/config.py` - Hyperparameter configuration
- [ ] Pre-trained encoder checkpoint saved
- [ ] Fine-tuned classifier with AUROC metric on test set
- [ ] Comparison metrics added to Streamlit dashboard
- [ ] Documentation with architecture diagrams

## Technical Specification

### Input Data (from V6 Pipeline)

```
Tensor Shape: (Batch, 24, 40)
├── 19 Clinical Values (Z-score normalized)
│   ├── Vitals (7): HR, SBP, DBP, MAP, RR, SpO2, Temp
│   ├── Labs (9): WBC, Creatinine, Platelets, Bilirubin, Glucose, BUN, Na, K, Hgb
│   └── Deltas (3): delta_HR, delta_SBP, delta_Temp
├── 19 Freshness Masks (0/1 indicating measurement recency)
└── 2 Static Features (Age, Gender)
```

### Architecture Components

#### Component A: Encoder (Time-Series Transformer)

```
Input: (Batch, 24, 40)
    │
    ▼
Linear Projection: 40 → 128 (d_model)
    │
    ▼
Learnable Positional Embedding (24 positions)
    │
    ▼
TransformerEncoder (3 layers, 4 heads, d_ff=512)
    │
    ▼
Output: (Batch, 24, 128) latent sequence
```

**Key Design Decisions:**
- `batch_first=True` for PyTorch Transformers
- Bidirectional attention on context window (no causal masking needed for encoding past)
- LayerNorm + GELU activation

#### Component B: Predictor (World Model)

```
Input: z_20 (latent at hour 20)
    │
    ▼
MLP: 128 → 256 → 128
    │
    ▼
Output: Predicted z_24
```

**Why MLP over Transformer for predictor?**
- Simpler architecture sufficient for single-step prediction
- Faster training, lower memory
- Following I-JEPA paper recommendations

#### Component C: JEPA Training Loop

```
Context Window: Hours 0-20 → Encoder → z_context
Target Window:  Hours 20-24 → Encoder (stop_grad) → z_target

z_20 = z_context[:, -1, :]  # Last context position
z_24 = z_target[:, -1, :]   # Target state

predicted_z = Predictor(z_20)

Loss = VICReg(predicted_z, z_24)
```

#### Component D: VICReg Loss

```python
L = λ_inv * invariance(pred, target)    # MSE alignment
  + λ_var * variance(pred) + variance(target)  # Prevent collapse
  + λ_cov * covariance(pred) + covariance(target)  # Decorrelation
```

Default coefficients: `λ_inv=1.0, λ_var=1.0, λ_cov=0.04`

#### Component E: Downstream Classifier

```
Pre-trained Encoder (frozen or low LR)
    │
    ▼
Mean Pooling over 24 timesteps
    │
    ▼
Linear: 128 → 1
    │
    ▼
Sigmoid → Sepsis Probability
```

### Hyperparameters (Initial)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 128 | Balance expressivity vs. efficiency |
| n_heads | 4 | 128/4 = 32 dim per head |
| n_layers | 3 | Sufficient for 24-step sequences |
| d_ff | 512 | 4x d_model (standard) |
| dropout | 0.1 | Regularization |
| context_len | 20 | Hours 0-19 |
| target_len | 4 | Hours 20-23 (predict into) |
| batch_size | 64 | GPU memory dependent |
| lr_pretrain | 1e-4 | AdamW with warmup |
| lr_finetune | 1e-5 | Lower for fine-tuning |
| epochs_pretrain | 100 | With early stopping |
| epochs_finetune | 50 | Less needed |

### File Structure

```
Sepsis/
└── jepa/
    ├── __init__.py
    ├── model.py          # SepsisJEPA, SepsisClassifier
    ├── loss.py           # VICRegLoss
    ├── config.py         # Hyperparameters
    ├── data.py           # PyTorch Dataset wrapper
    ├── train_pretrain.py # Self-supervised training
    ├── train_finetune.py # Supervised fine-tuning
    ├── evaluate.py       # Metrics and comparison
    └── checkpoints/      # Saved models
```

## Dependencies

### New PyTorch Dependencies
```
torch>=2.0.0
```

### Internal Dependencies
- V6 tensors (`sepsis_model_v6/tensors.npz` or regenerated)
- Existing scaler for consistent preprocessing

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PyTorch/TensorFlow version conflicts | Medium | Separate virtual environment or careful version pinning |
| Representation collapse during pre-training | High | VICReg loss with proper coefficients; monitor variance during training |
| Overfitting on small dataset | Medium | Dropout, weight decay, early stopping |
| Poor fine-tuning transfer | Medium | Try frozen vs. unfrozen encoder; learning rate warmup |
| Integration complexity with Streamlit | Low | Load PyTorch model separately; expose prediction function |

## References

1. [JEPA Paper (LeCun, 2022)](https://arxiv.org/abs/2206.02888) - Original JEPA formulation
2. [I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243) - Image JEPA implementation details
3. [VICReg (Bardes et al., 2022)](https://arxiv.org/abs/2105.04906) - Variance-Invariance-Covariance Regularization
4. MIMIC-IV v3.1 Documentation
5. Existing V6 training script (`train_model_v6.py`)

## Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Pre-training convergence | VICReg loss < 1.0 | < 0.5 |
| Fine-tuned AUROC | > 0.70 | > BiLSTM (0.71) |
| Inference latency | < 200ms | < 100ms |
| Code test coverage | 60% | 80% |
