# Implementation Plan: JEPA World Model for Sepsis Prediction

**Track ID:** 001-jepa-world-model
**Status:** active
**Started:** 2025-12-26
**Last Updated:** 2025-12-26

## Summary

Implement a PyTorch-based JEPA (Joint Embedding Predictive Architecture) as a separate module that learns patient physiological dynamics through self-supervised pre-training, then fine-tunes for sepsis prediction. The implementation follows a phased approach: core architecture → pre-training → fine-tuning → integration.

---

## Phases

### Phase 1: Project Setup & Core Architecture
**Status:** COMPLETED
**Checkpoint:** `SepsisJEPA` model runs forward pass on dummy data - VERIFIED

#### Tasks

- [x] **Task 1.1:** Create `jepa/` module directory structure
  - Created `jepa/__init__.py`, `jepa/config.py`
  - PyTorch 2.1.0 already installed
  - Verified no conflicts with TensorFlow

- [x] **Task 1.2:** Implement `jepa/config.py` - Hyperparameter configuration
  - Defined `JEPAConfig` dataclass (d_model=128, n_heads=4, n_layers=3)
  - Defined `TrainingConfig` dataclass (lr, epochs, batch_size)
  - Added `VICRegConfig` with coefficients (lambda_inv=1.0, lambda_var=1.0, lambda_cov=0.04)

- [x] **Task 1.3:** Implement `jepa/model.py` - Core JEPA architecture
  - Implemented `InputProjection` class (Linear 40->128 + LayerNorm)
  - Implemented `PositionalEncoding` with learnable embeddings
  - Implemented `Predictor` MLP (128->256->128 with GELU)
  - Implemented `SepsisJEPA` main class with forward pass logic
  - Implemented `SepsisClassifier` for downstream tasks

- [x] **Task 1.4:** Implement `jepa/loss.py` - VICReg loss function
  - Implemented invariance term (MSE between pred and target)
  - Implemented variance term with hinge loss (std > gamma)
  - Implemented covariance term (off-diagonal regularization)
  - Added `CollapseMonitor` utility class

- [x] **Task 1.5:** Write unit tests for core components
  - 9/9 tests passing
  - Verified forward pass shapes: (B, 24, 40) -> (B, 128)
  - Verified VICReg loss computation
  - Verified gradient flow (encoder + predictor have gradients)
  - Model has 735,360 parameters

---

### Phase 2: Data Pipeline & Pre-training
**Status:** pending
**Checkpoint:** Pre-trained encoder checkpoint saved with decreasing VICReg loss

#### Tasks

- [ ] **Task 2.1:** Implement `jepa/data.py` - PyTorch data utilities
  - Create `SepsisDataset(torch.utils.data.Dataset)` class
  - Load from V6 tensors (.npz format)
  - Convert numpy arrays to PyTorch tensors
  - Implement train/val/test split logic

- [ ] **Task 2.2:** Implement `jepa/train_pretrain.py` - Self-supervised training script
  - Set up argument parser (data path, config, output dir)
  - Initialize model, optimizer (AdamW), scheduler (warmup + cosine)
  - Training loop with VICReg loss
  - Logging: loss curves, variance/covariance metrics
  - Checkpoint saving (best model, latest model)
  - Early stopping based on validation loss

- [ ] **Task 2.3:** Run pre-training experiment
  - Load V6 tensors
  - Train for 100 epochs (or until convergence)
  - Monitor for representation collapse (check embedding variance)
  - Save best checkpoint to `jepa/checkpoints/pretrained.pt`

- [ ] **Task 2.4:** Visualize pre-training results
  - Plot VICReg loss components over epochs
  - t-SNE/UMAP of learned embeddings (color by sepsis label)
  - Save figures to `jepa/figures/`

---

### Phase 3: Fine-tuning & Evaluation
**Status:** pending
**Checkpoint:** Fine-tuned model achieves AUROC > 0.70 on test set

#### Tasks

- [ ] **Task 3.1:** Implement `jepa/finetune.py` - Classifier module
  - Implement `SepsisClassifier` class
  - Load pre-trained encoder from checkpoint
  - Add classification head (mean pool → Linear 128→1 → Sigmoid)
  - Support frozen vs. unfrozen encoder modes

- [ ] **Task 3.2:** Implement `jepa/train_finetune.py` - Supervised training script
  - Load pre-trained encoder
  - Create classifier with frozen encoder
  - Training loop with BCE loss
  - Use class weights for imbalanced data
  - Threshold optimization on validation set

- [ ] **Task 3.3:** Run fine-tuning experiments
  - Experiment A: Frozen encoder + linear probe
  - Experiment B: Unfrozen encoder with low LR (1e-5)
  - Compare both approaches

- [ ] **Task 3.4:** Implement `jepa/evaluate.py` - Metrics and comparison
  - Load both JEPA and BiLSTM models
  - Compute AUROC, sensitivity, specificity, confusion matrix
  - Generate comparison table
  - ROC curve overlay plot

---

### Phase 4: Dashboard Integration
**Status:** pending
**Checkpoint:** Dashboard shows A/B comparison between JEPA and BiLSTM

#### Tasks

- [ ] **Task 4.1:** Create JEPA inference utility
  - Implement `jepa/inference.py` with `predict_sepsis_jepa()` function
  - Handle model loading and caching
  - Ensure compatible with Streamlit's caching

- [ ] **Task 4.2:** Update Streamlit dashboard
  - Add "Model Selection" toggle (BiLSTM vs JEPA)
  - Show side-by-side metrics comparison
  - Display JEPA-specific visualizations (attention patterns if useful)

- [ ] **Task 4.3:** Update FastAPI endpoints (optional)
  - Add `/api/v1/predict/jepa` endpoint
  - Add `/api/v1/compare` endpoint for A/B comparison

---

### Phase 5: Documentation & Cleanup
**Status:** pending
**Checkpoint:** All acceptance criteria met, documentation complete

#### Tasks

- [ ] **Task 5.1:** Write module documentation
  - Update README with JEPA section
  - Add architecture diagram to `jepa/README.md`
  - Document training commands and hyperparameters

- [ ] **Task 5.2:** Code cleanup and review
  - Add type hints to all functions
  - Add Google-style docstrings
  - Run linter (flake8/ruff)

- [ ] **Task 5.3:** Final verification
  - Verify all acceptance criteria met
  - Run full pipeline from scratch
  - Document final metrics

- [ ] **Task 5.4:** Create checkpoint commit
  - Stage all jepa/ files
  - Create commit with summary of implementation
  - Tag with version (e.g., v0.1.0-jepa)

---

## Notes

### Architecture Decisions Made

1. **PyTorch over TensorFlow** - JEPA research is PyTorch-native; better self-supervised learning ecosystem
2. **Separate module** - Keeps existing TensorFlow code working; clean separation
3. **VICReg over contrastive** - Simpler, no negative sampling needed; proven for JEPA-style architectures
4. **MLP predictor** - Following I-JEPA paper; Transformer predictor is overkill for single-step prediction

### Potential Issues

1. **Memory** - Transformer attention is O(n²); 24 timesteps should be fine
2. **Dataset size** - Self-supervised pre-training usually benefits from large datasets; may need data augmentation
3. **Framework loading** - Loading both PyTorch and TensorFlow models in same process; test memory

### Data Augmentation Ideas (If Needed)

- Random temporal jitter (shift by 1-2 hours)
- Feature dropout (mask random features)
- Gaussian noise injection
- Time reversal (patient trajectory in reverse)

---

## Git Strategy

- **Branch:** `feature/jepa-world-model`
- **Commit pattern:** `feat(jepa): <description>` for new features
- **Merge strategy:** Squash merge to main after validation

---

## Estimated Task Sizes

| Phase | Tasks | Complexity |
|-------|-------|------------|
| Phase 1: Core Architecture | 5 | Medium |
| Phase 2: Pre-training | 4 | High |
| Phase 3: Fine-tuning | 4 | Medium |
| Phase 4: Integration | 3 | Low |
| Phase 5: Documentation | 4 | Low |
| **Total** | **20** | |
