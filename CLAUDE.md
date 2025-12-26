# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sepsis early prediction system using a Bidirectional LSTM neural network trained on MIMIC-IV clinical data. The project consists of:
1. **Training Pipeline V3** (`train_model_v3.py`) - 3-hour prediction gap, basic training
2. **Training Pipeline V4** (`train_model_v4.py`) - 6-hour early warning, comprehensive EDA, train/val/test split
3. **Streamlit Dashboard** (`streamlit_app.py`) - Interactive web interface for model visualization

## Commands

### Run the Streamlit Dashboard
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Train Model V4 (Recommended - 6-hour Early Warning)
```bash
# Full pipeline: extract data, run EDA, build tensors, train
python train_model_v4.py --extract --project YOUR_GCP_PROJECT_ID

# EDA only (no training)
python train_model_v4.py --extract --project YOUR_PROJECT --eda-only

# Build tensors without training
python train_model_v4.py --extract --project YOUR_PROJECT --tensors-only

# Train from pre-built tensors
python train_model_v4.py --load sepsis_model_v4/tensors_v4.npz --output sepsis_model_v4
```

### Train Model V3 (Legacy - 3-hour Prediction Gap)
```bash
python train_model_v3.py --load sepsis_model_v3/tensors.npz --output sepsis_model_v3
python train_model_v3.py --extract --project YOUR_GCP_PROJECT_ID
```

### Docker (HuggingFace Spaces Deployment)
```bash
docker build -t sepsis-prediction .
docker run -p 7860:7860 sepsis-prediction
```

## Architecture

### Model Architecture
- **Input**: 24 hourly timesteps × 32 features (15 clinical values + 15 freshness masks + 2 static features)
- **Architecture**: Bidirectional LSTM (64 units) → BatchNorm → Bidirectional LSTM (32 units) → BatchNorm → Dense(32, ReLU) → Dense(1, sigmoid)
- **Output**: Sepsis probability (binary classification)

### Key Innovation: Time-Aware Forward-Fill
The model uses a 6-hour freshness window to handle missing clinical data:
- Measurements < 6 hours old: `mask=1` (fresh)
- Measurements ≥ 6 hours old: `mask=0` (stale)
- Never-measured features: imputed with population medians, `mask=0`

This allows the model to learn different weights for fresh vs. stale measurements.

### Clinical Features (15)
- **Vitals (6)**: heart_rate, sbp, dbp, resp_rate, spo2, temperature
- **Labs (9)**: wbc, creatinine, platelets, bilirubin, glucose, bun, sodium, potassium, hemoglobin

### Data Pipeline Flow
1. Extract data from MIMIC-IV BigQuery tables
2. Build cohort (sepsis cases + age-matched controls using Sepsis-3 criteria)
3. Apply unit repairs (temperature F→C conversion, HR interval→BPM)
4. Clamp values to physiological bounds (defined in `BOUNDS` dict)
5. Build 24-hour time-series tensors with freshness masks
6. Scale features using StandardScaler (fit only on measured training values)
7. Train with class weights for imbalanced data

### Key Configuration Constants
Located at the top of training scripts:
- `PREDICTION_GAP` - Hours before sepsis onset to stop using data (V3: 3h, V4: 6h)
- `FRESHNESS_WINDOW` - Hours before measurements become stale (6h)
- `N_TIMESTEPS` - Lookback window in hours (24h)
- `FEATURE_MAP` - Maps MIMIC-IV itemids to feature names
- `BOUNDS` - Physiological bounds for outlier clamping
- `POPULATION_MEDIANS` - Imputation values for never-measured features

### V4 Improvements Over V3
- **6-hour early warning** (vs 3h) for more clinical utility
- **Train/Val/Test split** - Validation set for threshold tuning, test set held out
- **Comprehensive EDA** - 13 publication-quality figures generated before training
- **Subject-aware splitting** - Prevents data leakage across splits

## EDA Output (V4: sepsis_model_v4/eda/)
V4 generates publication-ready figures (PNG + PDF at 400 DPI):
1. `class_balance` - Cohort distribution
2. `age_distribution` - Age KDE by group
3. `gender_distribution` - Sex breakdown
4. `icu_los_hours` - Length of stay histogram
5. `sepsis_onset_hours` - Onset timing with median line
6. `measurement_counts_by_feature` - Total measurements (log scale)
7. `stays_with_measurement_by_feature` - Coverage per feature
8. `outlier_rate_by_feature` - Pre-clamping outlier rates
9. `measurement_heatmap_{control,sepsis}` - Temporal measurement density
10. `value_distributions_grid` - 5×3 histogram panel
11. `boxplots_grid` - 5×3 boxplot panel
12. `mean_trajectories_grid` - Feature trajectories with SE bands
13. `correlation_heatmap` - Spearman correlation (annotated, lower triangle)
14. `per_stay_median_violin` - Split violin plots

## Model Artifacts
### V3 (sepsis_model_v3/)
- `sepsis_model_v3.keras`, `sepsis_scaler_v3.pkl`, `config.pkl`
- `X_test.npy`, `y_test.npy` - Test data

### V4 (sepsis_model_v4/)
- `sepsis_model_v4.keras`, `sepsis_scaler_v4.pkl`, `config_v4.pkl`
- `X_test.npy`, `y_test.npy`, `X_val.npy`, `y_val.npy` - Val/test data
- `tensors_v4.npz` - Pre-built tensors
- `eda/` - All EDA figures and CSVs

## Maestro Context

This project uses **Claude Maestro** for spec-driven development.
See `maestro/` for product context, tech stack, and workflow preferences.

### Maestro Files
- `maestro/product.md` - Product context, users, and goals
- `maestro/tech-stack.md` - Technical stack configuration
- `maestro/workflow.md` - Development workflow preferences
- `maestro/guidelines.md` - Code style guidelines
- `maestro/tracks.md` - Track registry for feature development
- `maestro/tracks/<track-id>/` - Individual track specs and plans

### When Working on Features
1. Check `maestro/tracks.md` for current work
2. Follow specs in `maestro/tracks/<track-id>/spec.md`
3. Execute plans in `maestro/tracks/<track-id>/plan.md`
4. Use `/maestro:status` to see current progress

### Common Maestro Commands
```bash
/maestro:new "Feature description"  # Start a new track
/maestro:implement                   # Execute current track
/maestro:status                      # Check progress
/maestro:list                        # List all tracks
```
