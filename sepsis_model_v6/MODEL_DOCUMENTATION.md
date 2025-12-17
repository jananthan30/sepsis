# Sepsis Prediction Model V6 - Documentation

**Version**: 6.0
**Date**: December 2024
**Prediction Horizon**: 4 hours before sepsis onset
**Dataset**: MIMIC-IV v3.1

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Version History & Rationale](#2-version-history--rationale)
3. [Feature Set](#3-feature-set)
4. [Data Quality Pipeline](#4-data-quality-pipeline)
5. [Lag-Masking Strategy](#5-lag-masking-strategy)
6. [Delta Features](#6-delta-features)
7. [Model Architecture](#7-model-architecture)
8. [Training Strategy](#8-training-strategy)
9. [EDA & Data Quality Reports](#9-eda--data-quality-reports)
10. [Usage Instructions](#10-usage-instructions)
11. [Expected Performance](#11-expected-performance)
12. [File Structure](#12-file-structure)

---

## 1. Executive Summary

Model V6 is a strategic rebuild that combines the best elements of V3 (high AUC) and V5 (clinical utility at 4-hour horizon) while fixing the issues that caused V5's performance degradation.

### Key Improvements Over V5

| Issue in V5 | Fix in V6 |
|-------------|-----------|
| Feature bloat (25 features) | Pruned to 16 base + 3 delta = 19 features |
| GCS/Coag labs (weak signal) | Removed - reverted to V3 core features |
| Bilirubin ordering bias | Lag-masking (only use values >8h before onset) |
| Fixed validation split (70/10/20) | 5-fold cross-validation (80% train per fold) |
| Complex architecture Dense(64)→Dense(32) | Simplified to Dense(32) |
| Missing resp_rate clipping | Added back from V3 |
| No 0-padding preservation in clamping | Fixed - preserves imputed values |

### Target Performance

| Metric | V3 (3h) | V5 (4h) | V6 Target (4h) |
|--------|---------|---------|----------------|
| AUC | 0.726 | 0.691 | **0.71+** |
| Sensitivity | - | 92.4% | 85-90% |
| Specificity | - | 30.2% | 45-55% |

---

## 2. Version History & Rationale

### V3 (Baseline - 3-hour prediction)
- **AUC**: 0.726
- **Features**: 15 (vitals + basic labs)
- **Strength**: High discrimination
- **Weakness**: Only 3-hour warning (limited clinical utility)

### V4 (6-hour prediction)
- **AUC**: ~0.65
- **Features**: 15
- **Issue**: 6-hour gap too aggressive, lost too much signal

### V5 (4-hour prediction)
- **AUC**: 0.691
- **Features**: 25 (added GCS, coag labs, shock index)
- **Issues**:
  - Feature bloat increased noise
  - Fixed val split reduced training data
  - Bilirubin ordering bias (data leakage)

### V6 (4-hour prediction - optimized)
- **Target AUC**: 0.71+
- **Features**: 19 (V3 core + MAP + delta features)
- **Fixes**: All V5 issues addressed

---

## 3. Feature Set

### Base Features (16)

#### Vitals (7)
| Feature | Item IDs | Unit | Bounds | Population Median |
|---------|----------|------|--------|-------------------|
| heart_rate | 220045 | BPM | 20-250 | 84 |
| sbp | 220179, 220050 | mmHg | 40-250 | 118 |
| dbp | 220180, 220051 | mmHg | 30-200 | 62 |
| map | 220052, 220181 | mmHg | 30-200 | 77 |
| resp_rate | 220210 | /min | 5-60 | 18 |
| spo2 | 220277 | % | 50-100 | 97 |
| temperature | 223761, 223762 | °C | 30-45 | 37.0 |

#### Labs (9)
| Feature | Item ID | Unit | Bounds | Population Median |
|---------|---------|------|--------|-------------------|
| wbc | 51301 | K/uL | 0.1-100 | 9.5 |
| creatinine | 50912 | mg/dL | 0.1-30 | 1.0 |
| platelets | 51265 | K/uL | 5-1200 | 200 |
| bilirubin | 50885 | mg/dL | 0.1-60 | 0.6 |
| glucose | 50931 | mg/dL | 20-1200 | 120 |
| bun | 51006 | mg/dL | 1-250 | 18 |
| sodium | 50983 | mEq/L | 90-180 | 139 |
| potassium | 50971 | mEq/L | 1-15 | 4.1 |
| hemoglobin | 51222 | g/dL | 3-25 | 10.5 |

### Delta Features (3)
| Feature | Calculation | Bounds | Clinical Meaning |
|---------|-------------|--------|------------------|
| delta_hr | (HR_now - HR_4h_ago) / 4 | -50 to 50 | Tachycardia progression |
| delta_sbp | (SBP_now - SBP_4h_ago) / 4 | -50 to 50 | Hypotension development |
| delta_temp | (Temp_now - Temp_4h_ago) / 4 | -2 to 2 | Fever trajectory |

### Features NOT Included (and why)

| Feature | Reason for Exclusion |
|---------|---------------------|
| GCS (eye, verbal, motor) | Low variance in ICU (most patients score 14-15) |
| Shock Index | Redundant with HR and SBP |
| INR, PT, PTT | Late-stage indicators, weak early signal |
| Anion Gap, Bicarbonate | Added complexity without proportional benefit |
| Lactate | High ordering bias (ordered when sepsis suspected) |

### Total Input Shape
```
Per timestep: 19 values + 19 freshness masks + 2 static (age, gender) = 40 features
Full tensor: (24 timesteps, 40 features)
```

---

## 4. Data Quality Pipeline

### Pipeline Order (sequence matters!)

```
Raw Data
    │
    ▼
[1] HR RR Interval → BPM Conversion
    │
    ▼
[2] Temperature F → C Conversion (value-based)
    │
    ▼
[3] Resp Rate Artifact Clipping (>60 → 60)
    │
    ▼
[4] Zero/Negative Value Removal
    │
    ▼
[5] Drop NaN Rows
    │
    ▼
[6] Lag-Masking (Bilirubin)
    │
    ▼
[7] Delta Feature Computation
    │
    ▼
[8] Tensor Building (24h window, freshness masks)
    │
    ▼
[9] Physiological Clamping (with 0-padding preservation)
    │
    ▼
[10] Feature Scaling (train-only fit)
```

### 4.1 Heart Rate RR Interval Conversion

**Problem**: Some monitors store RR intervals in milliseconds instead of BPM.

**Detection**: Values > 300 are likely RR intervals (300 BPM is physiologically extreme).

**Fix**:
```python
HR_bpm = 60000 / RR_ms.clip(lower=200)
```

**Example**:
- RR = 750 ms → HR = 60000/750 = 80 BPM ✓
- RR = 500 ms → HR = 60000/500 = 120 BPM ✓

### 4.2 Temperature Conversion

**Problem**: MIMIC-IV has MISLABELED temperature item IDs:
- itemid 223761 labeled "Celsius" but contains **Fahrenheit** (median 98.4)
- itemid 223762 labeled "Fahrenheit" but contains **Celsius** (median 37.0)

**Fix**: Value-based detection (NOT itemid-based!)
```python
if value > 50 and value < 120:
    # Fahrenheit body temperature
    celsius = (fahrenheit - 32) * 5 / 9
```

**Why NOT itemid-based**: V3 used `FAHRENHEIT_ITEMIDS = {223762}` which is **WRONG** - it would convert already-Celsius values!

**Extreme Value Handling**:
- Values > 120: Remove (data error)
- Values ≤ 0: Remove (data error)

### 4.3 Respiratory Rate Clipping

**Problem**: Some resp_rate values are artifacts (ventilator rates, data errors).

**Fix**: Clip values > 60 to 60 (physiological maximum for adults).

```python
if resp_rate > 60:
    resp_rate = 60
```

### 4.4 Zero and Negative Value Removal

| Category | Features | Rule | Reason |
|----------|----------|------|--------|
| Vitals (cannot be zero) | HR, SBP, DBP, MAP, RR | ≤0 → NaN | Zero = death or placeholder |
| Blood Pressure | SBP, DBP, MAP | <0 → NaN | Negative BP impossible |
| SpO2 | spo2 | >100% or ≤0 → NaN | Physical limits |
| Labs | WBC, Hgb, Cr, Plt, Bili, Glucose | ≤0 → NaN | Below detection = missing |

### 4.5 Physiological Clamping

Applied to tensors after building, with key features:

1. **0-padding preservation**: Imputed/missing values (zeros) are NOT clamped
2. **Per-feature logging**: Each feature's outlier count is reported
3. **Bounds from clinical knowledge**: See feature tables above

```python
# Pseudocode
for each feature:
    nonzero_mask = values != 0  # Preserve padding
    outliers = (values < low | values > high) & nonzero_mask
    values[outliers] = clip(values[outliers], low, high)
```

---

## 5. Lag-Masking Strategy

### The Bilirubin Problem

**Issue**: Clinicians order bilirubin tests when they suspect liver dysfunction, which is associated with sepsis. The *presence* of a recent bilirubin order is itself a signal of sepsis suspicion.

**Evidence from V5 Data Quality Report**:
- Bilirubin ordered in 77% of sepsis patients
- Bilirubin ordered in 66% of control patients
- **+10.8% differential** = ordering bias

### The Solution: Lag-Masking

Only use bilirubin values measured **before** the clinician could reasonably suspect sepsis.

**Implementation**:
```
Prediction time = onset_time - 4 hours (prediction gap)
Lag-mask cutoff = prediction_time - 4 hours (lag window)
                = onset_time - 8 hours

Only bilirubin values with charttime < (onset_time - 8 hours) are used.
```

**Visual Timeline**:
```
ICU Admission ──────────────────────────────────> Sepsis Onset
                                    │            │
                           Lag Cutoff (-8h)  Pred Time (-4h)
                                    │            │
    ◄── Bilirubin OK ──►           │◄── MASKED ─►│◄── Gap ─►│
```

### Why This Works

- **Before -8h**: Bilirubin ordered as routine, not due to sepsis suspicion
- **-8h to -4h**: May be ordered due to early sepsis signs → MASKED
- **-4h to onset**: Too close to onset → already excluded by prediction gap

---

## 6. Delta Features

### Rationale

The LSTM captures temporal patterns, but the final Dense layer sees a "snapshot" representation. Delta features explicitly encode **rate of deterioration** which helps the Dense layer recognize:
- Rapid HR increase → compensatory tachycardia
- Falling BP → developing shock
- Rising temperature → worsening infection

### Computation

For each vital (HR, SBP, Temperature):
```python
delta = (current_value - value_4_hours_ago) / 4  # Per-hour rate
```

**Time Matching**: Uses 15-minute bins to find approximate matches (vitals often charted at slightly different times).

### Bounds

| Feature | Bounds | Interpretation |
|---------|--------|----------------|
| delta_hr | -50 to +50 | ±50 BPM change over 4 hours |
| delta_sbp | -50 to +50 | ±50 mmHg change over 4 hours |
| delta_temp | -2 to +2 | ±2°C change over 4 hours |

---

## 7. Model Architecture

### Simplified Architecture (V3-style)

```
Input: (24 timesteps, 40 features)
       │
       ▼
Bidirectional LSTM (64 units, return_sequences=True)
       │
       ▼
BatchNormalization
       │
       ▼
Bidirectional LSTM (32 units, return_sequences=False)
       │
       ▼
BatchNormalization
       │
       ▼
Dense (32 units, ReLU)          ◄── Single dense layer (not 64→32)
       │
       ▼
Dropout (0.3)
       │
       ▼
Dense (1 unit, Sigmoid)
       │
       ▼
Output: Sepsis probability (0-1)
```

### Why Simplified?

V5 used `Dense(64) → Dense(32)` which:
- Added more parameters to learn
- Increased overfitting risk
- No performance benefit with limited data

V3's single `Dense(32)` achieved AUC 0.726 - simpler is better.

---

## 8. Training Strategy

### 5-Fold Cross-Validation

**Why?**: V5 used a fixed 70/10/20 split, losing 12.5% of training data to the validation set.

**Benefits of 5-Fold CV**:
1. 80% training data per fold (matches V3)
2. More robust AUC estimate (mean ± std across folds)
3. Threshold optimization using CV predictions

### Split Strategy

```
Fold 1: [████████ Train ████████][Val][████ Test ████]
Fold 2: [██ Val ██][████████ Train ████████][████ Test ████]
Fold 3: [████████ Train ████████][Val][████ Test ████]
...

Final: [████████████ All Training Data ████████████][Test]
```

### Subject-Aware Splitting

All splits use `GroupKFold` or `GroupShuffleSplit` with `groups=subject_id` to prevent:
- Same patient appearing in train and test
- Data leakage from correlated stays

### Class Weighting

```python
weight_positive = n_samples / (2 * n_positive)
weight_negative = n_samples / (2 * n_negative)
```

Handles the ~50/50 class balance after matching.

### Callbacks

| Callback | Configuration |
|----------|---------------|
| EarlyStopping | monitor='val_AUC', patience=15, mode='max' |
| ReduceLROnPlateau | monitor='val_AUC', factor=0.5, patience=5 |

---

## 9. EDA & Data Quality Reports

### Generated Artifacts

#### Data Quality Reports (`sepsis_model_v6/data_quality/`)

| File | Contents |
|------|----------|
| `feature_statistics.csv` | Count, min, max, mean, std, percentiles (p1-p99), outlier rates |
| `outlier_summary.csv` | Pre-clamping outlier counts sorted by severity |

#### EDA Figures (`sepsis_model_v6/eda/`)

| Figure | Description |
|--------|-------------|
| `class_balance.png/pdf` | Cohort distribution bar chart |
| `age_distribution.png/pdf` | KDE plots by patient group |
| `feature_coverage.png/pdf` | Horizontal bar chart of % stays with each feature |
| `feature_distributions.png/pdf` | 4×4 histogram grid comparing sepsis vs control |
| `correlation_heatmap.png/pdf` | Spearman correlation matrix (lower triangle) |
| `outlier_rates.png/pdf` | Pre-clamping outlier rates by feature |
| `sepsis_onset_timing.png/pdf` | Distribution of onset hours after ICU admission |
| `bilirubin_lag_masking.png/pdf` | Visualization of lag-mask cutoff impact |

#### Data Exports (`sepsis_model_v6/eda/`)

| File | Contents |
|------|----------|
| `cohort.csv` | Full cohort with computed fields |
| `cohort_summary.csv` | Summary statistics by group |
| `feature_coverage.csv` | Coverage % and measurement counts |

---

## 10. Usage Instructions

### Full Pipeline (Extract + EDA + Train)

```bash
python train_model_v6.py --extract --project YOUR_GCP_PROJECT_ID
```

### EDA Only (No Training)

```bash
python train_model_v6.py --extract --project YOUR_PROJECT --eda-only
```

### Build Tensors Only (No Training)

```bash
python train_model_v6.py --extract --project YOUR_PROJECT --tensors-only
```

### Train from Pre-built Tensors

```bash
python train_model_v6.py --load sepsis_model_v6/tensors_v6.npz
```

### Skip EDA (Faster Iteration)

```bash
python train_model_v6.py --extract --project YOUR_PROJECT --no-eda
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--extract` | - | Extract data from BigQuery |
| `--project` | sepsis-prediction-2025 | GCP project ID |
| `--load` | - | Load pre-built tensors |
| `--output` | sepsis_model_v6 | Output directory |
| `--prediction-gap-hours` | 4 | Hours before onset to stop |
| `--n-folds` | 5 | Number of CV folds |
| `--no-eda` | False | Skip EDA generation |
| `--eda-only` | False | Run EDA only |
| `--tensors-only` | False | Build tensors only |
| `--batch-size` | 64 | Training batch size |
| `--epochs` | 100 | Max epochs |
| `--seed` | 42 | Random seed |

---

## 11. Expected Performance

### Compared to Previous Versions

| Metric | V3 (3h) | V5 (4h) | V6 Expected (4h) |
|--------|---------|---------|------------------|
| CV AUC | - | - | 0.70-0.72 |
| Test AUC | 0.726 | 0.691 | 0.70-0.72 |
| Sensitivity | ~90% | 92.4% | 85-90% |
| Specificity | ~40% | 30.2% | 45-55% |
| PPV | ~55% | 55.5% | 55-60% |
| NPV | ~85% | 80.9% | 80-85% |

### Clinical Operating Points

| Use Case | Threshold | Expected Sens | Expected Spec |
|----------|-----------|---------------|---------------|
| High sensitivity (screening) | ~0.40 | 90% | 35-40% |
| Balanced (Youden J) | ~0.50 | 75% | 55-60% |
| High specificity (reduce alerts) | ~0.60 | 60% | 70-75% |

### Traffic Light System (Operational Recommendation)

| Alert Level | Model | Threshold | Purpose |
|-------------|-------|-----------|---------|
| **Red** (Page Doctor) | V3 (3h) | High | "Act Now" - high precision |
| **Yellow** (Dashboard Flag) | V6 (4h) | Balanced | "Watch List" - early warning |

---

## 12. File Structure

```
sepsis_model_v6/
├── MODEL_DOCUMENTATION.md      # This file
├── tensors_v6.npz              # Pre-built tensors (X, y, subjects)
├── sepsis_model_v6.keras       # Trained Keras model
├── sepsis_scaler_v6.pkl        # StandardScaler (fitted on train)
├── config_v6.pkl               # Model configuration and metrics
├── X_test.npy                  # Test features (scaled)
├── y_test.npy                  # Test labels
├── training_history.csv        # Epoch-by-epoch metrics
├── training_curves.png/pdf     # Loss and AUC plots
├── roc_curve.png/pdf           # ROC curve with AUC
├── data_quality/
│   ├── feature_statistics.csv  # Per-feature statistics
│   └── outlier_summary.csv     # Outlier counts
└── eda/
    ├── cohort.csv              # Full cohort data
    ├── cohort_summary.csv      # Summary by group
    ├── feature_coverage.csv    # Coverage statistics
    ├── class_balance.png/pdf
    ├── age_distribution.png/pdf
    ├── feature_coverage.png/pdf
    ├── feature_distributions.png/pdf
    ├── correlation_heatmap.png/pdf
    ├── outlier_rates.png/pdf
    ├── sepsis_onset_timing.png/pdf
    └── bilirubin_lag_masking.png/pdf
```

---

## Appendix A: Configuration Constants

```python
# Prediction parameters
DEFAULT_PREDICTION_GAP_HOURS = 4
DEFAULT_MIN_DATA_HOURS = 3
DEFAULT_FRESHNESS_WINDOW_HOURS = 6
DEFAULT_N_TIMESTEPS = 24

# Lag-masking
LAG_MASKED_FEATURES = {"bilirubin"}
LAG_MASK_HOURS = 4  # Total lag = prediction_gap + lag_mask = 8 hours

# Training
DEFAULT_N_FOLDS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOP_PATIENCE = 15
DEFAULT_LR_REDUCE_PATIENCE = 5
```

---

## Appendix B: Changelog

### V6.0 (December 2024)
- Initial release
- Combined best practices from V3 + V5
- Added lag-masking for bilirubin
- Added delta features (HR, SBP, Temperature)
- Implemented 5-fold cross-validation
- Fixed temperature conversion (value-based, not itemid-based)
- Added resp_rate clipping
- Fixed clamping to preserve 0-padding
- Added comprehensive EDA and data quality reports

---

*Documentation generated for Sepsis Prediction Model V6*
*Author: Claude Code*
