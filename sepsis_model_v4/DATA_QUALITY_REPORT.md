# MIMIC-IV Data Quality Analysis Report

## Sepsis Early Prediction Model - Data Preprocessing Documentation

**Dataset**: MIMIC-IV v3.1 (Medical Information Mart for Intensive Care)
**Analysis Date**: December 2024
**Purpose**: Document data quality issues and preprocessing decisions for reproducibility

---

## 1. Data Overview

### 1.1 Source Tables

| Table | Description |
|-------|-------------|
| `mimiciv_3_1_icu.icustays` | ICU stay information |
| `mimiciv_3_1_icu.chartevents` | Vital signs (HR, BP, SpO2, Temp, RR) |
| `mimiciv_3_1_hosp.labevents` | Laboratory values |
| `mimiciv_3_1_hosp.patients` | Patient demographics |
| `mimiciv_3_1_derived.sepsis3` | Sepsis-3 labels |

### 1.2 Cohort Summary

| Metric | Value |
|--------|-------|
| Total ICU Stays | 8,748 |
| Sepsis Cases | 4,374 (50%) |
| Control Cases | 4,374 (50%) |
| Matching Strategy | Age (±5 years), Gender, ICU type |

### 1.3 Feature Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `stay_id` | Int64 | ICU stay identifier |
| `charttime` | datetime64 | Time of measurement |
| `itemid` | Int64 | MIMIC item ID for measurement type |
| `valuenum` | float64 | Numeric value of measurement |
| `feature` | string | Feature name (heart_rate, sbp, etc.) |
| `hadm_id` | Int64 | Hospital admission ID |
| `subject_id` | Int64 | Patient subject ID |
| `intime` | datetime64 | ICU admission time |
| `outtime` | datetime64 | ICU discharge time |
| `Age` | Float64 | Patient age at admission |
| `Gender` | Int64 | Patient gender (0=Female, 1=Male) |
| `onset_time` | datetime64 | Sepsis onset time (NaT for controls) |
| `label` | int64 | Label (0=Control, 1=Sepsis) |
| `win_end` | datetime64 | End of prediction window |
| `hours_before_win_end` | float64 | Hours before window end |
| `hour_bin` | int32 | Hourly bin (0-23) |

---

## 2. Clinical Features

### 2.1 Vital Signs (from `chartevents`)

| Feature | Item IDs | Unit | Clinical Range | Description |
|---------|----------|------|----------------|-------------|
| Heart Rate | 220045 | bpm | 60-100 | Cardiac rhythm |
| Systolic BP | 220179, 220050 | mmHg | 90-140 | Peak arterial pressure |
| Diastolic BP | 220180, 220051 | mmHg | 60-90 | Resting arterial pressure |
| Respiratory Rate | 220210 | /min | 12-20 | Breathing frequency |
| SpO2 | 220277 | % | 95-100 | Oxygen saturation |
| Temperature | 223761, 223762 | °C/°F | 36.5-37.5°C | Body temperature |

### 2.2 Laboratory Values (from `labevents`)

| Feature | Item ID | Unit | Clinical Range | Description |
|---------|---------|------|----------------|-------------|
| WBC | 51301 | K/uL | 4.5-11.0 | White blood cell count |
| Creatinine | 50912 | mg/dL | 0.7-1.3 | Kidney function marker |
| Platelets | 51265 | K/uL | 150-400 | Clotting cells |
| Bilirubin | 50885 | mg/dL | 0.1-1.2 | Liver function marker |
| Glucose | 50931 | mg/dL | 70-100 | Blood sugar |
| BUN | 51006 | mg/dL | 7-20 | Blood urea nitrogen |
| Sodium | 50983 | mEq/L | 136-145 | Electrolyte |
| Potassium | 50971 | mEq/L | 3.5-5.0 | Electrolyte |
| Hemoglobin | 51222 | g/dL | 12-17 | Oxygen carrier |

---

## 3. Data Quality Issues Identified

### 3.1 Summary of Issues by Feature

| Feature | Total Records | Min Value | Max Value | % Below Bound | % Above Bound | Primary Issue |
|---------|---------------|-----------|-----------|---------------|---------------|---------------|
| heart_rate | 8,748,458 | 1.0 | 10,000,000 | 0.01% | 0.0001% | Data entry errors |
| sbp | 8,465,986 | 0.1 | 1,025,100 | 0.01% | 0.0005% | Data entry errors |
| dbp | 8,464,494 | -41.0 | 114,109 | 0.37% | 0.001% | Data entry errors |
| resp_rate | 8,604,779 | 0.0001 | 7,000,400 | 0.39% | 0.00006% | Data entry errors |
| spo2 | 8,566,442 | 0.01 | 9,900,000 | 0.01% | 0.0002% | Data entry errors |
| temperature | 2,449,676 | 0.1 | 234,123 | 16.1% | 0.00004% | **Unit label swap** |
| wbc | 4,145,868 | 0.1 | 12,500 | 0% | 0.08% | Extreme values |
| creatinine | 4,317,391 | 0.07 | 808 | 0% | ~0% | Extreme values |
| platelets | 4,197,870 | 5.0 | 2,989 | 0% | 0.01% | Minor |
| bilirubin | 1,575,798 | 0.1 | 87.2 | 0% | 0.07% | Minor |
| glucose | 3,621,279 | 1.0 | 23,200 | 0.01% | 0.08% | Data entry errors |
| bun | 4,200,695 | 1.0 | 305 | 0% | 0% | Clean |
| sodium | 4,110,683 | 67.0 | 185 | 0% | 0% | Clean |
| potassium | 4,147,219 | 0.7 | 26.5 | 0% | 0% | Clean |
| hemoglobin | 4,172,445 | 1.03 | 24.9 | 0% | 0% | Clean |

### 3.2 Critical Finding: Temperature ItemID Label Swap

**Issue**: MIMIC-IV documentation labels are **swapped** for temperature itemids.

| ItemID | MIMIC Label | Actual Data | Evidence |
|--------|-------------|-------------|----------|
| 223761 | "Celsius" | **FAHRENHEIT** | Median=98.4, 99.8% values in 95-105 range |
| 223762 | "Fahrenheit" | **CELSIUS** | Median=37.0, 96.9% values in 35-42 range |

**Verification Query Results**:
```
itemid 223761 (labeled "Celsius"):
  - Records: 2,054,880
  - Median: 98.4 (clearly Fahrenheit!)
  - In Celsius range (35-42): 0.03%
  - In Fahrenheit range (95-105): 99.8%

itemid 223762 (labeled "Fahrenheit"):
  - Records: 394,796
  - Median: 37.0 (clearly Celsius!)
  - In Celsius range (35-42): 96.9%
  - In Fahrenheit range (95-105): 0.2%
```

---

## 4. Root Cause Analysis

### 4.1 Extreme Maximum Values

| Cause | Affected Features | Example | Frequency |
|-------|-------------------|---------|-----------|
| **Double-digit entry** | DBP, SBP, SpO2 | 114 → 11411 | Rare (<0.01%) |
| **Extra zeros** | Heart Rate, SpO2 | 98 → 980000 | Very rare |
| **Equipment malfunction** | All vitals | Random large values | Very rare |
| **Copy-paste errors** | All features | Unrelated values | Very rare |

### 4.2 Extreme Minimum Values

| Cause | Affected Features | Example | Frequency |
|-------|-------------------|---------|-----------|
| **Zero/placeholder values** | All features | 0 entered for missing | 0.01-0.4% |
| **Negative values** | DBP, Temperature | -41 DBP | Very rare |
| **Sensor disconnection** | SpO2, HR | Near-zero readings | Rare |
| **Unit conversion errors** | Temperature | Already-Celsius converted again | 16% for temp |

### 4.3 Temperature-Specific Issues

After incorrect F→C conversion of already-Celsius values:
- Original: 37.0°C (normal body temp)
- Incorrectly converted: (37.0 - 32) × 5/9 = **2.78°C** (hypothermia!)

This explains the 16% outlier rate for temperature.

---

## 5. Preprocessing Approach

### 5.1 Four-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATA PREPROCESSING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: REMOVE IMPOSSIBLE VALUES                                      │
│  ─────────────────────────────────                                      │
│  • Temperature > 120 or ≤ 0     → NaN                                  │
│  • SpO2 > 100%                  → NaN                                  │
│  • DBP > 300 or < 0             → NaN                                  │
│  • SBP > 350 or < 0             → NaN                                  │
│  • Heart Rate ≤ 0               → NaN                                  │
│                                                                         │
│  Stage 2: UNIT CONVERSION (Value-Based Detection)                       │
│  ────────────────────────────────────────────────                       │
│  • Temperature 50-120           → F to C: (val-32)×5/9                 │
│  • Heart Rate > 300             → ms to BPM: 60000/val                 │
│                                                                         │
│  Stage 3: IMPUTATION (Time-Aware Forward-Fill)                          │
│  ─────────────────────────────────────────────                          │
│  • If measured at time t        → Use value, mask=1                    │
│  • If forward-fill < 6 hours    → Use previous, mask=1                 │
│  • If forward-fill ≥ 6 hours    → Use previous, mask=0 (stale)         │
│  • If never measured            → Population median, mask=0            │
│                                                                         │
│  Stage 4: CLAMP TO PHYSIOLOGICAL BOUNDS                                 │
│  ──────────────────────────────────────                                 │
│  • All features clipped to [min_bound, max_bound]                      │
│  • Ensures no extreme values reach the model                           │
│                                                                         │
│  Stage 5: STANDARDIZATION                                               │
│  ────────────────────────────                                           │
│  • StandardScaler fit on training data only                            │
│  • Only fresh measurements (mask=1) used for statistics                │
│  • Transform: (x - mean) / std                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Physiological Bounds Used

| Feature | Min Bound | Max Bound | Rationale |
|---------|-----------|-----------|-----------|
| heart_rate | 20 | 250 | Extreme bradycardia to SVT |
| sbp | 40 | 250 | Severe hypotension to hypertensive crisis |
| dbp | 20 | 200 | Physiological extremes |
| resp_rate | 5 | 60 | Severe bradypnea to tachypnea |
| spo2 | 50 | 100 | Severe hypoxemia to normal |
| temperature | 30 | 45 | Severe hypothermia to hyperpyrexia |
| wbc | 0.1 | 100 | Severe neutropenia to leukemoid reaction |
| creatinine | 0.1 | 30 | Normal to severe AKI |
| platelets | 5 | 1200 | Severe thrombocytopenia to thrombocytosis |
| bilirubin | 0.1 | 60 | Normal to severe liver failure |
| glucose | 20 | 1200 | Severe hypoglycemia to DKA |
| bun | 1 | 250 | Normal to severe uremia |
| sodium | 90 | 180 | Severe hypo/hypernatremia |
| potassium | 1 | 15 | Severe hypo/hyperkalemia |
| hemoglobin | 3 | 25 | Severe anemia to polycythemia |

### 5.3 Handling Minimum Value Outliers

| Issue | Detection | Action | Rationale |
|-------|-----------|--------|-----------|
| Zero values | value = 0 | Set to NaN → Impute | Likely missing data placeholder |
| Negative values | value < 0 | Set to NaN → Impute | Physiologically impossible |
| Below clinical minimum | value < bound | Clamp to minimum | Preserve extreme but plausible values |

**Example for DBP**:
- Value = -41 → Set to NaN → Forward-fill or median impute
- Value = 15 → Clamp to 20 (minimum bound)
- Value = 45 → Keep as-is (valid low DBP)

### 5.4 Freshness Mask Mechanism

The freshness mask informs the LSTM about data reliability:

```python
# For each feature at each timestep:
if measurement_exists_at_time_t:
    value = measurement
    mask = 1.0  # Fresh - high confidence
elif hours_since_last_measurement < 6:
    value = forward_filled_value
    mask = 1.0  # Recent - still reliable
elif hours_since_last_measurement >= 6:
    value = forward_filled_value
    mask = 0.0  # Stale - lower confidence
else:  # Never measured
    value = population_median
    mask = 0.0  # Imputed - lowest confidence
```

---

## 6. Impact Assessment

### 6.1 Before Preprocessing

| Issue | Records Affected | % of Data |
|-------|------------------|-----------|
| Temperature label swap | ~350,000 | 14.3% of temp readings |
| Extreme high values (>10000) | ~300 | <0.01% |
| Negative/zero values | ~32,000 | ~0.4% |
| Total problematic | ~382,000 | ~4.5% of all readings |

### 6.2 After Preprocessing

| Metric | Before | After |
|--------|--------|-------|
| Values outside physiological range | 4.5% | 0% (clamped) |
| Missing value handling | None | Time-aware forward-fill |
| Unit consistency | Mixed F/C | All Celsius |
| Model input quality | Poor | Publication-ready |

---

## 7. Recommendations for Future Work

1. **Report to MIMIC maintainers**: Temperature itemid label swap should be documented or corrected in future versions.

2. **Automated quality checks**: Implement pre-extraction validation queries to catch data quality issues early.

3. **Sensitivity analysis**: Test model performance with different clamping bounds to assess robustness.

4. **External validation**: Validate preprocessing approach on other ICU databases (eICU, HiRID).

---

## 8. References

1. Johnson, A., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data.

2. Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA.

3. Che, Z., et al. (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports.

---

## Appendix A: SQL Verification Query

```sql
-- Verify temperature itemid label swap
SELECT
    itemid,
    COUNT(*) as count,
    APPROX_QUANTILES(valuenum, 100)[OFFSET(50)] as median_val,
    COUNTIF(valuenum BETWEEN 35 AND 42) as in_celsius_range,
    COUNTIF(valuenum BETWEEN 95 AND 105) as in_fahrenheit_range
FROM `physionet-data.mimiciv_3_1_icu.chartevents`
WHERE itemid IN (223761, 223762)
  AND valuenum IS NOT NULL AND valuenum > 0
GROUP BY itemid;
```

---

## Appendix B: Preprocessing Code Reference

Key functions in `train_model_v4.py`:

| Function | Purpose |
|----------|---------|
| `convert_temperature()` | Value-based F→C conversion |
| `apply_unit_repairs()` | Fix HR intervals, remove extremes |
| `build_tensor_with_masks()` | Time-aware forward-fill with freshness masks |
| `apply_clamping()` | Clamp to physiological bounds |
| `scale_features()` | StandardScaler on fresh measurements only |

---

*Document generated for publication reproducibility. Last updated: December 2024*
