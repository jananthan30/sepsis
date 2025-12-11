"""
Sepsis Prediction Model V3 - Training Script
=============================================

This script trains a Bidirectional LSTM model for early sepsis prediction using MIMIC-IV data.

Key Features:
- Time-aware forward-fill with 6-hour freshness window
- 15 clinical features + 15 freshness masks + 2 static features = 32 total
- Proper handling of missing data with mask=0 for never-measured/backfilled data
- Temperature F->C conversion with per-itemid logic
- Scaler fitted only on training-split measured values (no data leakage)
- Class weight balancing for imbalanced datasets

Data Pipeline:
1. Extract data from MIMIC-IV BigQuery
2. Build cohort (sepsis cases + age-matched controls)
3. Apply unit repairs (temperature F->C, HR interval->BPM, etc.)
4. Clamp values to physiological bounds
5. Build 24-hour time series tensors with freshness masks
6. Scale features (training data only)
7. Train Bidirectional LSTM model
8. Save model, scaler, and config

Author: Generated from High_Accurate_Sepsis_3_final.ipynb
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from tqdm import tqdm

# ML imports
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Masking, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature definitions
FINAL_FEATURES = [
    'heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature',
    'wbc', 'creatinine', 'platelets', 'bilirubin', 'glucose',
    'bun', 'sodium', 'potassium', 'hemoglobin'
]

# MIMIC-IV itemid to feature mapping
# Temperature itemids: 223761 (C), 223762 (F) - need separate handling
FEATURE_MAP = {
    # Vitals
    220045: 'heart_rate',
    220179: 'sbp', 220050: 'sbp',  # Systolic BP (invasive/non-invasive)
    220180: 'dbp', 220051: 'dbp',  # Diastolic BP
    220210: 'resp_rate',
    220277: 'spo2',
    223761: 'temperature',  # Celsius
    223762: 'temperature',  # Fahrenheit - needs conversion
    # Labs
    51301: 'wbc',
    50912: 'creatinine',
    51265: 'platelets',
    50885: 'bilirubin',
    50931: 'glucose',
    51006: 'bun',
    50983: 'sodium',
    50971: 'potassium',
    51222: 'hemoglobin'
}

# Itemids that are in Fahrenheit (need conversion to Celsius)
FAHRENHEIT_ITEMIDS = {223762}

# Physiological bounds for clamping outliers
BOUNDS = {
    'heart_rate': (20, 250),
    'sbp': (40, 250),
    'dbp': (30, 200),
    'resp_rate': (5, 60),
    'spo2': (50, 100),
    'temperature': (30, 45),  # Celsius
    'wbc': (0.1, 100),
    'creatinine': (0.1, 30),
    'platelets': (5, 1200),
    'bilirubin': (0.1, 60),
    'glucose': (20, 1200),
    'bun': (1, 250),
    'sodium': (90, 180),
    'potassium': (1, 15),
    'hemoglobin': (3, 25)
}

# Population medians for imputation (from MIMIC-IV)
POPULATION_MEDIANS = {
    'heart_rate': 84, 'sbp': 118, 'dbp': 62, 'resp_rate': 18,
    'spo2': 97, 'temperature': 37.0, 'wbc': 9.5, 'creatinine': 1.0,
    'platelets': 200, 'bilirubin': 0.6, 'glucose': 120, 'bun': 18,
    'sodium': 139, 'potassium': 4.1, 'hemoglobin': 10.5
}

# Training parameters
PREDICTION_GAP = 3  # Hours before sepsis onset to stop using data
MIN_DATA_HOURS = 3  # Minimum hours of data required
FRESHNESS_WINDOW = 6  # Hours - measurements older than this get mask=0
N_TIMESTEPS = 24  # 24-hour lookback window
BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 5

# Output directory
OUTPUT_DIR = Path("sepsis_model_v3")


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def convert_temperature(df_raw):
    """
    Convert Fahrenheit temperatures to Celsius.
    Handles per-itemid logic for temperature conversion.

    Args:
        df_raw: DataFrame with 'itemid' and 'valuenum' columns

    Returns:
        DataFrame with temperatures converted to Celsius
    """
    df = df_raw.copy()

    # Count conversions for logging
    f_to_c_count = 0
    high_temp_clipped = 0

    # Convert Fahrenheit itemids to Celsius
    f_mask = df['itemid'].isin(FAHRENHEIT_ITEMIDS)
    if f_mask.sum() > 0:
        f_to_c_count = f_mask.sum()
        df.loc[f_mask, 'valuenum'] = (df.loc[f_mask, 'valuenum'] - 32) * 5 / 9

    # Also check for Celsius itemid with suspiciously high values (>50)
    # These are likely Fahrenheit recorded with wrong itemid
    temp_mask = (df['feature'] == 'temperature')
    high_temp = temp_mask & (df['valuenum'] > 50)
    if high_temp.sum() > 0:
        high_temp_clipped = high_temp.sum()
        df.loc[high_temp, 'valuenum'] = (df.loc[high_temp, 'valuenum'] - 32) * 5 / 9

    print(f"   Temperature conversion: {f_to_c_count} F->C by itemid, {high_temp_clipped} high values converted")

    return df


def apply_unit_repairs(df_raw):
    """
    Apply unit repairs to raw data.

    Fixes:
    - Temperature: F->C conversion (handled by convert_temperature)
    - Heart Rate: RR interval (ms) -> BPM for values > 300
    - Resp Rate: Clip to max 60
    """
    df = df_raw.copy()

    # Fix Heart Rate (RR Interval in ms -> BPM)
    hr_mask = df['feature'] == 'heart_rate'
    high_hr = hr_mask & (df['valuenum'] > 300)
    if high_hr.sum() > 0:
        print(f"   Converting {high_hr.sum()} HR intervals (ms) to BPM")
        val_clip = df.loc[high_hr, 'valuenum'].clip(lower=1)
        df.loc[high_hr, 'valuenum'] = 60000 / val_clip

    # Fix Resp Rate artifacts
    rr_mask = df['feature'] == 'resp_rate'
    high_rr = rr_mask & (df['valuenum'] > 60)
    if high_rr.sum() > 0:
        print(f"   Clipping {high_rr.sum()} resp rates > 60")
        df.loc[high_rr, 'valuenum'] = 60

    return df


def apply_clamping(X, feature_names):
    """
    Clamp values to physiological bounds.

    Args:
        X: 3D tensor (samples, timesteps, features)
        feature_names: List of feature names

    Returns:
        Clamped tensor with 0-padding preserved
    """
    X_clamped = X.copy()
    total_outliers = 0

    for i, feat_name in enumerate(feature_names):
        if feat_name in BOUNDS:
            low, high = BOUNDS[feat_name]
            vals = X_clamped[:, :, i]
            mask_vals = vals != 0  # Preserve 0 padding

            outliers = ((vals < low) | (vals > high)) & mask_vals
            count = outliers.sum()

            if count > 0:
                total_outliers += count
                print(f"   Clamping {feat_name:<12}: {count} outliers -> [{low}, {high}]")
                X_clamped[:, :, i] = np.clip(vals, low, high)
                # Restore 0 padding
                X_clamped[:, :, i][~mask_vals] = 0

    print(f"   Total outliers clamped: {total_outliers}")
    return X_clamped


def build_tensor_with_masks(df_group, onset_time, prediction_gap=3, n_timesteps=24,
                            freshness_window=6, feature_names=FINAL_FEATURES):
    """
    Build a single patient tensor with time-aware forward-fill and freshness masks.

    Key Innovation: 6-hour freshness window
    - If measurement is < freshness_window hours old: mask=1 (fresh)
    - If measurement is >= freshness_window hours old: mask=0 (stale)
    - Never-measured features: mask=0 throughout
    - Backward-filled (before first measurement): mask=0

    Args:
        df_group: DataFrame for single patient with 'charttime', 'feature', 'valuenum'
        onset_time: Sepsis onset time (or end of stay for controls)
        prediction_gap: Hours before onset to stop using data
        n_timesteps: Number of hourly bins (default 24)
        freshness_window: Hours after which measurements are considered stale
        feature_names: List of feature names

    Returns:
        tensor: (n_timesteps, n_features*2 + 2) - values, masks, age, gender
        or None if insufficient data
    """
    n_features = len(feature_names)

    # Calculate window end (prediction_gap hours before onset)
    win_end = onset_time - pd.Timedelta(hours=prediction_gap)

    # Filter to data before window end
    g_win = df_group[df_group['charttime'] <= win_end].copy()
    if len(g_win) < 2:
        return None

    # Calculate hourly bins (0 = oldest, 23 = most recent)
    g_win['hours_before_end'] = (win_end - g_win['charttime']).dt.total_seconds() / 3600
    g_win['bin'] = g_win['hours_before_end'].apply(lambda x: int(x) if x >= 0 else -1)
    g_win = g_win[g_win['bin'] < n_timesteps]
    g_win = g_win[g_win['bin'] >= 0]
    g_win['bin'] = n_timesteps - 1 - g_win['bin']  # Reverse so 0=oldest, 23=newest

    if len(g_win) == 0:
        return None

    # Initialize arrays
    values = np.zeros((n_timesteps, n_features))
    masks = np.zeros((n_timesteps, n_features))
    last_measured = np.full(n_features, -np.inf)  # Track when each feature was last measured

    # Pivot to get measurements per bin
    pivot = g_win.pivot_table(
        index='bin',
        columns='feature',
        values='valuenum',
        aggfunc='mean'
    ).reindex(index=range(n_timesteps), columns=feature_names)

    # Track which features have ANY measurement
    ever_measured = {feat: False for feat in feature_names}

    # Process each timestep (forward in time: 0=oldest to 23=newest)
    for t in range(n_timesteps):
        for i, feat in enumerate(feature_names):
            current_val = pivot.loc[t, feat] if feat in pivot.columns else np.nan

            if pd.notna(current_val):
                # Fresh measurement at this timestep
                values[t, i] = current_val
                masks[t, i] = 1  # Fresh
                last_measured[i] = t
                ever_measured[feat] = True
            else:
                # No measurement at this timestep
                if last_measured[i] >= 0:
                    # We have a previous measurement - forward fill
                    hours_since = t - last_measured[i]
                    values[t, i] = values[int(last_measured[i]), i]

                    # Freshness check
                    if hours_since < freshness_window:
                        masks[t, i] = 1  # Still fresh
                    else:
                        masks[t, i] = 0  # Stale
                else:
                    # Never measured before - use population median, mask=0
                    values[t, i] = POPULATION_MEDIANS.get(feat, 0)
                    masks[t, i] = 0  # Never measured = not fresh

    # Handle backward-fill case: if first measurement is after t=0,
    # earlier values should have mask=0 (they were backfilled, not measured)
    for i, feat in enumerate(feature_names):
        if ever_measured[feat]:
            # Find first measurement time
            feat_measurements = g_win[g_win['feature'] == feat]['bin']
            if len(feat_measurements) > 0:
                first_t = feat_measurements.min()
                # Set mask=0 for times before first measurement (backfilled)
                for t in range(int(first_t)):
                    masks[t, i] = 0

    # Get static features (Age, Gender) - should be same across all rows
    age = df_group['Age'].iloc[0] if 'Age' in df_group.columns else 65
    gender = df_group['Gender'].iloc[0] if 'Gender' in df_group.columns else 1
    static = np.tile([age, gender], (n_timesteps, 1))

    # Combine: [values (15), masks (15), static (2)] = 32 features
    tensor = np.hstack([values, masks, static])

    return tensor


def scale_features(X_train, X_test, n_features=15):
    """
    Scale features using StandardScaler fitted ONLY on training data.

    Important: Only fit on measured values (non-zero) to avoid bias from
    imputed values. Guards against zero variance.

    Args:
        X_train: Training tensor (samples, timesteps, total_features)
        X_test: Test tensor
        n_features: Number of clinical features to scale (default 15)

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()

    # Extract only the value columns (first n_features)
    train_values = X_train[:, :, :n_features]
    test_values = X_test[:, :, :n_features]

    # Extract masks to identify measured values
    train_masks = X_train[:, :, n_features:2*n_features]

    # Flatten for scaling
    train_flat = train_values.reshape(-1, n_features)
    test_flat = test_values.reshape(-1, n_features)
    train_masks_flat = train_masks.reshape(-1, n_features)

    # Calculate mean and std from MEASURED values only (mask=1 in training data)
    means = []
    stds = []
    sample_counts = []

    for i in range(n_features):
        # Get measured values (where mask > 0)
        measured_mask = train_masks_flat[:, i] > 0
        measured_vals = train_flat[measured_mask, i]

        sample_counts.append(len(measured_vals))

        if len(measured_vals) > 0:
            means.append(np.mean(measured_vals))
            std_val = np.std(measured_vals)
            # Guard against zero variance
            stds.append(std_val if std_val > 1e-7 else 1.0)
        else:
            # No measured values - use population defaults
            feat_name = FINAL_FEATURES[i]
            means.append(POPULATION_MEDIANS.get(feat_name, 0))
            stds.append(1.0)

    # Set scaler attributes manually
    scaler.mean_ = np.array(means)
    scaler.scale_ = np.array(stds)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = np.array(sample_counts)

    print(f"   Scaler fitted on training data:")
    for i, feat in enumerate(FINAL_FEATURES):
        print(f"      {feat:<12}: mean={means[i]:.2f}, std={stds[i]:.2f}, n={sample_counts[i]}")

    # Apply scaling
    train_scaled = (train_flat - scaler.mean_) / scaler.scale_
    test_scaled = (test_flat - scaler.mean_) / scaler.scale_

    # Reshape back
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[:, :, :n_features] = train_scaled.reshape(X_train.shape[0], -1, n_features)
    X_test_scaled[:, :, :n_features] = test_scaled.reshape(X_test.shape[0], -1, n_features)

    return X_train_scaled, X_test_scaled, scaler


def build_model(input_shape=(24, 32), lstm_units=(64, 32), dropout=0.3):
    """
    Build Bidirectional LSTM model for sepsis prediction.

    Architecture:
    - Masking layer (ignores 0-padded timesteps)
    - Bidirectional LSTM (64 units) with dropout
    - BatchNormalization
    - Bidirectional LSTM (32 units) with dropout
    - BatchNormalization
    - Dense (32 units, ReLU)
    - Dropout
    - Dense (1 unit, Sigmoid) - binary classification

    Args:
        input_shape: (timesteps, features)
        lstm_units: Tuple of LSTM layer sizes
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Masking layer - ignores timesteps where all features are 0
        Masking(mask_value=0.0, input_shape=input_shape),

        # First LSTM layer
        Bidirectional(LSTM(lstm_units[0], return_sequences=True, dropout=dropout)),
        BatchNormalization(),

        # Second LSTM layer
        Bidirectional(LSTM(lstm_units[1], return_sequences=False, dropout=dropout)),
        BatchNormalization(),

        # Dense layers
        Dense(32, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    return model


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced dataset.
    Guards against division by zero.

    Args:
        y: Binary labels array

    Returns:
        Dictionary with class weights {0: weight, 1: weight}
    """
    n_samples = len(y)
    n_positive = np.sum(y)
    n_negative = n_samples - n_positive

    # Guard against zero
    if n_positive == 0 or n_negative == 0:
        print("   Warning: One class has zero samples, using equal weights")
        return {0: 1.0, 1: 1.0}

    # Balanced weights
    weight_positive = n_samples / (2 * n_positive)
    weight_negative = n_samples / (2 * n_negative)

    print(f"   Class weights: negative={weight_negative:.2f}, positive={weight_positive:.2f}")

    return {0: weight_negative, 1: weight_positive}


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model_from_tensors(X, y, subjects, output_dir=OUTPUT_DIR):
    """
    Train the sepsis prediction model from pre-built tensors.

    Args:
        X: Input tensor (samples, 24, 32)
        y: Labels (samples,)
        subjects: Subject IDs for group-aware splitting
        output_dir: Directory to save outputs

    Returns:
        model, scaler, history, metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("TRAINING SEPSIS PREDICTION MODEL V3")
    print("=" * 70)

    # 1. Split data (subject-aware to prevent data leakage)
    print("\n[1/5] Splitting data (subject-aware)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=subjects))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"   Train: {X_train.shape[0]} samples ({y_train.sum():.0f} sepsis)")
    print(f"   Test:  {X_test.shape[0]} samples ({y_test.sum():.0f} sepsis)")

    # 2. Apply clamping
    print("\n[2/5] Applying physiological clamping...")
    X_train = apply_clamping(X_train, FINAL_FEATURES)
    X_test = apply_clamping(X_test, FINAL_FEATURES)

    # 3. Scale features (training data only)
    print("\n[3/5] Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 4. Build model
    print("\n[4/5] Building model...")
    model = build_model(input_shape=(N_TIMESTEPS, 32))
    model.summary()

    # 5. Train
    print("\n[5/5] Training model...")

    class_weights = calculate_class_weights(y_train)

    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=EARLY_STOP_PATIENCE,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=LR_REDUCE_PATIENCE,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    y_pred = model.predict(X_test, verbose=0).flatten()
    auc = roc_auc_score(y_test, y_pred)

    # Binary predictions at optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    y_pred_binary = (y_pred >= optimal_threshold).astype(int)

    print(f"\nTest AUC: {auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=['Control', 'Sepsis']))

    # Calculate accuracy
    accuracy = np.mean(y_pred_binary == y_test)

    # Save outputs
    print("\n[Saving outputs...]")

    # Save model
    model.save(output_dir / "sepsis_model_v3.keras")
    print(f"   Model saved to {output_dir / 'sepsis_model_v3.keras'}")

    # Save scaler
    joblib.dump(scaler, output_dir / "sepsis_scaler_v3.pkl")
    print(f"   Scaler saved to {output_dir / 'sepsis_scaler_v3.pkl'}")

    # Save config
    config = {
        'freshness_window': FRESHNESS_WINDOW,
        'n_timesteps': N_TIMESTEPS,
        'features': FINAL_FEATURES,
        'population_medians': POPULATION_MEDIANS,
        'auc': auc,
        'accuracy': accuracy
    }
    with open(output_dir / "config.pkl", 'wb') as f:
        pickle.dump(config, f)
    print(f"   Config saved to {output_dir / 'config.pkl'}")

    # Save validation data
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    print(f"   Validation data saved")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - AUC: {auc:.4f}")
    print("=" * 70)

    return model, scaler, history, {'auc': auc, 'accuracy': accuracy}


# ============================================================================
# DATA EXTRACTION (requires BigQuery access)
# ============================================================================

def extract_mimic_data(project_id="sepsis-prediction-2025", chunk_size=5000):
    """
    Extract data from MIMIC-IV BigQuery.

    Requires:
    - Google Cloud credentials configured
    - Access to physionet-data MIMIC-IV tables

    Args:
        project_id: GCP project ID
        chunk_size: Number of stay_ids per query chunk

    Returns:
        df_raw: Raw data DataFrame
        cohort: Cohort DataFrame with labels
    """
    from google.cloud import bigquery

    print("=" * 70)
    print("EXTRACTING DATA FROM MIMIC-IV")
    print("=" * 70)

    client = bigquery.Client(project=project_id)

    # 1. Build cohort
    print("[1/2] Building cohort...")

    # Get ICU stays >= 24 hours
    stays_query = """
    SELECT s.stay_id, s.subject_id, s.intime, s.outtime,
           p.anchor_age as Age,
           CASE WHEN p.gender='F' THEN 0 ELSE 1 END as Gender
    FROM `physionet-data.mimiciv_3_1_icu.icustays` s
    JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
        ON s.subject_id = p.subject_id
    WHERE DATETIME_DIFF(s.outtime, s.intime, HOUR) >= 24
    """
    stays_df = client.query(stays_query).to_dataframe()
    stays_df['intime'] = pd.to_datetime(stays_df['intime'])

    # Get sepsis-3 cases
    s3_query = """
    SELECT stay_id, sofa_time as onset_time
    FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
    """
    s3_raw = client.query(s3_query).to_dataframe()

    # Merge and filter
    sepsis_df = s3_raw.merge(stays_df, on='stay_id', how='inner')
    sepsis_df['onset_time'] = pd.to_datetime(sepsis_df['onset_time'])
    sepsis_df['onset_hours'] = (sepsis_df['onset_time'] - sepsis_df['intime']).dt.total_seconds() / 3600
    sepsis_df = sepsis_df[sepsis_df['onset_hours'] > (MIN_DATA_HOURS + PREDICTION_GAP)].copy()
    sepsis_df['label'] = 1

    print(f"   Sepsis cases: {len(sepsis_df)}")

    # Match controls (age-matched)
    print(f"   Matching controls...")
    pool = stays_df[~stays_df['stay_id'].isin(sepsis_df['stay_id'])].copy()
    controls = []
    used = set()

    for _, row in tqdm(sepsis_df.iterrows(), total=len(sepsis_df), desc="Matching"):
        cand = pool[
            (pool['Age'].between(row['Age']-5, row['Age']+5)) &
            (~pool['stay_id'].isin(used))
        ]
        if not cand.empty:
            match = cand.sample(1).iloc[0]
            used.add(match['stay_id'])
            controls.append({
                'stay_id': match['stay_id'],
                'subject_id': match['subject_id'],
                'intime': match['intime'],
                'Age': match['Age'],
                'Gender': match['Gender'],
                'onset_time': match['intime'] + pd.to_timedelta(row['onset_hours'], unit='h'),
                'label': 0
            })

    cohort = pd.concat([
        sepsis_df[['stay_id', 'subject_id', 'onset_time', 'Age', 'Gender', 'label']],
        pd.DataFrame(controls)
    ])

    print(f"   Total cohort: {len(cohort)} ({cohort['label'].sum()} sepsis, {(1-cohort['label']).sum()} control)")

    # 2. Extract features
    print("\n[2/2] Extracting vitals & labs...")

    stay_ids = cohort['stay_id'].unique().tolist()
    dfs = []

    for i in tqdm(range(0, len(stay_ids), chunk_size), desc="Extracting"):
        chunk = ",".join(map(str, stay_ids[i:i+chunk_size]))
        itemids = ",".join(map(str, FEATURE_MAP.keys()))

        # Chart events (vitals)
        try:
            df_chart = client.query(f"""
                SELECT stay_id, charttime, itemid, valuenum
                FROM `physionet-data.mimiciv_3_1_icu.chartevents`
                WHERE stay_id IN ({chunk}) AND itemid IN ({itemids})
            """).to_dataframe()
            dfs.append(df_chart)
        except Exception as e:
            print(f"   Chart events error: {e}")

        # Lab events
        try:
            df_labs = client.query(f"""
                SELECT i.stay_id, l.charttime, l.itemid, l.valuenum
                FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
                JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
                    ON l.subject_id = i.subject_id
                WHERE i.stay_id IN ({chunk}) AND l.itemid IN ({itemids})
            """).to_dataframe()
            dfs.append(df_labs)
        except Exception as e:
            print(f"   Lab events error: {e}")

    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw['feature'] = df_raw['itemid'].map(FEATURE_MAP)
    df_raw['charttime'] = pd.to_datetime(df_raw['charttime'])
    df_raw = df_raw.merge(cohort, on='stay_id')

    print(f"\n   Raw data: {len(df_raw)} rows")

    return df_raw, cohort


def build_tensors_from_raw(df_raw, cohort):
    """
    Build tensors from raw extracted data.

    Args:
        df_raw: Raw data with features
        cohort: Cohort with labels

    Returns:
        X, y, subjects arrays
    """
    print("\n" + "=" * 70)
    print("BUILDING TENSORS")
    print("=" * 70)

    # Apply temperature conversion
    print("\n[1/3] Converting temperatures...")
    df_clean = convert_temperature(df_raw)

    # Apply unit repairs
    print("\n[2/3] Applying unit repairs...")
    df_clean = apply_unit_repairs(df_clean)

    # Build tensors
    print("\n[3/3] Building time-series tensors...")
    X_list, y_list, sub_list = [], [], []

    for stay_id, g in tqdm(df_clean.groupby('stay_id'), desc="Building tensors"):
        try:
            onset_time = g['onset_time'].iloc[0]
            tensor = build_tensor_with_masks(
                g, onset_time,
                prediction_gap=PREDICTION_GAP,
                n_timesteps=N_TIMESTEPS,
                freshness_window=FRESHNESS_WINDOW
            )

            if tensor is not None:
                X_list.append(tensor)
                y_list.append(g['label'].iloc[0])
                sub_list.append(g['subject_id'].iloc[0])
        except Exception as e:
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    subjects = np.array(sub_list)

    print(f"\n   Tensors built: {X.shape}")
    print(f"   Sepsis: {y.sum():.0f}, Control: {(1-y).sum():.0f}")

    return X, y, subjects


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Sepsis Prediction Model V3")
    parser.add_argument("--extract", action="store_true", help="Extract data from BigQuery")
    parser.add_argument("--project", type=str, default="sepsis-prediction-2025", help="GCP project ID")
    parser.add_argument("--load", type=str, help="Load pre-built tensors from .npz file")
    parser.add_argument("--output", type=str, default="sepsis_model_v3", help="Output directory")

    args = parser.parse_args()

    if args.extract:
        # Extract from BigQuery and train
        df_raw, cohort = extract_mimic_data(project_id=args.project)
        X, y, subjects = build_tensors_from_raw(df_raw, cohort)

        # Save intermediate tensors
        np.savez(f"{args.output}/tensors.npz", X=X, y=y, subjects=subjects)
        print(f"Tensors saved to {args.output}/tensors.npz")

        # Train
        model, scaler, history, metrics = train_model_from_tensors(X, y, subjects, args.output)

    elif args.load:
        # Load pre-built tensors and train
        data = np.load(args.load)
        X, y, subjects = data['X'], data['y'], data['subjects']
        model, scaler, history, metrics = train_model_from_tensors(X, y, subjects, args.output)

    else:
        print("Usage:")
        print("  python train_model_v3.py --extract --project YOUR_PROJECT_ID")
        print("  python train_model_v3.py --load tensors.npz")
        print("\nRun with --extract to download from BigQuery, or --load to use pre-built tensors.")
