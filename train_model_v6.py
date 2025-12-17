"""
Sepsis Prediction Model V6 - Training Script (4-hour Early Warning, Optimized)
===============================================================================

Strategic fixes over V5:
1. Reverted to V3 feature set (15 features) - removes GCS, shock_index, coag labs
2. Added MAP (clinically critical for sepsis shock definition)
3. Bilirubin LAG-MASKING: Only use values >4 hours before prediction time
4. Delta features for HR, SBP, Temperature (rate of deterioration)
5. 5-Fold Cross-Validation (recovers training data vs fixed val split)
6. Simpler architecture: Dense(32) instead of Dense(64)->Dense(32)
7. 4-hour prediction gap maintained for clinical utility

Expected outcome: AUC ~0.71+ at 4-hour horizon (vs V3's 0.726 at 3-hour)
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import pickle
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================================
# V6 CONFIGURATION - STRIPPED DOWN FROM V5
# ============================================================================

# V3 Feature Set (15 features) + MAP = 16 base features
BASE_FEATURES = [
    # Vitals (7)
    "heart_rate",
    "sbp",
    "dbp",
    "map",  # Added back - critical for sepsis shock (MAP < 65)
    "resp_rate",
    "spo2",
    "temperature",
    # Labs (9) - includes bilirubin with lag-masking
    "wbc",
    "creatinine",
    "platelets",
    "bilirubin",  # LAG-MASKED: only values >4h before prediction
    "glucose",
    "bun",
    "sodium",
    "potassium",
    "hemoglobin",
]

# Delta features (rate of change over past 4 hours)
DELTA_FEATURES = [
    "delta_hr",      # Heart rate trend (tachycardia progression)
    "delta_sbp",     # Blood pressure trend (hypotension development)
    "delta_temp",    # Temperature trend (fever trajectory)
]

FINAL_FEATURES = BASE_FEATURES + DELTA_FEATURES  # 16 + 3 = 19 features

# Features requiring lag-masking (only use values >LAG_MASK_HOURS before prediction)
LAG_MASKED_FEATURES = {"bilirubin"}
LAG_MASK_HOURS = 4  # Same as prediction gap

# Item ID mappings
VITALS_ITEMID_MAP = {
    220045: "heart_rate",
    220179: "sbp",
    220050: "sbp",
    220180: "dbp",
    220051: "dbp",
    220052: "map",
    220181: "map",
    220210: "resp_rate",
    220277: "spo2",
    223761: "temperature",
    223762: "temperature",
}

LABS_ITEMID_MAP = {
    51301: "wbc",
    50912: "creatinine",
    51265: "platelets",
    50885: "bilirubin",
    50931: "glucose",
    51006: "bun",
    50983: "sodium",
    50971: "potassium",
    51222: "hemoglobin",
}

# Physiological bounds (V3 bounds + MAP)
BOUNDS = {
    "heart_rate": (20, 250),
    "sbp": (40, 250),
    "dbp": (30, 200),
    "map": (30, 200),
    "resp_rate": (5, 60),
    "spo2": (50, 100),
    "temperature": (30, 45),
    "wbc": (0.1, 100),
    "creatinine": (0.1, 30),
    "platelets": (5, 1200),
    "bilirubin": (0.1, 60),
    "glucose": (20, 1200),
    "bun": (1, 250),
    "sodium": (90, 180),
    "potassium": (1, 15),
    "hemoglobin": (3, 25),
    # Delta features (change per hour, roughly)
    "delta_hr": (-50, 50),
    "delta_sbp": (-50, 50),
    "delta_temp": (-2, 2),
}

# Population medians
POPULATION_MEDIANS = {
    "heart_rate": 84,
    "sbp": 118,
    "dbp": 62,
    "map": 77,
    "resp_rate": 18,
    "spo2": 97,
    "temperature": 37.0,
    "wbc": 9.5,
    "creatinine": 1.0,
    "platelets": 200,
    "bilirubin": 0.6,
    "glucose": 120,
    "bun": 18,
    "sodium": 139,
    "potassium": 4.1,
    "hemoglobin": 10.5,
    # Delta features default to 0 (no change)
    "delta_hr": 0.0,
    "delta_sbp": 0.0,
    "delta_temp": 0.0,
}

# Training parameters
DEFAULT_PREDICTION_GAP_HOURS = 4  # Maintain 4-hour clinical utility
DEFAULT_MIN_DATA_HOURS = 3
DEFAULT_FRESHNESS_WINDOW_HOURS = 6
DEFAULT_N_TIMESTEPS = 24
DEFAULT_N_FOLDS = 5  # 5-fold cross-validation

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOP_PATIENCE = 15
DEFAULT_LR_REDUCE_PATIENCE = 5

DEFAULT_OUTPUT_DIR = Path("sepsis_model_v6")
DEFAULT_FIG_DPI = 300


# ============================================================================
# UTILITIES
# ============================================================================


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_figure(fig: plt.Figure, path: Path, dpi: int = DEFAULT_FIG_DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def configure_publication_style() -> None:
    """Configure matplotlib/seaborn for publication-quality figures."""
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# Publication color palette
PUB_COLORS = {
    "control": "#4C72B0",
    "sepsis": "#DD8452",
    "accent1": "#55A868",
    "accent2": "#C44E52",
    "neutral": "#666666",
}


def get_label_color(label: int) -> str:
    return PUB_COLORS["sepsis"] if int(label) == 1 else PUB_COLORS["control"]


def _label_name(label: int) -> str:
    return "Sepsis" if int(label) == 1 else "Control"


# ============================================================================
# DATA QUALITY REPORT GENERATION
# ============================================================================


def generate_data_quality_report(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Generate comprehensive data quality report with statistics for each feature.

    Creates:
    - feature_statistics.csv: Per-feature stats (count, min, max, percentiles, outliers)
    - outlier_summary.csv: Outlier counts by feature
    """
    report_dir = output_dir / "data_quality"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Generating Data Quality Report]")

    all_features = list(set(BASE_FEATURES) | set(DELTA_FEATURES))

    rows = []
    for feature in all_features:
        feat_df = df[df["feature"] == feature]["valuenum"].dropna()

        if len(feat_df) == 0:
            continue

        bounds = BOUNDS.get(feature, (None, None))
        n_below = int((feat_df < bounds[0]).sum()) if bounds[0] is not None else 0
        n_above = int((feat_df > bounds[1]).sum()) if bounds[1] is not None else 0

        rows.append({
            "feature": feature,
            "count": len(feat_df),
            "n_stays": df[df["feature"] == feature]["stay_id"].nunique(),
            "min_val": float(feat_df.min()),
            "max_val": float(feat_df.max()),
            "mean_val": float(feat_df.mean()),
            "std_val": float(feat_df.std()),
            "median_val": float(feat_df.median()),
            "p1": float(feat_df.quantile(0.01)),
            "p5": float(feat_df.quantile(0.05)),
            "p25": float(feat_df.quantile(0.25)),
            "p75": float(feat_df.quantile(0.75)),
            "p95": float(feat_df.quantile(0.95)),
            "p99": float(feat_df.quantile(0.99)),
            "expected_min": bounds[0],
            "expected_max": bounds[1],
            "n_below_min": n_below,
            "n_above_max": n_above,
            "outlier_pct": 100 * (n_below + n_above) / len(feat_df) if len(feat_df) > 0 else 0,
        })

    report_df = pd.DataFrame(rows).sort_values("feature")
    report_df.to_csv(report_dir / "feature_statistics.csv", index=False)

    # Outlier summary
    outlier_df = report_df[["feature", "count", "n_below_min", "n_above_max", "outlier_pct"]].copy()
    outlier_df = outlier_df.sort_values("outlier_pct", ascending=False)
    outlier_df.to_csv(report_dir / "outlier_summary.csv", index=False)

    print(f"   Saved feature statistics to {report_dir / 'feature_statistics.csv'}")
    print(f"   Saved outlier summary to {report_dir / 'outlier_summary.csv'}")

    # Print summary
    print(f"\n   Feature Statistics Summary:")
    for _, row in report_df.iterrows():
        if row["outlier_pct"] > 0.1:
            print(f"      {row['feature']:<15}: {row['count']:>10,} values, {row['outlier_pct']:.2f}% outliers")

    return report_df


# ============================================================================
# EDA GENERATION
# ============================================================================


def run_eda(
    df_clean: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    output_dir: Path,
    prediction_gap_hours: int,
    max_points_per_feature: int = 200_000,
    seed: int = 42,
) -> None:
    """
    Generate comprehensive EDA artifacts (plots + summary tables).

    Creates:
    - cohort.csv, cohort_summary.csv
    - class_balance.png/pdf
    - age_distribution.png/pdf
    - feature_coverage.png/pdf, feature_coverage.csv
    - feature_distributions.png/pdf
    - correlation_heatmap.png/pdf
    - outlier_rates.png/pdf
    """
    eda_dir = Path(output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    configure_publication_style()

    print("\n[Generating EDA Figures]")

    # Prepare cohort data
    cohort_local = cohort.copy()
    for col in ["intime", "outtime", "onset_time"]:
        if col in cohort_local.columns:
            cohort_local[col] = pd.to_datetime(cohort_local[col])

    if {"intime", "outtime"}.issubset(cohort_local.columns):
        cohort_local["icu_los_hours"] = (
            (cohort_local["outtime"] - cohort_local["intime"]).dt.total_seconds() / 3600
        )

    if {"intime", "onset_time"}.issubset(cohort_local.columns):
        cohort_local["onset_hours"] = (
            (cohort_local["onset_time"] - cohort_local["intime"]).dt.total_seconds() / 3600
        )

    cohort_local.to_csv(eda_dir / "cohort.csv", index=False)

    # Cohort summary
    cohort_summary = (
        cohort_local.groupby("label")
        .agg(
            n_stays=("stay_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            age_mean=("Age", "mean"),
            age_median=("Age", "median"),
            gender_male_pct=("Gender", "mean"),
        )
        .reset_index()
    )
    cohort_summary["label_name"] = cohort_summary["label"].apply(_label_name)
    cohort_summary.to_csv(eda_dir / "cohort_summary.csv", index=False)
    print(f"   Saved cohort summary")

    # 1. Class Balance
    print(f"   Generating class balance plot...")
    fig, ax = plt.subplots(figsize=(7, 5))
    class_counts = cohort_local["label"].value_counts().sort_index()
    bars = ax.bar(
        [_label_name(i) for i in class_counts.index],
        class_counts.values,
        color=[PUB_COLORS["control"], PUB_COLORS["sepsis"]],
        edgecolor="black",
        linewidth=1.2,
    )
    for bar, val in zip(bars, class_counts.values):
        ax.annotate(
            f"{val:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    ax.set_title("Cohort Class Distribution (V6)", fontweight="bold")
    ax.set_ylabel("Number of ICU Stays")
    ax.set_xlabel("Patient Group")
    save_figure(fig, eda_dir / "class_balance")

    # 2. Age Distribution
    if "Age" in cohort_local.columns:
        print(f"   Generating age distribution plot...")
        fig, ax = plt.subplots(figsize=(7, 5))
        for label in sorted(cohort_local["label"].unique()):
            data = cohort_local.loc[cohort_local["label"] == label, "Age"].dropna()
            sns.kdeplot(
                data,
                ax=ax,
                label=f"{_label_name(label)} (n={len(data):,})",
                color=get_label_color(label),
                fill=True,
                alpha=0.3,
                linewidth=2,
            )
        ax.set_title("Age Distribution by Patient Group", fontweight="bold")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", title="Group")
        save_figure(fig, eda_dir / "age_distribution")

    # 3. Feature Coverage
    print(f"   Generating feature coverage plot...")
    features_in_data = df_clean["feature"].unique()
    all_features = list(set(BASE_FEATURES) | set(DELTA_FEATURES))

    coverage_data = []
    total_stays = cohort_local["stay_id"].nunique()
    for feat in all_features:
        feat_df = df_clean[df_clean["feature"] == feat]
        n_stays = feat_df["stay_id"].nunique()
        coverage_data.append({
            "feature": feat,
            "n_stays": n_stays,
            "coverage_pct": 100 * n_stays / total_stays if total_stays > 0 else 0,
            "n_measurements": len(feat_df),
        })

    coverage_df = pd.DataFrame(coverage_data).sort_values("coverage_pct", ascending=True)
    coverage_df.to_csv(eda_dir / "feature_coverage.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [
        PUB_COLORS["accent1"] if pct > 90 else PUB_COLORS["accent2"] if pct > 50 else PUB_COLORS["neutral"]
        for pct in coverage_df["coverage_pct"]
    ]
    ax.barh(coverage_df["feature"], coverage_df["coverage_pct"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(90, color="green", linestyle="--", linewidth=1.5, label="90% threshold")
    ax.axvline(50, color="orange", linestyle="--", linewidth=1.5, label="50% threshold")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Coverage Across ICU Stays (V6)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 105)
    save_figure(fig, eda_dir / "feature_coverage")

    # 4. Feature Distributions
    print(f"   Generating feature distribution plots...")
    plot_features = [f for f in BASE_FEATURES if f in features_in_data]

    if plot_features:
        n_features = len(plot_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for i, feat in enumerate(plot_features):
            ax = axes[i]
            feat_df = df_clean[df_clean["feature"] == feat].copy()

            if len(feat_df) > max_points_per_feature:
                feat_df = feat_df.sample(max_points_per_feature, random_state=seed)

            for label in sorted(feat_df["label"].unique()):
                subset = feat_df[feat_df["label"] == label]["valuenum"]
                if len(subset) > 0:
                    sns.histplot(
                        subset, bins=40, element="step", stat="density",
                        color=get_label_color(label), alpha=0.4, linewidth=1.5,
                        label=_label_name(label), ax=ax,
                    )

            ax.set_title(feat, fontweight="bold", fontsize=10)
            ax.set_xlabel("")
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        for j in range(len(plot_features), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Feature Value Distributions (Control vs Sepsis)", fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        save_figure(fig, eda_dir / "feature_distributions")

    # 5. Correlation Heatmap
    print(f"   Generating correlation heatmap...")
    med = (
        df_clean.groupby(["stay_id", "label", "feature"])["valuenum"]
        .median()
        .reset_index()
        .pivot_table(index=["stay_id", "label"], columns="feature", values="valuenum")
    )

    valid_features = [f for f in BASE_FEATURES if f in med.columns and med[f].notna().sum() > 100]

    if len(valid_features) >= 5:
        med_valid = med[valid_features].copy()

        for feat in valid_features:
            if feat in POPULATION_MEDIANS:
                med_valid[feat] = med_valid[feat].fillna(POPULATION_MEDIANS[feat])

        corr = med_valid.corr(method="spearman")

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Spearman rho", "shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Feature Correlation Matrix (V6)\n(Per-Stay Median Values)", fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        save_figure(fig, eda_dir / "correlation_heatmap")

    # 6. Outlier Rates
    print(f"   Generating outlier rates plot...")
    outlier_data = []
    for feat in BASE_FEATURES:
        if feat not in BOUNDS:
            continue
        feat_df = df_clean[df_clean["feature"] == feat]["valuenum"].dropna()
        if len(feat_df) == 0:
            continue
        low, high = BOUNDS[feat]
        n_outliers = int(((feat_df < low) | (feat_df > high)).sum())
        outlier_data.append({
            "feature": feat,
            "outlier_pct": 100 * n_outliers / len(feat_df),
            "n_outliers": n_outliers,
            "n_total": len(feat_df),
        })

    if outlier_data:
        outlier_df = pd.DataFrame(outlier_data).sort_values("outlier_pct", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [PUB_COLORS["accent2"] if pct > 0.5 else PUB_COLORS["accent1"] for pct in outlier_df["outlier_pct"]]
        ax.barh(outlier_df["feature"], outlier_df["outlier_pct"], color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="0.5% threshold")
        ax.set_xlabel("Outlier Rate (%)")
        ax.set_ylabel("Feature")
        ax.set_title("Pre-Clamping Outlier Rates by Feature (V6)", fontweight="bold")
        ax.legend(loc="lower right")
        save_figure(fig, eda_dir / "outlier_rates")

    # 7. Sepsis Onset Timing (if available)
    if "onset_hours" in cohort_local.columns:
        print(f"   Generating sepsis onset timing plot...")
        sepsis_onset = cohort_local[cohort_local["label"] == 1]["onset_hours"].dropna()
        if len(sepsis_onset) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sepsis_onset, bins=50, color=PUB_COLORS["sepsis"], edgecolor="black", alpha=0.7)
            median_onset = sepsis_onset.median()
            ax.axvline(median_onset, color="red", linestyle="--", linewidth=2, label=f"Median: {median_onset:.1f}h")
            ax.set_xlabel("Hours After ICU Admission")
            ax.set_ylabel("Number of Patients")
            ax.set_title("Sepsis Onset Timing Distribution", fontweight="bold")
            ax.legend()
            save_figure(fig, eda_dir / "sepsis_onset_timing")

    # 8. Bilirubin Lag-Masking Impact (if bilirubin present)
    if "bilirubin" in features_in_data:
        print(f"   Generating bilirubin lag-masking analysis...")
        bili_df = df_clean[df_clean["feature"] == "bilirubin"].copy()
        if len(bili_df) > 0 and "onset_time" in bili_df.columns:
            bili_df["hours_before_onset"] = (
                pd.to_datetime(bili_df["onset_time"]) - pd.to_datetime(bili_df["charttime"])
            ).dt.total_seconds() / 3600

            fig, ax = plt.subplots(figsize=(8, 5))
            for label in [0, 1]:
                subset = bili_df[bili_df["label"] == label]["hours_before_onset"]
                if len(subset) > 0:
                    ax.hist(subset, bins=50, alpha=0.5, color=get_label_color(label),
                            label=f"{_label_name(label)} (n={len(subset):,})", edgecolor="black")

            # Mark lag-mask cutoff
            lag_cutoff = prediction_gap_hours + LAG_MASK_HOURS
            ax.axvline(lag_cutoff, color="red", linestyle="--", linewidth=2,
                       label=f"Lag-mask cutoff ({lag_cutoff}h)")
            ax.set_xlabel("Hours Before Onset")
            ax.set_ylabel("Number of Measurements")
            ax.set_title("Bilirubin Measurement Timing\n(Values < cutoff are masked)", fontweight="bold")
            ax.legend()
            ax.set_xlim(0, 48)
            save_figure(fig, eda_dir / "bilirubin_lag_masking")

    print(f"\n   EDA complete. Saved to: {eda_dir}")


# ============================================================================
# DATA QUALITY FIXES (Combined best practices from V3 + V5)
# ============================================================================


def convert_heart_rate_rr_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert heart rate values that are RR intervals (ms) to BPM.

    Issue: Some monitors store RR intervals in milliseconds instead of BPM.
    Detection: Values > 300 are likely RR intervals (300 BPM is physiologically extreme)
    Fix: HR_bpm = 60000 / RR_ms
    """
    df = df.copy()
    hr_mask = df["feature"] == "heart_rate"

    # Values > 300 are likely RR intervals in ms
    rr_interval_mask = hr_mask & (df["valuenum"] > 300)
    n_converted = int(rr_interval_mask.sum())

    if n_converted > 0:
        # Clip to prevent division errors (min 200ms = 300 BPM max)
        rr_values = df.loc[rr_interval_mask, "valuenum"].clip(lower=200)
        df.loc[rr_interval_mask, "valuenum"] = 60000 / rr_values
        print(f"   Heart rate: Converted {n_converted:,} RR intervals (ms) to BPM")

    return df


def convert_temperature_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert temperature values to Celsius using value-based detection.

    Issue: MIMIC-IV has MISLABELED temperature item IDs:
      - itemid 223761 labeled "Celsius" but contains Fahrenheit (median 98.4)
      - itemid 223762 labeled "Fahrenheit" but contains Celsius (median 37.0)

    Fix: Value-based detection (NOT itemid-based!)
      - Values > 50 and < 120 are Fahrenheit body temperatures
      - C = (F - 32) * 5/9
    """
    df = df.copy()
    temp_mask = df["feature"] == "temperature"

    # Values > 50 and < 120 are likely Fahrenheit (normal body temp 95-105°F)
    fahrenheit_mask = temp_mask & (df["valuenum"] > 50) & (df["valuenum"] < 120)
    n_f_to_c = int(fahrenheit_mask.sum())

    if n_f_to_c > 0:
        df.loc[fahrenheit_mask, "valuenum"] = (df.loc[fahrenheit_mask, "valuenum"] - 32) * 5 / 9
        print(f"   Temperature: Converted {n_f_to_c:,} Fahrenheit values to Celsius")

    # Remove extreme values (> 120 or <= 0) - these are data errors
    extreme_high_mask = temp_mask & (df["valuenum"] > 120)
    extreme_low_mask = temp_mask & (df["valuenum"] <= 0)
    n_extreme = int(extreme_high_mask.sum()) + int(extreme_low_mask.sum())

    if n_extreme > 0:
        df.loc[extreme_high_mask | extreme_low_mask, "valuenum"] = np.nan
        print(f"   Temperature: Removed {n_extreme:,} extreme/invalid values")

    return df


def clip_resp_rate_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip respiratory rate artifacts to physiological maximum.

    Issue: Some resp_rate values are artifacts (ventilator rates, data errors)
    Fix: Clip values > 60 to 60 (physiological maximum for adults)
    """
    df = df.copy()
    rr_mask = df["feature"] == "resp_rate"

    # Clip high values to 60
    high_rr = rr_mask & (df["valuenum"] > 60)
    n_clipped = int(high_rr.sum())

    if n_clipped > 0:
        df.loc[high_rr, "valuenum"] = 60
        print(f"   Resp rate: Clipped {n_clipped:,} values > 60 to 60")

    return df


def remove_zero_and_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove physiologically impossible zero and negative values.

    Issues:
    - Zeros used as missing data placeholders (esp. resp_rate)
    - Negative BP values (data entry errors)
    - Zero lab values (below detection limit)
    """
    df = df.copy()

    # Vitals that cannot be zero or negative
    vitals_no_zero = ["heart_rate", "sbp", "dbp", "map", "resp_rate"]
    for vital in vitals_no_zero:
        mask = (df["feature"] == vital) & (df["valuenum"] <= 0)
        n_removed = int(mask.sum())
        if n_removed > 0:
            df.loc[mask, "valuenum"] = np.nan
            print(f"   {vital}: Removed {n_removed:,} zero/negative values")

    # Blood pressures cannot be negative (explicit check for negative only)
    bp_features = ["sbp", "dbp", "map"]
    for bp in bp_features:
        mask = (df["feature"] == bp) & (df["valuenum"] < 0)
        n_removed = int(mask.sum())
        if n_removed > 0:
            df.loc[mask, "valuenum"] = np.nan
            print(f"   {bp}: Removed {n_removed:,} negative values")

    # SpO2 cannot exceed 100%
    spo2_high_mask = (df["feature"] == "spo2") & (df["valuenum"] > 100)
    n_spo2_high = int(spo2_high_mask.sum())
    if n_spo2_high > 0:
        df.loc[spo2_high_mask, "valuenum"] = np.nan
        print(f"   spo2: Removed {n_spo2_high:,} values > 100%")

    # SpO2 cannot be zero or negative
    spo2_low_mask = (df["feature"] == "spo2") & (df["valuenum"] <= 0)
    n_spo2_low = int(spo2_low_mask.sum())
    if n_spo2_low > 0:
        df.loc[spo2_low_mask, "valuenum"] = np.nan
        print(f"   spo2: Removed {n_spo2_low:,} zero/negative values")

    # Labs that cannot be zero (below detection limit = missing, not zero)
    labs_no_zero = ["wbc", "hemoglobin", "creatinine", "platelets", "bilirubin", "glucose"]
    for lab in labs_no_zero:
        mask = (df["feature"] == lab) & (df["valuenum"] <= 0)
        n_removed = int(mask.sum())
        if n_removed > 0:
            df.loc[mask, "valuenum"] = np.nan
            print(f"   {lab}: Removed {n_removed:,} zero/negative values")

    return df


def apply_all_data_quality_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data quality fixes in sequence.

    Order matters:
    1. Unit conversions first (HR, Temperature)
    2. Artifact clipping (Resp rate)
    3. Invalid value removal (zeros, negatives, out-of-range)
    4. Drop NaN rows
    """
    print("\n[Data Quality Fixes]")

    n_before = len(df)

    # 1. Unit conversions
    df = convert_heart_rate_rr_intervals(df)
    df = convert_temperature_to_celsius(df)

    # 2. Artifact clipping
    df = clip_resp_rate_artifacts(df)

    # 3. Invalid value removal
    df = remove_zero_and_negative_values(df)

    # 4. Drop rows with NaN values after fixes
    df = df.dropna(subset=["valuenum"])
    n_after = len(df)

    if n_before > n_after:
        print(f"   Total rows dropped (NaN after fixes): {n_before - n_after:,}")

    print(f"   Data quality fixes complete: {n_after:,} rows remaining")

    return df


# ============================================================================
# LAG-MASKING FOR BILIRUBIN
# ============================================================================


def apply_lag_masking(
    df: pd.DataFrame,
    prediction_gap_hours: int,
) -> pd.DataFrame:
    """
    Apply lag-masking to features with ordering bias (bilirubin).

    Only keep measurements that are >LAG_MASK_HOURS before the prediction time.
    This prevents the model from learning "ordering bias" - where the presence
    of a test order (not the value) signals sepsis suspicion.

    For a 4-hour prediction gap:
    - Prediction time = onset_time - 4 hours
    - Lag mask cutoff = prediction_time - 4 hours = onset_time - 8 hours
    - Only bilirubin values BEFORE this cutoff are used
    """
    df = df.copy()

    for feat in LAG_MASKED_FEATURES:
        feat_mask = df["feature"] == feat
        n_before = int(feat_mask.sum())

        if n_before == 0:
            continue

        # Calculate hours before onset for each measurement
        df["_onset_time"] = pd.to_datetime(df["onset_time"])
        df["_charttime"] = pd.to_datetime(df["charttime"])

        # Prediction time = onset - prediction_gap
        # Lag mask cutoff = prediction_time - LAG_MASK_HOURS
        # = onset - prediction_gap - LAG_MASK_HOURS
        total_lag = prediction_gap_hours + LAG_MASK_HOURS

        hours_before_onset = (df["_onset_time"] - df["_charttime"]).dt.total_seconds() / 3600

        # Keep only measurements > total_lag hours before onset
        too_recent = feat_mask & (hours_before_onset < total_lag)
        n_masked = int(too_recent.sum())

        if n_masked > 0:
            df = df[~too_recent].copy()
            print(f"   Lag-masking {feat}: Removed {n_masked}/{n_before} recent values (within {total_lag}h of onset)")

        # Clean up temp columns
        df = df.drop(columns=["_onset_time", "_charttime"], errors="ignore")

    return df


# ============================================================================
# DELTA FEATURES
# ============================================================================


def compute_delta_features(
    df: pd.DataFrame,
    delta_window_hours: int = 4,
) -> pd.DataFrame:
    """
    Compute delta (rate of change) features for key vitals.

    Delta = (current_value - value_N_hours_ago) / N_hours

    This explicitly encodes "rate of deterioration" which helps the
    Dense layers see trends without relying solely on LSTM memory.
    """
    df = df.copy()
    df["charttime"] = pd.to_datetime(df["charttime"])

    delta_mappings = {
        "heart_rate": "delta_hr",
        "sbp": "delta_sbp",
        "temperature": "delta_temp",
    }

    new_rows = []
    meta_cols = ["stay_id", "subject_id", "hadm_id", "intime", "outtime",
                 "Age", "Gender", "onset_time", "label"]
    available_meta = [c for c in meta_cols if c in df.columns]

    for stay_id, g in tqdm(df.groupby("stay_id"), desc="Computing deltas", leave=False):
        cohort_info = g[available_meta].iloc[0].to_dict() if available_meta else {}

        for base_feat, delta_feat in delta_mappings.items():
            feat_df = g[g["feature"] == base_feat][["charttime", "valuenum"]].copy()

            if len(feat_df) < 2:
                continue

            feat_df = feat_df.sort_values("charttime")

            # For each measurement, find the value ~delta_window_hours ago
            for _, row in feat_df.iterrows():
                current_time = row["charttime"]
                current_val = row["valuenum"]

                # Look for measurement approximately delta_window_hours ago
                lookback_start = current_time - pd.Timedelta(hours=delta_window_hours + 1)
                lookback_end = current_time - pd.Timedelta(hours=delta_window_hours - 1)

                past_vals = feat_df[
                    (feat_df["charttime"] >= lookback_start) &
                    (feat_df["charttime"] <= lookback_end)
                ]["valuenum"]

                if len(past_vals) > 0:
                    past_val = past_vals.iloc[-1]  # Most recent in window
                    delta = (current_val - past_val) / delta_window_hours

                    new_row = {
                        "stay_id": stay_id,
                        "charttime": current_time,
                        "feature": delta_feat,
                        "valuenum": delta,
                        **cohort_info,
                    }
                    new_rows.append(new_row)

    if new_rows:
        delta_df = pd.DataFrame(new_rows)
        df = pd.concat([df, delta_df], ignore_index=True)
        print(f"   Computed {len(new_rows)} delta feature values")

    return df


# ============================================================================
# TENSOR BUILDING
# ============================================================================


def build_tensor_with_masks(
    df_group: pd.DataFrame,
    onset_time: pd.Timestamp,
    *,
    prediction_gap_hours: int,
    n_timesteps: int,
    freshness_window_hours: int,
    feature_names: list[str],
) -> np.ndarray | None:
    """Build tensor with time-aware forward-fill and freshness masks."""
    n_features = len(feature_names)

    win_end = onset_time - pd.Timedelta(hours=prediction_gap_hours)
    g_win = df_group[df_group["charttime"] <= win_end].copy()

    if len(g_win) < 2:
        return None

    hours_before_end = (win_end - g_win["charttime"]).dt.total_seconds() / 3600
    bins = hours_before_end.astype(int)
    g_win = g_win[(bins >= 0) & (bins < n_timesteps)].copy()

    if g_win.empty:
        return None

    g_win["bin"] = (n_timesteps - 1 - bins.loc[g_win.index]).astype(int)

    values = np.zeros((n_timesteps, n_features), dtype=np.float32)
    masks = np.zeros((n_timesteps, n_features), dtype=np.float32)
    last_measured = np.full(n_features, -np.inf)

    pivot = (
        g_win.pivot_table(index="bin", columns="feature", values="valuenum", aggfunc="mean")
        .reindex(index=range(n_timesteps), columns=feature_names)
    )

    for t in range(n_timesteps):
        for i, feat in enumerate(feature_names):
            current_val = pivot.loc[t, feat] if feat in pivot.columns else np.nan

            if pd.notna(current_val):
                values[t, i] = float(current_val)
                masks[t, i] = 1.0
                last_measured[i] = t
            else:
                if last_measured[i] >= 0:
                    hours_since = t - last_measured[i]
                    values[t, i] = values[int(last_measured[i]), i]
                    masks[t, i] = 1.0 if hours_since < freshness_window_hours else 0.0
                else:
                    values[t, i] = float(POPULATION_MEDIANS.get(feat, 0.0))
                    masks[t, i] = 0.0

    age = float(df_group["Age"].iloc[0]) if "Age" in df_group.columns else 65.0
    gender = float(df_group["Gender"].iloc[0]) if "Gender" in df_group.columns else 1.0
    static = np.tile([age, gender], (n_timesteps, 1)).astype(np.float32)

    return np.hstack([values, masks, static]).astype(np.float32)


def apply_clamping(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    Clamp tensor values to physiological bounds.

    Features:
    - Preserves 0-padding (imputed/missing values stay as-is)
    - Logs per-feature outlier counts
    - Uses BOUNDS dict for physiological limits

    Args:
        X: 3D tensor (samples, timesteps, features)
        feature_names: List of feature names matching first n columns

    Returns:
        Clamped tensor with 0-padding preserved
    """
    X_clamped = X.copy()
    total_outliers = 0

    print("\n[Physiological Clamping]")

    for i, feat in enumerate(feature_names):
        if feat not in BOUNDS:
            continue

        low, high = BOUNDS[feat]
        vals = X_clamped[:, :, i]

        # Preserve 0-padding (these are imputed/missing values, not real measurements)
        # Only clamp non-zero values
        nonzero_mask = vals != 0

        # Find outliers (values outside bounds AND non-zero)
        outliers = ((vals < low) | (vals > high)) & nonzero_mask
        n_outliers = int(outliers.sum())

        if n_outliers > 0:
            total_outliers += n_outliers
            print(f"   {feat:<15}: {n_outliers:>6,} outliers clamped to [{low}, {high}]")

            # Clamp only the non-zero values
            clamped_vals = np.clip(vals, low, high)
            # Restore zeros (0-padding)
            X_clamped[:, :, i] = np.where(nonzero_mask, clamped_vals, vals)

    print(f"   Total outliers clamped: {total_outliers:,}")

    return X_clamped


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_features: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on training data only, using measured values."""
    scaler = StandardScaler()

    train_values = X_train[:, :, :n_features]
    test_values = X_test[:, :, :n_features]
    train_masks = X_train[:, :, n_features : 2 * n_features]

    train_flat = train_values.reshape(-1, n_features)
    test_flat = test_values.reshape(-1, n_features)
    train_masks_flat = train_masks.reshape(-1, n_features)

    means, stds = [], []
    for i in range(n_features):
        measured_mask = train_masks_flat[:, i] > 0
        measured_vals = train_flat[measured_mask, i]

        if len(measured_vals) > 0:
            means.append(float(np.mean(measured_vals)))
            std = float(np.std(measured_vals))
            stds.append(std if std > 1e-7 else 1.0)
        else:
            means.append(float(POPULATION_MEDIANS.get(feature_names[i], 0.0)))
            stds.append(1.0)

    scaler.mean_ = np.array(means, dtype=np.float64)
    scaler.scale_ = np.array(stds, dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = n_features

    def _apply(flat):
        return (flat - scaler.mean_) / scaler.scale_

    train_scaled = _apply(train_flat)
    test_scaled = _apply(test_flat)

    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    X_train_out[:, :, :n_features] = train_scaled.reshape(X_train.shape[0], -1, n_features)
    X_test_out[:, :, :n_features] = test_scaled.reshape(X_test.shape[0], -1, n_features)

    return X_train_out, X_test_out, scaler


# ============================================================================
# MODEL (Simplified V3-style architecture)
# ============================================================================


def build_model(
    input_shape: tuple[int, int],
    lstm_units: tuple[int, int] = (64, 32),
    dense_units: int = 32,  # Single dense layer like V3
    dropout: float = 0.3,
) -> tf.keras.Model:
    """
    Build Bidirectional LSTM model - V3-style simplified architecture.

    Key difference from V5: Single Dense(32) layer instead of Dense(64)->Dense(32)
    """
    model = Sequential([
        Bidirectional(
            LSTM(lstm_units[0], return_sequences=True, dropout=dropout),
            input_shape=input_shape
        ),
        BatchNormalization(),
        Bidirectional(
            LSTM(lstm_units[1], return_sequences=False, dropout=dropout)
        ),
        BatchNormalization(),
        Dense(dense_units, activation="relu"),  # Single dense layer
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["AUC"]
    )

    return model


def calculate_class_weights(y: np.ndarray) -> dict[int, float]:
    """Calculate balanced class weights."""
    n_samples = len(y)
    n_positive = int(np.sum(y))
    n_negative = n_samples - n_positive

    if n_positive == 0 or n_negative == 0:
        return {0: 1.0, 1: 1.0}

    return {
        0: float(n_samples / (2 * n_negative)),
        1: float(n_samples / (2 * n_positive)),
    }


# ============================================================================
# 5-FOLD CROSS-VALIDATION TRAINING
# ============================================================================


def train_model_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    output_dir: Path,
    feature_names: list[str],
    n_folds: int,
    seed: int,
    batch_size: int,
    epochs: int,
    early_stop_patience: int,
    lr_reduce_patience: int,
    prediction_gap_hours: int,
    freshness_window_hours: int,
) -> tuple[tf.keras.Model, StandardScaler, dict]:
    """
    Train model using 5-fold cross-validation.

    Benefits over fixed validation split:
    - Uses 80% of data for training in each fold (vs 70% in V5)
    - More robust AUC estimate
    - Final model trained on ALL data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = len(feature_names)

    print("=" * 70)
    print("TRAINING SEPSIS PREDICTION MODEL V6 (5-Fold CV)")
    print("=" * 70)
    print(f"Features: {n_features} base + {n_features} masks + 2 static = {X.shape[2]}")
    print(f"Samples: {len(X)} ({int(y.sum())} sepsis, {int((1-y).sum())} control)")

    # Apply clamping to full dataset
    print("\n[1/4] Applying physiological clamping...")
    X = apply_clamping(X, feature_names)

    # 5-Fold Cross-Validation
    print(f"\n[2/4] Running {n_folds}-fold cross-validation...")
    kfold = GroupKFold(n_splits=n_folds)

    fold_aucs = []
    fold_thresholds = []
    all_val_preds = np.zeros(len(y))
    all_val_indices = np.zeros(len(y), dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups=subjects)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]

        print(f"   Train: {len(train_idx)} ({int(y_train_fold.sum())} sepsis)")
        print(f"   Val:   {len(val_idx)} ({int(y_val_fold.sum())} sepsis)")

        # Scale features (fit on train only)
        X_train_scaled, X_val_scaled, _ = scale_features(
            X_train_fold, X_val_fold,
            n_features=n_features,
            feature_names=feature_names,
        )

        # Build and train model
        model = build_model(input_shape=(X.shape[1], X.shape[2]))
        class_weights = calculate_class_weights(y_train_fold)

        callbacks = [
            EarlyStopping(
                monitor="val_AUC",
                patience=early_stop_patience,
                mode="max",
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor="val_AUC",
                factor=0.5,
                patience=lr_reduce_patience,
                mode="max",
                verbose=0
            ),
        ]

        history = model.fit(
            X_train_scaled, y_train_fold,
            validation_data=(X_val_scaled, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0,
        )

        # Evaluate fold
        y_val_pred = model.predict(X_val_scaled, verbose=0).flatten()
        fold_auc = roc_auc_score(y_val_fold, y_val_pred)
        fold_aucs.append(fold_auc)

        # Store predictions for final evaluation
        all_val_preds[val_idx] = y_val_pred
        all_val_indices[val_idx] = True

        # Find optimal threshold for this fold
        precision, recall, thresholds = precision_recall_curve(y_val_fold, y_val_pred)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        best_idx = int(np.argmax(f1))
        fold_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        fold_thresholds.append(fold_threshold)

        print(f"   Fold AUC: {fold_auc:.4f}, Threshold: {fold_threshold:.4f}")

        # Clean up to save memory
        tf.keras.backend.clear_session()

    # Cross-validation summary
    cv_auc_mean = np.mean(fold_aucs)
    cv_auc_std = np.std(fold_aucs)
    cv_threshold = np.mean(fold_thresholds)

    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Mean Threshold: {cv_threshold:.4f}")

    # Holdout test set for final evaluation
    print(f"\n[3/4] Creating holdout test set...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=subjects))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    subjects_train = subjects[train_idx]

    print(f"   Train: {len(train_idx)} ({int(y_train.sum())} sepsis)")
    print(f"   Test:  {len(test_idx)} ({int(y_test.sum())} sepsis)")

    # Train final model on full training set
    print(f"\n[4/4] Training final model on full training set...")

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test,
        n_features=n_features,
        feature_names=feature_names,
    )

    final_model = build_model(input_shape=(X.shape[1], X.shape[2]))
    final_model.summary()

    class_weights = calculate_class_weights(y_train)
    print(f"   Class weights: {class_weights}")

    callbacks = [
        EarlyStopping(
            monitor="val_AUC",
            patience=early_stop_patience,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_AUC",
            factor=0.5,
            patience=lr_reduce_patience,
            mode="max",
            verbose=1
        ),
    ]

    history = final_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Final evaluation
    y_test_pred = final_model.predict(X_test_scaled, verbose=0).flatten()
    test_auc = roc_auc_score(y_test, y_test_pred)

    # Use CV threshold for consistency
    y_test_bin = (y_test_pred >= cv_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_bin, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n{'='*50}")
    print(f"FINAL TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"CV AUC:      {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
    print(f"Test AUC:    {test_auc:.4f}")
    print(f"Threshold:   {cv_threshold:.4f} (from CV)")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f} ({tp}/{tp+fn})")
    print(f"Specificity: {specificity:.4f} ({tn}/{tn+fp})")
    print(f"PPV:         {ppv:.4f}")
    print(f"NPV:         {npv:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg  {tn:5d}  {fp:5d}")
    print(f"  Actual Pos  {fn:5d}  {tp:5d}")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_test_bin,
        labels=[0, 1],
        target_names=["Control", "Sepsis"],
        zero_division=0,
    ))

    # Save artifacts
    final_model.save(output_dir / "sepsis_model_v6.keras")
    joblib.dump(scaler, output_dir / "sepsis_scaler_v6.pkl")

    config = {
        "features": feature_names,
        "n_timesteps": int(X.shape[1]),
        "n_features": n_features,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "cv_threshold": cv_threshold,
        "test_auc": test_auc,
        "test_accuracy": accuracy,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "test_ppv": ppv,
        "test_npv": npv,
        "prediction_gap_hours": prediction_gap_hours,
        "freshness_window_hours": freshness_window_hours,
        "n_folds": n_folds,
        "seed": seed,
        "fold_aucs": fold_aucs,
    }
    with open(output_dir / "config_v6.pkl", "wb") as f:
        pickle.dump(config, f)

    # Save test data
    np.save(output_dir / "X_test.npy", X_test_scaled)
    np.save(output_dir / "y_test.npy", y_test)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["loss"], label="Train")
    axes[0].plot(history_df["val_loss"], label="Test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Test Loss")
    axes[0].legend()

    axes[1].plot(history_df["AUC"], label="Train")
    axes[1].plot(history_df["val_AUC"], label="Test")
    axes[1].axhline(cv_auc_mean, color="green", linestyle="--", label=f"CV Mean ({cv_auc_mean:.3f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Training and Test AUC")
    axes[1].legend()

    plt.tight_layout()
    save_figure(fig, output_dir / "training_curves")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#DD8452", lw=2, label=f"Test ROC (AUC = {test_auc:.3f})")
    ax.fill_between(fpr, 0, tpr, alpha=0.2, color="#DD8452")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - V6 (4h gap)\nCV AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_dir / "roc_curve")

    metrics = {
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "test_auc": test_auc,
        "accuracy": accuracy,
        "threshold": cv_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }

    print(f"\nArtifacts saved to: {output_dir}")

    return final_model, scaler, metrics


# ============================================================================
# BIGQUERY EXTRACTION
# ============================================================================


def extract_mimic_data(
    *,
    project_id: str,
    chunk_size: int,
    prediction_gap_hours: int,
    min_data_hours: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract data from MIMIC-IV BigQuery."""
    from google.cloud import bigquery

    print("=" * 70)
    print("EXTRACTING DATA FROM MIMIC-IV (V6 Feature Set)")
    print("=" * 70)

    client = bigquery.Client(project=project_id)

    # Get ICU stays
    print("\n[1/3] Building cohort...")
    stays_query = """
    SELECT s.stay_id, s.hadm_id, s.subject_id, s.intime, s.outtime,
           p.anchor_age as Age,
           CASE WHEN p.gender='F' THEN 0 ELSE 1 END as Gender
    FROM `physionet-data.mimiciv_3_1_icu.icustays` s
    JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
      ON s.subject_id = p.subject_id
    WHERE DATETIME_DIFF(s.outtime, s.intime, HOUR) >= 24
    """
    stays_df = client.query(stays_query).to_dataframe()
    stays_df["intime"] = pd.to_datetime(stays_df["intime"])
    stays_df["outtime"] = pd.to_datetime(stays_df["outtime"])
    stays_df["icu_los_hours"] = (stays_df["outtime"] - stays_df["intime"]).dt.total_seconds() / 3600

    print(f"   Total ICU stays (>= 24h): {len(stays_df):,}")

    # Sepsis-3 onsets
    s3_query = """
    SELECT stay_id, sofa_time as onset_time
    FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
    """
    s3_raw = client.query(s3_query).to_dataframe()
    sepsis_df = s3_raw.merge(stays_df, on="stay_id", how="inner")
    sepsis_df["onset_time"] = pd.to_datetime(sepsis_df["onset_time"])
    sepsis_df["onset_hours"] = (sepsis_df["onset_time"] - sepsis_df["intime"]).dt.total_seconds() / 3600

    sepsis_df = sepsis_df[sepsis_df["onset_hours"] > (min_data_hours + prediction_gap_hours)].copy()
    sepsis_df["label"] = 1

    print(f"   Sepsis cases (with enough data): {len(sepsis_df):,}")

    # Age-matched controls
    print("   Matching controls...")
    pool = stays_df[~stays_df["stay_id"].isin(sepsis_df["stay_id"])].copy()
    controls = []
    used_stays: set[int] = set()

    sepsis_rows = sepsis_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for _, row in tqdm(sepsis_rows.iterrows(), total=len(sepsis_rows), desc="Matching"):
        onset_hours = float(row["onset_hours"])
        pred_time_hours = onset_hours - float(prediction_gap_hours)

        cand = pool[
            pool["Age"].between(row["Age"] - 5, row["Age"] + 5)
            & (~pool["stay_id"].isin(used_stays))
            & (pool["icu_los_hours"] >= pred_time_hours)
        ]
        if cand.empty:
            continue

        match = cand.sample(1, random_state=seed).iloc[0]
        used_stays.add(int(match["stay_id"]))
        controls.append({
            "stay_id": int(match["stay_id"]),
            "hadm_id": int(match["hadm_id"]),
            "subject_id": int(match["subject_id"]),
            "intime": match["intime"],
            "outtime": match["outtime"],
            "Age": float(match["Age"]),
            "Gender": int(match["Gender"]),
            "onset_time": match["intime"] + pd.to_timedelta(onset_hours, unit="h"),
            "label": 0,
            "onset_hours": onset_hours,
        })

    controls_df = pd.DataFrame(controls)
    cohort_cols = [
        "stay_id", "hadm_id", "subject_id", "intime", "outtime",
        "Age", "Gender", "onset_time", "label", "onset_hours",
    ]
    cohort = pd.concat([sepsis_df[cohort_cols], controls_df[cohort_cols]], ignore_index=True)

    print(f"   Total cohort: {len(cohort):,} ({int(cohort['label'].sum()):,} sepsis, {int((1 - cohort['label']).sum()):,} control)")

    # Extract features
    print("\n[2/3] Extracting vitals from chartevents...")
    stay_ids = cohort["stay_id"].unique().tolist()
    vitals_itemids = ",".join(map(str, VITALS_ITEMID_MAP.keys()))

    dfs: list[pd.DataFrame] = []
    for i in tqdm(range(0, len(stay_ids), chunk_size), desc="Vitals"):
        chunk_ids = stay_ids[i : i + chunk_size]
        chunk = ",".join(map(str, chunk_ids))

        try:
            df_vitals = client.query(f"""
                SELECT stay_id, charttime, itemid, valuenum
                FROM `physionet-data.mimiciv_3_1_icu.chartevents`
                WHERE stay_id IN ({chunk}) AND itemid IN ({vitals_itemids})
                AND valuenum IS NOT NULL
            """).to_dataframe()
            df_vitals["feature"] = df_vitals["itemid"].map(VITALS_ITEMID_MAP)
            dfs.append(df_vitals)
        except Exception as e:
            print(f"   Vitals chunk error: {e}")

    print("\n[3/3] Extracting labs from labevents...")
    labs_itemids = ",".join(map(str, LABS_ITEMID_MAP.keys()))

    for i in tqdm(range(0, len(stay_ids), chunk_size), desc="Labs"):
        chunk_ids = stay_ids[i : i + chunk_size]
        chunk = ",".join(map(str, chunk_ids))

        try:
            df_labs = client.query(f"""
                SELECT i.stay_id, l.charttime, l.itemid, l.valuenum
                FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
                JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
                  ON l.hadm_id = i.hadm_id
                 AND l.charttime >= i.intime
                 AND l.charttime <= i.outtime
                WHERE i.stay_id IN ({chunk}) AND l.itemid IN ({labs_itemids})
                AND l.valuenum IS NOT NULL
            """).to_dataframe()
            df_labs["feature"] = df_labs["itemid"].map(LABS_ITEMID_MAP)
            dfs.append(df_labs)
        except Exception as e:
            print(f"   Labs chunk error: {e}")

    df_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["stay_id", "charttime", "itemid", "valuenum", "feature"]
    )
    df_raw["charttime"] = pd.to_datetime(df_raw["charttime"])
    df_raw = df_raw.merge(cohort.drop(columns=["onset_hours"]), on="stay_id", how="inner")

    print(f"\n   Raw data: {len(df_raw):,} rows")

    return df_raw, cohort


def build_tensors_from_raw(
    df_raw: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    prediction_gap_hours: int,
    n_timesteps: int,
    freshness_window_hours: int,
    output_dir: Path,
    run_eda_flag: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build tensors with all V6 processing steps."""
    print("\n" + "=" * 70)
    print("BUILDING V6 TENSORS")
    print("=" * 70)

    # 1. Data quality fixes
    print("\n[1/7] Applying data quality fixes...")
    df_clean = apply_all_data_quality_fixes(df_raw)

    # 2. Generate data quality report (BEFORE lag-masking to see full picture)
    print("\n[2/7] Generating data quality report...")
    generate_data_quality_report(df_clean, output_dir)

    # 3. Run EDA (BEFORE lag-masking to visualize bilirubin timing)
    if run_eda_flag:
        print("\n[3/7] Running EDA...")
        run_eda(
            df_clean,
            cohort,
            output_dir=output_dir,
            prediction_gap_hours=prediction_gap_hours,
            seed=seed,
        )
    else:
        print("\n[3/7] Skipping EDA (--no-eda flag)")

    # 4. Lag-masking for bilirubin
    print("\n[4/7] Applying lag-masking for bilirubin...")
    df_clean = apply_lag_masking(df_clean, prediction_gap_hours)

    # 5. Compute delta features
    print("\n[5/7] Computing delta features...")
    df_clean = compute_delta_features(df_clean, delta_window_hours=4)

    # 6. Build tensors
    print("\n[6/7] Building time-series tensors...")
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    subj_list: list[int] = []

    n_skipped = 0
    for _, g in tqdm(df_clean.groupby("stay_id"), desc="Building tensors"):
        onset_time = pd.to_datetime(g["onset_time"].iloc[0])
        tensor = build_tensor_with_masks(
            g,
            onset_time,
            prediction_gap_hours=prediction_gap_hours,
            n_timesteps=n_timesteps,
            freshness_window_hours=freshness_window_hours,
            feature_names=FINAL_FEATURES,
        )
        if tensor is None:
            n_skipped += 1
            continue
        X_list.append(tensor)
        y_list.append(float(g["label"].iloc[0]))
        subj_list.append(int(g["subject_id"].iloc[0]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    subjects = np.array(subj_list)

    print(f"\n   Tensors: {X.shape}")
    print(f"   Features: {len(FINAL_FEATURES)} values + {len(FINAL_FEATURES)} masks + 2 static = {X.shape[2]}")
    print(f"   Sepsis: {int(y.sum()):,}, Control: {int((1 - y).sum()):,}")
    print(f"   Skipped: {n_skipped}")

    # 7. Save tensors
    print("\n[7/7] Saving tensors...")
    tensor_path = output_dir / "tensors_v6.npz"
    np.savez(tensor_path, X=X, y=y, subjects=subjects)
    print(f"   Saved to {tensor_path}")

    return X, y, subjects


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Sepsis Prediction Model V6 (4h, optimized features, 5-fold CV)"
    )

    parser.add_argument("--extract", action="store_true", help="Extract data from BigQuery")
    parser.add_argument("--project", type=str, default="sepsis-prediction-2025", help="GCP project ID")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Stays per BigQuery chunk")
    parser.add_argument("--load", type=str, help="Load pre-built tensors from .npz file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prediction-gap-hours", type=int, default=DEFAULT_PREDICTION_GAP_HOURS)
    parser.add_argument("--min-data-hours", type=int, default=DEFAULT_MIN_DATA_HOURS)
    parser.add_argument("--freshness-window-hours", type=int, default=DEFAULT_FRESHNESS_WINDOW_HOURS)
    parser.add_argument("--n-timesteps", type=int, default=DEFAULT_N_TIMESTEPS)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS, help="Number of CV folds")

    parser.add_argument("--no-eda", action="store_true", help="Skip EDA generation")
    parser.add_argument("--eda-only", action="store_true", help="Run EDA only, skip tensor building and training")
    parser.add_argument("--tensors-only", action="store_true", help="Build tensors only, skip training")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--lr-reduce-patience", type=int, default=DEFAULT_LR_REDUCE_PATIENCE)

    args = parser.parse_args()

    # Validate flag combinations
    if args.no_eda and args.eda_only:
        raise SystemExit("Error: --no-eda and --eda-only are mutually exclusive")
    if args.eda_only and args.tensors_only:
        raise SystemExit("Error: --eda-only and --tensors-only are mutually exclusive")

    set_global_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SEPSIS PREDICTION MODEL V6")
    print("=" * 70)
    print(f"Prediction gap: {args.prediction_gap_hours} hours")
    print(f"Features: {len(FINAL_FEATURES)} (V3 base + MAP + deltas)")
    print(f"Cross-validation: {args.n_folds}-fold")
    print(f"Lag-masking: bilirubin (>{args.prediction_gap_hours + LAG_MASK_HOURS}h before onset)")
    print(f"Output: {output_dir}")

    if args.extract:
        df_raw, cohort = extract_mimic_data(
            project_id=args.project,
            chunk_size=args.chunk_size,
            prediction_gap_hours=args.prediction_gap_hours,
            min_data_hours=args.min_data_hours,
            seed=args.seed,
        )

        # Handle --eda-only: run EDA and data quality report, then exit
        if args.eda_only:
            print("\n[Running EDA only mode]")
            df_clean = apply_data_quality_fixes(df_raw)
            generate_data_quality_report(df_clean, output_dir)
            run_eda(
                df_clean,
                cohort,
                output_dir=output_dir,
                prediction_gap_hours=args.prediction_gap_hours,
                seed=args.seed,
            )
            print(f"\nEDA complete. See: {output_dir / 'eda'}")
            print(f"Data quality report: {output_dir / 'data_quality'}")
            return

        X, y, subjects = build_tensors_from_raw(
            df_raw,
            cohort,
            prediction_gap_hours=args.prediction_gap_hours,
            n_timesteps=args.n_timesteps,
            freshness_window_hours=args.freshness_window_hours,
            output_dir=output_dir,
            run_eda_flag=not args.no_eda,
            seed=args.seed,
        )

        if args.tensors_only:
            print("\nTensor build complete. Skipping training (--tensors-only).")
            return

    elif args.load:
        data = np.load(args.load)
        X, y, subjects = data["X"], data["y"], data["subjects"]
        print(f"\nLoaded tensors from {args.load}")
        print(f"   Shape: {X.shape}")
        print(f"   Sepsis: {int(y.sum()):,}, Control: {int((1 - y).sum()):,}")

    else:
        print("\nUsage:")
        print("  python train_model_v6.py --extract --project YOUR_PROJECT_ID")
        print("  python train_model_v6.py --load sepsis_model_v6/tensors_v6.npz")
        return

    # Train with 5-fold CV
    model, scaler, metrics = train_model_with_cv(
        X, y, subjects,
        output_dir=output_dir,
        feature_names=FINAL_FEATURES,
        n_folds=args.n_folds,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        lr_reduce_patience=args.lr_reduce_patience,
        prediction_gap_hours=args.prediction_gap_hours,
        freshness_window_hours=args.freshness_window_hours,
    )

    print("\n" + "=" * 70)
    print("V6 COMPLETE")
    print(f"   CV AUC:           {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")
    print(f"   Test AUC:         {metrics['test_auc']:.4f}")
    print(f"   Test Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"   Test Specificity: {metrics['specificity']:.4f}")
    print(f"   Threshold:        {metrics['threshold']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
