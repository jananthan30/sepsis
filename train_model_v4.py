"""
Sepsis Prediction Model V4 - Training Script (6-hour Early Warning)
===================================================================

Changes vs V3:
- Predicts sepsis 6 hours before onset by default (`--prediction-gap-hours 6`)
- Adds extensive EDA + visualization BEFORE tensorization/scaling
- Uses subject-aware Train/Val/Test split (avoids using test set for early stopping)
- Selects decision threshold on validation set (avoids test-set tuning)
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
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
# CONFIGURATION
# ============================================================================

FINAL_FEATURES = [
    "heart_rate",
    "sbp",
    "dbp",
    "resp_rate",
    "spo2",
    "temperature",
    "wbc",
    "creatinine",
    "platelets",
    "bilirubin",
    "glucose",
    "bun",
    "sodium",
    "potassium",
    "hemoglobin",
]

FEATURE_MAP = {
    # Vitals
    220045: "heart_rate",
    220179: "sbp",
    220050: "sbp",
    220180: "dbp",
    220051: "dbp",
    220210: "resp_rate",
    220277: "spo2",
    # NOTE: MIMIC-IV labels are SWAPPED! Verified via BigQuery analysis:
    # - 223761 labeled "Celsius" but contains FAHRENHEIT (median 98.4, 99.8% in 95-105F)
    # - 223762 labeled "Fahrenheit" but contains CELSIUS (median 37.0, 96.9% in 35-42C)
    223761: "temperature",  # Actually Fahrenheit despite MIMIC label!
    223762: "temperature",  # Actually Celsius despite MIMIC label!
    # Labs
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

# IMPORTANT: MIMIC-IV has swapped labels! 223761 is actually Fahrenheit.
# We use value-based detection in convert_temperature() instead of relying on itemid.
FAHRENHEIT_ITEMIDS = {223761}  # Despite MIMIC labeling this as "Celsius"

BOUNDS = {
    "heart_rate": (20, 250),
    "sbp": (40, 250),
    "dbp": (30, 200),
    "resp_rate": (5, 60),
    "spo2": (50, 100),
    "temperature": (30, 45),  # Celsius
    "wbc": (0.1, 100),
    "creatinine": (0.1, 30),
    "platelets": (5, 1200),
    "bilirubin": (0.1, 60),
    "glucose": (20, 1200),
    "bun": (1, 250),
    "sodium": (90, 180),
    "potassium": (1, 15),
    "hemoglobin": (3, 25),
}

POPULATION_MEDIANS = {
    "heart_rate": 84,
    "sbp": 118,
    "dbp": 62,
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
}

DEFAULT_PREDICTION_GAP_HOURS = 6
DEFAULT_MIN_DATA_HOURS = 3
DEFAULT_FRESHNESS_WINDOW_HOURS = 6
DEFAULT_N_TIMESTEPS = 24

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOP_PATIENCE = 15
DEFAULT_LR_REDUCE_PATIENCE = 5

DEFAULT_OUTPUT_DIR = Path("sepsis_model_v4")
DEFAULT_FIG_DPI = 400
DEFAULT_FIG_FORMATS = ("png", "pdf")


# ============================================================================
# UTILITIES
# ============================================================================


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _safe_close(fig: plt.Figure) -> None:
    try:
        plt.close(fig)
    except Exception:
        pass


def configure_publication_style() -> None:
    """Configure matplotlib/seaborn for publication-quality figures."""
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update(
        {
            # Font settings for journals (Nature, Science, etc.)
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            # Embed fonts (required for publication)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Figure quality
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.dpi": 400,
            "figure.dpi": 150,
            # Clean appearance
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#cccccc",
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


# Publication color palette (colorblind-safe)
PUB_COLORS = {
    "control": "#4C72B0",   # Blue
    "sepsis": "#DD8452",    # Orange
    "accent1": "#55A868",   # Green
    "accent2": "#C44E52",   # Red
    "accent3": "#8172B3",   # Purple
    "accent4": "#937860",   # Brown
    "neutral": "#666666",   # Gray
}


def get_label_color(label: int) -> str:
    """Get consistent color for label."""
    return PUB_COLORS["sepsis"] if int(label) == 1 else PUB_COLORS["control"]


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int = DEFAULT_FIG_DPI,
    formats: tuple[str, ...] = DEFAULT_FIG_FORMATS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.with_suffix("") if path.suffix else path
    fig.tight_layout()
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    _safe_close(fig)


def _label_name(label: int) -> str:
    return "Sepsis" if int(label) == 1 else "Control"


# ============================================================================
# CLEANING
# ============================================================================


def convert_temperature(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert temperature values to Celsius using value-based detection.

    MIMIC-IV issue: itemid 223762 is labeled as Fahrenheit but often contains
    Celsius values. Instead of converting by itemid, we detect Fahrenheit by
    value range:
    - Fahrenheit body temps: typically 95-105°F (35-40.5°C)
    - Values > 50 that aren't plausible Celsius are likely Fahrenheit
    - Values in 35-45 range are already valid Celsius
    """
    df = df_raw.copy()
    temp_mask = df["feature"].eq("temperature")

    # Only convert values that are clearly in Fahrenheit range (> 50)
    # Normal body temp in F is ~95-104°F, fever up to ~108°F
    # Any temp > 50 that's a valid measurement must be Fahrenheit
    fahrenheit_mask = temp_mask & (df["valuenum"] > 50) & (df["valuenum"] < 120)
    f_to_c_count = int(fahrenheit_mask.sum())
    if f_to_c_count > 0:
        df.loc[fahrenheit_mask, "valuenum"] = (df.loc[fahrenheit_mask, "valuenum"] - 32) * 5 / 9

    # Handle extreme values (> 120) - likely data entry errors, set to NaN for later imputation
    extreme_mask = temp_mask & (df["valuenum"] > 120)
    extreme_count = int(extreme_mask.sum())
    if extreme_count > 0:
        df.loc[extreme_mask, "valuenum"] = np.nan

    # Handle impossible negative/zero temps - set to NaN
    invalid_low_mask = temp_mask & (df["valuenum"] <= 0)
    invalid_low_count = int(invalid_low_mask.sum())
    if invalid_low_count > 0:
        df.loc[invalid_low_mask, "valuenum"] = np.nan

    print(
        f"   Temperature conversion: {f_to_c_count} F->C (value-based), "
        f"{extreme_count} extreme values removed, {invalid_low_count} invalid low values removed"
    )
    return df


def apply_unit_repairs(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common unit/data entry errors in vital signs.

    Known issues in MIMIC-IV:
    - Heart rate sometimes stored as RR interval (ms) instead of BPM
    - Respiratory rate can have outliers > 60
    - DBP/SpO2 occasionally have extreme outliers (data entry errors)
    - Zero/negative values used as placeholders for missing data
    """
    df = df_raw.copy()

    # =========================================================================
    # MINIMUM VALUE OUTLIERS (zeros, negatives, impossibly low values)
    # =========================================================================

    # Remove zero/negative values for all vitals (likely missing data placeholders)
    vitals = ["heart_rate", "sbp", "dbp", "resp_rate", "spo2"]
    for vital in vitals:
        mask = df["feature"].eq(vital)
        invalid_low = mask & (df["valuenum"] <= 0)
        count = int(invalid_low.sum())
        if count > 0:
            print(f"   Removing {count} zero/negative {vital} values")
            df.loc[invalid_low, "valuenum"] = np.nan

    # DBP: Remove negative values (can occur from calculation errors)
    dbp_mask = df["feature"].eq("dbp")
    neg_dbp = dbp_mask & (df["valuenum"] < 0)
    if int(neg_dbp.sum()) > 0:
        print(f"   Removing {int(neg_dbp.sum())} negative DBP values")
        df.loc[neg_dbp, "valuenum"] = np.nan

    # =========================================================================
    # MAXIMUM VALUE OUTLIERS AND UNIT CONVERSIONS
    # =========================================================================

    # Heart rate: Convert RR intervals (ms) to BPM if > 300
    hr_mask = df["feature"].eq("heart_rate")
    high_hr = hr_mask & (df["valuenum"] > 300)
    if int(high_hr.sum()) > 0:
        print(f"   Converting {int(high_hr.sum())} HR intervals (ms) to BPM")
        val_clip = df.loc[high_hr, "valuenum"].clip(lower=1)
        df.loc[high_hr, "valuenum"] = 60000 / val_clip

    # Respiratory rate: Clip extreme values
    rr_mask = df["feature"].eq("resp_rate")
    high_rr = rr_mask & (df["valuenum"] > 60)
    if int(high_rr.sum()) > 0:
        print(f"   Clipping {int(high_rr.sum())} resp rates > 60")
        df.loc[high_rr, "valuenum"] = 60

    # DBP: Remove extreme outliers (> 300 is impossible)
    dbp_mask = df["feature"].eq("dbp")
    extreme_dbp = dbp_mask & (df["valuenum"] > 300)
    if int(extreme_dbp.sum()) > 0:
        print(f"   Removing {int(extreme_dbp.sum())} extreme DBP values (> 300)")
        df.loc[extreme_dbp, "valuenum"] = np.nan

    # SpO2: Remove extreme outliers (> 100 is impossible for percentage)
    spo2_mask = df["feature"].eq("spo2")
    extreme_spo2 = spo2_mask & (df["valuenum"] > 100)
    if int(extreme_spo2.sum()) > 0:
        print(f"   Removing {int(extreme_spo2.sum())} extreme SpO2 values (> 100)")
        df.loc[extreme_spo2, "valuenum"] = np.nan

    # SBP: Remove extreme outliers (> 350 is impossible)
    sbp_mask = df["feature"].eq("sbp")
    extreme_sbp = sbp_mask & (df["valuenum"] > 350)
    if int(extreme_sbp.sum()) > 0:
        print(f"   Removing {int(extreme_sbp.sum())} extreme SBP values (> 350)")
        df.loc[extreme_sbp, "valuenum"] = np.nan

    # =========================================================================
    # LAB VALUE OUTLIERS
    # =========================================================================

    # Remove zero/negative lab values (impossible for these tests)
    labs_no_zero = ["wbc", "creatinine", "platelets", "bilirubin", "glucose", "bun", "hemoglobin"]
    for lab in labs_no_zero:
        mask = df["feature"].eq(lab)
        invalid_low = mask & (df["valuenum"] <= 0)
        count = int(invalid_low.sum())
        if count > 0:
            print(f"   Removing {count} zero/negative {lab} values")
            df.loc[invalid_low, "valuenum"] = np.nan

    # Potassium: Can't be zero but can be very low
    k_mask = df["feature"].eq("potassium")
    invalid_k = k_mask & (df["valuenum"] <= 0)
    if int(invalid_k.sum()) > 0:
        print(f"   Removing {int(invalid_k.sum())} zero/negative potassium values")
        df.loc[invalid_k, "valuenum"] = np.nan

    # Sodium: Can't be below ~60 (incompatible with life)
    na_mask = df["feature"].eq("sodium")
    invalid_na = na_mask & (df["valuenum"] < 60)
    if int(invalid_na.sum()) > 0:
        print(f"   Removing {int(invalid_na.sum())} impossibly low sodium values (< 60)")
        df.loc[invalid_na, "valuenum"] = np.nan

    return df


# ============================================================================
# EDA
# ============================================================================


def run_eda(
    df_clean: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    output_dir: Path,
    prediction_gap_hours: int,
    n_timesteps: int,
    max_points_per_feature: int = 200_000,
    seed: int = 42,
) -> None:
    """
    Generate EDA artifacts (plots + summary tables) before tensorization/scaling.

    The primary "modeling window" used for many plots is:
      win_end = onset_time - prediction_gap_hours
      charttime in [win_end - n_timesteps, win_end]
    """
    rng = np.random.default_rng(seed)
    eda_dir = Path(output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    configure_publication_style()

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
        cohort_local["prediction_time_hours"] = cohort_local["onset_hours"] - prediction_gap_hours
    for optional_col in ["icu_los_hours", "onset_hours", "prediction_time_hours"]:
        if optional_col not in cohort_local.columns:
            cohort_local[optional_col] = np.nan

    cohort_local.to_csv(eda_dir / "cohort.csv", index=False)

    cohort_summary = (
        cohort_local.groupby("label")
        .agg(
            n_stays=("stay_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            age_mean=("Age", "mean"),
            age_median=("Age", "median"),
            gender_mean=("Gender", "mean"),
            icu_los_hours_mean=("icu_los_hours", "mean"),
            onset_hours_mean=("onset_hours", "mean"),
            prediction_time_hours_mean=("prediction_time_hours", "mean"),
        )
        .reset_index()
    )
    cohort_summary["label_name"] = cohort_summary["label"].apply(_label_name)
    cohort_summary.to_csv(eda_dir / "cohort_summary.csv", index=False)

    # -------------------------
    # Cohort plots (standardized 7x5 inch figures for single-column publication)
    # -------------------------
    FIG_SINGLE = (7, 5)  # Single-column figure size
    FIG_WIDE = (10, 5)   # Wide figure for bar charts with many categories
    FIG_GRID = (14, 16)  # Grid figure for multi-panel plots

    # 1. Class Balance
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    class_counts = cohort_local["label"].value_counts().sort_index()
    bars = ax.bar(
        [_label_name(i) for i in class_counts.index],
        class_counts.values,
        color=[PUB_COLORS["control"], PUB_COLORS["sepsis"]],
        edgecolor="black",
        linewidth=1.2,
    )
    # Add value labels on bars
    for bar, val in zip(bars, class_counts.values):
        ax.annotate(
            f"{val:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    ax.set_title("Cohort Class Distribution", fontweight="bold")
    ax.set_ylabel("Number of ICU Stays")
    ax.set_xlabel("Patient Group")
    save_figure(fig, eda_dir / "class_balance")

    # 2. Age Distribution
    if "Age" in cohort_local.columns:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
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

    # 3. Gender Distribution
    if "Gender" in cohort_local.columns:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        gender_counts = cohort_local.groupby(["label", "Gender"])["stay_id"].nunique().reset_index()
        gender_counts["Group"] = gender_counts["label"].apply(_label_name)
        gender_counts["Sex"] = gender_counts["Gender"].map({0: "Female", 1: "Male"})
        sns.barplot(
            data=gender_counts, x="Group", y="stay_id", hue="Sex",
            palette=[PUB_COLORS["accent3"], PUB_COLORS["accent1"]],
            edgecolor="black", linewidth=1.2, ax=ax,
        )
        ax.set_title("Sex Distribution by Patient Group", fontweight="bold")
        ax.set_ylabel("Number of ICU Stays")
        ax.set_xlabel("Patient Group")
        ax.legend(title="Sex")
        save_figure(fig, eda_dir / "gender_distribution")

    # 4. ICU Length of Stay
    if "icu_los_hours" in cohort_local.columns:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        for label in sorted(cohort_local["label"].unique()):
            data = cohort_local.loc[cohort_local["label"] == label, "icu_los_hours"].dropna()
            sns.histplot(
                data, bins=50, ax=ax, element="step",
                label=f"{_label_name(label)} (n={len(data):,})",
                color=get_label_color(label), alpha=0.5, linewidth=1.5,
            )
        ax.set_title("ICU Length of Stay Distribution", fontweight="bold")
        ax.set_xlabel("ICU Length of Stay (hours)")
        ax.set_ylabel("Number of Stays")
        ax.legend(title="Group")
        save_figure(fig, eda_dir / "icu_los_hours")

    # 5. Sepsis Onset Timing
    if "onset_hours" in cohort_local.columns:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        sepsis_data = cohort_local[cohort_local["label"] == 1]["onset_hours"].dropna()
        sns.histplot(
            sepsis_data, bins=50, ax=ax, element="step",
            color=PUB_COLORS["sepsis"], alpha=0.6, linewidth=1.5,
        )
        # Add median line
        median_onset = sepsis_data.median()
        ax.axvline(median_onset, color=PUB_COLORS["accent2"], linestyle="--", linewidth=2,
                   label=f"Median: {median_onset:.1f}h")
        ax.set_title("Sepsis Onset Time from ICU Admission", fontweight="bold")
        ax.set_xlabel("Hours from ICU Admission to Sepsis Onset")
        ax.set_ylabel("Number of Patients")
        ax.legend()
        save_figure(fig, eda_dir / "sepsis_onset_hours")

    # -------------------------
    # Feature measurement stats
    # -------------------------
    df_nonnull = df_clean.dropna(subset=["valuenum"]).copy()
    df_nonnull = df_nonnull[df_nonnull["feature"].isin(FINAL_FEATURES)]

    counts_by_feature = (
        df_nonnull.groupby("feature")["valuenum"].size().reindex(FINAL_FEATURES).fillna(0).astype(int)
    )
    stays_by_feature = (
        df_nonnull.groupby("feature")["stay_id"].nunique().reindex(FINAL_FEATURES).fillna(0).astype(int)
    )

    # Feature labels for publication
    FEATURE_LABELS = {
        "heart_rate": "Heart Rate", "sbp": "Systolic BP", "dbp": "Diastolic BP",
        "resp_rate": "Resp. Rate", "spo2": "SpO₂", "temperature": "Temp.",
        "wbc": "WBC", "creatinine": "Creatinine", "platelets": "Platelets",
        "bilirubin": "Bilirubin", "glucose": "Glucose", "bun": "BUN",
        "sodium": "Sodium", "potassium": "Potassium", "hemoglobin": "Hemoglobin",
    }

    # 6. Measurement Counts
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    x_labels = [FEATURE_LABELS.get(f, f) for f in FINAL_FEATURES]
    bars = ax.bar(x_labels, counts_by_feature.values, color=PUB_COLORS["control"],
                  edgecolor="black", linewidth=0.8)
    ax.set_title("Total Measurement Counts by Clinical Feature", fontweight="bold")
    ax.set_ylabel("Number of Measurements (log scale)")
    ax.set_xlabel("Clinical Feature")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    save_figure(fig, eda_dir / "measurement_counts_by_feature")

    # 7. Stays with Measurement
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    bars = ax.bar(x_labels, stays_by_feature.values, color=PUB_COLORS["accent1"],
                  edgecolor="black", linewidth=0.8)
    ax.set_title("ICU Stays with at Least One Measurement per Feature", fontweight="bold")
    ax.set_ylabel("Number of ICU Stays")
    ax.set_xlabel("Clinical Feature")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    # Add horizontal line for total stays
    total_stays = cohort_local["stay_id"].nunique()
    ax.axhline(total_stays, color=PUB_COLORS["neutral"], linestyle="--", linewidth=1.5,
               label=f"Total Stays: {total_stays:,}")
    ax.legend(loc="upper right")
    save_figure(fig, eda_dir / "stays_with_measurement_by_feature")

    # 8. Outlier rates (pre-clamping) by physiological bounds
    outlier_rows = []
    for feat in FINAL_FEATURES:
        low, high = BOUNDS[feat]
        vals = df_nonnull.loc[df_nonnull["feature"] == feat, "valuenum"]
        if len(vals) == 0:
            outlier_rows.append({"feature": feat, "outlier_rate": 0.0, "n": 0})
            continue
        outlier_rate = float(((vals < low) | (vals > high)).mean())
        outlier_rows.append({"feature": feat, "outlier_rate": outlier_rate, "n": int(len(vals))})
    outlier_df = pd.DataFrame(outlier_rows)
    outlier_df.to_csv(eda_dir / "outlier_rates.csv", index=False)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    bars = ax.bar(x_labels, outlier_df["outlier_rate"].values, color=PUB_COLORS["accent2"],
                  edgecolor="black", linewidth=0.8)
    ax.set_title("Pre-Clamping Outlier Rate by Feature", fontweight="bold")
    ax.set_ylabel("Fraction Outside Physiological Bounds")
    ax.set_xlabel("Clinical Feature")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    # Add percentage labels for high outlier rates
    for bar, rate in zip(bars, outlier_df["outlier_rate"].values):
        if rate > 0.01:  # Only label if > 1%
            ax.annotate(
                f"{rate:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8,
            )
    save_figure(fig, eda_dir / "outlier_rate_by_feature")

    # -------------------------
    # Windowed EDA: aligns to model horizon
    # -------------------------
    if df_nonnull.empty:
        print(f"\n[EDA] No non-null measurements found; saved cohort-level artifacts to: {eda_dir}")
        return

    df_win = df_nonnull.copy()
    df_win["win_end"] = pd.to_datetime(df_win["onset_time"]) - pd.to_timedelta(prediction_gap_hours, unit="h")
    df_win["hours_before_win_end"] = (df_win["win_end"] - pd.to_datetime(df_win["charttime"])).dt.total_seconds() / 3600
    df_win = df_win[(df_win["hours_before_win_end"] >= 0) & (df_win["hours_before_win_end"] < n_timesteps)].copy()
    if df_win.empty:
        print(f"\n[EDA] No measurements in modeling window; saved cohort-level artifacts to: {eda_dir}")
        return
    df_win["hour_bin"] = np.floor(df_win["hours_before_win_end"]).astype(int)  # 0=most recent hour

    try:
        df_win.to_parquet(eda_dir / "windowed_measurements.parquet", index=False)
    except Exception:
        # Parquet may not be available in some environments; EDA plots still work.
        df_win.to_csv(eda_dir / "windowed_measurements.csv", index=False)

    # 9. Heatmaps: measurement density by hour bin (sepsis vs control)
    for label in sorted(df_win["label"].unique()):
        tmp = df_win[df_win["label"] == label]
        if tmp.empty:
            continue
        heat = (
            tmp.groupby(["feature", "hour_bin"])["valuenum"]
            .size()
            .unstack("hour_bin")
            .reindex(index=FINAL_FEATURES, columns=range(0, n_timesteps))
            .fillna(0)
        )
        # Use feature labels for y-axis
        heat.index = [FEATURE_LABELS.get(f, f) for f in heat.index]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            heat, ax=ax, cmap="YlOrRd", linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Number of Measurements", "shrink": 0.8},
        )
        ax.set_title(f"Measurement Density: {_label_name(label)} Patients\n(0 = Most Recent Hour)", fontweight="bold")
        ax.set_xlabel("Hours Before Prediction Time")
        ax.set_ylabel("Clinical Feature")
        ax.tick_params(axis="y", rotation=0)
        save_figure(fig, eda_dir / f"measurement_heatmap_{_label_name(label).lower()}")

    # 10. Distributions in the modeling window (sampled for speed)
    sampled = []
    for feat in FINAL_FEATURES:
        sub = df_win[df_win["feature"] == feat][["valuenum", "label"]].dropna()
        if sub.empty:
            continue
        if len(sub) > max_points_per_feature:
            idx = rng.choice(sub.index.values, size=max_points_per_feature, replace=False)
            sub = sub.loc[idx]
        sampled.append(sub.assign(feature=feat))
    sampled_df = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()

    if not sampled_df.empty:
        # 10a. Distribution histograms
        fig, axes = plt.subplots(5, 3, figsize=FIG_GRID)
        axes = axes.flatten()
        for i, feat in enumerate(FINAL_FEATURES):
            ax = axes[i]
            d = sampled_df[sampled_df["feature"] == feat].copy()
            for lbl in sorted(d["label"].unique()):
                subset = d[d["label"] == lbl]["valuenum"]
                sns.histplot(
                    subset, bins=40, element="step", stat="density",
                    color=get_label_color(lbl), alpha=0.4, linewidth=1.5,
                    label=_label_name(lbl), ax=ax,
                )
            ax.set_title(FEATURE_LABELS.get(feat, feat), fontweight="bold", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("Density" if i % 3 == 0 else "")
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
        for j in range(len(FINAL_FEATURES), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Feature Value Distributions in 24-Hour Modeling Window", fontweight="bold", fontsize=14, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_figure(fig, eda_dir / "value_distributions_grid")

        # 10b. Boxplots
        fig, axes = plt.subplots(5, 3, figsize=FIG_GRID)
        axes = axes.flatten()
        for i, feat in enumerate(FINAL_FEATURES):
            ax = axes[i]
            d = sampled_df[sampled_df["feature"] == feat].copy()
            d["Group"] = d["label"].apply(_label_name)
            sns.boxplot(
                data=d, x="Group", y="valuenum",
                palette=[PUB_COLORS["control"], PUB_COLORS["sepsis"]],
                ax=ax, linewidth=1.2,
            )
            ax.set_title(FEATURE_LABELS.get(feat, feat), fontweight="bold", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("")
        for j in range(len(FINAL_FEATURES), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Feature Value Comparison: Control vs Sepsis", fontweight="bold", fontsize=14, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_figure(fig, eda_dir / "boxplots_grid")

    # 11. Trajectories: mean value per hour_bin for each label
    traj = df_win.groupby(["label", "feature", "hour_bin"])["valuenum"].mean().reset_index().rename(columns={"valuenum": "mean"})
    # Also compute standard error for confidence bands
    traj_se = df_win.groupby(["label", "feature", "hour_bin"])["valuenum"].sem().reset_index().rename(columns={"valuenum": "se"})
    traj = traj.merge(traj_se, on=["label", "feature", "hour_bin"])

    fig, axes = plt.subplots(5, 3, figsize=FIG_GRID, sharex=True)
    axes = axes.flatten()
    for i, feat in enumerate(FINAL_FEATURES):
        ax = axes[i]
        d = traj[traj["feature"] == feat]
        for label in sorted(d["label"].unique()):
            dd = d[d["label"] == label].sort_values("hour_bin")
            color = get_label_color(label)
            ax.plot(dd["hour_bin"], dd["mean"], label=_label_name(label), color=color, linewidth=2)
            # Add confidence band (±1 SE)
            ax.fill_between(
                dd["hour_bin"],
                dd["mean"] - dd["se"],
                dd["mean"] + dd["se"],
                color=color, alpha=0.2,
            )
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontweight="bold", fontsize=11)
        ax.set_ylabel("")
        if i >= 12:  # Bottom row
            ax.set_xlabel("Hours Before Prediction")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for j in range(len(FINAL_FEATURES), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Mean Feature Trajectories Over 24-Hour Window\n(shaded = ±1 SE)", fontweight="bold", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, eda_dir / "mean_trajectories_grid")

    # 12. Correlation (per-stay medians within window)
    med = (
        df_win.groupby(["stay_id", "label", "feature"])["valuenum"]
        .median()
        .reset_index()
        .pivot_table(index=["stay_id", "label"], columns="feature", values="valuenum")
        .reindex(columns=FINAL_FEATURES)
    )
    for feat in FINAL_FEATURES:
        med[feat] = med[feat].fillna(POPULATION_MEDIANS[feat])

    corr = med[FINAL_FEATURES].corr(method="spearman")
    # Use publication labels
    corr.index = [FEATURE_LABELS.get(f, f) for f in corr.index]
    corr.columns = [FEATURE_LABELS.get(f, f) for f in corr.columns]

    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Upper triangle mask
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix\n(Per-Stay Median Values)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    save_figure(fig, eda_dir / "correlation_heatmap")

    # 13. Violin plots for per-stay medians
    med_long = (
        med.reset_index()
        .melt(id_vars=["stay_id", "label"], value_vars=FINAL_FEATURES, var_name="feature", value_name="median")
    )
    med_long["Group"] = med_long["label"].apply(_label_name)
    med_long["Feature"] = med_long["feature"].map(FEATURE_LABELS)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=med_long, x="Feature", y="median", hue="Group",
        split=True, inner="quart",
        palette=[PUB_COLORS["control"], PUB_COLORS["sepsis"]],
        linewidth=1.2, ax=ax,
    )
    ax.set_title("Per-Stay Median Feature Values: Control vs Sepsis", fontweight="bold")
    ax.set_xlabel("Clinical Feature")
    ax.set_ylabel("Median Value")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend(title="Group", loc="upper right")
    save_figure(fig, eda_dir / "per_stay_median_violin")

    print(f"\n[EDA] Saved {13} publication-quality figures to: {eda_dir}")


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
    feature_names: list[str] = FINAL_FEATURES,
) -> np.ndarray | None:
    """
    Build one tensor with time-aware forward-fill and freshness masks.

    Output format per timestep:
      [15 values, 15 freshness masks, Age, Gender] => 32 features

    Freshness mask:
      - 1 if measured at this bin OR forward-filled from a measurement < freshness_window_hours ago
      - 0 if stale (>= freshness_window_hours), never measured, or imputed
    """
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

    g_win["bin"] = (n_timesteps - 1 - bins.loc[g_win.index]).astype(int)  # 0=oldest, n-1=newest

    values = np.zeros((n_timesteps, n_features), dtype=np.float32)
    masks = np.zeros((n_timesteps, n_features), dtype=np.float32)
    last_measured = np.full(n_features, -np.inf)

    pivot = (
        g_win.pivot_table(index="bin", columns="feature", values="valuenum", aggfunc="mean")
        .reindex(index=range(n_timesteps), columns=feature_names)
    )

    for t in range(n_timesteps):
        for i, feat in enumerate(feature_names):
            current_val = pivot.loc[t, feat]
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
    X_clamped = X.copy()
    total_outliers = 0

    for i, feat in enumerate(feature_names):
        if feat not in BOUNDS:
            continue
        low, high = BOUNDS[feat]
        vals = X_clamped[:, :, i]
        outliers = (vals < low) | (vals > high)
        count = int(outliers.sum())
        if count > 0:
            total_outliers += count
            X_clamped[:, :, i] = np.clip(vals, low, high)

    print(f"   Total outliers clamped: {total_outliers}")
    return X_clamped


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    n_features: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit scaling on train split only.

    Statistics are computed from rows where the corresponding freshness mask > 0.
    This approximates "measured/recent" values and avoids fitting on never-measured
    median imputations.
    """
    scaler = StandardScaler()

    train_values = X_train[:, :, :n_features]
    val_values = X_val[:, :, :n_features]
    test_values = X_test[:, :, :n_features]

    train_masks = X_train[:, :, n_features : 2 * n_features]

    train_flat = train_values.reshape(-1, n_features)
    val_flat = val_values.reshape(-1, n_features)
    test_flat = test_values.reshape(-1, n_features)
    train_masks_flat = train_masks.reshape(-1, n_features)

    means: list[float] = []
    stds: list[float] = []
    sample_counts: list[int] = []

    for i in range(n_features):
        measured_mask = train_masks_flat[:, i] > 0
        measured_vals = train_flat[measured_mask, i]
        sample_counts.append(int(len(measured_vals)))
        if len(measured_vals) > 0:
            means.append(float(np.mean(measured_vals)))
            std = float(np.std(measured_vals))
            stds.append(std if std > 1e-7 else 1.0)
        else:
            feat_name = FINAL_FEATURES[i]
            means.append(float(POPULATION_MEDIANS.get(feat_name, 0.0)))
            stds.append(1.0)

    scaler.mean_ = np.array(means, dtype=np.float64)
    scaler.scale_ = np.array(stds, dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = int(train_flat.shape[0])

    print("   Scaler (train-only):")
    for i, feat in enumerate(FINAL_FEATURES):
        print(f"      {feat:<12}: mean={means[i]:.2f}, std={stds[i]:.2f}, n={sample_counts[i]}")

    def _apply(flat: np.ndarray) -> np.ndarray:
        return (flat - scaler.mean_) / scaler.scale_

    train_scaled = _apply(train_flat)
    val_scaled = _apply(val_flat)
    test_scaled = _apply(test_flat)

    X_train_out = X_train.copy()
    X_val_out = X_val.copy()
    X_test_out = X_test.copy()
    X_train_out[:, :, :n_features] = train_scaled.reshape(X_train.shape[0], -1, n_features)
    X_val_out[:, :, :n_features] = val_scaled.reshape(X_val.shape[0], -1, n_features)
    X_test_out[:, :, :n_features] = test_scaled.reshape(X_test.shape[0], -1, n_features)

    return X_train_out, X_val_out, X_test_out, scaler


# ============================================================================
# MODEL + TRAINING
# ============================================================================


def build_model(
    input_shape: tuple[int, int],
    lstm_units: tuple[int, int] = (64, 32),
    dropout: float = 0.3,
) -> tf.keras.Model:
    model = Sequential(
        [
            Bidirectional(LSTM(lstm_units[0], return_sequences=True, dropout=dropout), input_shape=input_shape),
            BatchNormalization(),
            Bidirectional(LSTM(lstm_units[1], return_sequences=False, dropout=dropout)),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["AUC"])
    return model


def calculate_class_weights(y: np.ndarray) -> dict[int, float]:
    n_samples = len(y)
    n_positive = int(np.sum(y))
    n_negative = int(n_samples - n_positive)
    if n_positive == 0 or n_negative == 0:
        print("   Warning: One class has zero samples, using equal weights")
        return {0: 1.0, 1: 1.0}
    w_pos = n_samples / (2 * n_positive)
    w_neg = n_samples / (2 * n_negative)
    print(f"   Class weights: negative={w_neg:.2f}, positive={w_pos:.2f}")
    return {0: float(w_neg), 1: float(w_pos)}


def _find_group_split(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    n_tries: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=n_tries, test_size=test_size, random_state=random_state)
    X_dummy = np.zeros((len(y), 1))
    last = None
    for train_idx, test_idx in splitter.split(X_dummy, y, groups=groups):
        last = (train_idx, test_idx)
        if len(np.unique(y[train_idx])) >= 2 and len(np.unique(y[test_idx])) >= 2:
            return train_idx, test_idx
    assert last is not None
    print("   Warning: Could not find a split with both classes in each partition; using last tried split.")
    return last


def train_model_from_tensors(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    output_dir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    early_stop_patience: int,
    lr_reduce_patience: int,
) -> tuple[tf.keras.Model, StandardScaler, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING SEPSIS PREDICTION MODEL V4")
    print("=" * 70)

    # Subject-aware Train/Val/Test split
    print("\n[1/6] Splitting data (subject-aware train/val/test)...")
    trainval_idx, test_idx = _find_group_split(y, subjects, test_size=0.2, random_state=seed)
    y_trainval = y[trainval_idx]
    subjects_trainval = subjects[trainval_idx]

    val_rel = 0.125  # => ~10% of full data when test_size=0.2
    train_idx_rel, val_idx_rel = _find_group_split(y_trainval, subjects_trainval, test_size=val_rel, random_state=seed + 1)

    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"   Train: {len(train_idx)} samples ({int(y_train.sum())} sepsis)")
    print(f"   Val:   {len(val_idx)} samples ({int(y_val.sum())} sepsis)")
    print(f"   Test:  {len(test_idx)} samples ({int(y_test.sum())} sepsis)")

    # Clamp
    print("\n[2/6] Applying physiological clamping...")
    X_train = apply_clamping(X_train, FINAL_FEATURES)
    X_val = apply_clamping(X_val, FINAL_FEATURES)
    X_test = apply_clamping(X_test, FINAL_FEATURES)

    # Scale (train-only)
    print("\n[3/6] Scaling features (train-only)...")
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test, n_features=len(FINAL_FEATURES))

    # Build model
    print("\n[4/6] Building model...")
    model = build_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()

    # Train
    print("\n[5/6] Training model...")
    class_weights = calculate_class_weights(y_train)
    callbacks = [
        EarlyStopping(monitor="val_AUC", patience=early_stop_patience, mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_AUC", factor=0.5, patience=lr_reduce_patience, mode="max", verbose=1),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Evaluate on test (threshold from validation)
    print("\n[6/6] Evaluating (test set; threshold chosen on val)...")
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()

    if len(np.unique(y_val)) >= 2:
        precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        best = int(np.argmax(f1))
        threshold = float(thresholds[best]) if best < len(thresholds) else 0.5
    else:
        threshold = 0.5

    if len(np.unique(y_test)) >= 2:
        test_auc = float(roc_auc_score(y_test, y_test_pred))
    else:
        print("   Warning: Test set has only one class; AUC undefined")
        test_auc = 0.5

    y_test_bin = (y_test_pred >= threshold).astype(int)
    test_accuracy = float(np.mean(y_test_bin == y_test))

    print(f"\nTest AUC: {test_auc:.4f}")
    print(f"Threshold (val-optimized): {threshold:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_bin, labels=[0, 1]))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_test_bin,
            labels=[0, 1],
            target_names=["Control", "Sepsis"],
            zero_division=0,
        )
    )

    # Save artifacts
    model_path = output_dir / "sepsis_model_v4.keras"
    scaler_path = output_dir / "sepsis_scaler_v4.pkl"
    config_path = output_dir / "config_v4.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    config = {
        "features": FINAL_FEATURES,
        "n_timesteps": int(X.shape[1]),
        "val_threshold": threshold,
        "test_auc": test_auc,
        "test_accuracy": test_accuracy,
    }
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)

    metrics = {"auc": test_auc, "accuracy": test_accuracy, "threshold": threshold}
    return model, scaler, metrics


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
    from google.cloud import bigquery

    print("=" * 70)
    print("EXTRACTING DATA FROM MIMIC-IV")
    print("=" * 70)

    client = bigquery.Client(project=project_id)

    # ICU stays
    print("[1/2] Building cohort...")
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

    # Sepsis-3 onsets
    s3_query = """
    SELECT stay_id, sofa_time as onset_time
    FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
    """
    s3_raw = client.query(s3_query).to_dataframe()
    sepsis_df = s3_raw.merge(stays_df, on="stay_id", how="inner")
    sepsis_df["onset_time"] = pd.to_datetime(sepsis_df["onset_time"])
    sepsis_df["onset_hours"] = (sepsis_df["onset_time"] - sepsis_df["intime"]).dt.total_seconds() / 3600

    # Need at least min_data_hours BEFORE prediction time
    sepsis_df = sepsis_df[sepsis_df["onset_hours"] > (min_data_hours + prediction_gap_hours)].copy()
    sepsis_df["label"] = 1

    print(f"   Sepsis cases: {len(sepsis_df)}")

    # Controls: age-matched and long enough to reach prediction time (win_end)
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
        controls.append(
            {
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
            }
        )

    controls_df = pd.DataFrame(controls)
    cohort_cols = [
        "stay_id",
        "hadm_id",
        "subject_id",
        "intime",
        "outtime",
        "Age",
        "Gender",
        "onset_time",
        "label",
        "onset_hours",
    ]
    cohort = pd.concat([sepsis_df[cohort_cols], controls_df[cohort_cols]], ignore_index=True)

    print(f"   Total cohort: {len(cohort)} ({int(cohort['label'].sum())} sepsis, {int((1 - cohort['label']).sum())} control)")

    # Extract features
    print("\n[2/2] Extracting vitals & labs...")
    stay_ids = cohort["stay_id"].unique().tolist()
    itemids = ",".join(map(str, FEATURE_MAP.keys()))

    dfs: list[pd.DataFrame] = []
    num_chunks = (len(stay_ids) + chunk_size - 1) // chunk_size
    for i in tqdm(range(0, len(stay_ids), chunk_size), desc="Extracting"):
        chunk_ids = stay_ids[i : i + chunk_size]
        chunk = ",".join(map(str, chunk_ids))
        print(f"   Chunk {i // chunk_size + 1}/{num_chunks}: {len(chunk_ids)} stays")

        # Chart events
        try:
            df_chart = client.query(
                f"""
                SELECT stay_id, charttime, itemid, valuenum
                FROM `physionet-data.mimiciv_3_1_icu.chartevents`
                WHERE stay_id IN ({chunk}) AND itemid IN ({itemids})
                """
            ).to_dataframe()
            dfs.append(df_chart)
        except Exception as e:
            print(f"   Chart events error: {e}")

        # Lab events (restricted to ICU window)
        try:
            df_labs = client.query(
                f"""
                SELECT i.stay_id, l.charttime, l.itemid, l.valuenum
                FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
                JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
                  ON l.hadm_id = i.hadm_id
                 AND l.charttime >= i.intime
                 AND l.charttime <= i.outtime
                WHERE i.stay_id IN ({chunk}) AND l.itemid IN ({itemids})
                """
            ).to_dataframe()
            dfs.append(df_labs)
        except Exception as e:
            print(f"   Lab events error: {e}")

    df_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])
    df_raw["feature"] = df_raw["itemid"].map(FEATURE_MAP)
    df_raw["charttime"] = pd.to_datetime(df_raw["charttime"])
    df_raw = df_raw.merge(cohort.drop(columns=["onset_hours"]), on="stay_id", how="inner")

    print(f"\n   Raw data: {len(df_raw)} rows")
    return df_raw, cohort


def build_tensors_from_raw(
    df_raw: pd.DataFrame,
    *,
    prediction_gap_hours: int,
    n_timesteps: int,
    freshness_window_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("\n" + "=" * 70)
    print("BUILDING TENSORS")
    print("=" * 70)

    print("\n[1/2] Cleaning units...")
    df_clean = apply_unit_repairs(convert_temperature(df_raw))

    print("\n[2/2] Building tensors...")
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

    print(f"\n   Tensors built: {X.shape} (skipped stays: {n_skipped})")
    print(f"   Sepsis: {int(y.sum())}, Control: {int((1 - y).sum())}")
    return X, y, subjects


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Sepsis Prediction Model V4 (6h early warning + EDA)")
    parser.add_argument("--extract", action="store_true", help="Extract data from BigQuery")
    parser.add_argument("--project", type=str, default="sepsis-prediction-2025", help="GCP project ID")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Rows per stay chunk when pulling from BigQuery")
    parser.add_argument("--load", type=str, help="Load pre-built tensors from a .npz file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--prediction-gap-hours",
        type=int,
        default=DEFAULT_PREDICTION_GAP_HOURS,
        help="Hours before onset to stop using data",
    )
    parser.add_argument(
        "--min-data-hours",
        type=int,
        default=DEFAULT_MIN_DATA_HOURS,
        help="Minimum hours of data before prediction time",
    )
    parser.add_argument(
        "--freshness-window-hours",
        type=int,
        default=DEFAULT_FRESHNESS_WINDOW_HOURS,
        help="Hours until measurement becomes stale",
    )
    parser.add_argument("--n-timesteps", type=int, default=DEFAULT_N_TIMESTEPS, help="Lookback window length (hours)")

    parser.add_argument("--no-eda", action="store_true", help="Disable EDA/visualization step")
    parser.add_argument("--eda-only", action="store_true", help="Run extraction + EDA, then exit (no tensors/training)")
    parser.add_argument("--tensors-only", action="store_true", help="Run extraction + EDA + tensor build, then exit (no training)")
    parser.add_argument(
        "--eda-max-points-per-feature",
        type=int,
        default=200_000,
        help="Max points per feature for distribution plots",
    )

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--lr-reduce-patience", type=int, default=DEFAULT_LR_REDUCE_PATIENCE)

    args = parser.parse_args()

    if args.eda_only and args.no_eda:
        raise SystemExit("Invalid flags: --eda-only cannot be combined with --no-eda")
    if args.eda_only and args.tensors_only:
        raise SystemExit("Invalid flags: choose either --eda-only or --tensors-only (or neither)")

    set_global_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.extract:
        df_raw, cohort = extract_mimic_data(
            project_id=args.project,
            chunk_size=args.chunk_size,
            prediction_gap_hours=args.prediction_gap_hours,
            min_data_hours=args.min_data_hours,
            seed=args.seed,
        )

        # EDA explicitly happens before tensorization/scaling
        if not args.no_eda:
            df_clean = apply_unit_repairs(convert_temperature(df_raw))
            run_eda(
                df_clean,
                cohort,
                output_dir=output_dir,
                prediction_gap_hours=args.prediction_gap_hours,
                n_timesteps=args.n_timesteps,
                max_points_per_feature=args.eda_max_points_per_feature,
                seed=args.seed,
            )

        if args.eda_only:
            print(f"\nEDA complete. See: {output_dir / 'eda'}")
            return

        X, y, subjects = build_tensors_from_raw(
            df_raw,
            prediction_gap_hours=args.prediction_gap_hours,
            n_timesteps=args.n_timesteps,
            freshness_window_hours=args.freshness_window_hours,
        )

        np.savez(output_dir / "tensors_v4.npz", X=X, y=y, subjects=subjects)
        print(f"Tensors saved to {output_dir / 'tensors_v4.npz'}")

        if args.tensors_only:
            print("\nTensor build complete. Skipping training (--tensors-only).")
            return

    elif args.load:
        data = np.load(args.load)
        X, y, subjects = data["X"], data["y"], data["subjects"]
    else:
        print("Usage:")
        print("  python train_model_v4.py --extract --project YOUR_PROJECT_ID")
        print("  python train_model_v4.py --load tensors_v4.npz")
        return

    model, scaler, metrics = train_model_from_tensors(
        X,
        y,
        subjects,
        output_dir=output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        lr_reduce_patience=args.lr_reduce_patience,
    )

    # Enrich config with data-generation + run settings
    config_path = output_dir / "config_v4.pkl"
    try:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    except Exception:
        config = {}

    config.update(
        {
            "prediction_gap_hours": int(args.prediction_gap_hours),
            "freshness_window_hours": int(args.freshness_window_hours),
            "n_timesteps": int(args.n_timesteps),
            "min_data_hours": int(args.min_data_hours),
            "seed": int(args.seed),
            "metrics": metrics,
        }
    )
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    print("\n" + "=" * 70)
    print(f"V4 COMPLETE - Test AUC: {metrics['auc']:.4f} | Test Acc: {metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
