"""
Sepsis Prediction REST API
FastAPI backend exposing REST endpoints for model inference and analysis.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans

# ============================================================================
# API CONFIGURATION
# ============================================================================
app = FastAPI(
    title="Sepsis Early Prediction API",
    description="REST API for sepsis prediction using MIMIC-IV trained deep learning model",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL LOADING
# ============================================================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "sepsis_model_v6"

# Global model and scaler (loaded once at startup)
model = None
scaler = None
config = None

# V6 Features (19 total)
FEATURES_19 = [
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
    "wbc", "creatinine", "platelets", "bilirubin", "glucose",
    "bun", "sodium", "potassium", "hemoglobin",
    "delta_hr", "delta_sbp", "delta_temp"
]

# Population medians for imputation
POPULATION_MEDIANS = {
    'heart_rate': 86.0, 'sbp': 119.0, 'dbp': 60.0, 'map': 77.0,
    'resp_rate': 18.0, 'spo2': 97.0, 'temperature': 36.9,
    'wbc': 10.4, 'creatinine': 1.0, 'platelets': 199.0,
    'bilirubin': 0.8, 'glucose': 128.0, 'bun': 21.0,
    'sodium': 139.0, 'potassium': 4.1, 'hemoglobin': 10.2,
    'delta_hr': 0.0, 'delta_sbp': 0.0, 'delta_temp': 0.0
}

def load_model():
    """Load model, scaler, and config at startup."""
    global model, scaler, config

    model_path = MODEL_DIR / "sepsis_model_v6.keras"
    scaler_path = MODEL_DIR / "sepsis_scaler_v6.pkl"
    config_path = MODEL_DIR / "config_v6.pkl"

    if model_path.exists():
        model = tf.keras.models.load_model(str(model_path))

    if scaler_path.exists():
        scaler = joblib.load(str(scaler_path))

    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

@app.on_event("startup")
async def startup_event():
    """Load model on API startup."""
    load_model()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class PatientFeatures(BaseModel):
    """Patient feature input for prediction."""
    heart_rate: float = Field(default=80.0, ge=30, le=200, description="Heart rate (bpm)")
    sbp: float = Field(default=120.0, ge=50, le=250, description="Systolic blood pressure (mmHg)")
    dbp: float = Field(default=80.0, ge=20, le=150, description="Diastolic blood pressure (mmHg)")
    map: Optional[float] = Field(default=None, description="Mean arterial pressure (mmHg)")
    resp_rate: float = Field(default=16.0, ge=5, le=50, description="Respiratory rate (/min)")
    spo2: float = Field(default=98.0, ge=50, le=100, description="Oxygen saturation (%)")
    temperature: float = Field(default=37.0, ge=32, le=43, description="Temperature (Celsius)")
    wbc: float = Field(default=8.0, ge=0, le=50, description="White blood cell count (x10^3/uL)")
    creatinine: float = Field(default=1.0, ge=0, le=15, description="Creatinine (mg/dL)")
    platelets: float = Field(default=250.0, ge=0, le=1000, description="Platelet count (x10^3/uL)")
    bilirubin: float = Field(default=1.0, ge=0, le=30, description="Bilirubin (mg/dL)")
    glucose: float = Field(default=100.0, ge=20, le=600, description="Glucose (mg/dL)")
    bun: float = Field(default=15.0, ge=0, le=150, description="Blood urea nitrogen (mg/dL)")
    sodium: float = Field(default=140.0, ge=100, le=180, description="Sodium (mEq/L)")
    potassium: float = Field(default=4.0, ge=1.5, le=8, description="Potassium (mEq/L)")
    hemoglobin: float = Field(default=14.0, ge=3, le=25, description="Hemoglobin (g/dL)")
    delta_hr: float = Field(default=0.0, ge=-30, le=30, description="Heart rate trend (bpm/hour)")
    delta_sbp: float = Field(default=0.0, ge=-30, le=30, description="SBP trend (mmHg/hour)")
    delta_temp: float = Field(default=0.0, ge=-2, le=2, description="Temperature trend (C/hour)")
    age: float = Field(default=65.0, ge=18, le=120, description="Patient age (years)")
    gender: int = Field(default=1, ge=0, le=1, description="Gender (0=Female, 1=Male)")

class PredictionResponse(BaseModel):
    """Prediction response."""
    probability: float
    risk_level: str
    threshold: float
    model_version: str
    features_used: int

class MetricsResponse(BaseModel):
    """Model metrics response."""
    auc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    threshold: float
    n_samples: int

class CalibrationResponse(BaseModel):
    """Calibration data response."""
    bin_edges: List[float]
    bin_centers: List[float]
    observed_rate: List[Optional[float]]
    predicted_rate: List[Optional[float]]
    bin_counts: List[int]
    brier_score: float
    ece: float
    mce: float
    n_bins: int

class PermutationImportanceResponse(BaseModel):
    """Permutation importance response."""
    feature_names: List[str]
    importance_mean: List[float]
    importance_std: List[float]
    n_iter: int
    baseline_auc: float

class ClusterResponse(BaseModel):
    """Cluster analysis response."""
    cluster_id: List[int]
    embeddings: List[List[float]]
    summaries: List[Dict[str, Any]]
    k: int
    n_samples: int

class MetadataResponse(BaseModel):
    """Model metadata response."""
    model_version: str
    features: List[str]
    n_features: int
    n_timesteps: int
    prediction_gap_hours: int
    freshness_window_hours: int
    cv_auc_mean: Optional[float]
    cv_auc_std: Optional[float]
    test_auc: Optional[float]
    training_date: Optional[str]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_patient(features: PatientFeatures) -> np.ndarray:
    """Preprocess patient features for model input."""
    # Calculate MAP if not provided
    if features.map is None:
        pulse_pressure = features.sbp - features.dbp
        features.map = features.dbp + (pulse_pressure / 3)

    # Create feature array in correct order
    feature_values = np.array([[
        features.heart_rate, features.sbp, features.dbp, features.map,
        features.resp_rate, features.spo2, features.temperature,
        features.wbc, features.creatinine, features.platelets,
        features.bilirubin, features.glucose, features.bun,
        features.sodium, features.potassium, features.hemoglobin,
        features.delta_hr, features.delta_sbp, features.delta_temp
    ]])

    # Scale features
    if scaler is not None:
        feature_values = scaler.transform(feature_values)

    # Create tensor: [19 values, 19 masks, age, gender] = 40 features
    X = np.zeros((1, 24, 40))
    X[:, :, :19] = feature_values  # Scaled values
    X[:, :, 19:38] = 1.0  # All masks = 1 (fresh data)
    X[:, :, 38] = features.age  # Raw age
    X[:, :, 39] = features.gender  # Gender

    return X.astype(np.float32)

def get_risk_level(probability: float, threshold: float = 0.3853) -> str:
    """Convert probability to risk level."""
    if probability < 0.30:
        return "Low"
    elif probability < threshold:
        return "Moderate"
    else:
        return "High"

def load_test_data():
    """Load test data for analysis endpoints."""
    x_path = MODEL_DIR / "X_test.npy"
    y_path = MODEL_DIR / "y_test.npy"

    if x_path.exists() and y_path.exists():
        X = np.load(str(x_path))
        y = np.load(str(y_path))
        return X, y
    return None, None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Sepsis Early Prediction API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "endpoints": [
            "/api/v1/metadata",
            "/api/v1/predict",
            "/api/v1/analysis/metrics",
            "/api/v1/analysis/calibration",
            "/api/v1/analysis/permutation-importance",
            "/api/v1/analysis/clusters"
        ]
    }

@app.get("/api/v1/metadata", response_model=MetadataResponse)
async def get_metadata():
    """
    Get model metadata including version, features, and training configuration.
    """
    if config is None:
        raise HTTPException(status_code=500, detail="Model configuration not loaded")

    return MetadataResponse(
        model_version="v6",
        features=FEATURES_19,
        n_features=len(FEATURES_19),
        n_timesteps=config.get('n_timesteps', 24),
        prediction_gap_hours=config.get('prediction_gap_hours', 4),
        freshness_window_hours=config.get('freshness_window_hours', 6),
        cv_auc_mean=float(config.get('cv_auc_mean', 0)) if config.get('cv_auc_mean') is not None else None,
        cv_auc_std=float(config.get('cv_auc_std', 0)) if config.get('cv_auc_std') is not None else None,
        test_auc=float(config.get('test_auc', 0)) if config.get('test_auc') is not None else None,
        training_date=None
    )

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(features: PatientFeatures):
    """
    Generate sepsis risk prediction for a patient.

    Accepts patient vital signs and lab values, returns probability and risk level.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Preprocess and predict
    X = preprocess_patient(features)
    probability = float(model.predict(X, verbose=0)[0][0])

    threshold = 0.3853  # V6 optimal threshold from CV
    risk_level = get_risk_level(probability, threshold)

    return PredictionResponse(
        probability=round(probability, 4),
        risk_level=risk_level,
        threshold=threshold,
        model_version="v6",
        features_used=19
    )

@app.get("/api/v1/analysis/metrics", response_model=MetricsResponse)
async def get_metrics(threshold: float = Query(default=0.3853, ge=0.0, le=1.0)):
    """
    Get model performance metrics at a specified threshold.

    Parameters:
    - threshold: Classification threshold (default: 0.3853)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X_test, y_test = load_test_data()
    if X_test is None:
        raise HTTPException(status_code=500, detail="Test data not available")

    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred_class = (y_pred_prob >= threshold).astype(int)
    y_true = y_test.astype(int)

    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return MetricsResponse(
        auc=round(auc_score, 4),
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        ppv=round(ppv, 4),
        npv=round(npv, 4),
        threshold=threshold,
        n_samples=len(y_true)
    )

@app.get("/api/v1/analysis/calibration", response_model=CalibrationResponse)
async def get_calibration(n_bins: int = Query(default=10, ge=5, le=20)):
    """
    Get calibration (reliability) curve data.

    Parameters:
    - n_bins: Number of bins for calibration analysis (default: 10)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X_test, y_test = load_test_data()
    if X_test is None:
        raise HTTPException(status_code=500, detail="Test data not available")

    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_true = y_test.astype(int)

    # Calculate calibration bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
    bin_indices = np.digitize(y_pred_prob, bin_edges[1:-1])

    observed_rate = []
    predicted_rate = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        bin_counts.append(count)

        if count > 0:
            observed_rate.append(round(float(y_true[mask].mean()), 4))
            predicted_rate.append(round(float(y_pred_prob[mask].mean()), 4))
        else:
            observed_rate.append(None)
            predicted_rate.append(None)

    # Calculate calibration metrics
    brier_score = float(np.mean((y_pred_prob - y_true) ** 2))

    # ECE and MCE
    valid_mask = np.array(bin_counts) > 0
    obs = np.array([o if o is not None else 0 for o in observed_rate])
    pred = np.array([p if p is not None else 0 for p in predicted_rate])
    counts = np.array(bin_counts)

    if valid_mask.sum() > 0:
        weights = counts[valid_mask] / counts[valid_mask].sum()
        ece = float(np.sum(weights * np.abs(obs[valid_mask] - pred[valid_mask])))
        mce = float(np.max(np.abs(obs[valid_mask] - pred[valid_mask])))
    else:
        ece = 0.0
        mce = 0.0

    return CalibrationResponse(
        bin_edges=bin_edges.tolist(),
        bin_centers=bin_centers,
        observed_rate=observed_rate,
        predicted_rate=predicted_rate,
        bin_counts=bin_counts,
        brier_score=round(brier_score, 4),
        ece=round(ece, 4),
        mce=round(mce, 4),
        n_bins=n_bins
    )

@app.get("/api/v1/analysis/permutation-importance", response_model=PermutationImportanceResponse)
async def get_permutation_importance(
    n_iter: int = Query(default=5, ge=1, le=20),
    top_k: int = Query(default=19, ge=1, le=19)
):
    """
    Get feature importance via permutation importance.

    Parameters:
    - n_iter: Number of permutation iterations (default: 5)
    - top_k: Number of top features to return (default: 19, all features)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X_test, y_test = load_test_data()
    if X_test is None:
        raise HTTPException(status_code=500, detail="Test data not available")

    y_true = y_test.astype(int)

    # Baseline AUC
    y_pred_base = model.predict(X_test, verbose=0).flatten()
    baseline_auc = roc_auc_score(y_true, y_pred_base)

    # Permutation importance for each feature
    importance_scores = []

    for feat_idx in range(19):  # 19 features
        delta_aucs = []

        for _ in range(n_iter):
            X_permuted = X_test.copy()
            # Permute feature across all samples
            np.random.shuffle(X_permuted[:, :, feat_idx])

            y_pred_perm = model.predict(X_permuted, verbose=0).flatten()
            perm_auc = roc_auc_score(y_true, y_pred_perm)
            delta_aucs.append(baseline_auc - perm_auc)

        importance_scores.append({
            'feature': FEATURES_19[feat_idx],
            'mean': np.mean(delta_aucs),
            'std': np.std(delta_aucs)
        })

    # Sort by importance and get top_k
    importance_scores.sort(key=lambda x: x['mean'], reverse=True)
    top_features = importance_scores[:top_k]

    return PermutationImportanceResponse(
        feature_names=[f['feature'] for f in top_features],
        importance_mean=[round(f['mean'], 4) for f in top_features],
        importance_std=[round(f['std'], 4) for f in top_features],
        n_iter=n_iter,
        baseline_auc=round(baseline_auc, 4)
    )

@app.get("/api/v1/analysis/clusters", response_model=ClusterResponse)
async def get_clusters(k: int = Query(default=3, ge=2, le=10)):
    """
    Cluster patients based on LSTM embeddings.

    Parameters:
    - k: Number of clusters (default: 3)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X_test, y_test = load_test_data()
    if X_test is None:
        raise HTTPException(status_code=500, detail="Test data not available")

    # Create embedding model from penultimate layer
    # The model architecture: LSTM -> BatchNorm -> LSTM -> BatchNorm -> Dense(32) -> Dense(1)
    # We want the output of the Dense(32) layer as embeddings
    embedding_layer_idx = -2  # Second to last layer (Dense 32)
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[embedding_layer_idx].output
    )

    # Extract embeddings
    embeddings = embedding_model.predict(X_test, verbose=0)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Generate cluster summaries
    y_true = y_test.astype(int)
    summaries = []

    for cluster_idx in range(k):
        cluster_mask = cluster_labels == cluster_idx
        cluster_size = int(cluster_mask.sum())
        cluster_sepsis_rate = float(y_true[cluster_mask].mean()) if cluster_size > 0 else 0.0

        # Get mean feature values for this cluster (first 19 features are clinical values)
        cluster_features = X_test[cluster_mask, -1, :19]  # Last timestep, first 19 features
        feature_means = cluster_features.mean(axis=0) if cluster_size > 0 else np.zeros(19)

        # Find top 3 distinguishing features (highest absolute z-score from overall mean)
        overall_means = X_test[:, -1, :19].mean(axis=0)
        overall_stds = X_test[:, -1, :19].std(axis=0) + 1e-8
        z_scores = (feature_means - overall_means) / overall_stds
        top_feature_indices = np.argsort(np.abs(z_scores))[-3:][::-1]

        distinguishing_features = []
        for idx in top_feature_indices:
            direction = "high" if z_scores[idx] > 0 else "low"
            distinguishing_features.append({
                "feature": FEATURES_19[idx],
                "direction": direction,
                "z_score": round(float(z_scores[idx]), 2)
            })

        summaries.append({
            "cluster_id": cluster_idx,
            "size": cluster_size,
            "sepsis_rate": round(cluster_sepsis_rate, 3),
            "risk_level": "High" if cluster_sepsis_rate > 0.6 else ("Moderate" if cluster_sepsis_rate > 0.4 else "Low"),
            "distinguishing_features": distinguishing_features
        })

    # Reduce embeddings to 2D for visualization (using first 2 principal components)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    return ClusterResponse(
        cluster_id=cluster_labels.tolist(),
        embeddings=embeddings_2d.tolist(),
        summaries=summaries,
        k=k,
        n_samples=len(cluster_labels)
    )

# ============================================================================
# HEALTH CHECK
# ============================================================================
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "config_loaded": config is not None
    }

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
