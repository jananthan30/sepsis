# Code Style Guidelines

## Python Style

### General Standards
- Follow **PEP 8** for code style
- Use **PEP 484** type hints for function signatures
- Maximum line length: **100 characters** (relaxed from 79 for readability)
- Use **4 spaces** for indentation (no tabs)

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `patient_data`, `sepsis_risk` |
| Functions | snake_case | `calculate_auroc()`, `build_tensors()` |
| Classes | PascalCase | `PatientInput`, `PredictionResponse` |
| Constants | SCREAMING_SNAKE_CASE | `PREDICTION_GAP`, `FRESHNESS_WINDOW` |
| Private | _leading_underscore | `_internal_helper()` |

### Import Organization
```python
# Standard library
import os
from pathlib import Path
from typing import Optional, List, Dict

# Third-party
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI

# Local
from utils import helper_function
```

### Docstring Format (Google Style)
```python
def predict_sepsis(patient_data: np.ndarray, threshold: float = 0.5) -> dict:
    """Predict sepsis risk for a patient.

    Args:
        patient_data: Array of shape (24, 32) containing hourly clinical features.
        threshold: Classification threshold for positive prediction.

    Returns:
        Dictionary containing:
            - probability: float, raw model probability
            - prediction: bool, thresholded binary prediction
            - risk_level: str, one of 'low', 'medium', 'high'

    Raises:
        ValueError: If patient_data has incorrect shape.

    Example:
        >>> result = predict_sepsis(patient_array)
        >>> print(f"Risk: {result['probability']:.2%}")
    """
```

### Type Hints
```python
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd

def process_vitals(
    df: pd.DataFrame,
    features: List[str],
    window_hours: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

## Clinical Domain Conventions

### Variable Naming for Clinical Features
```python
# Use full clinical names for clarity
heart_rate = ...      # Not: hr
temperature = ...     # Not: temp
creatinine = ...      # Not: creat
platelets = ...       # Not: plt (conflicts with matplotlib)

# Abbreviations acceptable in dictionaries/configs
FEATURE_MAP = {
    'sbp': 'Systolic Blood Pressure',
    'dbp': 'Diastolic Blood Pressure',
    'spo2': 'Oxygen Saturation',
    'wbc': 'White Blood Cell Count',
    'bun': 'Blood Urea Nitrogen',
}
```

### Bounds and Thresholds
```python
# Document physiological rationale
BOUNDS = {
    'heart_rate': (20, 300),      # Min survivable to max physiological
    'temperature': (25, 45),       # Hypothermia to extreme hyperthermia (Celsius)
    'creatinine': (0.1, 30.0),     # Normal low to severe renal failure
}
```

### Time Units
```python
# Always specify units in variable names or comments
prediction_gap_hours = 6
freshness_window_hours = 6
n_timesteps_hours = 24
icu_los_hours = ...  # Length of stay in hours
```

## TensorFlow/Keras Conventions

### Model Definition
```python
def build_model(n_features: int, n_timesteps: int) -> tf.keras.Model:
    """Build BiLSTM model for sepsis prediction.

    Architecture follows the pattern:
    BiLSTM -> BatchNorm -> BiLSTM -> BatchNorm -> Dense -> Dense
    """
    inputs = tf.keras.Input(shape=(n_timesteps, n_features))

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # ... continue building

    return tf.keras.Model(inputs, outputs, name="sepsis_bilstm")
```

### Model Artifacts
```python
# Consistent naming pattern
model.save(output_dir / "sepsis_model_v6.keras")
joblib.dump(scaler, output_dir / "sepsis_scaler_v6.pkl")
with open(output_dir / "config_v6.pkl", "wb") as f:
    pickle.dump(config, f)
```

## FastAPI Conventions

### Endpoint Naming
```python
# Use RESTful patterns with versioned prefixes
@app.get("/api/v1/health")
@app.post("/api/v1/predict")
@app.post("/api/v1/predict/batch")
@app.get("/api/v1/analysis/clusters")
```

### Pydantic Models
```python
from pydantic import BaseModel, Field

class PatientInput(BaseModel):
    """Single patient input for prediction."""

    heart_rate: Optional[float] = Field(
        None,
        ge=20, le=300,
        description="Heart rate in BPM"
    )
    temperature: Optional[float] = Field(
        None,
        ge=25, le=45,
        description="Body temperature in Celsius"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "heart_rate": 85,
                "temperature": 37.2
            }
        }
```

## Streamlit Conventions

### Layout Organization
```python
# Use clear section headers
st.header("Model Performance")

# Group related inputs
with st.sidebar:
    st.subheader("Configuration")
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5)

# Use columns for side-by-side content
col1, col2 = st.columns(2)
with col1:
    st.metric("AUROC", f"{auroc:.3f}")
with col2:
    st.metric("Sensitivity", f"{sensitivity:.1%}")
```

### Caching
```python
@st.cache_data  # For data that doesn't change
def load_test_data():
    ...

@st.cache_resource  # For models and connections
def load_model():
    ...
```

## Data Processing Conventions

### DataFrame Operations
```python
# Use method chaining for clarity
processed_df = (
    raw_df
    .query("itemid in @valid_itemids")
    .assign(
        charttime=lambda x: pd.to_datetime(x['charttime']),
        hours_since_admit=lambda x: (x['charttime'] - x['intime']).dt.total_seconds() / 3600
    )
    .dropna(subset=['valuenum'])
)
```

### NumPy Array Documentation
```python
# Document array shapes in comments
X_train = ...  # Shape: (n_samples, n_timesteps, n_features) = (N, 24, 32)
y_train = ...  # Shape: (n_samples,) = (N,)
```

## Error Handling

### Clinical Validation
```python
def validate_clinical_values(data: dict) -> dict:
    """Validate and clamp clinical values to physiological bounds."""
    errors = []

    for feature, (low, high) in BOUNDS.items():
        if feature in data and data[feature] is not None:
            if not low <= data[feature] <= high:
                # Log warning but don't fail - clamp instead
                logger.warning(f"{feature}={data[feature]} outside bounds [{low}, {high}]")
                data[feature] = np.clip(data[feature], low, high)

    return data
```

### API Error Responses
```python
from fastapi import HTTPException

if not model_loaded:
    raise HTTPException(
        status_code=503,
        detail="Model not available. Please try again later."
    )
```

## File Organization

### Script Structure
```python
"""
Module docstring explaining purpose.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import ...

# ============================================================================
# CONSTANTS
# ============================================================================
PREDICTION_GAP = 6
FRESHNESS_WINDOW = 6

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def helper_function():
    ...

# ============================================================================
# MAIN LOGIC
# ============================================================================
def main():
    ...

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
```
