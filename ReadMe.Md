---
title: Sepsis Early Prediction Model
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.28.0
app_file: streamlit_app.py
pinned: false
license: apache-2.0
---

# Sepsis Early Prediction Model

A deep learning model for early prediction of sepsis in ICU patients using MIMIC-IV clinical data.

## Model Overview

- **Architecture**: Bidirectional LSTM neural network
- **Input**: 24 hours of clinical data (15 features + 15 freshness masks)
- **Output**: Probability of sepsis onset
- **Performance**: AUC 0.91 on internal validation

## Features Used

**Vital Signs (6)**: Heart Rate, Systolic BP, Diastolic BP, Respiratory Rate, SpO2, Temperature

**Laboratory Values (9)**: WBC, Creatinine, Platelets, Bilirubin, Glucose, BUN, Sodium, Potassium, Hemoglobin

**Freshness Masks (15)**: Binary indicators showing if each measurement is recent (within 6 hours)

## Data Source

Trained on MIMIC-IV v3.1 (Medical Information Mart for Intensive Care) database using Sepsis-3 clinical criteria:
- Sepsis-3 Definition: Suspected infection + organ dysfunction (SOFA >= 2)
- Dataset: 15,332 ICU stays (3,833 sepsis, 11,499 control)
- Validation: 80/20 stratified train/test split

## Key Innovation: Time-Aware Preprocessing

Traditional forward-fill carries stale values indefinitely. Our approach uses a 6-hour freshness window:
- If measurement is less than 6 hours old: use value, mark as fresh (mask=1)
- If measurement is 6+ hours old: use value, mark as stale (mask=0)
- Model learns to weight recent measurements more heavily

## Disclaimer

This model is for research and educational purposes only. It has been internally validated on MIMIC-IV data. External validation on independent datasets is required before any clinical deployment.

## Dashboard Features

1. **Model Performance**: ROC curves, PR curves, confusion matrix, and metrics
2. **Feature Analysis**: Correlation matrix and feature importance visualizations
3. **Patient Explainer**: Individual patient predictions with SHAP explanations
4. **Data Pipeline**: Training process and data distribution visualizations

## Technical Stack

- TensorFlow/Keras for deep learning
- Streamlit for the web interface
- Plotly for interactive visualizations
- SHAP for model explainability

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## License

Apache License 2.0 - See LICENSE file for details.
