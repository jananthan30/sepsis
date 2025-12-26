# Product Context

## Project Name
Sepsis Early Prediction System

## Description
A clinical decision support system using Bidirectional LSTM deep learning to predict sepsis onset 6 hours before clinical manifestation, trained on MIMIC-IV ICU data. The system provides both an interactive Streamlit dashboard for clinicians and a REST API for EHR integration.

## Target Users

### Primary: Healthcare Clinicians
- **ICU Physicians**: Real-time sepsis risk monitoring and early warning alerts
- **ICU Nurses**: Dashboard for continuous patient monitoring
- **Clinical Decision Makers**: Risk stratification for resource allocation

### Secondary: ML Researchers
- **Clinical ML Researchers**: Model development and validation methodologies
- **Healthcare Data Scientists**: Feature engineering for clinical time-series
- **Academic Collaborators**: Reproducible experiments for publications

## Key Features

### Current Features
1. **6-Hour Early Warning Prediction**: Sepsis prediction 6 hours before onset using BiLSTM
2. **Time-Aware Missing Data Handling**: Freshness masks distinguish fresh vs. stale measurements
3. **Interactive Dashboard**: Streamlit-based visualization with model performance metrics
4. **REST API**: FastAPI endpoints for model inference and patient risk scoring
5. **SHAP Explainability**: Feature importance visualization for model interpretability
6. **Comprehensive EDA Pipeline**: Publication-ready exploratory data analysis

### Core Clinical Inputs
- **6 Vital Signs**: Heart rate, SBP, DBP, respiratory rate, SpO2, temperature
- **9 Lab Values**: WBC, creatinine, platelets, bilirubin, glucose, BUN, sodium, potassium, hemoglobin

## Project Goals

### Short-Term Goals
1. **Production Readiness**: Validation, reliability testing, and deployment hardening
2. **API Completeness**: Full API coverage for EHR integration workflows
3. **Model V6 Optimization**: Performance improvements and extended feature set

### Long-Term Goals
1. **Clinical Validation**: Prospective validation study in partnership with clinical sites
2. **Research Publication**: Academic publication with reproducible methodology
3. **Multi-Site Deployment**: Scalable architecture for multiple hospital systems
4. **Regulatory Pathway**: Documentation for FDA/CE clearance as clinical decision support

## Success Metrics
- **AUROC**: Target >0.85 on held-out test set
- **Sensitivity**: >80% at clinically useful specificity
- **API Latency**: <100ms for single patient prediction
- **Uptime**: 99.9% availability for production deployment

## Constraints
- **Data Privacy**: HIPAA compliance required for any real patient data
- **Clinical Safety**: Must not replace clinical judgment; decision support only
- **Reproducibility**: All experiments must be fully reproducible from documented pipelines
