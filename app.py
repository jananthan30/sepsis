import logging
import pandas as pd
import numpy as np
import os
from typing import Optional, List, Dict
from flask import Flask, render_template, request, jsonify
from pydantic import BaseModel, Field, ValidationError
from waitress import serve
import torch
import copy
import shap
import sys

from config import Config
from models_utils import load_model, preprocess_input

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SepsisGuardian")

# --- INPUT VALIDATION ---
class PatientInput(BaseModel):
    # Vitals (Optional because model handles missingness)
    hr: Optional[float] = Field(None, ge=0, le=300)
    sbp: Optional[float] = Field(None, ge=0, le=300)
    o2: Optional[float] = Field(None, ge=0, le=100)
    rr: Optional[float] = Field(None, ge=0, le=100)
    temp: Optional[float] = Field(None, ge=10, le=45)  # Celsius
    
    # Labs
    lactate: Optional[float] = Field(None, ge=0)
    wbc: Optional[float] = Field(None, ge=0)
    creat: Optional[float] = Field(None, ge=0)
    glucose: Optional[float] = Field(None, ge=0)
    trop: Optional[float] = Field(None, ge=0)
    lipase: Optional[float] = Field(None, ge=0)
    bili: Optional[float] = Field(None, ge=0)
    plt: Optional[float] = Field(None, ge=0)
    bicarb: Optional[float] = Field(None, ge=0)
    bun: Optional[float] = Field(None, ge=0)
    
    # Demographics
    age: float = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(M|F|m|f)$")

# --- APP INITIALIZATION ---
app = Flask(__name__)

# Load model at startup
logger.info("Initializing Sepsis Guardian...")
model = load_model()
explainer = None

if not model:
    logger.critical("Failed to load model. Application will be unstable.")
else:
    try:
        # Initialize SHAP GradientExplainer (Verified working with LSTM)
        logger.info("Initializing SHAP GradientExplainer...")
        background = torch.zeros((5, 24, 32))
        explainer = shap.GradientExplainer(model, background)
        logger.info("SHAP Explainer ready.")
    except Exception as e:
        logger.error(f"Failed to init SHAP: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for orchestrators."""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200

@app.route('/api/patient_data/<patient_id>')
def get_patient_data(patient_id):
    """
    Fetches simulated patient data (24h timeline) for the demo mode.
    """
    try:
        # Sanitize input to prevent directory traversal
        safe_id = os.path.basename(patient_id)
        file_path = os.path.join('patient_data', f"{safe_id}.psv")
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Patient not found'}), 404
            
        # Read PSV (Pipe Separated)
        df = pd.read_csv(file_path, sep='|')
        
        # Replace NaN with None (null in JSON)
        # Robust replacement for JSON serialization
        df = df.replace({np.nan: None})
        
        # Convert to list of dicts
        data = df.to_dict(orient='records')
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching patient data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        logger.error("Predict called but model is not loaded.")
        return jsonify({'error': 'Model service unavailable'}), 503

    try:
        # 1. Validate Input using Pydantic
        json_data = request.get_json()
        if not json_data:
             return jsonify({'error': 'Empty request body'}), 400
             
        patient_data = PatientInput(**json_data)
        
        # 2. Preprocess (Pydantic model -> Dict -> Tensor)
        clean_data = patient_data.model_dump() 
        tensor = preprocess_input(clean_data)
        
        # 3. Inference
        with torch.no_grad():
            # Model output shape is (Batch, 1) -> squeeze to scalar
            out = model(tensor)
            prob = torch.sigmoid(out).item()
        
        # 4. SHAP Feature Attribution (GradientExplainer)
        shap_data = []
        if explainer:
            try:
                # GradientExplainer requires gradients to be enabled
                tensor.requires_grad = True
                shap_vals = explainer.shap_values(tensor)
                
                # Handle Output Format (List vs Tensor)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0] # Take first output class if list
                
                # Convert to numpy if it's a tensor (GradientExplainer often returns numpy, but verify)
                if hasattr(shap_vals, 'cpu'):
                    shap_vals = shap_vals.cpu().detach().numpy()
                
                # Shape: (1, 24, 32) -> Sum over Time (Axis 1) -> (1, 32)
                shap_sum = np.sum(shap_vals[0], axis=0) # Shape (32,)
                
                # Aggregate Interleaved Features (Value + Mask)
                feature_scores = []
                
                # Dynamic Features
                for i, key in enumerate(Config.INPUT_FEATURES):
                    idx_val = i * 2
                    idx_mask = i * 2 + 1
                    score = shap_sum[idx_val] + shap_sum[idx_mask]
                    
                    feature_scores.append({
                        'feature': Config.FEATURE_NAMES_PRETTY.get(key, key),
                        'score': float(score)
                    })
                    
                # Static Features
                feature_scores.append({
                    'feature': 'Age',
                    'score': float(shap_sum[-2])
                })
                feature_scores.append({
                    'feature': 'Gender',
                    'score': float(shap_sum[-1])
                })
                
                # Sort by magnitude
                shap_data = sorted(feature_scores, key=lambda x: abs(x['score']), reverse=True)
                
            except Exception as e:
                logger.error(f"SHAP calculation failed: {e}")

        # 5. Clinical Logic (using Config)
        if prob < Config.RISK_LOW:
            status = "🟢 LOW RISK"
            color = "success"
            msg = "Patient stable. Continue standard monitoring."
        elif prob < Config.RISK_HIGH:
            status = "🟡 SCREENING ALERT"
            color = "warning"
            msg = "Risk Elevated. Nurse verification required."
        else:
            status = "🔴 SEPSIS ALERT"
            color = "danger"
            msg = "High Probability (>95% Specificity). Initiate Sepsis Protocol."

        # 6. Data Sufficiency Check
        alerts = Config.CLINICAL_ALERTS['abnormal_vitals']
        crit_labs = Config.CLINICAL_ALERTS['critical_labs']
        
        abnormal_vitals = []
        if patient_data.hr and patient_data.hr > alerts['hr_high']: abnormal_vitals.append('HR')
        if patient_data.sbp and patient_data.sbp < alerts['sbp_low']: abnormal_vitals.append('SBP')
        if patient_data.rr and patient_data.rr > alerts['rr_high']: abnormal_vitals.append('RR')
        if patient_data.temp and (patient_data.temp > alerts['temp_high'] or patient_data.temp < alerts['temp_low']): abnormal_vitals.append('Temp')
        if patient_data.o2 and patient_data.o2 < alerts['o2_low']: abnormal_vitals.append('O2')

        missing_critical_labs = []
        p_dict = patient_data.model_dump()
        for lab in crit_labs:
            if p_dict.get(lab) is None:
                missing_critical_labs.append(lab.capitalize())

        recommendations = []
        if abnormal_vitals and missing_critical_labs and prob < Config.RISK_HIGH:
            status = "⚠️ INSUFFICIENT DATA"
            color = "secondary" 
            msg = f"Abnormal vitals ({', '.join(abnormal_vitals)}) detected. Model requires lab data for accurate risk assessment."
            recommendations = missing_critical_labs

        logger.info(f"Prediction success. Risk: {prob:.4f}. Status: {status}")
        
        return jsonify({
            'probability': round(prob * 100, 2),
            'status': status,
            'color': color,
            'message': msg,
            'recommendations': recommendations,
            'shap_data': shap_data
        })

    except ValidationError as e:
        logger.warning(f"Validation error: {e.errors()}")
        return jsonify({'error': 'Validation Error', 'details': e.errors()}), 400
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Generates a full 24-hour risk trend for the Patient Journey visualization.
    Accepts a list of 24 hourly states.
    """
    if not model:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data_list = request.get_json()
        if not isinstance(data_list, list):
            return jsonify({'error': 'Expected a list of records'}), 400

        risk_trend = []
        
        # Optimization: Run inference in a loop (Batch size=1 is fine for 24 items)
        # Ideally we would stack them into a batch of 24, but looping is simpler for now
        for hour_data in data_list:
            # Convert keys if necessary, or assume frontend sends correct 'hr', 'sbp' keys
            # Since frontend sends the same structure as single predict, we reuse preprocess
            tensor = preprocess_input(hour_data)
            
            with torch.no_grad():
                out = model(tensor)
                prob = torch.sigmoid(out).item()
                risk_trend.append(round(prob * 100, 2))

        return jsonify({'risk_trend': risk_trend})

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if Config.ENV == 'development':
        logger.info(f"Starting Flask Development Server on port {Config.PORT}")
        app.run(debug=True, port=Config.PORT)
    else:
        logger.info(f"Starting Waitress Production Server on port {Config.PORT}")
        serve(app, host='0.0.0.0', port=Config.PORT)
