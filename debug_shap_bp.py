import torch
import shap
import numpy as np
from models_utils import load_model, preprocess_input
from config import Config

# Load Model
model = load_model()
explainer = shap.GradientExplainer(model, torch.zeros((5, 24, 32)))

def run_shap_test(data, label):
    tensor = preprocess_input(data)
    tensor.requires_grad = True
    
    # Prediction
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
        
    # SHAP
    shap_vals = explainer.shap_values(tensor)
    shap_sum = np.sum(shap_vals[0], axis=0) # (32,)
    
    # Check SBP impact (Index 1 * 2 = 2 for value, 3 for mask)
    sbp_val_idx = Config.INPUT_FEATURES.index('sbp') * 2
    sbp_mask_idx = sbp_val_idx + 1
    sbp_impact = shap_sum[sbp_val_idx] + shap_sum[sbp_mask_idx]
    
    print(f"\n--- {label} ---")
    print(f"Input SBP: {data['sbp']}")
    print(f"Risk Prob: {prob:.4f}")
    print(f"SBP SHAP Impact: {sbp_impact:.4f} ({'RED/RISK' if sbp_impact > 0 else 'GREEN/SAFE'})")

# Case 1: SBP 100 (Normal-ish)
case_1 = {
    'age': 65, 'gender': 'M', 'hr': 80, 'sbp': 100, 'o2': 98, 'rr': 12, 'temp': 37,
    'lactate': None, 'wbc': None, 'creat': None, 'glucose': None
}

# Case 2: SBP 50 (Critical Hypotension)
case_2 = case_1.copy()
case_2['sbp'] = 50

run_shap_test(case_1, "Case 1: SBP 100")
run_shap_test(case_2, "Case 2: SBP 50")
