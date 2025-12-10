import torch
import shap
import numpy as np
from models_utils import load_model, preprocess_input
from config import Config

# 1. Load Model
print("Loading Model...")
model = load_model()
if not model:
    print("❌ Failed to load model")
    exit()

# 2. Initialize Explainer (Same logic as app.py)
print("Initializing SHAP Explainer...")
try:
    background = torch.zeros((5, 24, 32))
    explainer = shap.DeepExplainer(model, background)
    print("✅ Explainer Initialized")
except Exception as e:
    print(f"❌ Explainer Init Failed: {e}")
    exit()

# 3. Create Dummy Input (Abnormal Case)
data = {
    'age': 65, 'gender': 'M',
    'hr': 120, 'sbp': 90, 'temp': 39.0,
    'lactate': 4.0, 'wbc': 15.0,
    'glucose': 200.0
}
tensor = preprocess_input(data)
print(f"Input Tensor Shape: {tensor.shape}")

# 4. Run Inference & SHAP
print("Running Prediction...")
with torch.no_grad():
    prob = torch.sigmoid(model(tensor)).item()
print(f"Probability: {prob:.4f}")

print("Running SHAP Calculation...")
try:
    # Note: shap_values might return a list or a tensor depending on the model output
    shap_vals = explainer.shap_values(tensor)
    
    if isinstance(shap_vals, list):
        print(f"SHAP Output is a List of length {len(shap_vals)}")
        shap_vals = shap_vals[0]
    else:
        print("SHAP Output is a Tensor")
        
    print(f"SHAP Values Shape: {shap_vals.shape}")
    
    # Check if values are all zero
    if np.all(shap_vals == 0):
        print("⚠️ WARNING: All SHAP values are ZERO!")
    else:
        print("✅ SHAP values generated.")
        print(f"Max SHAP Val: {np.max(shap_vals)}")
        print(f"Min SHAP Val: {np.min(shap_vals)}")

except Exception as e:
    print(f"❌ SHAP Calculation Error: {e}")
