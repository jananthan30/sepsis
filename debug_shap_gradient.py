import torch
import shap
import numpy as np
from models_utils import load_model, preprocess_input

# 1. Load Model
print("Loading Model...")
model = load_model()
if not model:
    print("❌ Failed to load model")
    exit()

# 2. Initialize Gradient Explainer
# GradientExplainer is better suited for PyTorch models than DeepExplainer for some RNNs
print("Initializing GradientExplainer...")
try:
    # Background: 5 samples of zeros
    background = torch.zeros((5, 24, 32))
    # Note: GradientExplainer requires the model, the background, 
    # and sometimes handles LSTM better if we don't flatten immediately.
    explainer = shap.GradientExplainer(model, background)
    print("✅ GradientExplainer Initialized")
except Exception as e:
    print(f"❌ GradientExplainer Init Failed: {e}")
    exit()

# 3. Create Dummy Input
data = {
    'age': 65, 'gender': 'M',
    'hr': 120, 'sbp': 90, 'temp': 39.0,
    'lactate': 4.0, 'wbc': 15.0,
    'glucose': 200.0
}
tensor = preprocess_input(data)
print(f"Input Tensor Shape: {tensor.shape}")
tensor.requires_grad = True # Required for GradientExplainer

# 4. Run SHAP
print("Running SHAP Calculation...")
try:
    # gradient_shap returns a list of arrays (one per output) or a single array
    shap_vals = explainer.shap_values(tensor)
    
    if isinstance(shap_vals, list):
        print(f"SHAP Output is a List of length {len(shap_vals)}")
        shap_vals = shap_vals[0]
    else:
        print("SHAP Output is a Tensor")
        
    print(f"SHAP Values Shape: {shap_vals.shape}")
    
    if np.all(shap_vals == 0):
        print("⚠️ WARNING: All SHAP values are ZERO!")
    else:
        print("✅ SHAP values generated.")
        print(f"Max SHAP Val: {np.max(shap_vals)}")
        print(f"Min SHAP Val: {np.min(shap_vals)}")
        
except Exception as e:
    print(f"❌ SHAP Calculation Error: {e}")
