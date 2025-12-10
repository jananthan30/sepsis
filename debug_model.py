import torch
import numpy as np
from models_utils import load_model, preprocess_input
from config import Config

# Mock the input the user provided
# 55, Male
# HR 60, SBP 120, O2 99, RR 12, Temp 37
data = {
    'age': 55,
    'gender': 'M',
    'hr': 60,
    'sbp': 120,
    'o2': 99,
    'rr': 12,
    'temp': 37,
    # Labs are missing (None)
    'lactate': None,
    'wbc': None,
    'creat': None,
    'glucose': None,
    'trop': None,
    'lipase': None,
    'bili': None,
    'plt': None,
    'bicarb': None,
    'bun': None
}

print("--- Debugging Sepsis Model ---")
print(f"Input Data: {data}")

# 1. Preprocess
tensor = preprocess_input(data)
print(f"\nTensor Shape: {tensor.shape}")
print(f"Tensor Mean: {tensor.mean().item():.4f}")
print(f"Tensor Max: {tensor.max().item():.4f}")
print(f"Tensor Min: {tensor.min().item():.4f}")

# Inspect specific features
# Config.INPUT_FEATURES indices: 0-14 (Values), 15-29 (Masks), 30 (Age), 31 (Gender)
# HR is index 0. Mean 85, Std 25. Val 60. -> (60-85)/25 = -1.0
print(f"\nNormalized HR (idx 0): {tensor[0, 0, 0].item():.4f} (Expected approx -1.0)")
# Temp is index 4. Mean 36.8, Std 0.8. Val 37. -> (37-36.8)/0.8 = 0.25
print(f"Normalized Temp (idx 4): {tensor[0, 0, 4].item():.4f} (Expected approx 0.25)")
# Mask for HR (idx 15): Should be 1.0
print(f"Mask HR (idx 15): {tensor[0, 0, 15].item():.4f}")
# Mask for Lactate (idx 15+5 = 20): Should be 0.0
print(f"Mask Lactate (idx 20): {tensor[0, 0, 20].item():.4f}")
# Age (idx 30): Mean 65, Std 20. Val 55. -> (55-65)/20 = -0.5
print(f"Normalized Age (idx 30): {tensor[0, 0, 30].item():.4f}")

# 2. Load Model
model = load_model()
if not model:
    print("Failed to load model")
    exit()

# 3. Inference
with torch.no_grad():
    # Check output of LSTM layer directly if possible, but let's check final output first
    output = model(tensor)
    prob = torch.sigmoid(output).item()
    
    print(f"\nRaw Logit (Pre-Sigmoid): {output.item():.4f}")
    print(f"Probability: {prob:.4f} ({prob*100:.2f}%)")

# Test what happens if we zero out everything
zero_tensor = torch.zeros_like(tensor)
with torch.no_grad():
    out_zero = model(zero_tensor)
    prob_zero = torch.sigmoid(out_zero).item()
    print(f"\nZero Input Probability: {prob_zero:.4f}")
