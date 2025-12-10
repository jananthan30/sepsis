import torch
import numpy as np
from models_utils import load_model, preprocess_input

# User's specific case from the logs
# Case A: HR 100 (Tachy), Glucose 400, O2 92
case_a = {
    'age': 34, 'gender': 'M',
    'hr': 100, 'sbp': 120, 'o2': 92, 'rr': 12, 'temp': 37,
    'lactate': 1.0, 'wbc': 11.0, 'creat': 1.0, 'glucose': 400.0,
    'trop': None, 'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

# Case B: HR 80 (Normal), Glucose 400, O2 92 (Still Sick)
case_b = case_a.copy()
case_b['hr'] = 80

# Case C: HR 80, Glucose 100 (Normal), O2 98 (Normal) -> Should be healthy
case_c = case_b.copy()
case_c['glucose'] = 100.0
case_c['o2'] = 98.0

model = load_model()

def predict(data, label):
    tensor = preprocess_input(data)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    print(f"{label}: {prob*100:.2f}%")

print("--- Diagnosis Analysis ---")
predict(case_a, "Case A (User Input 1)")
predict(case_b, "Case B (User Input 2 - Lower HR)")
predict(case_c, "Case C (Hypothetical - Fix Glucose/O2)")
