import torch
import numpy as np
from models_utils import load_model, preprocess_input

# Case 1: User's Input (Abnormal Vitals, No Labs)
case_1 = {
    'age': 55, 'gender': 'M', 
    'hr': 50, 'sbp': 90, 'o2': 90, 'rr': 24, 'temp': 38,
    'lactate': None, 'wbc': None, 'creat': None, 'glucose': None, 
    'trop': None, 'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

# Case 2: Same Vitals + High Lactate (4.0) + High WBC (18.0)
case_2 = case_1.copy()
case_2['lactate'] = 4.0
case_2['wbc'] = 18.0

model = load_model()

def predict(data, label):
    tensor = preprocess_input(data)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    print(f"{label}: {prob*100:.2f}%")

print("--- Sensitivity Test ---")
predict(case_1, "Case 1 (Vitals Only)")
predict(case_2, "Case 2 (Vitals + Labs)")
