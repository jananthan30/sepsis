import torch
from models_utils import load_model, preprocess_input

# Case: Healthy Vitals
base = {
    'age': 34, 'gender': 'M',
    'hr': 80, 'sbp': 120, 'o2': 98, 'rr': 12, 'temp': 37,
    'lactate': None, 'wbc': None, 'creat': None, 'glucose': None,
    'trop': None, 'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

# Test 1: Glucose 100 (Normal)
t1 = base.copy()
t1['glucose'] = 100.0

# Test 2: Glucose 400 (Severe)
t2 = base.copy()
t2['glucose'] = 400.0

model = load_model()

def predict(data, label):
    tensor = preprocess_input(data)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    print(f"{label}: {prob*100:.2f}%")

print("--- Glucose Analysis ---")
predict(base, "Vitals Only")
predict(t1, "Glucose 100")
predict(t2, "Glucose 400")
