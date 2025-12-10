import torch
from models_utils import load_model, preprocess_input

# Case D (WBC 6, Lactate 1, Creat 1, Glucose 100, Vitals Normal) was 96%.
# Why is a perfectly healthy patient rated 96%?
# Maybe the 'masks' are triggering it? (The fact that labs WERE ordered implies sickness?)
# Or maybe "Lactate 1.0" is considered high by this specific model? (MIMIC mean is 2.0, but maybe median is lower?)
# Or Creatinine 1.0?

case_d = {
    'age': 34, 'gender': 'M',
    'hr': 80, 'sbp': 120, 'o2': 98, 'rr': 12, 'temp': 37,
    'lactate': 1.0, 'wbc': 6.0, 'creat': 1.0, 'glucose': 100.0,
    'trop': None, 'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

# Test 1: Lactate 0.5 (Lower)
t1 = case_d.copy()
t1['lactate'] = 0.5

# Test 2: Remove Lactate (Make it None)
t2 = case_d.copy()
t2['lactate'] = None

# Test 3: Remove Creatinine
t3 = case_d.copy()
t3['creat'] = None

model = load_model()

def predict(data, label):
    tensor = preprocess_input(data)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    print(f"{label}: {prob*100:.2f}%")

print("--- Feature Attribution ---")
predict(case_d, "Baseline (All Normal Labs)")
predict(t1, "Lactate 0.5")
predict(t2, "Lactate None")
predict(t3, "Creatinine None")
