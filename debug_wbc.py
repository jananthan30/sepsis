import torch
from models_utils import load_model, preprocess_input

# Previous tests showed Case C (Normal Glucose/O2) still at 98%.
# Let's investigate WBC 11.0 (High Normal) and Lactate 1.0 (Normal).

case_c = {
    'age': 34, 'gender': 'M',
    'hr': 80, 'sbp': 120, 'o2': 98, 'rr': 12, 'temp': 37,
    'lactate': 1.0, 'wbc': 11.0, 'creat': 1.0, 'glucose': 100.0,
    'trop': None, 'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

# Case D: Lower WBC to 6.0 (Perfectly Normal)
case_d = case_c.copy()
case_d['wbc'] = 6.0

# Case E: Remove Labs entirely (Vitals Only)
case_e = {
    'age': 34, 'gender': 'M',
    'hr': 80, 'sbp': 120, 'o2': 98, 'rr': 12, 'temp': 37,
    'lactate': None, 'wbc': None, 'creat': None, 'glucose': None,
    'trop': None, 'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

model = load_model()

def predict(data, label):
    tensor = preprocess_input(data)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    print(f"{label}: {prob*100:.2f}%")

print("--- Deep Dive Analysis ---")
predict(case_c, "Case C (WBC 11)")
predict(case_d, "Case D (WBC 6)")
predict(case_e, "Case E (Vitals Only)")
