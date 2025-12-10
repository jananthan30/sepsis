import torch
import json
from app import app
from config import Config

client = app.test_client()

def test_prediction(data, label):
    print(f"\n--- Testing: {label} ---")
    print(f"Input: {json.dumps(data, indent=2)}")
    
    response = client.post('/predict', json=data)
    result = response.get_json()
    
    print(f"Response Status: {result.get('status')}")
    print(f"Probability: {result.get('probability')}%)")
    print(f"Message: {result.get('message')}")
    return result

# 1. Abnormal Case (Sepsis)
abnormal_data = {
    'age': 65, 'gender': 'M',
    'hr': 120, 'sbp': 90, 'temp': 39.0,
    'lactate': 4.0, 'wbc': 15.0,
    # Others None
    'o2': None, 'rr': None, 'creat': None, 'glucose': None, 'trop': None, 
    'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

# 2. Normal Case (Healthy)
normal_data = {
    'age': 65, 'gender': 'M',
    'hr': 80, 'sbp': 120, 'temp': 37.0,
    'lactate': 1.0, 'wbc': 7.0,
    # Others None
    'o2': None, 'rr': None, 'creat': None, 'glucose': None, 'trop': None, 
    'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

# Run Sequence
res1 = test_prediction(abnormal_data, "Step 1: Abnormal Patient")
res2 = test_prediction(normal_data, "Step 2: Normal Patient (Reset)")

if res2['probability'] > 30.0:
    print("\n❌ FAILURE: Prediction stuck on High Risk!")
else:
    print("\n✅ SUCCESS: Prediction correctly dropped to Low Risk.")
