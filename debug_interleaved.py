import torch
import numpy as np
from models_utils import load_model
from config import Config

# ... (Same data setup as before) ...
data = {
    'age': 55, 'gender': 'M', 'hr': 60, 'sbp': 120, 'o2': 99, 'rr': 12, 'temp': 37,
    'lactate': None, 'wbc': None, 'creat': None, 'glucose': None, 'trop': None, 
    'lipase': None, 'bili': None, 'plt': None, 'bicarb': None, 'bun': None
}

def get_interleaved_tensor(data):
    features = []
    # Interleaved: Value, Mask, Value, Mask...
    for f in Config.INPUT_FEATURES:
        val = data.get(f)
        stats = Config.NORMALIZATION_STATS.get(f, {'mean': 0, 'std': 1})
        if val is None:
            features.append(0.0) # Value (Mean)
            features.append(0.0) # Mask (Missing)
        else:
            norm_val = (float(val) - stats['mean']) / stats['std']
            features.append(norm_val)
            features.append(1.0)
            
    # Static at the end
    age_stats = Config.NORMALIZATION_STATS['age']
    age = (float(data.get('age', 60)) - age_stats['mean']) / age_stats['std']
    is_male = 1.0 if str(data.get('gender', '')).upper().startswith('M') else 0.0
    
    features.append(age)
    features.append(is_male)
    
    feats = np.array(features, dtype=np.float32)
    # Shape: (32,)
    # Tile to (1, 24, 32)
    return torch.tensor(np.tile(feats, (24, 1))).unsqueeze(0)

print("--- Testing Interleaved Layout ---")
model = load_model()
tensor = get_interleaved_tensor(data)

with torch.no_grad():
    logit = model(tensor).item()
    prob = torch.sigmoid(model(tensor)).item()
    print(f"Logit: {logit:.4f}")
    print(f"Prob:  {prob:.4f}")
