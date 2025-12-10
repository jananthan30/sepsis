import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Optional, Any
from config import Config

logger = logging.getLogger(__name__)

# --- MODEL ARCHITECTURE ---
class SepsisLSTM_TimeAware(nn.Module):
    def __init__(self):
        super(SepsisLSTM_TimeAware, self).__init__()
        # Input 32: (15 Values + 15 Masks) + 2 Static
        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM, 
            hidden_size=Config.HIDDEN_SIZE, 
            num_layers=Config.NUM_LAYERS, 
            batch_first=True, 
            dropout=0.0
        )
        self.fc = nn.Linear(Config.HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- HELPERS ---
def load_model(path: str = Config.MODEL_PATH) -> Optional[SepsisLSTM_TimeAware]:
    """
    Loads the PyTorch model from disk.
    """
    device = torch.device("cpu")
    model = SepsisLSTM_TimeAware()
    try:
        # Use weights_only=True if supported by the torch version for security, 
        # but sticking to standard load for compatibility with 2.1.0
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}", exc_info=True)
        return None

def preprocess_input(data: Dict[str, Any]) -> torch.Tensor:
    """
    Transforms validated dictionary -> Time-Aware Tensor (Values + Masks + Static)
    Applies Z-Score normalization using clinical statistics.
    Layout: Interleaved [Val1, Mask1, Val2, Mask2, ..., Age, Gender]
    """
    final_vector = []
    
    # 1. Process Dynamic Features (Interleaved Value + Mask)
    for f in Config.INPUT_FEATURES:
        val = data.get(f)
        stats = Config.NORMALIZATION_STATS.get(f, {'mean': 0, 'std': 1})
        
        if val is None:
            # Missing: Value = Mean (0.0 in Z-space), Mask = 0.0
            final_vector.append(0.0) 
            final_vector.append(0.0)  
        else:
            try:
                raw_val = float(val)
                # Z-Score Normalization: (x - u) / s
                norm_val = (raw_val - stats['mean']) / stats['std']
                final_vector.append(norm_val)
                final_vector.append(1.0) # Mask = Present
            except (ValueError, TypeError):
                # Fallback
                final_vector.append(0.0)
                final_vector.append(0.0)

    # 2. Process Static Features (At the end)
    try:
        age_raw = float(data.get('age', 60))
        age_stats = Config.NORMALIZATION_STATS['age']
        age = (age_raw - age_stats['mean']) / age_stats['std']
    except (ValueError, TypeError):
        age = 0.0 
        
    gender_val = data.get('gender', 'F')
    is_male = 1.0 if str(gender_val).upper().startswith('M') else 0.0
    
    final_vector.append(age)
    final_vector.append(is_male)
    
    # 3. Create Tensor (Batch=1, Time=24, Feat=32)
    # We simulate 24h history by repeating the current state
    feats = np.array(final_vector, dtype=np.float32)
    tensor = torch.tensor(np.tile(feats, (24, 1))).unsqueeze(0)
    
    return tensor
