import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Server Config
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    ENV = os.getenv('FLASK_ENV', 'production')

    # Model Config
    MODEL_PATH = os.getenv('MODEL_PATH', 'sepsis_model_time_aware.pth')
    INPUT_FEATURES = [
        'hr', 'sbp', 'o2', 'rr', 'temp', 
        'lactate', 'wbc', 'creat', 
        'glucose', 'trop', 'lipase', 'bili', 'plt', 'bicarb', 'bun'
    ]
    
    # Clinical Thresholds
    RISK_LOW = float(os.getenv('RISK_LOW_THRESHOLD', 0.25))
    RISK_HIGH = float(os.getenv('RISK_HIGH_THRESHOLD', 0.30))
    
    # Model Parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    INPUT_DIM = 32  # (15 values + 15 masks + 2 static)

    # Normalization Statistics (Approximate Clinical Distributions from MIMIC-IV)
    # Used to zero-center inputs for the neural network
    NORMALIZATION_STATS = {
        'hr': {'mean': 85.0, 'std': 25.0},
        'sbp': {'mean': 120.0, 'std': 25.0},
        'o2': {'mean': 97.0, 'std': 3.0},
        'rr': {'mean': 20.0, 'std': 6.0},
        'temp': {'mean': 36.8, 'std': 0.8},
        'lactate': {'mean': 2.0, 'std': 2.0},
        'wbc': {'mean': 11.0, 'std': 6.0},
        'creat': {'mean': 1.3, 'std': 1.0},
        'glucose': {'mean': 140.0, 'std': 50.0},
        'trop': {'mean': 0.05, 'std': 0.1},
        'lipase': {'mean': 60.0, 'std': 40.0},
        'bili': {'mean': 0.9, 'std': 1.0},
        'plt': {'mean': 220.0, 'std': 100.0},
        'bicarb': {'mean': 24.0, 'std': 5.0},
        'bun': {'mean': 25.0, 'std': 20.0},
        'age': {'mean': 65.0, 'std': 20.0}
    }

    # Clinical Logic for Data Sufficiency
    CLINICAL_ALERTS = {
        'abnormal_vitals': {
            'hr_high': 90,
            'sbp_low': 100,
            'rr_high': 20,
            'temp_high': 38.0,
            'temp_low': 36.0,
            'o2_low': 92
        },
        'critical_labs': ['lactate', 'wbc', 'creat']
    }

    # Pretty names for SHAP Visualization
    FEATURE_NAMES_PRETTY = {
        'hr': 'Heart Rate',
        'sbp': 'Systolic BP',
        'o2': 'O2 Saturation',
        'rr': 'Resp. Rate',
        'temp': 'Temperature',
        'lactate': 'Lactate',
        'wbc': 'WBC Count',
        'creat': 'Creatinine',
        'glucose': 'Glucose',
        'trop': 'Troponin',
        'lipase': 'Lipase',
        'bili': 'Bilirubin',
        'plt': 'Platelets',
        'bicarb': 'Bicarbonate',
        'bun': 'BUN',
        'age': 'Age',
        'gender': 'Gender'
    }


