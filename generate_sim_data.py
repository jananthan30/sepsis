import pandas as pd
import numpy as np
import os

# Define the columns exactly as PhysioNet uses them
COLUMNS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
    'HospAdmTime', 'ICULOS', 'SepsisLabel'
]

def generate_patient(filename, condition):
    """
    Generates a 24-hour timeline of vitals/labs for a simulated patient.
    condition: 'healthy', 'sepsis_fast', 'sepsis_slow', 'recovering'
    """
    hours = 24
    data = []
    
    # Baselines
    age = np.random.randint(40, 80)
    gender = np.random.randint(0, 2) # 0=Female, 1=Male
    
    # Initial Vitals (Healthy-ish)
    hr = 75.0
    sbp = 120.0
    temp = 37.0
    resp = 16.0
    o2 = 98.0
    
    # Labs (Initially Missing or Normal)
    lactate = np.nan
    wbc = np.nan
    creat = np.nan
    glucose = np.nan
    
    for i in range(hours):
        row = {col: np.nan for col in COLUMNS}
        row['Age'] = age
        row['Gender'] = gender
        row['ICULOS'] = i + 1
        row['SepsisLabel'] = 0 # Default
        
        # Dynamic Evolution based on Condition
        if condition == 'healthy':
            # Stable vitals, random noise
            hr = max(60, min(100, hr + np.random.normal(0, 2)))
            sbp = max(110, min(130, sbp + np.random.normal(0, 2)))
            temp = 37.0 + np.random.normal(0, 0.1)
            # Rarely measure labs
            if i == 10: # Routine check
                row['Glucose'] = 100 + np.random.normal(0, 5)
                row['WBC'] = 7.0
        
        elif condition == 'sepsis_fast':
            # Rapid deterioration starting hour 10
            if i > 10:
                hr += 3.0  # Tachycardia
                sbp -= 2.0 # Hypotension
                temp += 0.1 # Fever
                resp += 0.5
                row['SepsisLabel'] = 1 if i > 16 else 0
                
                # Labs ordered frequently
                if i % 2 == 0:
                    lactate = 2.0 + (i-10)*0.3 # Rising Lactate
                    wbc = 12.0 + (i-10)*1.0    # Rising WBC
                    row['Lactate'] = lactate
                    row['WBC'] = wbc
                    row['Creatinine'] = 1.2 + (i-10)*0.1
            else:
                # Stable before onset
                hr += np.random.normal(0, 1)
        
        elif condition == 'recovering':
            # Starts sick, gets better
            if i == 0:
                hr = 110
                sbp = 90
                temp = 38.5
            
            if i < 12:
                # Improving Phase
                hr -= 2.0
                sbp += 1.5
                temp -= 0.1
                
                # Labs every 4 hours while sick
                if i % 4 == 0:
                    row['WBC'] = max(6.0, 15.0 - i*0.5)
                    row['Lactate'] = max(1.0, 4.0 - i*0.2)
            else:
                # Stable / Recovered Phase
                hr = 80 + np.random.normal(0, 2)
                sbp = 115 + np.random.normal(0, 2)
                temp = 37.0
                
                # Confirm recovery with Frequent Normal Labs
                if i % 2 == 0:
                    row['WBC'] = 7.0 + np.random.normal(0, 0.5) # Normal
                    row['Lactate'] = 1.0 + np.random.normal(0, 0.1) # Normal


        # Assign Vitals
        row['HR'] = hr
        row['SBP'] = sbp
        row['Temp'] = temp
        row['Resp'] = resp
        row['O2Sat'] = o2
        
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join('patient_data', filename), sep='|', index=False)
    print(f"Generated {filename} ({condition})")

# Generate 5 Patients
generate_patient('p100001.psv', 'healthy')
generate_patient('p100002.psv', 'sepsis_fast')
generate_patient('p100003.psv', 'recovering')
generate_patient('p100004.psv', 'healthy')
generate_patient('p100005.psv', 'sepsis_fast')
