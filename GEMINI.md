# Sepsis Guardian

## Project Overview

**Sepsis Guardian** is a research prototype web application designed to predict the onset of sepsis in ICU patients up to 4 hours in advance. It utilizes a deep learning model (Time-Aware LSTM) that specifically addresses "clinical missingness" by incorporating the presence or absence of lab tests as features.

### Key Technologies
*   **Backend:** Python (Flask)
*   **Machine Learning:** PyTorch (LSTM Architecture)
*   **Frontend:** HTML/Bootstrap (via Jinja2 templates)

### Architecture
The core logic resides in a Flask application that serves a web interface and an inference API.
*   **Model:** `SepsisLSTM_TimeAware` (in `models_utils.py`). Takes 32 input features:
    *   15 Dynamic Clinical Values (Vitals & Labs)
    *   15 "Presence Masks" (Boolean flags indicating if data is available)
    *   2 Static Features (Age, Gender)
*   **Inference:** The application simulates a 24-hour history for single-point predictions by tiling the input vector.

## Building and Running

### Prerequisites
*   Python 3.x
*   `pip` (Python package manager)

### Installation
1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Ensure the pre-trained model file `sepsis_model_time_aware.pth` is present in the root directory.

### Running the Application
1.  Start the Flask development server:
    ```bash
    python app.py
    ```
2.  Access the application in your browser at:
    `http://localhost:5000`

## Key Files

*   **`app.py`**: The main entry point. Initializes the Flask app, loads the PyTorch model, and defines the `/predict` API endpoint. Implements the "Traffic Light" clinical protocol logic.
*   **`models_utils.py`**: Contains the PyTorch model definition (`SepsisLSTM_TimeAware`) and data preprocessing logic (`preprocess_input`). This file handles the fusion of clinical values, missingness masks, and static context.
*   **`sepsis_model_time_aware.pth`**: The serialized PyTorch model state dictionary.
*   **`model_config_time_aware.json`**: Configuration file (likely defining hyperparameters or feature lists).
*   **`templates/index.html`**: The frontend user interface for inputting patient data and viewing risk predictions.

## Development Conventions

*   **Clinical Protocols:** The application uses hardcoded probability thresholds for decision support:
    *   **< 0.25:** Low Risk (Green)
    *   **0.25 - 0.30:** Screening Alert (Yellow)
    *   **> 0.30:** Sepsis Alert (Red)
*   **Data Handling:** The `preprocess_input` function in `models_utils.py` is critical. It transforms raw form inputs into the specific 32-dimensional tensor format expected by the LSTM, handling missing values by zero-filling and updating the corresponding mask bit.
*   **Dependencies:** Managed via `requirements.txt`.
