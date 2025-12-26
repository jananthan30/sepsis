# Tech Stack

## Languages
| Language | Version | Purpose |
|----------|---------|---------|
| Python | 3.11 | Primary development language |

## ML/AI Framework
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Deep Learning | TensorFlow | 2.20.0 | Model training and inference |
| Keras | Keras | 3.12.0 | High-level neural network API |
| ML Utilities | scikit-learn | 1.3.0 | Preprocessing, metrics, clustering |
| Explainability | SHAP | 0.44.1 | Model interpretability |

## Data Processing
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| DataFrames | Pandas | 2.1.0 | Data manipulation and analysis |
| Numerical | NumPy | 1.26.0 | Array operations and linear algebra |
| Serialization | joblib | 1.3.0+ | Model artifact persistence |
| Serialization | pickle | (stdlib) | Config persistence |

## Web Frameworks
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Dashboard | Streamlit | 1.32.0+ | Interactive web interface |
| REST API | FastAPI | 0.109.0+ | RESTful API endpoints |
| ASGI Server | Uvicorn | 0.27.0+ | Production ASGI server |
| Validation | Pydantic | 2.5.0+ | Request/response validation |

## Visualization
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Interactive | Plotly | 5.18.0 | Interactive charts in Streamlit |
| Static | Matplotlib | 3.8.0 | Publication-quality figures |
| Statistical | Seaborn | 0.13.0 | Statistical visualizations |

## Data Sources
| Component | Technology | Purpose |
|-----------|------------|---------|
| Database | Google BigQuery | MIMIC-IV data extraction |
| Dataset | MIMIC-IV | Clinical ICU data (PhysioNet) |

## Deployment
| Component | Technology | Purpose |
|-----------|------------|---------|
| Containerization | Docker | Reproducible deployment |
| Platform | HuggingFace Spaces | Public demo hosting |
| Port Configuration | 7860 (Streamlit), 8000 (FastAPI) | Service endpoints |

## Model Architecture
```
Input (24 timesteps x 32 features)
    │
    ├── 15 clinical values (vitals + labs)
    ├── 15 freshness masks (time-aware)
    └── 2 static features (age, gender)
    │
    ▼
Bidirectional LSTM (64 units)
    │
    ▼
BatchNormalization
    │
    ▼
Bidirectional LSTM (32 units)
    │
    ▼
BatchNormalization
    │
    ▼
Dense (32, ReLU)
    │
    ▼
Dense (1, Sigmoid)
    │
    ▼
Output: Sepsis Probability [0, 1]
```

## Project Structure
```
Sepsis/
├── streamlit_app.py      # Main dashboard application
├── api.py                # FastAPI REST endpoints
├── train_model_v3.py     # Legacy 3-hour gap training
├── train_model_v4.py     # 6-hour gap with EDA
├── train_model_v6.py     # Latest model version
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── start.sh              # Multi-service startup script
├── sepsis_model_v3/      # V3 model artifacts
├── sepsis_model_v4/      # V4 model artifacts + EDA
├── sepsis_model_v6/      # V6 model artifacts (production)
└── maestro/              # Spec-driven development context
```

## Development Environment
- **Python Environment**: pip with requirements.txt
- **IDE**: Any (no IDE-specific configs detected)
- **Version Control**: Git
- **Remote Repository**: GitHub
