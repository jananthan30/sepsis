# 🏥 Sepsis Guardian: Time-Aware Deep Learning for ICU

**Author:** [Your Name]  
**Status:** Research Prototype (v3.0)  
**Architecture:** Context-Aware LSTM with Informative Missingness

---

## 📋 Executive Summary

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. Traditional scoring systems (SOFA, qSOFA) are often reactive, detecting sepsis only after organ damage has begun.

**Sepsis Guardian** is a Deep Learning system designed to predict sepsis onset **4 hours in advance**. It utilizes a **Time-Aware LSTM architecture** that explicitly models "clinical missingness" (the frequency of lab tests) as a feature, allowing it to adapt to different hospital protocols without retraining.

## 🚀 Key Performance Indicators (KPIs)

| Metric                       | Sepsis Guardian (Time-Aware) | Standard Clinical Tool (SOFA) |
| :--------------------------- | :--------------------------- | :---------------------------- |
| **Internal AUC (MIMIC-IV)**  | **0.853**                    | ~0.74                         |
| **External AUC (PhysioNet)** | **0.802**                    | ~0.70                         |
| **Screening Sensitivity**    | **95.0%** (at Risk > 0.25)   | Varies                        |
| **Diagnostic Specificity**   | **96.5%** (at Risk > 0.40)   | ~90%                          |

## 🧠 Model Architecture

The model solves the "Domain Shift" problem common in medical AI using a **32-Feature Input Fusion** strategy:

1.  **Dynamic Values (15):** Vitals (HR, BP) + Labs (Lactate, Troponin, Glucose).
2.  **Presence Masks (15):** Boolean flags indicating if a test was ordered at time $t$. This prevents the model from hallucinating health when data is missing.
3.  **Static Context (2):** Age and Gender fused at every time step.

## 🛠 Tech Stack

- **Training:** PyTorch, Google BigQuery, MIMIC-IV Pipeline.
- **Deployment:** Flask (Python), Bootstrap 5 (Frontend).
- **Validation:** Tested on 40,000+ patients across 3 distinct hospital systems.

## 🚦 Clinical Protocol

The dashboard implements a "Traffic Light" protocol based on calibrated probability thresholds:

- **🟢 GREEN (<25%):** High Safety. Patient is stable.
- **🟡 YELLOW (25-30%):** Screening Alert. Nurse check required.
- **🔴 RED (>30%):** Code Sepsis. Immediate diagnostic workup.

## 💻 How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Place the model file `sepsis_model_time_aware.pth` in the root directory.
3. Run the server: `python app.py`
4. Open browser at `http://localhost:5000`
