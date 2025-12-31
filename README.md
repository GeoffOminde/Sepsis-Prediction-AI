# 🏥 Privacy-Preserving Sepsis Prediction using Generative AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Generative_AI-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This project implements an end-to-end AI pipeline to predict Sepsis risk in ICU patients. Unlike standard predictive models, this solution prioritizes **Patient Privacy** and **AI Ethics**. 

It utilizes a **Variational Autoencoder (VAE)** to learn the distribution of patient vital signs and generate **synthetic data**. The final classification model is trained on this privacy-compliant synthetic dataset, ensuring no raw Personal Health Information (PHI) is exposed during model training.

The system is deployed as an interactive web application using **Streamlit**, allowing clinicians to input vital signs and receive instant risk assessments with explainable insights.

## 🚀 Key Features

### 1. Generative AI for Data Augmentation
* **Architecture:** PyTorch-based Variational Autoencoder (VAE).
* **Purpose:** Upsampling rare sepsis cases and generating privacy-preserving synthetic patient records.
* **Benefit:** solves the "Class Imbalance" problem common in medical datasets (where 95% of patients are healthy).

### 2. Clinical Feature Engineering
* **Shock Index (SI):** Implemented `Heart Rate / Systolic BP` as a derived feature.
* **Reality Clamp:** The pipeline includes physics-based constraints to prevent "hallucinations" (e.g., negative white blood cell counts) often found in generative models.

### 3. AI Ethics & Responsibility
* **Fairness Audit:** Automated checks to ensure the model performs equally well across different age groups (Seniors vs. Adults).
* **Explainability:** Provides actionable recommendations (e.g., "Call Rapid Response") rather than just opaque probabilities.

### 4. Deployment
* **Git LFS:** Uses Git Large File Storage to handle high-fidelity model artifacts (>100MB).
* **Streamlit App:** A user-friendly interface for real-time inference.

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* Git (with Git LFS installed)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/GeoffOminde/Sepsis-Prediction-AI.git](https://github.com/GeoffOminde/Sepsis-Prediction-AI.git)
    cd Sepsis-Prediction-AI
    ```

2.  **Pull the Large Model File:**
    This project uses Git LFS for the model file (`sepsis_model_v1.pkl`).
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

### Running the Web App
To launch the interactive dashboard:
```bash
streamlit run app.py
```
The app will open in your browser at http://localhost:8501.

You can adjust sliders for Heart Rate, BP, and Temp to see how risk scores change in real-time.

Using the Model in Python
Python

import joblib
import pandas as pd

# Load the model
model = joblib.load('sepsis_model_v1.pkl')

# Predict for a new patient
data = pd.DataFrame([{
    'Age': 72, 'HeartRate': 110, 'SysBP': 90, 'Temp': 34.5, 'WBC': 18.0,
    'ShockIndex': 1.22  # Calculated as HR / SysBP
}])
risk = model.predict_proba(data)[0][1]
print(f"Sepsis Risk: {risk:.2%}")
📂 Project Structure
Plaintext

Sepsis-Prediction-AI/
│
├── app.py                # Streamlit Web Application entry point
├── sepsis_model_v1.pkl   # Trained Random Forest Model (Stored via LFS)
├── notebook.ipynb        # Jupyter Notebook used for training & VAE generation
├── requirements.txt      # Python dependencies
└── README.md             # Project Documentation
📊 Model Performance
Metric Focus: Recall (Sensitivity) to minimize false negatives (missed cases).

Imbalance Handling: Generative upsampling improved minority class detection.

Calibration: Logic implemented to detect "Cold Sepsis" (Hypothermia) patterns.

⚠️ Ethical Statement
This tool is designed as a Clinical Decision Support System (CDSS) and is not a replacement for professional medical judgment.

Human-in-the-Loop: High-risk scores trigger a recommendation for human verification.

Bias Check: The model has been audited for age-related performance disparities.

📜 License
MIT License. Data usage adheres to PhysioNet/MIMIC data use agreements.