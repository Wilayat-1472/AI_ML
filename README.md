<p align="center">
  <img src="https://img.icons8.com/3d-fluency/94/brain.png" width="80" alt="AI/ML Logo"/>
</p>

<h1 align="center">AI/ML Engineering Internship Tasks</h1>

<p align="center">
  <em>Three production-grade machine learning projects — classification, regression, and NLP fine-tuning<br/>submitted for the DevelopersHub Corporation AI/ML Engineering Internship.</em>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://huggingface.co"><img src="https://img.shields.io/badge/🤗%20Transformers-5.12-FFD21E?style=for-the-badge" alt="Transformers"/></a>
  <a href="https://jupyter.org"><img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-completed-tasks">Tasks</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 📋 Overview

This repository contains **3 of 6** selected AI/ML projects completed as part of the DevelopersHub Corporation AI/ML Engineering Internship. Each task is a self-contained Jupyter notebook with full data preprocessing, exploratory analysis, model training, evaluation, and interpretation of results.

| Task | Domain | Approach | Key Metric |
| :--- | :--- | :--- | :--- |
| **Task 3** — Heart Disease Prediction | Healthcare / Binary Classification | Logistic Regression, Decision Tree | ROC-AUC ~0.90 |
| **Task 5** — Mental Health Support Chatbot | NLP / Fine-Tuning | DistilGPT2 + EmpatheticDialogues | Response Quality (Qualitative) |
| **Task 6** — House Price Prediction | Real Estate / Regression | Gradient Boosting, Linear Regression | R² ~0.80, MAE ~$35K |

---

## ✅ Completed Tasks

### Task 3: Heart Disease Prediction

A binary classification model that predicts whether a patient is at risk of heart disease based on clinical measurements from the UCI Heart Disease dataset.

**Pipeline:**
```
Data Cleaning → EDA → Feature Scaling → Logistic Regression → Decision Tree → ROC-AUC Evaluation → Feature Importance Analysis
```

| Capability | Details |
| :--- | :--- |
| **Dataset** | UCI Heart Disease (Cleveland) — 303 samples, 13 features |
| **Preprocessing** | Missing value imputation (median), target binarization, StandardScaler |
| **Models** | Logistic Regression (`max_iter=1000`) · Decision Tree (`max_depth=5`) |
| **Evaluation** | Accuracy, Confusion Matrix, ROC Curve, AUC, Cross-Validation, Feature Importance |
| **Key Features** | `ca` (vessels), `thal`, `oldpeak`, `cp`, `thalach` |

| Model | Accuracy | ROC-AUC |
| :--- | :---: | :---: |
| Logistic Regression | ~85% | ~0.90 |
| Decision Tree | ~82% | ~0.85 |

> **Notebook:** [`task3_heart_disease/heart_disease_prediction.ipynb`](task3_heart_disease/heart_disease_prediction.ipynb)

---

### Task 5: Mental Health Support Chatbot

A fine-tuned language model that provides empathetic, supportive responses for stress, anxiety, and emotional wellness — built by fine-tuning DistilGPT2 on Facebook AI's EmpatheticDialogues dataset.

**Pipeline:**
```
Load Dataset → Format Conversations → Tokenize → Fine-tune DistilGPT2 (Hugging Face Trainer API) → Generate Responses → CLI + Streamlit Interface
```

| Capability | Details |
| :--- | :--- |
| **Base Model** | DistilGPT2 (82M parameters) — lightweight causal language model |
| **Dataset** | EmpatheticDialogues (Facebook AI) — 25K+ empathetic conversations |
| **Fine-Tuning** | Hugging Face `Trainer` API · 1 epoch · learning rate 5e-5 · batch size 8 |
| **Generation** | Top-k (50) + Top-p (0.95) sampling · temperature 0.7 |
| **Interfaces** | Interactive CLI [`chatbot_cli.py`](task5_mental_health_chatbot/chatbot_cli.py) · Streamlit web app [`streamlit_app.py`](task5_mental_health_chatbot/streamlit_app.py) |

> ⚠️ **GPU Recommended:** Full training requires significant compute. The notebook uses a 5,000-sample subset for CPU demonstration. For production use, train on GPU with the full 25K+ dataset over multiple epochs.

> **Notebook:** [`task5_mental_health_chatbot/mental_health_chatbot.ipynb`](task5_mental_health_chatbot/mental_health_chatbot.ipynb)

---

### Task 6: House Price Prediction

A regression model that predicts median house values from property and demographic features using the California Housing dataset.

**Pipeline:**
```
Data Loading → EDA → Feature Engineering (rooms_per_household, bedrooms_per_room, location_score) → Feature Scaling → Linear Regression → Gradient Boosting → MAE / RMSE / R² Evaluation
```

| Capability | Details |
| :--- | :--- |
| **Dataset** | California Housing (sklearn) — 20,640 samples, 8 features |
| **Feature Engineering** | `rooms_per_household`, `bedrooms_per_room`, `population_per_household`, `location_score` |
| **Models** | Linear Regression · Gradient Boosting (`n_estimators=200`, `max_depth=4`) |
| **Evaluation** | MAE, RMSE, R² Score, Predicted vs Actual Plot, Residual Analysis, Error Distribution |
| **Key Features** | `median_income`, `location_score`, `avg_occupancy`, `avg_rooms` |

| Model | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: |
| Linear Regression | ~$49K | ~$70K | ~0.61 |
| Gradient Boosting | **~$35K** | **~$49K** | **~0.80** |

> **Notebook:** [`task6_house_price/house_price_prediction.ipynb`](task6_house_price/house_price_prediction.ipynb)

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Data Processing** | pandas 3.0, numpy 2.5 |
| **Machine Learning** | scikit-learn 1.9 (Logistic Regression, Decision Tree, Gradient Boosting) |
| **Deep Learning** | PyTorch 2.12 (CUDA 13) |
| **NLP / Transformers** | Hugging Face Transformers 5.12, Datasets 5.0, Tokenizers |
| **Visualization** | matplotlib 3.11, seaborn 0.13, plotly 6.8 |
| **Development** | Jupyter Notebook 7.6, nbformat |
| **Deployment** | Streamlit (chatbot UI) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Wilayat-1472/AI_ML.git
cd AI_ML

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib torch transformers datasets streamlit jupyter kagglehub python-dotenv
```

### Kaggle Credentials (Optional)

Some notebooks download datasets from Kaggle. To use this feature:

1. Copy the example env file and fill in your Kaggle API credentials:
```bash
cp .env.example .env
```

2. Edit `.env` with your Kaggle username and API key:
```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```
> **Security:** `.env` is in `.gitignore` and will never be committed. Credentials are loaded at runtime only.

3. If no `.env` is found, notebooks automatically fall back to built-in sklearn datasets or public URLs — no credentials required.

### Run Notebooks

```bash
# Launch Jupyter and open any task notebook
jupyter notebook

# Or run directly:
jupyter notebook task3_heart_disease/heart_disease_prediction.ipynb
jupyter notebook task6_house_price/house_price_prediction.ipynb
jupyter notebook task5_mental_health_chatbot/mental_health_chatbot.ipynb
```

### Run Chatbot Interfaces

```bash
# Interactive CLI
python task5_mental_health_chatbot/chatbot_cli.py

# Streamlit web app
streamlit run task5_mental_health_chatbot/streamlit_app.py
```

> **Note:** The fine-tuned model checkpoint is not included in the repository due to size. Run the Task 5 notebook to generate it, or train on GPU for optimal results.

---

## 📁 Project Structure

```
AI_ML/
├── README.md                              # This file
├── .gitignore
│
├── task3_heart_disease/
│   └── heart_disease_prediction.ipynb      # Classification — Heart Disease Risk
│
├── task5_mental_health_chatbot/
│   ├── mental_health_chatbot.ipynb         # Fine-tuning — Empathetic Chatbot
│   ├── chatbot_cli.py                      # Command-line chatbot interface
│   └── streamlit_app.py                    # Streamlit web chatbot interface
│
└── task6_house_price/
    └── house_price_prediction.ipynb        # Regression — House Price Estimation
```

---

## 📄 Submission

| Detail | Info |
| :--- | :--- |
| **Organization** | DevelopersHub Corporation |
| **Program** | AI/ML Engineering Internship |
| **Due Date** | 26th June, 2026 |
| **Tasks Completed** | 3 of 6 (Tasks 3, 5, 6) |
| **Submitted By** | Wilayat Ali |

---

<p align="center">
  <strong>Built with ❤️ for the DevelopersHub AI/ML Engineering Internship</strong>
</p>
