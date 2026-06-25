# AI/ML Engineering Internship Tasks - DevelopersHub Corporation

## Overview
This repository contains my submissions for the AI/ML Engineering Internship at DevelopersHub Corporation. I completed **3 out of 6 tasks** covering classification, regression, and NLP/fine-tuning.

## Completed Tasks

### Task 3: Heart Disease Prediction
- **Objective**: Predict heart disease risk using health metrics
- **Dataset**: Heart Disease UCI Dataset
- **Models**: Logistic Regression, Decision Tree Classifier
- **Key Results**:
  - Logistic Regression: ~85% accuracy, AUC ~0.90
  - Decision Tree: ~82% accuracy, AUC ~0.85
  - Top features: `ca` (vessels), `thal`, `oldpeak`, `cp`
- **Notebook**: `task3_heart_disease/heart_disease_prediction.ipynb`

### Task 5: Mental Health Support Chatbot
- **Objective**: Fine-tune an LLM for empathetic mental health support
- **Model**: DistilGPT2 fine-tuned on EmpatheticDialogues (Facebook AI)
- **Features**:
  - Fine-tuned with Hugging Face Trainer API
  - Generates supportive, empathetic responses
  - CLI and Streamlit interfaces included
- **Note**: Full training on GPU is recommended for production use. The notebook uses a subset for demonstration on CPU.
- **Notebook**: `task5_mental_health_chatbot/mental_health_chatbot.ipynb`
- **Web App**: `task5_mental_health_chatbot/streamlit_app.py`

### Task 6: House Price Prediction
- **Objective**: Predict house prices from property features
- **Dataset**: California Housing Dataset
- **Models**: Linear Regression, Gradient Boosting Regressor
- **Key Results**:
  - Gradient Boosting: R² ~0.80, MAE ~$35K
  - Linear Regression: R² ~0.61, MAE ~$49K
  - Top features: `median_income`, `location`, `avg_occupancy`
- **Notebook**: `task6_house_price/house_price_prediction.ipynb`

## Technologies Used
- Python 3.12
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- PyTorch, Hugging Face Transformers
- Jupyter Notebook
- Streamlit

## How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib torch transformers datasets streamlit

# Run a notebook
jupyter notebook task3_heart_disease/heart_disease_prediction.ipynb

# Run Streamlit chatbot app
streamlit run task5_mental_health_chatbot/streamlit_app.py
```

## Submission
- **Due Date**: 26th June, 2026
- **Submitted by**: Wilayat Ali

---

*DevelopersHub Corporation - AI/ML Engineering Internship Tasks*
