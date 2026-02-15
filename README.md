# Week 12 Capstone Improvement

This repository is being enhanced as part of the Week 12 Capstone Challenge.  
The goal is to transform the original Credit Risk Probability Model into a production-grade, finance-ready ML system with improved engineering, testing, and documentation standards.

---

## Project Overview
This project implements an end-to-end **credit risk scoring system**.
A proxy target is engineered using customer transaction behavior, a machine
learning model is trained and tracked with MLflow, and predictions are served
through a FastAPI application deployed with Docker.

---

## Data & Target Engineering
Since labeled default data is unavailable, a **proxy target** is created using
RFM (Recency, Frequency, Monetary) features:

- Recency: Days since last transaction
- Frequency: Number of transactions
- Monetary: Total transaction amount

KMeans clustering is applied to RFM features, and the cluster with the lowest
average monetary value is labeled as **high risk**.

---

## Model Training
- Model: Logistic Regression
- Preprocessing: StandardScaler + OneHotEncoder
- Experiment tracking: MLflow
- Output: Trained model saved as `models/credit_model.pkl`

### Train the model
```bash
python src/train.py
