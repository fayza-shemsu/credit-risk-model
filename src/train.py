"""
train.py - Train a credit risk model and track with MLflow

This script:
1. Creates a synthetic credit-like dataset
2. Preprocesses the data (encoding + scaling)
3. Trains a Logistic Regression model
4. Logs parameters, metrics, and model to MLflow
5. Saves the trained model locally for API usage
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ==============================
# IMPORT CONFIGURATION
# ==============================
from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_NAME,
)

# ==============================
# SETUP LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================
# DATA LOADING (Synthetic Dataset)
# ==============================
def load_data():
    """Create a synthetic credit-like dataset."""
    logger.info("Creating synthetic dataset...")
    np.random.seed(RANDOM_STATE)
    n_samples = 1000

    duration = np.random.randint(6, 72, n_samples)
    credit_amount = np.random.randint(500, 20000, n_samples)
    age = np.random.randint(18, 75, n_samples)
    installment_commitment = np.random.randint(1, 5, n_samples)
    residence_since = np.random.randint(1, 5, n_samples)
    existing_credits = np.random.randint(1, 4, n_samples)
    num_dependents = np.random.randint(1, 3, n_samples)

    checking_status = np.random.choice(["low", "medium", "high", "none"], n_samples)
    credit_history = np.random.choice(["bad", "ok", "good"], n_samples)
    purpose = np.random.choice(["car", "electronics", "furniture", "education", "other"], n_samples)
    savings_status = np.random.choice(["low", "medium", "high", "none"], n_samples)
    employment = np.random.choice(["short", "medium", "long", "unemployed"], n_samples)
    personal_status = np.random.choice(["single", "married", "other"], n_samples)
    other_parties = np.random.choice(["none", "co_applicant", "guarantor"], n_samples)
    property_magnitude = np.random.choice(["car", "real_estate", "other"], n_samples)
    other_payment_plans = np.random.choice(["none", "bank", "store"], n_samples)
    housing = np.random.choice(["rent", "own", "free"], n_samples)
    job = np.random.choice(["unskilled", "skilled", "management"], n_samples)
    own_telephone = np.random.choice(["yes", "no"], n_samples)
    foreign_worker = np.random.choice(["yes", "no"], n_samples)

    X = pd.DataFrame({
        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment": employment,
        "installment_commitment": installment_commitment,
        "personal_status": personal_status,
        "other_parties": other_parties,
        "residence_since": residence_since,
        "property_magnitude": property_magnitude,
        "age": age,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "own_telephone": own_telephone,
        "foreign_worker": foreign_worker
    })

    risk_score = (
        (credit_amount > 10000).astype(int) * 0.3 +
        (duration < 12).astype(int) * 0.2 +
        (age < 25).astype(int) * 0.2 +
        (checking_status == "low").astype(int) * 0.3 +
        np.random.random(n_samples) * 0.3
    )

    y = (risk_score > 0.5).astype(int)

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")

    return X, y


# ==============================
# PREPROCESSING
# ==============================
def create_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor


# ==============================
# MODEL TRAINING
# ==============================
def train_model(X_train, y_train, preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    logger.info("Training completed.")

    return pipeline


# ==============================
# EVALUATION
# ==============================
def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info("\nClassification Report:\n" + report)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    return accuracy, roc_auc, report


# ==============================
# SAVE MODEL
# ==============================
def save_model(model):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{MODEL_NAME}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved at {model_path}")


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = create_preprocessor(X_train)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk-demo")

    with mlflow.start_run():

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("stratified_split", True)

        model = train_model(X_train, y_train, preprocessor)

        accuracy, roc_auc, report = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_text(report, "classification_report.txt")

        mlflow.sklearn.log_model(model, "model")

        save_model(model)

        logger.info("MLflow tracking complete.")
        logger.info("Run 'mlflow ui' to view experiments.")


if __name__ == "__main__":
    main()
