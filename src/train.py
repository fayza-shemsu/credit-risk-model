"""
train.py - Train a credit risk model and track with MLflow

This script:
1. Creates a synthetic credit-like dataset (simpler than downloading)
2. Prepares the data (encode categories, scale numbers)
3. Trains a simple LogisticRegression model
4. Logs everything to MLflow (parameters, metrics, model)
5. Saves the model to a file for the API to use
"""

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
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================
# SETUP LOGGING
# ============================================================
# This helps us see what's happening when the script runs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    """
    Create a synthetic credit-like dataset.
    
    This is simpler than downloading external data and works offline.
    The dataset has both numeric and categorical features,
    similar to real credit data.
    """
    logger.info("Creating synthetic credit dataset...")
    
    # Set random seed for reproducibility
    # This ensures we get the same "random" data every time we run
    np.random.seed(42)
    n_samples = 1000  # Number of fake customers to generate
    
    # Create numeric features
    duration = np.random.randint(6, 72, n_samples)  # Loan duration in months
    credit_amount = np.random.randint(500, 20000, n_samples)  # Loan amount
    age = np.random.randint(18, 75, n_samples)  # Customer age
    installment_commitment = np.random.randint(1, 5, n_samples)  # 1-4 scale
    residence_since = np.random.randint(1, 5, n_samples)  # Years at address
    existing_credits = np.random.randint(1, 4, n_samples)  # Number of credits
    num_dependents = np.random.randint(1, 3, n_samples)  # Number of dependents
    
    # Create categorical features with simple, readable values
    # These are much easier to explain than the original credit dataset codes
    checking_status = np.random.choice(
        ["low", "medium", "high", "none"], n_samples
    )
    credit_history = np.random.choice(
        ["bad", "ok", "good"], n_samples
    )
    purpose = np.random.choice(
        ["car", "electronics", "furniture", "education", "other"], n_samples
    )
    savings_status = np.random.choice(
        ["low", "medium", "high", "none"], n_samples
    )
    employment = np.random.choice(
        ["short", "medium", "long", "unemployed"], n_samples
    )
    personal_status = np.random.choice(
        ["single", "married", "other"], n_samples
    )
    other_parties = np.random.choice(
        ["none", "co_applicant", "guarantor"], n_samples
    )
    property_magnitude = np.random.choice(
        ["car", "real_estate", "other"], n_samples
    )
    other_payment_plans = np.random.choice(
        ["none", "bank", "store"], n_samples
    )
    housing = np.random.choice(
        ["rent", "own", "free"], n_samples
    )
    job = np.random.choice(
        ["unskilled", "skilled", "management"], n_samples
    )
    own_telephone = np.random.choice(["yes", "no"], n_samples)
    foreign_worker = np.random.choice(["yes", "no"], n_samples)
    
    # Create DataFrame
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
    
    # Create target variable (credit risk)
    # We define a simple rule: higher risk if loan is big, short, young, or low checking
    # In real ML, you wouldn't know this formula - the model learns it from data
    risk_score = (
        (credit_amount > 10000).astype(int) * 0.3 +  # Big loans are risky
        (duration < 12).astype(int) * 0.2 +          # Short duration = risky
        (age < 25).astype(int) * 0.2 +               # Young = risky
        (checking_status == "low").astype(int) * 0.3 +  # Low balance = risky
        np.random.random(n_samples) * 0.3            # Random noise (real data is messy)
    )
    # Convert score to binary: 1 = high risk (will default), 0 = low risk (safe)
    y = (risk_score > 0.5).astype(int)
    
    logger.info(f"Dataset created: {X.shape[0]} rows, {X.shape[1]} columns")
    logger.info(f"Target distribution: 0 (good)={sum(y==0)}, 1 (bad)={sum(y==1)}")
    
    return X, y


def create_preprocessing_pipeline(X):
    """
    Create a preprocessing pipeline that handles:
    - Numeric columns: fill missing values, then scale
    - Categorical columns: fill missing values, then one-hot encode
    """
    # Find which columns are numbers and which are categories
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    
    logger.info(f"Numeric columns: {numeric_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")
    
    # ColumnTransformer applies different transformations to different columns
    # WHY: ML models need numbers. Categories like "low" must become numbers.
    # WHY: Numbers should be on similar scales (age 25 vs amount 10000 would confuse model)
    preprocessor = ColumnTransformer(
        transformers=[
            # StandardScaler: transforms to mean=0, std=1 (e.g., age 25 → -0.3)
            ("num", StandardScaler(), numeric_cols),
            # OneHotEncoder: "low" → [1,0,0,0], "high" → [0,0,1,0]
            # handle_unknown="ignore" prevents errors on new categories at prediction time
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    
    return preprocessor


def train_model(X_train, y_train, preprocessor):
    """
    Create and train the full pipeline:
    1. Preprocessing (scaling + encoding)
    2. Logistic Regression classifier
    """
    logger.info("Creating model pipeline...")
    
    # Pipeline chains preprocessing and model together
    # WHY: One object to save, load, and use. No risk of forgetting preprocessing steps.
    # When you call pipeline.predict(X), it automatically preprocesses first, then predicts.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # Step 1: scale numbers, encode categories
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),  # Step 2: predict
        ]
    )
    
    logger.info("Training model...")
    # fit() learns from training data: which scaling factors, which category mappings,
    # and which coefficients best predict y from X
    pipeline.fit(X_train, y_train)
    logger.info("Training complete!")
    
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the model and return metrics.
    
    We calculate:
    - Accuracy: how many predictions are correct
    - ROC-AUC: how well the model ranks predictions
    """
    logger.info("Evaluating model...")
    
    # Get predictions
    y_pred = pipeline.predict(X_test)  # Returns 0 or 1 for each sample
    # predict_proba returns probabilities: [[prob_class_0, prob_class_1], ...]
    # We take [:, 1] to get just the probability of class 1 (high risk / bad)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    return {"accuracy": accuracy, "roc_auc": roc_auc}


def save_model(pipeline, filepath="models/credit_model.pkl"):
    """
    Save the trained model to a file.
    The API will load this file to make predictions.
    """
    joblib.dump(pipeline, filepath)
    logger.info(f"Model saved to {filepath}")


def main():
    """
    Main function that runs the entire training process.
    """
    # ========================================================
    # STEP 1: Load data
    # ========================================================
    X, y = load_data()
    
    # ========================================================
    # STEP 2: Split into train and test sets
    # ========================================================
    # We use 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ========================================================
    # STEP 3: Create preprocessing pipeline
    # ========================================================
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # ========================================================
    # STEP 4: Train model with MLflow tracking
    # ========================================================
    # Tell MLflow where to save experiment data (local folder)
    # You could also use a remote server like "http://mlflow-server:5000"
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Experiments group related runs (e.g., "credit-risk-demo" vs "fraud-detection")
    mlflow.set_experiment("credit-risk-demo")
    
    # start_run() creates a new "run" - a single training attempt
    # Everything logged inside this block is saved together
    with mlflow.start_run():
        
        # Log parameters (settings we used) - these are inputs to the experiment
        # Later you can compare runs: "what happens if I change test_size to 0.3?"
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Train the model
        pipeline = train_model(X_train, y_train, preprocessor)
        
        # Evaluate and log metrics - these are outputs/results of the experiment
        # Later you can sort runs by accuracy to find the best one
        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metric("accuracy", metrics["accuracy"])  # % correct predictions
        mlflow.log_metric("roc_auc", metrics["roc_auc"])    # ranking quality (0.5=random, 1.0=perfect)
        
        # Log the model to MLflow (for tracking/versioning)
        # This saves the model artifact inside mlruns/ so you can reload it later
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Also save locally as a .pkl file for the FastAPI to load
        # In production, you might load directly from MLflow instead
        save_model(pipeline)
        
        logger.info("MLflow tracking complete!")
        logger.info("Run 'mlflow ui' to see the experiment dashboard")


if __name__ == "__main__":
    main()
