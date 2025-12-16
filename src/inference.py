"""
inference.py - Load the model and make predictions

This module provides helper functions that the API uses:
1. load_model() - Load the trained model from file
2. predict() - Make a prediction for new customer data
"""

import joblib
import pandas as pd
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Where our trained model is saved
MODEL_PATH = "models/credit_model.pkl"

# Global variable to store the loaded model
# WHY: Loading a model from disk is slow (~100ms). We do it once at startup,
# then reuse the same object for all predictions (fast, ~1ms each)
_model = None


def load_model():
    """
    Load the trained model from disk.
    
    Returns the sklearn pipeline that includes:
    - Preprocessing (scaling + encoding)
    - Classifier (LogisticRegression)
    """
    global _model  # Use the module-level variable
    
    # Only load if not already loaded (singleton pattern)
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}...")
        # joblib.load() reads the .pkl file and reconstructs the Python object
        # This gives us back the exact same Pipeline we saved during training
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully!")
    
    return _model


def predict(features: dict) -> dict:
    """
    Make a prediction for a single customer.
    
    Args:
        features: Dictionary with customer data, e.g.:
            {
                "checking_status": "<0",
                "duration": 12,
                "credit_amount": 1000,
                ...
            }
    
    Returns:
        Dictionary with prediction results:
            {
                "default_probability": 0.25,
                "risk_label": "low"
            }
    """
    # Load model (uses cached version if already loaded)
    model = load_model()
    
    # Convert the input dictionary to a DataFrame
    # WHY: sklearn models expect DataFrame input (same format as training)
    # [features] wraps the dict in a list to create a single-row DataFrame
    input_df = pd.DataFrame([features])
    
    # Get probability of default (class 1 = bad credit)
    # predict_proba() returns array like [[0.7, 0.3]] meaning:
    #   - 70% chance of class 0 (low risk)
    #   - 30% chance of class 1 (high risk / default)
    probabilities = model.predict_proba(input_df)
    # [0] = first (only) row, [1] = probability of class 1
    default_probability = float(probabilities[0][1])
    
    # Convert probability to a simple label
    # If probability > 0.5, we consider it high risk
    risk_label = "high" if default_probability > 0.5 else "low"
    
    logger.info(f"Prediction: probability={default_probability:.4f}, label={risk_label}")
    
    return {
        "default_probability": round(default_probability, 4),
        "risk_label": risk_label
    }
