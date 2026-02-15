# src/config.py

RANDOM_STATE = 42

RAW_DATA_PATH = "data/raw/data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

TARGET_COLUMN = "is_high_risk"

# Model training configuration
TEST_SIZE = 0.2
N_CLUSTERS = 3

# Model settings
MODEL_NAME = "credit_risk_model"

# Explainability settings
SHAP_SAMPLE_SIZE = 100
