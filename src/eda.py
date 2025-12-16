"""
eda.py - Simple EDA for the synthetic credit dataset

This script:
1. Reuses the load_data() function from train.py
2. Shows basic information about the dataset
3. Gives you a feeling for the features and target

Run it with:
    python -m src.eda
"""

import logging

import pandas as pd

from src.train import load_data


# Set up logging so we can see what is happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run a very simple EDA on the synthetic credit dataset."""
    # 1. Load the same data that training uses
    X, y = load_data()

    logger.info("=== BASIC SHAPE ===")
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target length: {len(y)}")

    # 2. Show first few rows
    logger.info("=== FIRST 5 ROWS ===")
    print(X.head())

    # 3. Data types
    logger.info("=== DATA TYPES ===")
    print(X.dtypes)

    # 4. Target distribution
    logger.info("=== TARGET DISTRIBUTION (0 = good, 1 = bad) ===")
    print(y.value_counts())
    print("Proportions:")
    print(y.value_counts(normalize=True))

    # 5. Summary stats for numeric columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    logger.info("=== NUMERIC SUMMARY STATISTICS ===")
    print(X[numeric_cols].describe())

    # 6. Value counts for a few important categorical columns
    cat_cols = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
    ]

    for col in cat_cols:
        if col in X.columns:
            logger.info(f"=== VALUE COUNTS FOR {col} ===")
            print(X[col].value_counts())


if __name__ == "__main__":
    main()
