"""
test_api.py - Simple tests for the API

These tests check that:
1. The health endpoint works
2. The predict endpoint accepts valid data and returns predictions

To run these tests:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# ============================================================
# CREATE TEST CLIENT
# ============================================================
# TestClient lets us send requests to our API without starting a server
client = TestClient(app)


# ============================================================
# SAMPLE TEST DATA
# ============================================================
# This is a valid example of customer data for testing
SAMPLE_CUSTOMER = {
    "checking_status": "low",
    "duration": 12,
    "credit_history": "ok",
    "purpose": "car",
    "credit_amount": 1000,
    "savings_status": "low",
    "employment": "medium",
    "installment_commitment": 4,
    "personal_status": "single",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "car",
    "age": 30,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes"
}


# ============================================================
# TESTS
# ============================================================

def test_health_endpoint():
    """
    Test that the health endpoint returns OK.
    
    This is a simple test to verify the API is running.
    """
    response = client.get("/health")
    
    # Check that we get a 200 OK response
    assert response.status_code == 200
    
    # Check the response content
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] == True


def test_predict_endpoint():
    """
    Test that the predict endpoint works with valid data.
    
    We send sample customer data and check that we get
    a valid prediction back.
    """
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    
    # Check that we get a 200 OK response
    assert response.status_code == 200
    
    # Check the response structure
    data = response.json()
    assert "default_probability" in data
    assert "risk_label" in data
    
    # Check that probability is between 0 and 1
    assert 0 <= data["default_probability"] <= 1
    
    # Check that risk_label is valid
    assert data["risk_label"] in ["low", "high"]


def test_predict_returns_probability():
    """
    Test that the model returns reasonable probabilities.
    
    We just verify that the probability is a number, not that
    it's a specific value (since that depends on the model).
    """
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    
    data = response.json()
    probability = data["default_probability"]
    
    # Probability should be a float between 0 and 1
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0
