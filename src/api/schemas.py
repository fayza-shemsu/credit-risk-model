"""
schemas.py - Pydantic models for request and response validation

Pydantic models define the structure of data that the API accepts and returns.
This gives us:
1. Automatic validation (wrong data types get rejected)
2. Automatic documentation (Swagger UI shows these schemas)
3. Type hints for our code
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


# BaseModel is from Pydantic - it gives us automatic validation
# When FastAPI receives JSON, it tries to create a CreditFeatures object
# If the JSON is missing required fields or has wrong types, it returns 422 error
class CreditFeatures(BaseModel):
    """
    Input data for credit risk prediction.
    
    These are the features from the German Credit dataset.
    Each field has a description and example value.
    """
    
    # Pydantic configuration (new style)
    # extra="ignore" means: if JSON has extra fields we don't define, just ignore them
    model_config = ConfigDict(extra="ignore")
    
    # Account status
    # Field() lets us add metadata: description shows in /docs, examples help users
    checking_status: str = Field(
        ...,  # ... (Ellipsis) means this field is REQUIRED - no default value
        description="Status of checking account (balance level)",
        examples=["low", "medium", "high", "none"]  # Shown in Swagger UI
    )
    
    # Loan details
    duration: int = Field(
        ...,
        description="Duration of the loan in months",
        examples=[12, 24, 36]
    )
    
    credit_history: str = Field(
        ...,
        description="Overall credit history quality",
        examples=["bad", "ok", "good"]
    )
    
    purpose: str = Field(
        ...,
        description="Purpose of the loan",
        examples=["car", "electronics", "furniture", "education", "other"]
    )
    
    credit_amount: float = Field(
        ...,
        description="Amount of the loan",
        examples=[1000, 5000, 10000]
    )
    
    savings_status: str = Field(
        ...,
        description="Savings level",
        examples=["low", "medium", "high", "none"]
    )
    
    employment: str = Field(
        ...,
        description="Employment duration category",
        examples=["short", "medium", "long", "unemployed"]
    )
    
    installment_commitment: int = Field(
        ...,
        description="Installment rate as percentage of income",
        examples=[1, 2, 3, 4]
    )
    
    personal_status: str = Field(
        ...,
        description="Family / personal status",
        examples=["single", "married", "other"]
    )
    
    other_parties: str = Field(
        ...,
        description="Other debtors or guarantors",
        examples=["none", "co_applicant", "guarantor"]
    )
    
    residence_since: int = Field(
        ...,
        description="Years at current residence",
        examples=[1, 2, 3, 4]
    )
    
    property_magnitude: str = Field(
        ...,
        description="Main property owned",
        examples=["car", "real_estate", "other"]
    )
    
    age: int = Field(
        ...,
        description="Age in years",
        examples=[25, 35, 45]
    )
    
    other_payment_plans: str = Field(
        ...,
        description="Other payment plans",
        examples=["none", "bank", "store"]
    )
    
    housing: str = Field(
        ...,
        description="Housing situation",
        examples=["rent", "own", "free"]
    )
    
    existing_credits: int = Field(
        ...,
        description="Number of existing credits at this bank",
        examples=[1, 2, 3]
    )
    
    job: str = Field(
        ...,
        description="Job type",
        examples=["unskilled", "skilled", "management"]
    )
    
    num_dependents: int = Field(
        ...,
        description="Number of dependents",
        examples=[1, 2]
    )
    
    own_telephone: str = Field(
        ...,
        description="Has telephone registered",
        examples=["yes", "no"]
    )
    
    foreign_worker: str = Field(
        ...,
        description="Is foreign worker",
        examples=["yes", "no"]
    )



class PredictionResponse(BaseModel):
    """
    Response from the prediction endpoint.
    
    Contains the model's prediction results.
    """
    
    default_probability: float = Field(
        ...,
        description="Probability that the customer will default (0 to 1)",
        examples=[0.25, 0.75]
    )
    
    risk_label: str = Field(
        ...,
        description="Risk category: 'low' or 'high'",
        examples=["low", "high"]
    )


class HealthResponse(BaseModel):
    """
    Response from the health check endpoint.
    """
    
    status: str = Field(
        ...,
        description="Service status",
        examples=["ok"]
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready",
        examples=[True]
    )
