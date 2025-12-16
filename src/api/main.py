"""
main.py - FastAPI application for serving the credit risk model

This is the main API file that:
1. Loads the trained model when the server starts
2. Provides a /health endpoint to check if the service is running
3. Provides a /predict endpoint to get credit risk predictions
"""

import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Import our custom modules
from src.inference import load_model, predict
from src.api.schemas import CreditFeatures, PredictionResponse, HealthResponse

# ============================================================
# SETUP LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# APPLICATION STARTUP
# ============================================================
# Lifespan context manager handles startup and shutdown events
# WHY: We want to load the model ONCE when the server starts, not on every request
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs when the application starts and stops.
    
    We use it to load the model once at startup,
    so we don't have to load it for every request.
    """
    # STARTUP: Load the model before accepting requests
    logger.info("Starting up the API...")
    try:
        load_model()
        logger.info("Model loaded, ready to serve predictions!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield  # The application runs here
    
    # SHUTDOWN: Cleanup (nothing to do in our case)
    logger.info("Shutting down the API...")


# ============================================================
# CREATE THE FASTAPI APP
# ============================================================
# Create the FastAPI application instance
# These settings appear in the auto-generated docs at /docs
app = FastAPI(
    title="Credit Risk Prediction API",
    description="A simple API that predicts credit default risk using a trained ML model.",
    version="1.0.0",
    lifespan=lifespan  # Connect our startup/shutdown handler
)


# ============================================================
# ENDPOINTS
# ============================================================

# @app.get() decorator turns this function into a GET endpoint
# response_model tells FastAPI what shape the response should have
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    
    Use this to verify the service is running and the model is loaded.
    Returns: {"status": "ok", "model_loaded": true}
    """
    return HealthResponse(
        status="ok",
        model_loaded=True
    )


# @app.post() = this endpoint accepts POST requests (sending data)
# features: CreditFeatures = FastAPI automatically:
#   1. Parses JSON from request body
#   2. Validates it matches CreditFeatures schema
#   3. Returns 422 error if validation fails
@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(features: CreditFeatures):
    """
    Predict credit risk for a customer.
    
    Send customer data and receive:
    - default_probability: Chance of default (0 to 1)
    - risk_label: "low" or "high"
    
    Example request body:
    {
        "checking_status": "<0",
        "duration": 12,
        "credit_history": "existing paid",
        "purpose": "radio/tv",
        "credit_amount": 1000,
        ...
    }
    """
    try:
        # Convert Pydantic model to dictionary for our predict function
        # model_dump() turns CreditFeatures object into a plain Python dict
        features_dict = features.model_dump()
        
        # Call our inference function (this runs the ML model)
        result = predict(features_dict)
        
        # Return the response
        return PredictionResponse(
            default_probability=result["default_probability"],
            risk_label=result["risk_label"]
        )
    
    except Exception as e:
        # If something goes wrong, return a 500 error with details
        # HTTPException is FastAPI's way to return error responses
        # status_code=500 means "Internal Server Error"
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# RUN THE APP (for development)
# ============================================================
# You can run this with: python -m src.api.main
# Or with: uvicorn src.api.main:app --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
