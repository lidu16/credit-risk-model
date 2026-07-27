import os
import sys
import joblib
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
import pandas as pd
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load the model
MODEL_PATH = "models/best_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and save the model first.")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Credit Risk Model API", description="Predicts probability of a customer being high-risk", version="1.0.0")

@app.get("/")
def root():
    return {"message": "Credit Risk Model API is running. Use /predict endpoint."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert input to numpy array
        input_data = np.array(request.features).reshape(1, -1)
        
        # Get probability of high-risk (class 1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0, 1]
        else:
            # For models without predict_proba, use decision_function or similar
            proba = model.predict(input_data)[0]
        
        # Get class prediction
        pred_class = int(proba >= 0.5) if isinstance(proba, float) else int(model.predict(input_data)[0])
        
        return PredictionResponse(risk_probability=float(proba), risk_class=pred_class)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from config import APIConfig
from pydantic_models import PredictionRequest, PredictionResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model reference
_model = None

# Create FastAPI app
app = FastAPI(
    title="Credit Risk Model API",
    description="Real-time credit risk scoring for buy-now-pay-later customers",
    version="1.0.0"
)


@app.on_event("startup")
async def load_model() -> None:
    """
    Load the best model from disk on application startup.
    """
    global _model
    model_path = Path(__file__).parent.parent.parent / "models" / "best_model.pkl"
    
    if model_path.exists():
        try:
            _model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _model = None
    else:
        logger.warning(f"Model file not found at {model_path}")
        _model = None


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify service is running.
    
    Returns:
        Dict with status and model availability.
    """
    return {
        "status": "healthy",
        "model_loaded": str(_model is not None)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict credit risk probability for a new customer.
    
    Args:
        request: PredictionRequest with customer features.
    
    Returns:
        PredictionResponse with risk probability, class, and status message.
    
    Raises:
        HTTPException: If model is unavailable or prediction fails.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert request features to DataFrame
        input_df = pd.DataFrame([request.features])
        
        # Ensure feature columns match training (optional: handle missing)
        # Here we assume the request includes all required columns.
        # In practice, you may want to validate and preprocess here.
        
        # Predict probability
        proba = _model.predict_proba(input_df)[0, 1]
        risk_class = int(proba >= 0.5)
        
        return PredictionResponse(
            probability=float(proba),
            risk_class=risk_class,
            message="Prediction successful"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint providing basic information.
    """
    return {
        "message": "Credit Risk Model API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    config = APIConfig()
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload
    )