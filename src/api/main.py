import os
import sys
import joblib
import numpy as np
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
