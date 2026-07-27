"""FastAPI application for the credit risk model."""
import logging
from pathlib import Path
from typing import Any, Dict
from src.shap_explainer import SHAPExplainer
from src.data_processing import load_and_clean_data

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
_shap_explainer = None
_X_train = None 
try:
    from .pydantic_models import PredictionRequest, PredictionResponse
except ImportError:  # pragma: no cover - fallback when run as a script
    from pydantic_models import PredictionRequest, PredictionResponse

try:
    from ..config import APIConfig
except ImportError:  # pragma: no cover - fallback when run as a script
    from config import APIConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Model API",
    description="Real-time credit risk scoring for buy-now-pay-later customers",
    version="1.0.0",
)

_model: Any = None


def _get_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "best_model.pkl"


@app.on_event("startup")
async def load_model() -> None:
    """Load the best model from disk on application startup if it exists."""
    global _model
    model_path = _get_model_path()

    if model_path.exists():
        try:
            _model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Failed to load model: %s", exc)
            _model = None
    else:
        logger.warning("Model file not found at %s", model_path)
        _model = None


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify service is running."""
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict credit risk probability for a new customer."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later.",
        )

    try:
        input_df = pd.DataFrame([request.features])
        proba = float(_model.predict_proba(input_df)[0, 1])
        risk_class = int(proba >= 0.5)
        return PredictionResponse(
            probability=proba,
            risk_class=risk_class,
            message="Prediction successful",
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve a lightweight interactive dashboard page."""
    return HTMLResponse(content="""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Credit Risk Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
        <style>
          body { font-family: Arial, sans-serif; margin: 0; padding: 2rem; background: #f4f7fb; color: #14213d; }
          .card { background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 8px 25px rgba(0,0,0,0.08); margin-bottom: 1rem; }
          form { display: grid; gap: 0.75rem; max-width: 320px; }
          input, select, button { padding: 0.6rem; border-radius: 8px; border: 1px solid #cfd8e3; }
          button { background: #2563eb; color: white; cursor: pointer; }
          .result { font-weight: 600; margin-top: 0.5rem; }
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Interactive credit risk dashboard</h2>
          <p>Use the form below to score a customer and inspect the live risk state.</p>
        </div>
        <div class="card">
          <form id="score-form">
            <input name="Transaction_Count" type="number" value="10" placeholder="Transaction Count" />
            <input name="Average_Amount" type="number" step="0.01" value="150.5" placeholder="Average Amount" />
            <input name="Recency" type="number" value="30" placeholder="Recency" />
            <input name="ChannelId" type="number" value="1" placeholder="Channel ID" />
            <select name="ProductCategory">
              <option value="Electronics">Electronics</option>
              <option value="Retail">Retail</option>
              <option value="Travel">Travel</option>
            </select>
            <button type="submit">Score customer</button>
          </form>
          <div class="result" id="score-result">Waiting for input...</div>
        </div>
        <div class="card">
          <canvas id="risk-chart" height="120"></canvas>
        </div>
        <script>
          const form = document.getElementById('score-form');
          const result = document.getElementById('score-result');
          const ctx = document.getElementById('risk-chart');
          const chart = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: ['Model status', 'Risk score', 'High risk'],
              datasets: [{ label: 'Dashboard snapshot', data: [1, 0.4, 0.2], backgroundColor: ['#60a5fa', '#f59e0b', '#ef4444'] }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
          });

          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const payload = Object.fromEntries(new FormData(form).entries());
            payload.Transaction_Count = Number(payload.Transaction_Count);
            payload.Average_Amount = Number(payload.Average_Amount);
            payload.Recency = Number(payload.Recency);
            payload.ChannelId = Number(payload.ChannelId);
            const response = await fetch('/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ features: payload })
            });
            const data = await response.json();
            result.textContent = response.ok ? `Probability: ${data.probability.toFixed(3)} | Risk class: ${data.risk_class}` : `Error: ${data.detail}`;
            chart.data.datasets[0].data = [response.ok ? 1 : 0, response.ok ? data.probability : 0, response.ok ? data.risk_class : 0];
            chart.update();
          });
        </script>
      </body>
    </html>
    """)


@app.get("/dashboard-data")
async def dashboard_data() -> JSONResponse:
    """Return simple metrics for the dashboard chart."""
    return JSONResponse({
        "model_loaded": _model is not None,
        "sample_probability": 0.42 if _model is not None else 0.0,
        "sample_risk_class": 1 if _model is not None else 0,
    })


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint providing basic information."""
    return {"message": "Credit Risk Model API", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    config = APIConfig()
    uvicorn.run("main:app", host=config.host, port=config.port, reload=config.reload)
@app.on_event("startup")
async def load_model_and_shap() -> None:
    """Load model and initialize SHAP explainer on startup."""
    global _model, _shap_explainer, _X_train
    
    # ... existing model loading code ...
    
    # If model is loaded, initialize SHAP explainer
    if _model is not None:
        try:
            # Load training data (or a sample for SHAP)
            # For production, you'd store this from training
            import pandas as pd
            df = pd.read_csv("data/processed_data.csv")
            X_train = df.drop(columns=['is_high_risk'])
            _X_train = X_train.sample(min(100, len(X_train)), random_state=42)
            
            # Create SHAP explainer
            _shap_explainer = SHAPExplainer(_model, _X_train)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            _shap_explainer = None


@app.get("/shap/feature-importance")
async def get_shap_importance() -> Dict[str, Any]:
    """
    Get global feature importance from SHAP.
    
    Returns:
        Dictionary with feature names and importance scores.
    """
    if _shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available"
        )
    
    importance = _shap_explainer.get_feature_importance()
    return {
        "features": importance['feature'].tolist(),
        "importance": importance['importance'].tolist()
    }


@app.post("/shap/explain")
async def explain_prediction(request: PredictionRequest) -> Dict[str, Any]:
    """
    Get SHAP explanation for a specific prediction.
    
    Args:
        request: PredictionRequest with features.
    
    Returns:
        Dictionary with SHAP values, base value, and feature contributions.
    """
    if _shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available"
        )
    
    try:
        input_df = pd.DataFrame([request.features])
        # Ensure columns match training data
        input_df = input_df[_X_train.columns]
        explanation = _shap_explainer.explain_prediction(input_df)
        return explanation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SHAP explanation failed: {str(e)}"
        )