"""
FastAPI application for the credit risk model with SHAP explainability.
"""
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from .pydantic_models import PredictionRequest, PredictionResponse
except ImportError:
    from pydantic_models import PredictionRequest, PredictionResponse

try:
    from ..config import APIConfig
except ImportError:
    from config import APIConfig

try:
    from ..shap_explainer import SHAPExplainer
except ImportError:
    from shap_explainer import SHAPExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Model API",
    description="Real-time credit risk scoring for buy-now-pay-later customers",
    version="1.0.0",
)

_model: Any = None
_shap_explainer: Any = None
_X_train: Any = None


def _get_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "best_model.pkl"


def _get_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed_data.csv"


@app.on_event("startup")
async def load_model_and_shap() -> None:
    """Load model and initialize SHAP explainer on startup."""
    global _model, _shap_explainer, _X_train

    # Load model
    model_path = _get_model_path()
    if model_path.exists():
        try:
            _model = joblib.load(model_path)
            logger.info("✅ Model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            _model = None
    else:
        logger.warning("Model file not found at %s", model_path)
        _model = None

    # Load SHAP explainer if model is loaded
    if _model is not None:
        try:
            data_path = _get_data_path()
            if data_path.exists():
                df = pd.read_csv(data_path)
                # Check if target column exists
                if 'is_high_risk' in df.columns:
                    X_train = df.drop(columns=['is_high_risk'])
                else:
                    # If no target column, use all columns
                    X_train = df
                _X_train = X_train.sample(min(100, len(X_train)), random_state=42)
                _shap_explainer = SHAPExplainer(_model, _X_train)
                logger.info("✅ SHAP explainer initialized successfully")
            else:
                logger.warning("❌ Training data not found for SHAP at %s", data_path)
                _shap_explainer = None
        except Exception as e:
            logger.warning("❌ Failed to initialize SHAP explainer: %s", e)
            _shap_explainer = None


# -------------------- Health & Root --------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify service is running."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "shap_available": _shap_explainer is not None
    }


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint providing basic information."""
    return {
        "message": "Credit Risk Model API",
        "docs": "/docs",
        "health": "/health"
    }


# -------------------- Prediction --------------------
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
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        )


# -------------------- SHAP Endpoints --------------------
@app.get("/shap/feature-importance")
async def get_shap_importance() -> Dict[str, Any]:
    """Get global feature importance from SHAP."""
    if _shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available. Check if model and data are loaded."
        )

    try:
        importance = _shap_explainer.get_feature_importance()
        return {
            "features": importance['feature'].tolist(),
            "importance": importance['importance'].tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get SHAP importance: {str(e)}"
        )


@app.post("/shap/explain")
async def explain_prediction(request: PredictionRequest) -> Dict[str, Any]:
    """Get SHAP explanation for a specific prediction."""
    if _shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available. Check if model and data are loaded."
        )

    try:
        input_df = pd.DataFrame([request.features])
        # Ensure columns match training data
        if _X_train is not None:
            input_df = input_df[_X_train.columns]
        explanation = _shap_explainer.explain_prediction(input_df)
        return explanation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SHAP explanation failed: {str(e)}"
        )


@app.get("/shap/dependence/{feature}")
async def get_shap_dependence(feature: str) -> Dict[str, Any]:
    """Get SHAP dependence plot data for a specific feature."""
    if _shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available. Check if model and data are loaded."
        )

    if _X_train is None or feature not in _X_train.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature '{feature}' not found in training data. Available: {list(_X_train.columns) if _X_train is not None else []}"
        )

    try:
        shap_vals = _shap_explainer._extract_positive_class_shap(_shap_explainer.shap_values)
        feature_idx = _X_train.columns.get_loc(feature)
        feature_values = _X_train[feature].tolist()
        shap_values_feature = shap_vals[:, feature_idx].tolist()
        return {
            "feature": feature,
            "feature_values": feature_values,
            "shap_values": shap_values_feature
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get SHAP dependence data: {str(e)}"
        )


# -------------------- Dashboard --------------------
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
        "shap_available": _shap_explainer is not None,
        "sample_probability": 0.42 if _model is not None else 0.0,
        "sample_risk_class": 1 if _model is not None else 0,
    })


if __name__ == "__main__":
    config = APIConfig()
    uvicorn.run("main:app", host=config.host, port=config.port, reload=config.reload)