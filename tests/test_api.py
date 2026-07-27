import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import():
    from src.api.main import app
    assert app.title == "Credit Risk Model API"
    """
Unit tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_root_endpoint(client):
    """Test root endpoint returns expected structure."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "health" in data


def test_predict_missing_features(client):
    """Test prediction with missing features."""
    request_data = {}
    response = client.post("/predict", json=request_data)
    # Should return 422 for validation error
    assert response.status_code == 422


def test_predict_valid_request_structure(client):
    """Test prediction with valid structure but without model loaded."""
    request_data = {
        "features": {
            "Transaction_Count": 10,
            "Average_Amount": 150.5,
            "Recency": 30,
            "ChannelId": 1
        }
    }
    response = client.post("/predict", json=request_data)
    
    # Either 200 or 503 depending on model availability
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "probability" in data
        assert "risk_class" in data
        assert "message" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_class"] in [0, 1]


def test_predict_invalid_features_type(client):
    """Test prediction with invalid feature types."""
    request_data = {
        "features": {
            "Transaction_Count": "invalid",  # Should be numeric
            "Average_Amount": 150.5,
            "Recency": 30
        }
    }
    response = client.post("/predict", json=request_data)
    assert response.status_code in [422, 400, 503]