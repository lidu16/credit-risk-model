"""Pydantic models for API request and response validation."""
from typing import Any, Dict
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for the /predict endpoint."""

    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary matching the model's input columns",
        json_schema_extra={
            "example": {
                "Transaction_Count": 10,
                "Average_Amount": 150.5,
                "Recency": 30,
                "ChannelId": 1,
                "ProductCategory": "Electronics",
            }
        },
    )


class PredictionResponse(BaseModel):
    """Response model for the /predict endpoint."""

    probability: float = Field(
        ...,
        description="Predicted probability of being high-risk",
        ge=0.0,
        le=1.0,
    )
    risk_class: int = Field(
        ...,
        description="Binary class: 1 = high-risk, 0 = low-risk",
        ge=0,
        le=1,
    )
    message: str = Field(..., description="Status or informational message")