from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_class: int  # 0 = low risk, 1 = high risk
    """
Pydantic models for API request and response validation.
"""
from typing import Dict, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request model for the /predict endpoint.
    
    Attributes:
        features: Dictionary of feature names to values.
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary matching the model's input columns",
        example={
            "Transaction_Count": 10,
            "Average_Amount": 150.5,
            "Recency": 30,
            "ChannelId": 1,
            "ProductCategory": "Electronics"
        }
    )


class PredictionResponse(BaseModel):
    """
    Response model for the /predict endpoint.
    
    Attributes:
        probability: Predicted probability of being high-risk (0-1).
        risk_class: Binary class (1 = high-risk, 0 = low-risk).
        message: Status message.
    """
    probability: float = Field(
        ...,
        description="Predicted probability of being high-risk",
        ge=0.0,
        le=1.0
    )
    risk_class: int = Field(
        ...,
        description="Binary class: 1 = high-risk, 0 = low-risk",
        ge=0,
        le=1
    )
    message: str = Field(
        ...,
        description="Status or informational message"
    )