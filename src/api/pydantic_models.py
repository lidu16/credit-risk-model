from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_class: int  # 0 = low risk, 1 = high risk