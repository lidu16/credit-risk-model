"""
Configuration constants and dataclasses for the credit risk model.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PathConfig:
    """File paths configuration."""
    raw_data_path: str = "data/raw_transactions.csv"
    processed_data_path: str = "data/processed_data.csv"
    model_save_path: str = "models/best_model.pkl"
    model_registry_path: str = "models/mlflow"


@dataclass
class RFMConfig:
    """RFM clustering configuration."""
    n_clusters: int = 3
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model training configuration."""
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


# Named constants (replace magic numbers)
DEFAULT_RANDOM_STATE = 42
TARGET_COLUMN = "is_high_risk"
TEST_SIZE = 0.2
MAX_ITER = 1000
N_JOBS = -1  # Use all CPU cores