"""
Utility functions for logging, file I/O, and common helpers.
"""
import logging
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import joblib


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists; create if it doesn't.
    
    Args:
        path: Directory path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save a dictionary as JSON file.
    
    Args:
        data: Dictionary to save.
        filepath: Output path.
    """
    ensure_directory(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file into dictionary.
    
    Args:
        filepath: Path to JSON file.
    
    Returns:
        Dict with contents.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to disk using joblib.
    
    Args:
        model: Trained model object.
        filepath: Output path.
    """
    ensure_directory(Path(filepath).parent)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load model from disk.
    
    Args:
        filepath: Path to model file.
    
    Returns:
        Loaded model.
    """
    return joblib.load(filepath)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save.
        filepath: Output path.
    """
    ensure_directory(Path(filepath).parent)
    df.to_csv(filepath, index=False)


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load CSV into DataFrame.
    
    Args:
        filepath: Path to CSV file.
    
    Returns:
        DataFrame.
    """
    return pd.read_csv(filepath)