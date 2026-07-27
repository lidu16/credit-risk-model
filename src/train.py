"""
Model training and evaluation module with MLflow tracking.
Trains multiple models, compares performance, and registers the best one.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from typing import Dict, Any, Tuple, Optional
import logging
import joblib
from dataclasses import asdict

try:
    from .config import DEFAULT_RANDOM_STATE, TARGET_COLUMN, ModelConfig
except ImportError:  # pragma: no cover - fallback when run as a script
    from config import DEFAULT_RANDOM_STATE, TARGET_COLUMN, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    config: Optional[ModelConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.
    
    Args:
        df: Full DataFrame with features and target.
        target_col: Name of the target column.
        config: ModelConfig with test_size and random_state.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if config is None:
        config = ModelConfig()
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Splitting data: {len(df)} rows, {len(X.columns)} features")
    
    return train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )


def get_models() -> Dict[str, Any]:
    """
    Define a dictionary of models to train and compare.
    
    Returns:
        Dict mapping model names to sklearn-compatible estimators.
    """
    return {
        'LogisticRegression': LogisticRegression(
            random_state=DEFAULT_RANDOM_STATE,
            max_iter=1000
        ),
        'DecisionTree': DecisionTreeClassifier(
            random_state=DEFAULT_RANDOM_STATE,
            max_depth=6
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=DEFAULT_RANDOM_STATE,
            eval_metric='logloss'
        )
    }


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str
) -> Any:
    """
    Train a single model and log with MLflow.
    
    Args:
        model: Unfitted sklearn-compatible estimator.
        X_train: Training features.
        y_train: Training labels.
        model_name: Name identifier for the model.
    
    Returns:
        Fitted model.
    """
    logger.info(f"Training {model_name}...")
    
    with mlflow.start_run(run_name=model_name, nested=True):
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train model
        model.fit(X_train, y_train)
        
        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test labels.
    
    Returns:
        Dict with metrics: accuracy, f1, roc_auc, precision, recall.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    # Log metrics with MLflow
    mlflow.log_metrics(metrics)
    
    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> Dict[str, Tuple[float, float]]:
    """
    Perform cross-validation and return mean and std of metrics.
    
    Args:
        model: Unfitted sklearn-compatible estimator.
        X: Feature matrix.
        y: Target vector.
        cv: Number of folds.
    
    Returns:
        Dict with metric names mapping to (mean, std) tuples.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    
    metrics = {}
    
    for metric_name, scorer in [
        ('accuracy', 'accuracy'),
        ('f1', 'f1_weighted'),
        ('roc_auc', 'roc_auc')
    ]:
        scores = cross_val_score(model, X, y, cv=skf, scoring=scorer)
        metrics[metric_name] = (scores.mean(), scores.std())
    
    return metrics


def train_and_compare_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Train all models, evaluate, and return comparison results.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
    
    Returns:
        Dict mapping model names to their evaluation metrics.
    """
    models = get_models()
    results = {}
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        logger.info(f"Training and evaluating {name}...")
        
        # Train
        fitted_model = train_model(model, X_train, y_train, name)
        
        # Evaluate
        metrics = evaluate_model(fitted_model, X_test, y_test)
        results[name] = metrics
        
        # Track best model
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = fitted_model
    
    # Log best model info
    logger.info(f"Best model: {best_model.__class__.__name__} with F1: {best_f1:.4f}")
    
    return results


def save_best_model(model: Any, filepath: str) -> None:
    """
    Save the best model to disk.
    
    Args:
        model: Trained classifier.
        filepath: Path to save the model.
    """
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def register_best_model_with_mlflow(model: Any, model_name: str, metrics: Dict[str, float]) -> None:
    """
    Register the best model in MLflow Model Registry.
    
    Args:
        model: Trained classifier.
        model_name: Name for the model in registry.
        metrics: Evaluation metrics to log.
    """
    with mlflow.start_run(run_name=f"{model_name}_registration") as run:
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Register model
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        mlflow.register_model(model_uri, model_name)
        
        logger.info(f"Model {model_name} registered in MLflow Model Registry")