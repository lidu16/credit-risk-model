"""
Model Training Pipeline with MLflow Experiment Tracking
Trains, tunes, and evaluates classification models for credit risk prediction
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn

from src.data_processing import process_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_evaluate_model(model, params, X_train, X_test, y_train, y_test, 
                             search_type='grid', cv=5, n_iter=10):
    """
    Train model with hyperparameter tuning and return best model with metrics
    
    Parameters:
    -----------
    model : estimator
        Sklearn model to train
    params : dict
        Hyperparameter grid
    X_train, X_test : arrays
        Training and test features
    y_train, y_test : arrays
        Training and test targets
    search_type : str
        'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    cv : int
        Cross-validation folds
    n_iter : int
        Number of iterations for RandomSearch
    
    Returns:
    --------
    best_model, best_params, metrics
    """
    logger.info(f"Training with {search_type} search...")
    
    if search_type == 'grid':
        search = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            model, params, n_iter=n_iter, cv=cv, 
            scoring='roc_auc', random_state=42, n_jobs=-1
        )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
        'train_roc_auc': roc_auc_score(y_train, y_train_proba),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba),
        'best_cv_score': search.best_score_
    }
    
    logger.info(f"Best CV Score: {search.best_score_:.4f}")
    logger.info(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    
    return best_model, search.best_params_, metrics


def log_model_to_mlflow(model, params, metrics, model_name, X_test, y_test):
    """Log model, parameters, and metrics to MLflow"""
    
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_model")
    
    # Log additional artifacts
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Logged {model_name} to MLflow")


def main(raw_data_path='data/data.csv', test_size=0.2, random_state=42):
    """
    Complete training pipeline
    """
    logger.info("=" * 50)
    logger.info("Starting Credit Risk Model Training Pipeline")
    logger.info("=" * 50)
    
    # Set MLflow experiment
    mlflow.set_experiment("Credit_Risk_Classification")
    
    # Process data
    logger.info("Processing data...")
    X_processed, y, preprocessor, df_model = process_data(raw_data_path)
    
    # Train-test split
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    logger.info(f"Target distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # Store metrics for comparison
    all_models = {}
    
    # ==================== LOGISTIC REGRESSION ====================
    logger.info("\n" + "=" * 50)
    logger.info("Training Logistic Regression")
    logger.info("=" * 50)
    
    with mlflow.start_run(run_name="LogisticRegression_GridSearch"):
        lr_params = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [200]
        }
        
        lr = LogisticRegression(random_state=random_state)
        best_lr, best_lr_params, lr_metrics = train_and_evaluate_model(
            lr, lr_params, X_train, X_test, y_train, y_test, search_type='grid'
        )
        
        log_model_to_mlflow(best_lr, best_lr_params, lr_metrics, "Logistic Regression", X_test, y_test)
        all_models['LogisticRegression'] = {
            'model': best_lr,
            'metrics': lr_metrics,
            'params': best_lr_params
        }
    
    # ==================== RANDOM FOREST ====================
    logger.info("\n" + "=" * 50)
    logger.info("Training Random Forest")
    logger.info("=" * 50)
    
    with mlflow.start_run(run_name="RandomForest_RandomSearch"):
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        best_rf, best_rf_params, rf_metrics = train_and_evaluate_model(
            rf, rf_params, X_train, X_test, y_train, y_test, 
            search_type='random', n_iter=10
        )
        
        log_model_to_mlflow(best_rf, best_rf_params, rf_metrics, "Random Forest", X_test, y_test)
        all_models['RandomForest'] = {
            'model': best_rf,
            'metrics': rf_metrics,
            'params': best_rf_params
        }
    
    # ==================== GRADIENT BOOSTING ====================
    logger.info("\n" + "=" * 50)
    logger.info("Training Gradient Boosting")
    logger.info("=" * 50)
    
    with mlflow.start_run(run_name="GradientBoosting_RandomSearch"):
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        gb = GradientBoostingClassifier(random_state=random_state)
        best_gb, best_gb_params, gb_metrics = train_and_evaluate_model(
            gb, gb_params, X_train, X_test, y_train, y_test,
            search_type='random', n_iter=10
        )
        
        log_model_to_mlflow(best_gb, best_gb_params, gb_metrics, "Gradient Boosting", X_test, y_test)
        all_models['GradientBoosting'] = {
            'model': best_gb,
            'metrics': gb_metrics,
            'params': best_gb_params
        }
    
    # ==================== MODEL SELECTION ====================
    logger.info("\n" + "=" * 50)
    logger.info("Model Comparison and Selection")
    logger.info("=" * 50)
    
    # Compare models
    comparison = pd.DataFrame({
        'Model': list(all_models.keys()),
        'Test_Accuracy': [all_models[m]['metrics']['test_accuracy'] for m in all_models],
        'Test_Precision': [all_models[m]['metrics']['test_precision'] for m in all_models],
        'Test_Recall': [all_models[m]['metrics']['test_recall'] for m in all_models],
        'Test_F1': [all_models[m]['metrics']['test_f1'] for m in all_models],
        'Test_ROC_AUC': [all_models[m]['metrics']['test_roc_auc'] for m in all_models]
    })
    
    logger.info("\nModel Comparison:")
    logger.info(comparison.to_string(index=False))
    
    # Select best model
    best_model_name = comparison.loc[comparison['Test_ROC_AUC'].idxmax(), 'Model']
    best_model_obj = all_models[best_model_name]['model']
    best_model_metrics = all_models[best_model_name]['metrics']
    
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"Test ROC-AUC: {best_model_metrics['test_roc_auc']:.4f}")
    logger.info(f"Test F1 Score: {best_model_metrics['test_f1']:.4f}")
    
    # Save best model
    joblib.dump(best_model_obj, 'models/best_model.pkl')
    logger.info("Best model saved to models/best_model.pkl")
    
    # Register in MLflow
    try:
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
        model_details = mlflow.register_model(model_uri, "CreditRiskModel")
        logger.info(f"Model registered: CreditRiskModel v{model_details.version}")
    except Exception as e:
        logger.warning(f"Could not register model in MLflow: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 50)
    
    return best_model_obj, preprocessor, comparison


if __name__ == "__main__":
    best_model, preprocessor, comparison = main()
