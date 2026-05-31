"""
Prediction utility for credit risk model
"""

import joblib
import numpy as np
import pandas as pd
from src.data_processing import process_data


def load_model(model_path='models/best_model.pkl'):
    """Load trained model from disk"""
    return joblib.load(model_path)


def predict_risk(model, features):
    """
    Predict risk probability for customer features
    
    Parameters:
    -----------
    model : estimator
        Trained sklearn model
    features : array-like
        Customer features (must be preprocessed)
    
    Returns:
    --------
    risk_probability : float
        Probability of being high-risk (0-1)
    risk_class : int
        Binary classification (0=low-risk, 1=high-risk)
    """
    # Get probability for positive class
    proba = model.predict_proba(features)[0, 1]
    
    # Get class prediction
    pred_class = int(proba >= 0.5)
    
    return {
        'risk_probability': float(proba),
        'risk_class': pred_class
    }


def batch_predict(model, features_df):
    """
    Predict risk for batch of customers
    
    Parameters:
    -----------
    model : estimator
        Trained sklearn model
    features_df : DataFrame
        Customer features dataframe
    
    Returns:
    --------
    predictions : DataFrame
        Predictions with risk_probability and risk_class
    """
    probas = model.predict_proba(features_df)[:, 1]
    classes = (probas >= 0.5).astype(int)
    
    return pd.DataFrame({
        'risk_probability': probas,
        'risk_class': classes
    })


if __name__ == "__main__":
    # Example usage
    model = load_model()
    print("Model loaded successfully!")
