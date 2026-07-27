"""
SHAP explainability module for model interpretation.
Provides global feature importance and individual prediction explanations.
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wrapper class for SHAP explanations.
    Handles both global (feature importance) and local (individual) explanations.
    """
    
    def __init__(self, model, X_train: pd.DataFrame):
        """
        Initialize SHAP explainer with trained model and training data.
        
        Args:
            model: Trained sklearn-compatible model.
            X_train: Training data used to fit the explainer.
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        self._fit_explainer()
    
    def _fit_explainer(self) -> None:
        """
        Fit the SHAP explainer using TreeExplainer (for tree-based models).
        For non-tree models, fallback to KernelExplainer.
        """
        try:
            # Use TreeExplainer for XGBoost, RandomForest, DecisionTree
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.X_train)
            logger.info("Using TreeExplainer for SHAP explanations")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Falling back to KernelExplainer.")
            # Use KernelExplainer for other models (requires subset for speed)
            sample_size = min(100, len(self.X_train))
            X_sample = self.X_train.sample(n=sample_size, random_state=42)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
            self.shap_values = self.explainer.shap_values(self.X_train)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance (mean absolute SHAP values).
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if isinstance(self.shap_values, list):
            # For binary classification, use positive class SHAP values
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': np.abs(shap_vals).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def plot_summary(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Generate SHAP summary plot (bee swarm plot).
        
        Args:
            figsize: Figure dimensions.
        
        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.summary_plot(shap_vals, self.X_train, show=False, max_display=15)
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_bar(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Generate SHAP bar plot (mean absolute SHAP values).
        
        Args:
            figsize: Figure dimensions.
        
        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.summary_plot(shap_vals, self.X_train, plot_type="bar", show=False, max_display=15)
        plt.title("Top 15 Features by SHAP Importance", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_dependence(self, feature: str, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Generate SHAP dependence plot for a specific feature.
        
        Args:
            feature: Feature name to analyze.
            figsize: Figure dimensions.
        
        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.dependence_plot(feature, shap_vals, self.X_train, show=False)
        plt.title(f"SHAP Dependence Plot: {feature}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def explain_prediction(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Get SHAP explanation for a single prediction (force plot data).
        
        Args:
            X_sample: Single row of feature data.
        
        Returns:
            Dictionary with SHAP values, base value, and features.
        """
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Get SHAP values for the sample
        sample_shap = self.explainer.shap_values(X_sample)
        
        if isinstance(sample_shap, list):
            sample_shap = sample_shap[1] if len(sample_shap) > 1 else sample_shap[0]
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Predicted probability
        pred_proba = self.model.predict_proba(X_sample)[0, 1]
        
        # Create feature contributions
        contributions = []
        for i, feature in enumerate(self.X_train.columns):
            contributions.append({
                'feature': feature,
                'value': float(X_sample.iloc[0, i]),
                'shap_value': float(sample_shap[0, i])
            })
        
        return {
            'base_value': float(base_value),
            'predicted_probability': float(pred_proba),
            'contributions': sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        }
    
    def get_force_plot_html(self, X_sample: pd.DataFrame) -> str:
        """
        Generate HTML for SHAP force plot.
        
        Args:
            X_sample: Single row of feature data.
        
        Returns:
            HTML string for rendering force plot.
        """
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Get SHAP values for the sample
        sample_shap = self.explainer.shap_values(X_sample)
        
        if isinstance(sample_shap, list):
            sample_shap = sample_shap[1] if len(sample_shap) > 1 else sample_shap[0]
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Generate force plot
        force_plot = shap.force_plot(
            base_value,
            sample_shap,
            X_sample,
            matplotlib=False,
            show=False
        )
        return str(force_plot)