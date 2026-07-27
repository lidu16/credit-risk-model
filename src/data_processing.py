"""
Data processing module for credit risk model.
Handles feature engineering, RFM clustering, and preprocessing.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import DEFAULT_RANDOM_STATE, TARGET_COLUMN, RFMConfig
except ImportError:  # pragma: no cover - fallback when run as a script
    from config import DEFAULT_RANDOM_STATE, TARGET_COLUMN, RFMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    numerical_cols: List[str]
    categorical_cols: List[str]
    target_col: str = TARGET_COLUMN


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw transaction data and perform initial cleaning.
    
    Args:
        filepath: Path to the CSV file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    
    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Convert timestamp to datetime
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Drop rows with missing critical columns
    required_cols = ['CustomerId', 'Amount']
    df = df.dropna(subset=required_cols)
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def calculate_rfm_features(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary features per customer.
    
    Args:
        df: Raw transaction DataFrame.
        snapshot_date: Reference date for recency calculation.
                       Defaults to max date in data.
    
    Returns:
        pd.DataFrame: RFM features with CustomerId as index.
    """
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max()
    
    # Group by customer
    rfm = (
        df.groupby('CustomerId')
        .agg(
            Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        )
        .reset_index()
        .set_index('CustomerId')
    )
    
    # Handle potential missing values
    rfm = rfm.fillna(0)
    return rfm


def cluster_customers(rfm_df: pd.DataFrame, config: RFMConfig) -> pd.Series:
    """
    Cluster customers using K-Means on RFM features.
    
    Args:
        rfm_df: DataFrame with Recency, Frequency, Monetary.
        config: RFMConfig with number of clusters and random_state.
    
    Returns:
        pd.Series: Cluster labels per customer.
    """
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    
    # Apply K-Means
    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=10
    )
    clusters = kmeans.fit_predict(rfm_scaled)
    return pd.Series(clusters, index=rfm_df.index, name='cluster')


def identify_high_risk_cluster(rfm_df: pd.DataFrame, clusters: pd.Series) -> int:
    """Identify the cluster with highest risk (lowest engagement)."""
    cluster_stats = rfm_df.copy()
    cluster_stats['cluster'] = pd.Series(clusters.to_numpy(), index=rfm_df.index)
    cluster_stats = cluster_stats.groupby('cluster').mean()

    cluster_stats['risk_score'] = (
        -cluster_stats['Frequency']
        - cluster_stats['Monetary']
        + cluster_stats['Recency']
    )
    return int(cluster_stats['risk_score'].idxmax())


def create_target_variable(df: pd.DataFrame, rfm_clusters: pd.Series, risk_cluster: int) -> pd.DataFrame:
    """Add a binary target column aligned to the input rows."""
    result = df.copy()

    if 'CustomerId' in result.columns and hasattr(rfm_clusters, 'index'):
        cluster_map = pd.Series(rfm_clusters.to_numpy(), index=rfm_clusters.index)
        result['is_high_risk'] = (
            result['CustomerId'].map(cluster_map).astype(int) == risk_cluster
        ).astype(int)
    else:
        cluster_series = pd.Series(rfm_clusters.reset_index(drop=True).to_numpy(), index=result.index)
        result['is_high_risk'] = (cluster_series == risk_cluster).astype(int)

    return result


def build_preprocessing_pipeline(config: FeatureConfig) -> Pipeline:
    """
    Build a scikit-learn Pipeline for numerical and categorical features.
    
    Args:
        config: FeatureConfig with column lists.
    
    Returns:
        Pipeline: Configured preprocessing pipeline.
    """
    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, config.numerical_cols),
            ('cat', categorical_transformer, config.categorical_cols)
        ],
        remainder='drop'
    )
    
    return Pipeline([('preprocessor', preprocessor)])