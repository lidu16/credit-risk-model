"""
Unit tests for data_processing module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import (
    load_and_clean_data,
    calculate_rfm_features,
    cluster_customers,
    identify_high_risk_cluster,
    create_target_variable,
    build_preprocessing_pipeline,
    FeatureConfig,
    RFMConfig
)


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    customer_ids = [f'CUST_{i}' for i in range(1, 6)]
    data = []
    for cust in customer_ids:
        for _ in range(5):
            data.append({
                'CustomerId': cust,
                'TransactionId': f'TX_{cust}_{_}',
                'Amount': np.random.uniform(10, 500),
                'TransactionStartTime': np.random.choice(dates)
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_rfm_data():
    """Create RFM-like data."""
    return pd.DataFrame({
        'Recency': [5, 10, 30, 45, 60, 90, 120, 150],
        'Frequency': [50, 30, 20, 15, 10, 5, 2, 1],
        'Monetary': [1000, 800, 600, 400, 200, 100, 50, 10]
    })


def test_load_and_clean_data(tmp_path, sample_transaction_data):
    """Test that data loading and cleaning works correctly."""
    # Save sample data to temp file
    filepath = tmp_path / "test_data.csv"
    sample_transaction_data.to_csv(filepath, index=False)
    
    df = load_and_clean_data(str(filepath))
    assert isinstance(df, pd.DataFrame)
    assert 'CustomerId' in df.columns
    assert 'Amount' in df.columns
    assert df['TransactionStartTime'].dtype == 'datetime64[ns]'


def test_calculate_rfm_features(sample_transaction_data):
    """Test RFM feature calculation."""
    rfm = calculate_rfm_features(sample_transaction_data)
    
    assert isinstance(rfm, pd.DataFrame)
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert rfm.index.name == 'CustomerId'
    assert len(rfm) == sample_transaction_data['CustomerId'].nunique()


def test_cluster_customers(sample_rfm_data):
    """Test that K-means clustering returns labels."""
    config = RFMConfig(n_clusters=3, random_state=42)
    clusters = cluster_customers(sample_rfm_data, config)
    
    assert isinstance(clusters, pd.Series)
    assert len(clusters) == len(sample_rfm_data)
    assert clusters.min() >= 0
    assert clusters.max() < config.n_clusters


def test_identify_high_risk_cluster(sample_rfm_data):
    """Test that the high-risk cluster is correctly identified."""
    config = RFMConfig(n_clusters=3, random_state=42)
    clusters = cluster_customers(sample_rfm_data, config)
    
    high_risk = identify_high_risk_cluster(sample_rfm_data, clusters)
    assert isinstance(high_risk, int)
    assert 0 <= high_risk < config.n_clusters


def test_create_target_variable(sample_transaction_data):
    """Test that target variable is added correctly."""
    rfm = calculate_rfm_features(sample_transaction_data)
    config = RFMConfig(n_clusters=3, random_state=42)
    clusters = cluster_customers(rfm, config)
    risk_cluster = identify_high_risk_cluster(rfm, clusters)
    
    # Create a copy with CustomerId
    df = sample_transaction_data.copy()
    result = create_target_variable(df, clusters, risk_cluster)
    
    assert 'is_high_risk' in result.columns
    assert result['is_high_risk'].isin([0, 1]).all()


def test_build_preprocessing_pipeline():
    """Test that the preprocessing pipeline is constructed correctly."""
    config = FeatureConfig(
        numerical_cols=['Transaction_Count', 'Average_Amount'],
        categorical_cols=['ChannelId', 'ProductCategory']
    )
    pipeline = build_preprocessing_pipeline(config)
    
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'transform')
    assert 'preprocessor' in pipeline.named_steps