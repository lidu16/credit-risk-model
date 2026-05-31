"""
Unit Tests for Data Processing and Feature Engineering
Tests ensure data pipeline integrity and correct feature engineering
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import calculate_rfm, create_proxy_target, engineer_features, build_preprocessing_pipeline


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing"""
    data = {
        'CustomerId': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'TransactionStartTime': [
            '2024-01-01 10:00:00',
            '2024-01-15 14:30:00',
            '2024-02-01 09:15:00',
            '2024-01-10 11:00:00',
            '2024-01-20 15:30:00',
            '2023-12-01 08:00:00',
            '2024-01-05 12:00:00',
            '2024-02-10 16:00:00',
            '2024-02-20 10:30:00'
        ],
        'Value': [100.0, 150.0, 200.0, 50.0, 75.0, 300.0, 250.0, 180.0, 220.0],
        'Amount': [100.0, 150.0, 200.0, 50.0, 75.0, 300.0, 250.0, 180.0, 220.0],
        'ProductCategory': ['Electronics', 'Electronics', 'Clothing', 'Electronics', 'Clothing',
                           'Electronics', 'Clothing', 'Electronics', 'Clothing'],
        'ChannelId': ['web', 'mobile', 'web', 'mobile', 'mobile', 'web', 'web', 'mobile', 'web'],
        'FraudResult': [0, 0, 0, 0, 1, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df


class TestRFMCalculation:
    """Test RFM metric calculation"""
    
    def test_rfm_shape(self, sample_transaction_data):
        """Test that RFM output has correct shape"""
        rfm = calculate_rfm(sample_transaction_data)
        assert rfm.shape[0] == 3, "Should have 3 unique customers"
        assert 'Recency' in rfm.columns, "Should have Recency column"
        assert 'Frequency' in rfm.columns, "Should have Frequency column"
        assert 'Monetary' in rfm.columns, "Should have Monetary column"
    
    def test_rfm_values(self, sample_transaction_data):
        """Test that RFM values are calculated correctly"""
        rfm = calculate_rfm(sample_transaction_data)
        
        # Customer 1 should have 3 transactions
        cust1 = rfm[rfm['CustomerId'] == 1].iloc[0]
        assert cust1['Frequency'] == 3, "Customer 1 should have 3 transactions"
        assert cust1['Monetary'] == 450.0, "Customer 1 monetary value should be 450"
    
    def test_rfm_non_null(self, sample_transaction_data):
        """Test that RFM calculation produces non-null values"""
        rfm = calculate_rfm(sample_transaction_data)
        assert rfm.isnull().sum().sum() == 0, "RFM should not have null values"


class TestProxyTargetCreation:
    """Test proxy target variable creation"""
    
    def test_target_binary(self, sample_transaction_data):
        """Test that target variable is binary"""
        rfm = calculate_rfm(sample_transaction_data)
        rfm = create_proxy_target(rfm)
        
        assert rfm['is_high_risk'].isin([0, 1]).all(), "Target should be binary (0 or 1)"
        assert 'Cluster' in rfm.columns, "Should have Cluster column"
    
    def test_target_distribution(self, sample_transaction_data):
        """Test that target has reasonable distribution"""
        rfm = calculate_rfm(sample_transaction_data)
        rfm = create_proxy_target(rfm)
        
        target_sum = rfm['is_high_risk'].sum()
        assert target_sum > 0, "Should have at least some high-risk customers"
        assert target_sum < len(rfm), "Should have at least some low-risk customers"


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_feature_engineering_output_shape(self, sample_transaction_data):
        """Test feature engineering produces customer-level data"""
        features = engineer_features(sample_transaction_data)
        
        # Should have customer-level aggregation
        assert features.shape[0] == 3, "Should have 3 customers"
        assert 'CustomerId' in features.columns, "Should have CustomerId"
    
    def test_feature_engineering_columns(self, sample_transaction_data):
        """Test that feature engineering creates expected columns"""
        features = engineer_features(sample_transaction_data)
        
        # Check for aggregate features
        assert any('Value_sum' in col for col in features.columns), "Should have Value_sum"
        assert any('Value_mean' in col for col in features.columns), "Should have Value_mean"
        assert any('count' in col for col in features.columns), "Should have transaction count"


class TestPreprocessingPipeline:
    """Test preprocessing pipeline creation"""
    
    def test_pipeline_creation(self):
        """Test that preprocessing pipeline is created successfully"""
        numerical_features = ['feature1', 'feature2']
        categorical_features = ['cat1', 'cat2']
        
        pipeline = build_preprocessing_pipeline(numerical_features, categorical_features)
        
        assert pipeline is not None, "Pipeline should be created"
        assert hasattr(pipeline, 'fit_transform'), "Pipeline should have fit_transform method"
    
    def test_pipeline_handles_missing_values(self):
        """Test that pipeline handles missing values"""
        # Create test data with missing values
        X = pd.DataFrame({
            'num1': [1.0, 2.0, np.nan, 4.0],
            'num2': [5.0, np.nan, 7.0, 8.0],
            'cat1': ['a', 'b', 'a', np.nan]
        })
        
        numerical_features = ['num1', 'num2']
        categorical_features = ['cat1']
        
        pipeline = build_preprocessing_pipeline(numerical_features, categorical_features)
        X_transformed = pipeline.fit_transform(X)
        
        # Check no NaN in output
        assert not np.isnan(X_transformed).any(), "Transformed data should not have NaN values"


# Integration tests
class TestEndToEndPipeline:
    """End-to-end integration tests"""
    
    def test_rfm_to_target_pipeline(self, sample_transaction_data):
        """Test complete pipeline from data to target"""
        # Calculate RFM
        rfm = calculate_rfm(sample_transaction_data)
        assert rfm.shape[0] > 0, "RFM should have customers"
        
        # Create target
        rfm = create_proxy_target(rfm)
        assert 'is_high_risk' in rfm.columns, "Should have target column"
        
        # Check consistency
        assert len(rfm) == len(rfm['CustomerId'].unique()), "One row per customer"
        assert rfm['is_high_risk'].dtype in [np.int64, np.int32], "Target should be integer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
