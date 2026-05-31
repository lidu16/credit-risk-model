"""
Data Processing Pipeline for Credit Risk Model
Handles feature engineering, RFM calculation, and proxy target creation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath):
    """Load transaction data from CSV"""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    return df


def calculate_rfm(df, snapshot_date=None):
    """
    Calculate RFM metrics for each customer
    
    Parameters:
    -----------
    df : DataFrame
        Transaction data with 'CustomerId', 'Value', 'TransactionStartTime'
    snapshot_date : datetime
        Reference date for RFM calculation (default: max date in data)
    
    Returns:
    --------
    rfm_df : DataFrame
        Customer-level RFM features
    """
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max()
    
    logger.info(f"Calculating RFM metrics using snapshot date: {snapshot_date}")
    
    rfm_data = []
    for customer_id in df['CustomerId'].unique():
        customer_data = df[df['CustomerId'] == customer_id]
        
        # Recency: Days since last transaction
        last_transaction = customer_data['TransactionStartTime'].max()
        recency = (snapshot_date - last_transaction).days if pd.notna(last_transaction) else np.nan
        
        # Frequency: Number of transactions
        frequency = len(customer_data)
        
        # Monetary: Total transaction value
        monetary = customer_data['Value'].sum()
        
        rfm_data.append({
            'CustomerId': customer_id,
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary
        })
    
    rfm_df = pd.DataFrame(rfm_data)
    logger.info(f"RFM calculation complete. Shape: {rfm_df.shape}")
    return rfm_df


def create_proxy_target(rfm_df, n_clusters=3, random_state=42):
    """
    Create binary proxy target variable using K-Means clustering on RFM
    
    High-risk customers are identified as the cluster with lowest engagement
    (low frequency + low monetary value)
    
    Parameters:
    -----------
    rfm_df : DataFrame
        RFM features for customers
    n_clusters : int
        Number of clusters for segmentation
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    rfm_df : DataFrame
        Updated with 'Cluster' and 'is_high_risk' columns
    """
    logger.info("Creating proxy target variable with K-Means clustering")
    
    # Prepare features for clustering
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_features = rfm_features.fillna(rfm_features.median())
    
    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (lowest engagement)
    cluster_analysis = rfm_df.groupby('Cluster').agg({
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # High-risk = lowest frequency and monetary
    cluster_analysis['Risk_Score'] = (1 / (cluster_analysis['Frequency'] + 1)) * \
                                     (1 / (cluster_analysis['Monetary'] + 1))
    
    high_risk_cluster = cluster_analysis['Risk_Score'].idxmax()
    logger.info(f"High-risk cluster identified: {high_risk_cluster}")
    
    # Create binary target
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    
    logger.info(f"Target distribution:\n{rfm_df['is_high_risk'].value_counts()}")
    
    return rfm_df


def engineer_features(df):
    """
    Engineer temporal and aggregate features from transaction data
    
    Parameters:
    -----------
    df : DataFrame
        Transaction data
    
    Returns:
    --------
    customer_features : DataFrame
        Customer-level engineered features
    """
    logger.info("Engineering temporal and aggregate features")
    
    # Extract temporal features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    
    # Aggregate features per customer
    customer_features = df.groupby('CustomerId').agg({
        'Value': ['sum', 'mean', 'std', 'min', 'max', 'count'],
        'Amount': ['sum', 'mean', 'std'],
        'TransactionHour': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'ProductCategory': 'nunique',
        'ChannelId': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'FraudResult': ['sum', 'mean']
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                 for col in customer_features.columns.values]
    
    logger.info(f"Feature engineering complete. Shape: {customer_features.shape}")
    return customer_features


def build_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Build sklearn preprocessing pipeline
    
    Parameters:
    -----------
    numerical_features : list
        Names of numerical feature columns
    categorical_features : list
        Names of categorical feature columns
    
    Returns:
    --------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    """
    logger.info("Building preprocessing pipeline")
    
    # Numerical pipeline
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    logger.info("Preprocessing pipeline created")
    return preprocessor


def process_data(raw_data_path, output_dir='../data/processed/', save_artifacts=True):
    """
    Complete data processing pipeline
    
    Parameters:
    -----------
    raw_data_path : str
        Path to raw transaction data
    output_dir : str
        Directory to save processed data
    save_artifacts : bool
        Whether to save intermediate artifacts
    
    Returns:
    --------
    X_processed : array
        Processed feature matrix
    y : array
        Target variable
    preprocessor : ColumnTransformer
        Fitted preprocessor
    """
    # Load data
    df = load_data(raw_data_path)
    
    # Calculate RFM
    rfm_df = calculate_rfm(df)
    
    # Create proxy target
    rfm_df = create_proxy_target(rfm_df)
    
    # Engineer features
    customer_features = engineer_features(df)
    
    # Merge features with target
    df_model = customer_features.merge(
        rfm_df[['CustomerId', 'is_high_risk']], 
        on='CustomerId', 
        how='inner'
    )
    
    logger.info(f"Model dataset shape: {df_model.shape}")
    
    # Identify feature types
    numerical_features = df_model.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('CustomerId')
    if 'is_high_risk' in numerical_features:
        numerical_features.remove('is_high_risk')
    
    categorical_features = df_model.select_dtypes(include=['object']).columns.tolist()
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Prepare features and target
    X = df_model.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = df_model['is_high_risk'].values
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    logger.info(f"Processed features shape: {X_processed.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")
    
    # Save artifacts if requested
    if save_artifacts:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        processed_df = pd.DataFrame(X_processed)
        processed_df['is_high_risk'] = y
        processed_df.to_csv(f'{output_dir}/processed_features.csv', index=False)
        
        joblib.dump(preprocessor, f'{output_dir}/preprocessor.pkl')
        logger.info(f"Artifacts saved to {output_dir}")
    
    return X_processed, y, preprocessor, df_model


if __name__ == "__main__":
    # Example usage
    X_processed, y, preprocessor, df_model = process_data('../data/data.csv')
    print(f"\nProcessing complete!")
    print(f"Features shape: {X_processed.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
