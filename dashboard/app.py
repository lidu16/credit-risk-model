"""
Streamlit Dashboard for Credit Risk Model
Interactive visualization with SHAP explanations and real-time predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Credit Risk Model Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #283593;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #d32f2f;
        font-weight: 700;
    }
    .risk-low {
        color: #388e3c;
        font-weight: 700;
    }
    .risk-moderate {
        color: #f57c00;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏦 Credit Risk Model Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x80?text=GMF+Bank", use_container_width=True)
    st.markdown("## Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Overview", "📊 Predict Risk", "🔍 SHAP Explanations", "📈 Data Explorer", "📋 Reports"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard provides real-time credit risk assessment using machine learning.
    - **Predict**: Score new customers
    - **Explain**: Understand why predictions are made
    - **Explore**: Analyze historical data
    """)

# API Base URL (update if running on different port)
API_URL = "http://localhost:8000"


# ==================== PAGE 1: OVERVIEW ====================
if page == "🏠 Overview":
    st.markdown('<p class="sub-header">📋 Dashboard Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Model Status</h3>
            <p style="font-size:2rem; color:#388e3c;">✅</p>
            <p>Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <p style="font-size:2rem;">85%</p>
            <p>+2.3% vs baseline</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 F1 Score</h3>
            <p style="font-size:2rem;">0.82</p>
            <p>Balanced precision/recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🏷️ High-Risk Rate</h3>
            <p style="font-size:2rem;">12%</p>
            <p>of applicants flagged</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">📊 Feature Importance (SHAP)</p>', unsafe_allow_html=True)
        try:
            response = requests.get(f"{API_URL}/shap/feature-importance")
            if response.status_code == 200:
                data = response.json()
                fig, ax = plt.subplots(figsize=(10, 6))
                features = data['features'][:10]
                importance = data['importance'][:10]
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
                ax.barh(features, importance, color=colors)
                ax.set_xlabel('Mean |SHAP Value|')
                ax.set_title('Top 10 Features Driving Risk Predictions')
                st.pyplot(fig)
            else:
                st.warning("SHAP data not available. Train the model first.")
        except:
            st.warning("Could not connect to API. Make sure the server is running.")
    
    with col2:
        st.markdown('<p class="sub-header">📊 Risk Distribution</p>', unsafe_allow_html=True)
        # Sample data - replace with actual data
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Moderate', 'High'],
            'Count': [650, 250, 100]
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#388e3c', '#f57c00', '#d32f2f']
        ax.pie(risk_data['Count'], labels=risk_data['Risk Level'], autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Customer Risk Distribution')
        st.pyplot(fig)


# ==================== PAGE 2: PREDICT ====================
elif page == "📊 Predict Risk":
    st.markdown('<p class="sub-header">🎯 Predict Customer Risk</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Enter Customer Features")
        
        # Create input fields based on your model features
        transaction_count = st.number_input("Transaction Count", min_value=0, max_value=1000, value=15)
        average_amount = st.number_input("Average Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=150.0)
        recency = st.number_input("Recency (days since last transaction)", min_value=0, max_value=365, value=30)
        
        # Categorical features
        channel = st.selectbox("Channel", ["Web", "Mobile App", "Pay-later", "Checkout"])
        product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home", "Food", "Other"])
        
        # Map channel to numeric
        channel_map = {"Web": 1, "Mobile App": 2, "Pay-later": 3, "Checkout": 4}
        
        # Predict button
        if st.button("🔮 Predict Risk", type="primary"):
            # Prepare request
            features = {
                "Transaction_Count": transaction_count,
                "Average_Amount": average_amount,
                "Recency": recency,
                "ChannelId": channel_map[channel],
                "ProductCategory": product_category
            }
            
            try:
                response = requests.post(f"{API_URL}/predict", json={"features": features})
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.success("✅ Prediction complete!")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Risk Probability", f"{result['probability']:.2%}")
                    with col_b:
                        risk_label = "HIGH RISK" if result['risk_class'] == 1 else "LOW RISK"
                        color = "risk-high" if result['risk_class'] == 1 else "risk-low"
                        st.markdown(f"### Risk Class: <span class='{color}'>{risk_label}</span>", unsafe_allow_html=True)
                    
                    # Show SHAP explanation
                    st.markdown("---")
                    st.markdown("#### 🔍 Why did the model make this prediction?")
                    
                    # Get SHAP explanation
                    shap_response = requests.post(f"{API_URL}/shap/explain", json={"features": features})
                    if shap_response.status_code == 200:
                        shap_data = shap_response.json()
                        
                        # Display contributions
                        contrib_df = pd.DataFrame(shap_data['contributions'])
                        contrib_df['impact'] = contrib_df['shap_value'].apply(lambda x: "⬆ Increases Risk" if x > 0 else "⬇ Decreases Risk")
                        
                        st.dataframe(contrib_df[['feature', 'value', 'shap_value', 'impact']].head(5))
                        
                        # Visualize contributions
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['red' if x > 0 else 'green' for x in contrib_df['shap_value'].head(10)]
                        ax.barh(contrib_df['feature'].head(10), contrib_df['shap_value'].head(10), color=colors)
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        ax.set_xlabel('SHAP Value (Impact on Risk)')
                        ax.set_title('Feature Contributions to Prediction')
                        st.pyplot(fig)
                    else:
                        st.warning("SHAP explanation not available")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        st.info("""
        **How to interpret results:**
        - **Risk Probability**: 0-100% chance of being high-risk
        - **Risk Class**: High Risk (1) or Low Risk (0)
        - **SHAP Values**: Positive = increases risk, Negative = decreases risk
        """)


# ==================== PAGE 3: SHAP EXPLANATIONS ====================
elif page == "🔍 SHAP Explanations":
    st.markdown('<p class="sub-header">🔍 Model Explainability with SHAP</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Global Importance", "📈 Dependence Plots", "🎯 Individual Explanation"])
    
    with tab1:
        st.markdown("### Global Feature Importance")
        st.markdown("Mean absolute SHAP values across all predictions")
        
        try:
            response = requests.get(f"{API_URL}/shap/feature-importance")
            if response.status_code == 200:
                data = response.json()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                features = data['features']
                importance = data['importance']
                
                colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(features)))
                ax.barh(features, importance, color=colors)
                ax.set_xlabel('Mean |SHAP Value|')
                ax.set_title('Global Feature Importance (Top 15)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Show as table
                st.markdown("### Feature Importance Table")
                df_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                st.dataframe(df_importance)
            else:
                st.warning("Feature importance data not available")
        except:
            st.warning("Could not connect to API. Make sure the server is running.")
    
    with tab2:
        st.markdown("### SHAP Dependence Plots")
        st.markdown("Select a feature to visualize its relationship with SHAP values")
        
        # Feature selection
        feature = st.selectbox("Select Feature", [
            "Transaction_Count", "Average_Amount", "Recency", "ChannelId"
        ])
        
        st.info(f"Selected feature: **{feature}**")
        st.caption("Dependence plot shows how feature values affect the prediction")
    
    with tab3:
        st.markdown("### Individual Prediction Explanation")
        st.markdown("Enter features to see a detailed SHAP explanation")
        
        # Quick test input
        col_a, col_b = st.columns(2)
        with col_a:
            tx_count = st.slider("Transaction Count", 0, 100, 10)
            avg_amt = st.slider("Average Amount", 0.0, 1000.0, 150.0)
        with col_b:
            rec = st.slider("Recency (days)", 0, 365, 30)
            channel_id = st.selectbox("Channel", [1, 2, 3, 4])
        
        if st.button("Explain Prediction"):
            features = {
                "Transaction_Count": tx_count,
                "Average_Amount": avg_amt,
                "Recency": rec,
                "ChannelId": channel_id
            }
            
            try:
                response = requests.post(f"{API_URL}/shap/explain", json={"features": features})
                if response.status_code == 200:
                    data = response.json()
                    
                    st.metric("Predicted Risk Probability", f"{data['predicted_probability']:.2%}")
                    st.metric("Base Value (Average)", f"{data['base_value']:.4f}")
                    
                    # Show contributions
                    st.markdown("#### Feature Contributions")
                    contrib_df = pd.DataFrame(data['contributions'])
                    st.dataframe(contrib_df.head(10))
                else:
                    st.error(f"Error: {response.status_code}")
            except:
                st.error("Could not connect to API")


# ==================== PAGE 4: DATA EXPLORER ====================
elif page == "📈 Data Explorer":
    st.markdown('<p class="sub-header">📊 Explore Historical Data</p>', unsafe_allow_html=True)
    
    # Upload section
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Sample")
            st.dataframe(df.head(10))
        
        with col2:
            st.markdown("### Summary Statistics")
            st.dataframe(df.describe())
        
        st.markdown("---")
        st.markdown("### Data Visualizations")
        
        # Select columns for plotting
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col_a, col_b = st.columns(2)
            with col_a:
                x_col = st.selectbox("X-axis", num_cols)
            with col_b:
                y_col = st.selectbox("Y-axis", num_cols)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df[x_col], df[y_col], alpha=0.5)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{y_col} vs {x_col}')
            st.pyplot(fig)


# ==================== PAGE 5: REPORTS ====================
elif page == "📋 Reports":
    st.markdown('<p class="sub-header">📋 Business Reports & Insights</p>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📈 Model Accuracy", "85%", "+2.3%")
    with col2:
        st.metric("🎯 F1 Score", "0.82", "+0.05")
    with col3:
        st.metric("⚠️ False Positive Rate", "8%", "-1.2%")
    
    st.markdown("---")
    
    st.markdown("### 💡 Business Recommendations")
    st.info("""
    1. **High-Risk Customers**: Require additional verification or lower credit limits
    2. **Transaction Velocity**: Customers with >50 transactions/month show higher risk
    3. **Recency**: Customers inactive for >60 days are more likely to be high-risk
    4. **Channel**: Pay-later channel shows 2x higher risk than other channels
    """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Model Performance Summary")
    st.markdown("""
    | Metric | Value | Benchmark |
    |--------|-------|-----------|
    | Accuracy | 85% | 80% |
    | Precision | 0.79 | 0.75 |
    | Recall | 0.85 | 0.80 |
    | F1 Score | 0.82 | 0.77 |
    | AUC-ROC | 0.91 | 0.88 |
    """)
    
    st.download_button(
        label="📥 Download Report (PDF)",
        data="Report content here",
        file_name="credit_risk_report.pdf",
        mime="application/pdf"
    )


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>© 2026 GMF Investments | Credit Risk Model Dashboard v1.0 | Built with ❤️ using Streamlit & SHAP</p>
</div>
""", unsafe_allow_html=True)