"""
Portfolio-style Streamlit dashboard for credit risk analytics with SHAP explainability.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import requests

sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(page_title="Credit Risk Portfolio Dashboard", page_icon="🏦", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    .hero {
        background: linear-gradient(135deg, #0f172a, #2563eb);
        padding: 1.5rem 1.75rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
    }
    .card {
        background: #ffffff;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
    }
    .metric-label { font-size: 0.85rem; color: #64748b; }
    .metric-value { font-size: 1.4rem; font-weight: 700; }
    .sidebar-section { margin-top: 1rem; border-top: 1px solid #e2e8f0; padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "customer_features_with_risk.csv"
API_URL = "http://localhost:8000"

# -------------------- Data Loading --------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def make_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.copy()
    if "is_high_risk" not in summary.columns:
        summary["is_high_risk"] = 0
    summary["risk_band"] = np.where(summary["is_high_risk"] == 1, "High Risk", "Low Risk")
    return summary


# -------------------- Plotting Helpers --------------------
def plot_distribution(df: pd.DataFrame) -> None:
    counts = df["risk_band"].value_counts()
    color_map = {"Low Risk": "#2563eb", "High Risk": "#f59e0b"}
    colors = [color_map.get(str(idx), "#2563eb") for idx in counts.index]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Portfolio Risk Distribution")
    ax.set_ylabel("Customers")
    plt.tight_layout()
    st.pyplot(fig)


def plot_amount_vs_transactions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.scatter(df["Transaction_Count"], df["Amount_Sum"], c=df["is_high_risk"], cmap="coolwarm", alpha=0.7)
    ax.set_title("Transaction Volume vs Portfolio Exposure")
    ax.set_xlabel("Transaction Count")
    ax.set_ylabel("Amount Sum")
    plt.tight_layout()
    st.pyplot(fig)


# -------------------- Main Portfolio Page --------------------
def build_portfolio_story(df: pd.DataFrame) -> None:
    high_risk = int(df["is_high_risk"].mean() * 100)
    avg_amount = round(float(df["Amount_Sum"].mean()), 2)
    avg_transactions = round(float(df["Transaction_Count"].mean()), 1)

    st.markdown(
        f"""
        <div class="hero">
          <h1>Credit Risk Portfolio Dashboard</h1>
          <p>This view is designed as a polished portfolio showcase for a credit-risk analytics project.</p>
          <p><b>Portfolio signal:</b> {high_risk}% of customers in the sample are flagged as high risk. Average exposure is <b>${avg_amount}</b> with <b>{avg_transactions}</b> transactions per customer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="card"><div class="metric-label">Portfolio Size</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="metric-label">High Risk Share</div><div class="metric-value">{high_risk}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="metric-label">Avg Exposure</div><div class="metric-value">${avg_amount:.2f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><div class="metric-label">Avg Transactions</div><div class="metric-value">{avg_transactions:.1f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.2, 0.8])
    with left:
        plot_distribution(df)
    with right:
        st.markdown("### Portfolio Insights")
        st.info(
            "- High-risk customers tend to cluster around larger exposures and concentrated transaction behavior.\n"
            "- The dashboard can be extended with SHAP or a live API scoring endpoint for production use."
        )
        st.caption("This presentation is intentionally polished for interviews, demos, and project portfolios.")

    st.markdown("---")
    st.subheader("Customer-level explorer")
    filter_value = st.slider(
        "Show customers with transaction count at least",
        0, 100, 5,
        key="filter_transaction_count"
    )
    filtered = df[df["Transaction_Count"] >= filter_value]
    st.dataframe(filtered.head(20), use_container_width=True)


# -------------------- SHAP Pages --------------------
def show_shap_global_importance():
    st.markdown("## 📊 Global Feature Importance (SHAP)")
    st.markdown("Mean absolute SHAP values across all customers")
    try:
        response = requests.get(f"{API_URL}/shap/feature-importance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            fig, ax = plt.subplots(figsize=(10, 6))
            features = data['features'][:10]
            importance = data['importance'][:10]
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, max(1, len(features))))
            ax.barh(features, importance, color=colors)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
            ax.set_title('Top Features Driving Credit Risk', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            # Table
            df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
            st.dataframe(df_imp, use_container_width=True)
        else:
            st.warning(f"SHAP data not available (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Cannot connect to API. Make sure FastAPI is running:")
        st.code("python -m uvicorn src.api.main:app --reload", language="bash")
    except Exception as e:
        st.error(f"Error: {e}")


def show_shap_predict_explain():
    st.markdown("## 🎯 Predict Risk & Explain")
    st.markdown("Enter customer features to see SHAP explanation")

    col1, col2 = st.columns(2)
    with col1:
        tx_count = st.number_input("Transaction Count", 0, 100, 15, key="shap_tx_count")
        avg_amt = st.number_input("Average Amount ($)", 0.0, 1000.0, 150.0, key="shap_avg_amt")
    with col2:
        recency = st.number_input("Recency (days since last tx)", 0, 365, 30, key="shap_recency")
        amt_sum = st.number_input("Total Amount Sum ($)", 0.0, 50000.0, 2250.0, key="shap_amt_sum")

    if st.button("🔮 Predict & Explain", type="primary", key="predict_explain_btn"):
        features = {
            "Transaction_Count": tx_count,
            "Average_Amount": avg_amt,
            "Recency": recency,
            "Amount_Sum": amt_sum
        }
        try:
            pred_response = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=5)
            shap_response = requests.post(f"{API_URL}/shap/explain", json={"features": features}, timeout=5)

            if pred_response.status_code == 200 and shap_response.status_code == 200:
                pred = pred_response.json()
                shap_data = shap_response.json()

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Risk Probability", f"{pred['probability']:.2%}")
                with col_b:
                    risk_label = "🔴 HIGH RISK" if pred['risk_class'] == 1 else "🟢 LOW RISK"
                    st.metric("Risk Class", risk_label)

                st.markdown("### 📊 SHAP Feature Contributions")
                contrib_df = pd.DataFrame(shap_data['contributions'])
                contrib_df['Impact'] = contrib_df['shap_value'].apply(
                    lambda x: "⬆ Increases Risk" if x > 0 else "⬇ Decreases Risk"
                )
                st.dataframe(contrib_df[['feature', 'value', 'shap_value', 'Impact']].head(10), use_container_width=True)

                # Visualize
                fig, ax = plt.subplots(figsize=(10, 5))
                top = contrib_df.head(10)
                colors = ['red' if x > 0 else 'green' for x in top['shap_value']]
                ax.barh(top['feature'], top['shap_value'], color=colors)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('SHAP Value (Impact on Risk)')
                ax.set_title('Feature Contributions to This Prediction')
                st.pyplot(fig)
            else:
                st.error(f"Error: Prediction {pred_response.status_code}, SHAP {shap_response.status_code}")
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ Cannot connect to API. Make sure FastAPI is running:")
            st.code("python -m uvicorn src.api.main:app --reload", language="bash")
        except Exception as e:
            st.error(f"Error: {e}")


def show_shap_dependence():
    st.markdown("## 📈 SHAP Dependence Plots")
    st.markdown("Shows how feature values affect risk predictions")
    feature = st.selectbox(
        "Select Feature",
        ["Recency", "Transaction_Count", "Average_Amount", "Amount_Sum"],
        key="dep_feature_selector"
    )
    
    try:
        response = requests.get(f"{API_URL}/shap/dependence/{feature}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            x = np.array(data["feature_values"])
            y = np.array(data["shap_values"])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sc = ax.scatter(x, y, c=y, cmap="coolwarm", alpha=0.8, edgecolors="none")
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_xlabel(f"{feature} (Feature Value)", fontsize=11)
            ax.set_ylabel("SHAP Value (Impact on Risk)", fontsize=11)
            ax.set_title(f"SHAP Dependence: {feature}", fontsize=13, fontweight="bold")
            fig.colorbar(sc, ax=ax, label="SHAP Value")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"Dependence data unavailable (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Cannot connect to API. Make sure FastAPI is running:")
        st.code("python -m uvicorn src.api.main:app --reload", language="bash")
    except Exception as e:
        st.error(f"Error loading dependence plot: {e}")


# -------------------- Main App with Navigation --------------------
def main():
    # Load data
    df = load_data()
    if df.empty:
        return

    summary = make_risk_summary(df)

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #0f172a, #2563eb); padding: 1rem; border-radius: 12px; text-align: center; color: white; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: white;">🏦 Credit Analytics</h3>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.85;">Risk Scoring & SHAP Explanations</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Portfolio Overview", "📊 SHAP Global Importance", "🎯 Predict & Explain", "📈 Dependence Plots"],
            index=0,
            key="main_nav"
        )
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This dashboard provides credit risk analytics with SHAP explainability.\n"
            "- **Portfolio Overview**: Static metrics and visualizations\n"
            "- **SHAP**: Global and local explanations"
        )

    # Page routing
    if page == "🏠 Portfolio Overview":
        build_portfolio_story(summary)
        st.markdown("---")
        st.subheader("Visual analytics")
        col1, col2 = st.columns(2)
        with col1:
            plot_amount_vs_transactions(summary)
        with col2:
            amount_col = "Amount_Mean" if "Amount_Mean" in summary.columns else ("Average_Amount" if "Average_Amount" in summary.columns else summary.select_dtypes(include=[np.number]).columns[0])
            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            ax.hist(summary[amount_col], bins=20, color="#2563eb", edgecolor="black", alpha=0.7)
            ax.set_title("Distribution of Average Amounts")
            ax.set_xlabel(amount_col)
            ax.set_ylabel("Customers")
            plt.tight_layout()
            st.pyplot(fig)

    elif page == "📊 SHAP Global Importance":
        show_shap_global_importance()

    elif page == "🎯 Predict & Explain":
        show_shap_predict_explain()

    elif page == "📈 Dependence Plots":
        show_shap_dependence()


if __name__ == "__main__":
    main()