"""Portfolio-style Streamlit dashboard for credit risk analytics."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Credit Risk Portfolio Dashboard", page_icon="🏦", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "customer_features_with_risk.csv"

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


def plot_distribution(df: pd.DataFrame) -> None:
    counts = df["risk_band"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=["#2563eb", "#f59e0b"])
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
        st.markdown('<div class="card"><div class="metric-label">Portfolio Size</div><div class="metric-value">'+str(len(df))+'</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="metric-label">High Risk Share</div><div class="metric-value">'+f"{high_risk}%"+'</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div class="metric-label">Avg Exposure</div><div class="metric-value">$'+f"{avg_amount:.2f}"+'</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><div class="metric-label">Avg Transactions</div><div class="metric-value">'+f"{avg_transactions:.1f}"+'</div></div>', unsafe_allow_html=True)

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
    filter_value = st.slider("Show customers with transaction count at least", 0, 100, 5)
    filtered = df[df["Transaction_Count"] >= filter_value]
    st.dataframe(filtered.head(20), width="stretch")


def main() -> None:
    df = load_data()
    if df.empty:
        return

    summary = make_risk_summary(df)
    build_portfolio_story(summary)

    st.markdown("---")
    st.subheader("Visual analytics")
    col1, col2 = st.columns(2)
    with col1:
        plot_amount_vs_transactions(summary)
    with col2:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.hist(summary["Amount_Mean"], bins=20, color="#2563eb", edgecolor="black", alpha=0.7)
        ax.set_title("Distribution of Average Amounts")
        ax.set_xlabel("Average Amount")
        ax.set_ylabel("Customers")
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
