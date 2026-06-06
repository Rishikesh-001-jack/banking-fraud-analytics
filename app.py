import streamlit as st
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Banking Fraud Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD CSS
# ==========================================================

def load_css():
    css_file = "style.css"

    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

load_css()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    logo_path = "assets/logo.png"

    try:

        if os.path.exists(logo_path):
            st.image(
                logo_path,
                use_container_width=True
            )
        else:
            st.markdown(
                """
                # 🏦
                ## Banking Fraud
                ### Intelligence Platform
                """
            )

    except Exception:

        st.markdown(
            """
            # 🏦
            ## Banking Fraud
            ### Intelligence Platform
            """
        )

    st.markdown("---")

    st.success("Analytics Modules")

    st.markdown("""
    📊 Overview Dashboard

    🚨 Fraud Analysis

    ⚠ Risk Segmentation

    📈 Transaction Patterns

    🤖 AI Insights

    🧠 ML Predictions

    📑 Executive Report
    """)

# ==========================================================
# MAIN PAGE
# ==========================================================

st.title("🏦 Banking Fraud Intelligence Platform")

st.markdown("""
Advanced Banking Fraud Detection and Analytics System

### Features
- Fraud Detection Analytics
- Risk Segmentation
- Transaction Pattern Analysis
- AI-Powered Insights
- Machine Learning Predictions
- Executive Reporting
""")

st.markdown("---")

# ==========================================================
# DATA STATUS
# ==========================================================

data_path = "data/banking_transactions.csv"

if os.path.exists(data_path):

    st.success(
        "Dataset Loaded Successfully"
    )

else:

    st.error(
        "Dataset not found: data/banking_transactions.csv"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Banking Fraud Intelligence Platform © 2026"
)
