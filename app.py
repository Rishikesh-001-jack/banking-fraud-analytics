# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import (
    load_processed_data,
    get_kpi_summary
)

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
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main-header {
    font-size:40px;
    font-weight:bold;
    color:#1E3A8A;
}

.sub-header {
    font-size:20px;
    color:#6B7280;
}

.metric-card {
    background-color:#F8FAFC;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_processed_data()

kpis = get_kpi_summary(df)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "assets/logo.png",
        use_container_width=True
    )

    st.title("🏦 Fraud Intelligence")

    st.markdown("---")

    st.success("Navigation Available")

    st.markdown("""
### Pages

📊 Overview Dashboard

🚨 Fraud Analysis

⚠ Risk Segmentation

📈 Transaction Patterns

🤖 AI Insights

🧠 ML Predictions

📑 Executive Report
""")

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<p class="main-header">🏦 Banking Fraud Intelligence Platform</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">Enterprise Fraud Detection, Risk Analytics and AI Insights</p>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# KPI SECTION
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Transactions",
        f"{kpis['total_transactions']:,}"
    )

with col2:
    st.metric(
        "Fraud Cases",
        f"{kpis['total_fraud']:,}"
    )

with col3:
    st.metric(
        "Fraud Rate",
        f"{kpis['fraud_rate']}%"
    )

with col4:
    st.metric(
        "Total Volume",
        f"₹ {kpis['total_amount']:,.0f}"
    )

st.divider()

# ==========================================================
# OVERVIEW CHARTS
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    fraud_summary = pd.DataFrame({
        "Category": ["Legitimate", "Fraud"],
        "Count": [
            len(df) - int(df["fraud_flag"].sum()),
            int(df["fraud_flag"].sum())
        ]
    })

    fig = px.pie(
        fraud_summary,
        names="Category",
        values="Count",
        hole=0.5,
        title="Fraud Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    channel_df = (
        df.groupby("payment_channel")
        .size()
        .reset_index(name="Transactions")
    )

    fig = px.bar(
        channel_df,
        x="payment_channel",
        y="Transactions",
        color="payment_channel",
        title="Transactions by Payment Channel"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# RISK OVERVIEW
# ==========================================================

st.subheader("⚠ Risk Overview")

risk_df = (
    df["risk_category"]
    .value_counts()
    .reset_index()
)

risk_df.columns = [
    "Risk Category",
    "Count"
]

fig = px.bar(
    risk_df,
    x="Risk Category",
    y="Count",
    color="Risk Category",
    title="Risk Segmentation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TOP FRAUD CHANNELS
# ==========================================================

st.subheader("🚨 Fraud by Payment Channel")

fraud_channel = (
    df.groupby("payment_channel")
    ["fraud_flag"]
    .sum()
    .reset_index()
)

fig = px.bar(
    fraud_channel,
    x="payment_channel",
    y="fraud_flag",
    color="payment_channel",
    title="Fraud Cases by Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

st.subheader("📑 Executive Summary")

fraud_rate = kpis["fraud_rate"]

if fraud_rate < 2:
    status = "Low Risk Environment"

elif fraud_rate < 5:
    status = "Moderate Risk Environment"

else:
    status = "High Risk Environment"

st.info(f"""
### Banking Fraud Intelligence Summary

• Total Transactions: {kpis['total_transactions']:,}

• Fraud Cases: {kpis['total_fraud']:,}

• Fraud Rate: {fraud_rate}%

• Average Device Risk Score: {kpis['avg_risk']}

• Average Anomaly Score: {kpis['avg_anomaly']}

• Current Status: {status}
""")

# ==========================================================
# QUICK NAVIGATION
# ==========================================================

st.subheader("🚀 Analytics Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
### Fraud Analysis

- Fraud Trends
- Fraud Channels
- Fraud Heatmaps
- Fraud KPIs
""")

with col2:
    st.warning("""
### Risk Analytics

- Risk Segmentation
- Device Risk
- Velocity Analysis
- Geo Risk
""")

with col3:
    st.info("""
### AI & ML

- AI Insights
- Fraud Prediction
- Executive Reports
- Risk Recommendations
""")

# ==========================================================
# DATA PREVIEW
# ==========================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption("""
Banking Fraud Intelligence Platform v1.0

Built with Streamlit • Plotly • Scikit-Learn • Pandas
""")
