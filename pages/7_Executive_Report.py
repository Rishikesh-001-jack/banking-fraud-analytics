# pages/7_Executive_Report.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Executive Report",
    page_icon="📑",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/banking_transactions.csv")

df = load_data()

# ==========================================================
# TITLE
# ==========================================================

st.title("📑 Executive Banking Fraud Report")
st.markdown(
    "Executive Summary, Risk Intelligence, Fraud Metrics and Strategic Recommendations"
)

st.divider()

# ==========================================================
# KPI CALCULATIONS
# ==========================================================

total_transactions = len(df)

total_fraud = int(df["fraud_flag"].sum())

fraud_rate = round(
    (total_fraud / total_transactions) * 100,
    2
)

total_amount = round(
    df["transaction_amount"].sum(),
    2
)

avg_amount = round(
    df["transaction_amount"].mean(),
    2
)

avg_risk = round(
    df["device_risk_score"].mean(),
    2
)

avg_anomaly = round(
    df["anomaly_score"].mean(),
    2
)

avg_velocity = round(
    df["transaction_velocity_score"].mean(),
    2
)

# ==========================================================
# EXECUTIVE SCORE
# ==========================================================

security_score = round(
    100 -
    (
        fraud_rate * 2 +
        avg_risk * 0.3 +
        avg_anomaly * 0.2
    ),
    2
)

security_score = max(
    0,
    min(100, security_score)
)

# ==========================================================
# KPI CARDS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Cases",
    f"{total_fraud:,}"
)

col3.metric(
    "Fraud Rate",
    f"{fraud_rate}%"
)

col4.metric(
    "Security Score",
    f"{security_score}/100"
)

st.divider()

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

st.subheader("📌 Executive Summary")

summary = f"""
Total transactions processed: {total_transactions:,}

Total fraudulent transactions detected: {total_fraud:,}

Fraud rate observed: {fraud_rate}%

Average transaction value: ₹ {avg_amount:,.2f}

Average device risk score: {avg_risk}

Average anomaly score: {avg_anomaly}

Overall security score: {security_score}/100
"""

st.info(summary)

# ==========================================================
# SECURITY SCORE
# ==========================================================

st.subheader("🛡 Security Health Score")

st.progress(security_score / 100)

if security_score >= 80:
    st.success("Strong Banking Security Posture")

elif security_score >= 60:
    st.warning("Moderate Banking Security Risk")

else:
    st.error("High Banking Security Risk")

# ==========================================================
# FRAUD OVERVIEW
# ==========================================================

st.subheader("🚨 Fraud Overview")

fraud_df = pd.DataFrame({
    "Category": ["Legitimate", "Fraud"],
    "Count": [
        total_transactions - total_fraud,
        total_fraud
    ]
})

fig = px.pie(
    fraud_df,
    names="Category",
    values="Count",
    hole=0.5,
    title="Fraud Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# CHANNEL ANALYSIS
# ==========================================================

st.subheader("💳 Payment Channel Analysis")

channel_analysis = (
    df.groupby("payment_channel")
    ["fraud_flag"]
    .sum()
    .reset_index()
)

fig = px.bar(
    channel_analysis,
    x="payment_channel",
    y="fraud_flag",
    color="payment_channel",
    title="Fraud by Payment Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

top_channel = (
    channel_analysis
    .sort_values(
        by="fraud_flag",
        ascending=False
    )
    .iloc[0]["payment_channel"]
)

# ==========================================================
# AUTHENTICATION ANALYSIS
# ==========================================================

st.subheader("🔐 Authentication Analysis")

auth_analysis = (
    df.groupby("authentication_type")
    ["fraud_flag"]
    .sum()
    .reset_index()
)

fig = px.bar(
    auth_analysis,
    x="authentication_type",
    y="fraud_flag",
    color="authentication_type",
    title="Fraud by Authentication Type"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# RISK SEGMENTATION
# ==========================================================

st.subheader("⚠ Risk Segmentation")

df["risk_score"] = (
    df["device_risk_score"] * 0.4 +
    df["anomaly_score"] * 0.4 +
    df["transaction_velocity_score"] * 0.2
)

df["risk_category"] = pd.cut(
    df["risk_score"],
    bins=[0, 40, 70, 100],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

risk_df = (
    df["risk_category"]
    .value_counts()
    .reset_index()
)

risk_df.columns = [
    "Risk Category",
    "Count"
]

fig = px.pie(
    risk_df,
    names="Risk Category",
    values="Count",
    hole=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TOP RISKS
# ==========================================================

st.subheader("📈 Key Risk Indicators")

col1, col2, col3 = st.columns(3)

high_risk = len(
    df[
        df["risk_category"] == "High Risk"
    ]
)

col1.metric(
    "High Risk Transactions",
    high_risk
)

col2.metric(
    "Avg Risk Score",
    round(
        df["risk_score"].mean(),
        2
    )
)

col3.metric(
    "Avg Velocity Score",
    avg_velocity
)

# ==========================================================
# EXECUTIVE INSIGHTS
# ==========================================================

st.subheader("🧠 Executive Insights")

insights = []

if fraud_rate > 5:
    insights.append(
        "Fraud rate exceeds acceptable banking thresholds."
    )

if avg_risk > 60:
    insights.append(
        "Device risk levels indicate elevated cyber exposure."
    )

if avg_anomaly > 60:
    insights.append(
        "Anomaly patterns suggest abnormal transaction behavior."
    )

insights.append(
    f"Highest fraud concentration occurs in {top_channel} payment channel."
)

insights.append(
    "High-risk transactions should be prioritized for manual review."
)

for insight in insights:
    st.warning(insight)

# ==========================================================
# EXECUTIVE RECOMMENDATIONS
# ==========================================================

st.subheader("✅ Strategic Recommendations")

recommendations = [

    "Deploy real-time fraud detection models.",

    "Increase monitoring of high-risk payment channels.",

    "Implement adaptive multi-factor authentication.",

    "Strengthen anomaly detection controls.",

    "Introduce transaction velocity monitoring.",

    "Enhance geo-location verification processes.",

    "Establish executive fraud review committees.",

    "Perform quarterly fraud risk assessments."
]

for rec in recommendations:
    st.success(rec)

# ==========================================================
# TOP HIGH-RISK TRANSACTIONS
# ==========================================================

st.subheader("🚨 Highest Risk Transactions")

top_risk = df.sort_values(
    by="risk_score",
    ascending=False
)

st.dataframe(
    top_risk.head(20),
    use_container_width=True
)

# ==========================================================
# REPORT GENERATION
# ==========================================================

st.subheader("📄 Report Export")

report_df = pd.DataFrame({

    "Metric": [

        "Total Transactions",
        "Fraud Cases",
        "Fraud Rate",
        "Total Transaction Value",
        "Average Transaction",
        "Average Device Risk",
        "Average Anomaly Score",
        "Security Score"

    ],

    "Value": [

        total_transactions,
        total_fraud,
        fraud_rate,
        total_amount,
        avg_amount,
        avg_risk,
        avg_anomaly,
        security_score

    ]
})

st.dataframe(
    report_df,
    use_container_width=True
)

csv = report_df.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Executive Report (CSV)",
    data=csv,
    file_name="Executive_Fraud_Report.csv",
    mime="text/csv"
)

# ==========================================================
# REPORT TIMESTAMP
# ==========================================================

st.markdown("---")

st.caption(
    f"""
    Banking Fraud Intelligence Platform
    | Executive Reporting Suite
    | Generated on {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}
    """
)
