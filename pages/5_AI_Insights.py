# pages/5_AI_Insights.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Insights",
    page_icon="🤖",
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
# PAGE TITLE
# ==========================================================

st.title("🤖 AI Fraud Intelligence Center")
st.markdown(
    "Automated Fraud Detection Insights, Risk Intelligence, Business Analytics & Recommendations"
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

avg_amount = round(
    df["transaction_amount"].mean(),
    2
)

total_amount = round(
    df["transaction_amount"].sum(),
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

# ==========================================================
# KPI DASHBOARD
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
    "Total Volume",
    f"₹ {total_amount:,.0f}"
)

st.divider()

# ==========================================================
# RISK SCORE
# ==========================================================

df["risk_score"] = (
    df["device_risk_score"] * 0.4 +
    df["anomaly_score"] * 0.4 +
    df["transaction_velocity_score"] * 0.2
)

# ==========================================================
# OVERALL HEALTH SCORE
# ==========================================================

health_score = round(
    100 -
    (
        fraud_rate * 2 +
        avg_risk * 0.3 +
        avg_anomaly * 0.2
    ),
    2
)

health_score = max(0, min(100, health_score))

st.subheader("🏦 Banking Security Health Score")

st.progress(health_score / 100)

if health_score >= 80:
    st.success(f"Health Score : {health_score}/100")
elif health_score >= 60:
    st.warning(f"Health Score : {health_score}/100")
else:
    st.error(f"Health Score : {health_score}/100")

# ==========================================================
# FRAUD INTELLIGENCE
# ==========================================================

st.subheader("🚨 Fraud Intelligence Summary")

fraud_transactions = df[df["fraud_flag"] == 1]

fraud_amount = fraud_transactions[
    "transaction_amount"
].sum()

avg_fraud_amount = fraud_transactions[
    "transaction_amount"
].mean()

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"Fraudulent Transaction Value: ₹ {fraud_amount:,.0f}"
    )

with col2:

    st.info(
        f"Average Fraud Amount: ₹ {avg_fraud_amount:,.2f}"
    )

# ==========================================================
# PAYMENT CHANNEL INSIGHTS
# ==========================================================

st.subheader("💳 Payment Channel Intelligence")

channel_fraud = (
    df.groupby("payment_channel")
    ["fraud_flag"]
    .sum()
    .reset_index()
)

fig = px.bar(
    channel_fraud,
    x="payment_channel",
    y="fraud_flag",
    color="payment_channel",
    title="Fraud Cases by Payment Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

top_channel = (
    channel_fraud.sort_values(
        by="fraud_flag",
        ascending=False
    )
    .iloc[0]["payment_channel"]
)

st.warning(
    f"Highest fraud concentration detected in {top_channel}"
)

# ==========================================================
# AUTHENTICATION ANALYSIS
# ==========================================================

st.subheader("🔐 Authentication Intelligence")

auth_fraud = (
    df.groupby("authentication_type")
    ["fraud_flag"]
    .sum()
    .reset_index()
)

fig = px.pie(
    auth_fraud,
    names="authentication_type",
    values="fraud_flag",
    hole=0.5,
    title="Fraud by Authentication Method"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# DEVICE RISK INSIGHTS
# ==========================================================

st.subheader("📱 Device Risk Intelligence")

high_risk_devices = len(
    df[
        df["device_risk_score"] > 75
    ]
)

high_risk_fraud = int(
    df[
        (df["device_risk_score"] > 75)
        &
        (df["fraud_flag"] == 1)
    ].shape[0]
)

col1, col2 = st.columns(2)

col1.metric(
    "High Risk Transactions",
    high_risk_devices
)

col2.metric(
    "High Risk Frauds",
    high_risk_fraud
)

# ==========================================================
# ANOMALY ANALYSIS
# ==========================================================

st.subheader("📈 Anomaly Intelligence")

fig = px.histogram(
    df,
    x="anomaly_score",
    color="fraud_flag",
    nbins=50,
    title="Anomaly Score Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# RISK SEGMENTATION
# ==========================================================

st.subheader("⚠️ Risk Segmentation")

df["risk_category"] = pd.cut(
    df["risk_score"],
    bins=[0, 40, 70, 100],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

risk_data = (
    df["risk_category"]
    .value_counts()
    .reset_index()
)

risk_data.columns = [
    "Risk Category",
    "Count"
]

fig = px.pie(
    risk_data,
    names="Risk Category",
    values="Count",
    hole=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TOP SUSPICIOUS TRANSACTIONS
# ==========================================================

st.subheader("🚨 Top Suspicious Transactions")

suspicious = df.sort_values(
    by="risk_score",
    ascending=False
)

st.dataframe(
    suspicious.head(20),
    use_container_width=True
)

# ==========================================================
# EXECUTIVE AI INSIGHTS
# ==========================================================

st.subheader("🧠 Executive AI Insights")

insights = []

if fraud_rate > 5:
    insights.append(
        "Fraud rate exceeds recommended banking threshold."
    )

if avg_risk > 60:
    insights.append(
        "Average device risk score is elevated."
    )

if avg_anomaly > 60:
    insights.append(
        "Anomaly score indicates unusual transaction patterns."
    )

if high_risk_fraud > 0:
    insights.append(
        "High-risk devices are strongly associated with fraud."
    )

insights.append(
    f"Most vulnerable payment channel is {top_channel}."
)

for insight in insights:
    st.warning(insight)

# ==========================================================
# RECOMMENDATIONS
# ==========================================================

st.subheader("✅ AI Recommendations")

recommendations = [
    "Implement adaptive multi-factor authentication.",
    "Increase monitoring of high-risk devices.",
    "Review transactions with anomaly score above 80.",
    "Apply real-time fraud scoring before approval.",
    "Introduce velocity-based fraud detection controls.",
    "Enhance geo-location anomaly monitoring.",
    "Deploy ML-driven fraud prediction models."
]

for rec in recommendations:
    st.success(rec)

# ==========================================================
# FRAUD RISK METER
# ==========================================================

st.subheader("🎯 Fraud Risk Meter")

overall_risk = round(
    (
        fraud_rate * 0.4 +
        avg_risk * 0.3 +
        avg_anomaly * 0.3
    ),
    2
)

st.progress(overall_risk / 100)

if overall_risk < 30:
    st.success(f"Risk Level : LOW ({overall_risk})")

elif overall_risk < 60:
    st.warning(f"Risk Level : MEDIUM ({overall_risk})")

else:
    st.error(f"Risk Level : HIGH ({overall_risk})")

# ==========================================================
# DOWNLOAD INSIGHTS REPORT
# ==========================================================

st.divider()

report = suspicious.head(100)

csv = report.to_csv(index=False)

st.download_button(
    label="📥 Download AI Insights Report",
    data=csv,
    file_name="AI_Insights_Report.csv",
    mime="text/csv"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.caption(
    "Banking Fraud Intelligence Platform | AI Insights Engine"
)
