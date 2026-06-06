# pages/1_Overview_Dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Overview Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():
    df = pd.read_csv("data/banking_transactions.csv")
    return df

df = load_data()

# ==========================
# PAGE TITLE
# ==========================

st.title("🏦 Banking Fraud Analytics Dashboard")
st.markdown("### Executive Overview of Banking Transactions")

st.markdown("---")

# ==========================
# KPI CALCULATIONS
# ==========================

total_transactions = len(df)

total_fraud = df["fraud_flag"].sum()

fraud_rate = round(
    (total_fraud / total_transactions) * 100,
    2
)

avg_transaction = round(
    df["transaction_amount"].mean(),
    2
)

total_amount = round(
    df["transaction_amount"].sum(),
    2
)

avg_risk_score = round(
    df["device_risk_score"].mean(),
    2
)

# ==========================
# KPI SECTION
# ==========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Fraud Cases",
        f"{total_fraud:,}"
    )

with col3:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate}%"
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Average Transaction",
        f"₹ {avg_transaction:,.2f}"
    )

with col5:
    st.metric(
        "Total Amount",
        f"₹ {total_amount:,.2f}"
    )

with col6:
    st.metric(
        "Avg Device Risk",
        avg_risk_score
    )

st.markdown("---")

# ==========================
# FRAUD DISTRIBUTION
# ==========================

col1, col2 = st.columns(2)

with col1:

    fraud_dist = (
        df["fraud_flag"]
        .value_counts()
        .reset_index()
    )

    fraud_dist.columns = [
        "Fraud Status",
        "Count"
    ]

    fraud_dist["Fraud Status"] = fraud_dist[
        "Fraud Status"
    ].map(
        {
            0: "Legitimate",
            1: "Fraud"
        }
    )

    fig = px.pie(
        fraud_dist,
        names="Fraud Status",
        values="Count",
        hole=0.5,
        title="Fraud Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    channel_data = (
        df.groupby("payment_channel")
        .size()
        .reset_index(name="Transactions")
    )

    fig = px.bar(
        channel_data,
        x="payment_channel",
        y="Transactions",
        color="payment_channel",
        title="Transactions by Channel"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# TRANSACTION AMOUNT
# ==========================

st.subheader("💰 Transaction Amount Distribution")

fig = px.histogram(
    df,
    x="transaction_amount",
    nbins=50,
    title="Transaction Amount Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# DEVICE RISK ANALYSIS
# ==========================

col1, col2 = st.columns(2)

with col1:

    fig = px.box(
        df,
        x="fraud_flag",
        y="device_risk_score",
        color="fraud_flag",
        title="Device Risk Score vs Fraud"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        df,
        x="transaction_amount",
        y="device_risk_score",
        color="fraud_flag",
        title="Transaction Amount vs Risk Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# AUTHENTICATION ANALYSIS
# ==========================

st.subheader("🔐 Authentication Analysis")

auth_df = (
    df.groupby("authentication_type")
    .size()
    .reset_index(name="Count")
)

fig = px.bar(
    auth_df,
    x="authentication_type",
    y="Count",
    color="authentication_type",
    title="Authentication Type Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# PAYMENT CHANNEL ANALYSIS
# ==========================

st.subheader("💳 Payment Channel Analysis")

channel_fraud = (
    df.groupby("payment_channel")["fraud_flag"]
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

# ==========================
# TOP HIGH RISK TRANSACTIONS
# ==========================

st.subheader("🚨 Top High Risk Transactions")

high_risk = df.sort_values(
    by="anomaly_score",
    ascending=False
).head(20)

st.dataframe(
    high_risk,
    use_container_width=True
)

# ==========================
# INSIGHTS PANEL
# ==========================

st.subheader("🤖 AI Generated Insights")

insights = []

if fraud_rate > 5:
    insights.append(
        "High fraud rate detected. Immediate investigation recommended."
    )

if avg_risk_score > 60:
    insights.append(
        "Average device risk score is elevated."
    )

if df["transaction_amount"].mean() > 10000:
    insights.append(
        "Large transaction values dominate the dataset."
    )

if total_fraud > 0:
    insights.append(
        f"{total_fraud} fraudulent transactions identified."
    )

for insight in insights:
    st.info(insight)

# ==========================
# FOOTER
# ==========================

st.markdown("---")
st.caption(
    "Banking Fraud Intelligence Platform | Executive Dashboard"
)
