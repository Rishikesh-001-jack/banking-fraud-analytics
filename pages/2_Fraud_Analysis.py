# pages/2_Fraud_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fraud Analysis",
    page_icon="🚨",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/banking_transactions.csv")

df = load_data()

# =====================================================
# TITLE
# =====================================================

st.title("🚨 Fraud Analysis Dashboard")
st.markdown("Deep Investigation of Fraudulent Banking Transactions")

st.divider()

# =====================================================
# FILTERS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    selected_channel = st.multiselect(
        "Payment Channel",
        options=df["payment_channel"].unique(),
        default=df["payment_channel"].unique()
    )

with col2:
    selected_auth = st.multiselect(
        "Authentication Type",
        options=df["authentication_type"].unique(),
        default=df["authentication_type"].unique()
    )

with col3:
    fraud_filter = st.selectbox(
        "Transaction Type",
        ["All", "Fraud Only", "Non Fraud"]
    )

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["payment_channel"].isin(selected_channel)
]

filtered_df = filtered_df[
    filtered_df["authentication_type"].isin(selected_auth)
]

if fraud_filter == "Fraud Only":
    filtered_df = filtered_df[
        filtered_df["fraud_flag"] == 1
    ]

elif fraud_filter == "Non Fraud":
    filtered_df = filtered_df[
        filtered_df["fraud_flag"] == 0
    ]

# =====================================================
# KPIs
# =====================================================

total_transactions = len(filtered_df)

fraud_count = filtered_df["fraud_flag"].sum()

fraud_rate = round(
    (fraud_count / total_transactions) * 100,
    2
) if total_transactions > 0 else 0

fraud_amount = filtered_df[
    filtered_df["fraud_flag"] == 1
]["transaction_amount"].sum()

avg_fraud_amount = filtered_df[
    filtered_df["fraud_flag"] == 1
]["transaction_amount"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Cases",
    f"{int(fraud_count):,}"
)

col3.metric(
    "Fraud Rate",
    f"{fraud_rate}%"
)

col4.metric(
    "Fraud Amount",
    f"₹ {fraud_amount:,.0f}"
)

st.divider()

# =====================================================
# FRAUD VS NON FRAUD
# =====================================================

col1, col2 = st.columns(2)

with col1:

    fraud_distribution = (
        filtered_df["fraud_flag"]
        .value_counts()
        .reset_index()
    )

    fraud_distribution.columns = [
        "Category",
        "Count"
    ]

    fraud_distribution["Category"] = fraud_distribution[
        "Category"
    ].replace({
        0: "Legitimate",
        1: "Fraud"
    })

    fig = px.pie(
        fraud_distribution,
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

    fraud_channel = (
        filtered_df.groupby(
            "payment_channel"
        )["fraud_flag"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        fraud_channel,
        x="payment_channel",
        y="fraud_flag",
        color="payment_channel",
        title="Fraud Cases by Payment Channel"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# FRAUD BY AUTHENTICATION
# =====================================================

col1, col2 = st.columns(2)

with col1:

    auth_fraud = (
        filtered_df.groupby(
            "authentication_type"
        )["fraud_flag"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        auth_fraud,
        x="authentication_type",
        y="fraud_flag",
        color="authentication_type",
        title="Fraud by Authentication Method"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        filtered_df,
        x="fraud_flag",
        y="transaction_amount",
        color="fraud_flag",
        title="Transaction Amount vs Fraud"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# ANOMALY SCORE ANALYSIS
# =====================================================

st.subheader("📈 Anomaly Score Analysis")

fig = px.histogram(
    filtered_df,
    x="anomaly_score",
    color="fraud_flag",
    nbins=40,
    barmode="overlay",
    title="Anomaly Score Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# DEVICE RISK ANALYSIS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        filtered_df,
        x="device_risk_score",
        y="transaction_amount",
        color="fraud_flag",
        title="Risk Score vs Transaction Amount"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        filtered_df,
        x="fraud_flag",
        y="device_risk_score",
        color="fraud_flag",
        title="Device Risk Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# GEO DISTANCE ANALYSIS
# =====================================================

st.subheader("🌎 Geo Distance Fraud Analysis")

fig = px.scatter(
    filtered_df,
    x="geo_distance_km",
    y="transaction_amount",
    color="fraud_flag",
    size="device_risk_score",
    title="Geo Distance vs Transaction Amount"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# FRAUD HEATMAP
# =====================================================

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Fraud Correlation Matrix"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# TOP FRAUD TRANSACTIONS
# =====================================================

st.subheader("🚨 Top Fraud Transactions")

fraud_transactions = filtered_df[
    filtered_df["fraud_flag"] == 1
]

fraud_transactions = fraud_transactions.sort_values(
    by="anomaly_score",
    ascending=False
)

st.dataframe(
    fraud_transactions.head(25),
    use_container_width=True
)

# =====================================================
# HIGH RISK PROFILE
# =====================================================

st.subheader("⚠️ High Risk Profile")

high_risk = filtered_df[
    filtered_df["device_risk_score"] > 75
]

col1, col2, col3 = st.columns(3)

col1.metric(
    "High Risk Transactions",
    len(high_risk)
)

col2.metric(
    "High Risk Fraud Cases",
    int(high_risk["fraud_flag"].sum())
)

col3.metric(
    "Average Risk Score",
    round(
        high_risk["device_risk_score"].mean(),
        2
    )
)

# =====================================================
# AI INSIGHTS
# =====================================================

st.subheader("🤖 Fraud Intelligence Insights")

insights = []

if fraud_rate > 10:
    insights.append(
        "Fraud rate exceeds 10%. Immediate monitoring recommended."
    )

if filtered_df["anomaly_score"].mean() > 70:
    insights.append(
        "Average anomaly score is unusually high."
    )

if filtered_df["device_risk_score"].mean() > 60:
    insights.append(
        "Device risk levels indicate elevated threat activity."
    )

top_channel = (
    filtered_df.groupby("payment_channel")
    ["fraud_flag"]
    .sum()
    .idxmax()
)

insights.append(
    f"Highest fraud concentration detected in {top_channel} channel."
)

for insight in insights:
    st.warning(insight)

# =====================================================
# DOWNLOAD FRAUD DATA
# =====================================================

st.divider()

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="📥 Download Fraud Analysis Data",
    data=csv,
    file_name="fraud_analysis.csv",
    mime="text/csv"
)
