# pages/3_Risk_Segmentation.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Risk Segmentation",
    page_icon="⚠️",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/banking_transactions.csv")

df = load_data()

# ============================================================
# PAGE TITLE
# ============================================================

st.title("⚠️ Risk Segmentation Dashboard")
st.markdown("Customer & Transaction Risk Intelligence Analysis")

st.divider()

# ============================================================
# CREATE RISK SCORE
# ============================================================

df["combined_risk_score"] = (
    df["device_risk_score"] * 0.4 +
    df["anomaly_score"] * 0.4 +
    df["transaction_velocity_score"] * 0.2
)

# ============================================================
# CREATE RISK LEVELS
# ============================================================

df["risk_level"] = pd.cut(
    df["combined_risk_score"],
    bins=[0, 40, 70, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

channel_filter = st.sidebar.multiselect(
    "Payment Channel",
    options=df["payment_channel"].unique(),
    default=df["payment_channel"].unique()
)

filtered_df = df[
    (df["risk_level"].isin(risk_filter)) &
    (df["payment_channel"].isin(channel_filter))
]

# ============================================================
# KPIs
# ============================================================

total_transactions = len(filtered_df)

high_risk_count = len(
    filtered_df[
        filtered_df["risk_level"] == "High Risk"
    ]
)

medium_risk_count = len(
    filtered_df[
        filtered_df["risk_level"] == "Medium Risk"
    ]
)

low_risk_count = len(
    filtered_df[
        filtered_df["risk_level"] == "Low Risk"
    ]
)

fraud_cases = filtered_df["fraud_flag"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "High Risk",
    f"{high_risk_count:,}"
)

col3.metric(
    "Medium Risk",
    f"{medium_risk_count:,}"
)

col4.metric(
    "Fraud Cases",
    f"{int(fraud_cases):,}"
)

st.divider()

# ============================================================
# RISK DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)

with col1:

    risk_dist = (
        filtered_df["risk_level"]
        .value_counts()
        .reset_index()
    )

    risk_dist.columns = [
        "Risk Level",
        "Count"
    ]

    fig = px.pie(
        risk_dist,
        names="Risk Level",
        values="Count",
        hole=0.5,
        title="Risk Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.bar(
        risk_dist,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        title="Risk Category Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# FRAUD VS RISK LEVEL
# ============================================================

st.subheader("🚨 Fraud vs Risk Level")

fraud_risk = (
    filtered_df.groupby("risk_level")["fraud_flag"]
    .sum()
    .reset_index()
)

fig = px.bar(
    fraud_risk,
    x="risk_level",
    y="fraud_flag",
    color="risk_level",
    title="Fraud Cases Across Risk Segments"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# RISK SCORE DISTRIBUTION
# ============================================================

st.subheader("📈 Combined Risk Score Distribution")

fig = px.histogram(
    filtered_df,
    x="combined_risk_score",
    color="risk_level",
    nbins=50,
    title="Risk Score Histogram"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# DEVICE RISK ANALYSIS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    fig = px.box(
        filtered_df,
        x="risk_level",
        y="device_risk_score",
        color="risk_level",
        title="Device Risk Score by Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        filtered_df,
        x="risk_level",
        y="anomaly_score",
        color="risk_level",
        title="Anomaly Score by Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# PAYMENT CHANNEL RISK
# ============================================================

st.subheader("💳 Payment Channel Risk Analysis")

channel_risk = (
    filtered_df.groupby(
        ["payment_channel", "risk_level"]
    )
    .size()
    .reset_index(name="Count")
)

fig = px.bar(
    channel_risk,
    x="payment_channel",
    y="Count",
    color="risk_level",
    barmode="group",
    title="Risk Levels by Payment Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# GEO DISTANCE RISK ANALYSIS
# ============================================================

st.subheader("🌎 Geo Distance Risk Analysis")

fig = px.scatter(
    filtered_df,
    x="geo_distance_km",
    y="combined_risk_score",
    color="risk_level",
    size="transaction_amount",
    hover_data=["fraud_flag"],
    title="Geo Distance vs Risk Score"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# VELOCITY SCORE ANALYSIS
# ============================================================

st.subheader("⚡ Transaction Velocity Analysis")

fig = px.box(
    filtered_df,
    x="risk_level",
    y="transaction_velocity_score",
    color="risk_level",
    title="Velocity Score by Risk Level"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# HIGH RISK TRANSACTIONS
# ============================================================

st.subheader("🔴 Top High Risk Transactions")

high_risk_df = filtered_df[
    filtered_df["risk_level"] == "High Risk"
]

high_risk_df = high_risk_df.sort_values(
    by="combined_risk_score",
    ascending=False
)

st.dataframe(
    high_risk_df.head(25),
    use_container_width=True
)

# ============================================================
# RISK HEATMAP
# ============================================================

st.subheader("🔥 Risk Correlation Matrix")

numeric_cols = filtered_df.select_dtypes(
    include=np.number
)

corr_matrix = numeric_cols.corr()

fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# AI INSIGHTS
# ============================================================

st.subheader("🤖 AI Risk Insights")

high_risk_percent = round(
    (high_risk_count / total_transactions) * 100,
    2
)

avg_risk_score = round(
    filtered_df["combined_risk_score"].mean(),
    2
)

insights = []

if high_risk_percent > 20:
    insights.append(
        f"{high_risk_percent}% transactions belong to High Risk category."
    )

if avg_risk_score > 60:
    insights.append(
        "Average risk score is above acceptable threshold."
    )

if fraud_cases > 0:
    insights.append(
        f"{int(fraud_cases)} fraud cases detected in selected segments."
    )

highest_risk_channel = (
    filtered_df.groupby("payment_channel")
    ["combined_risk_score"]
    .mean()
    .idxmax()
)

insights.append(
    f"Highest average risk observed in {highest_risk_channel} payment channel."
)

for insight in insights:
    st.warning(insight)

# ============================================================
# DOWNLOAD SEGMENTATION REPORT
# ============================================================

st.divider()

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="📥 Download Risk Segmentation Report",
    data=csv,
    file_name="risk_segmentation_report.csv",
    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Banking Fraud Intelligence Platform | Risk Segmentation Dashboard"
)
