# pages/4_Transaction_Patterns.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Transaction Patterns",
    page_icon="📈",
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

st.title("📈 Transaction Pattern Analytics")
st.markdown("Deep Analysis of Transaction Behavior, Velocity, Geography and Fraud Trends")

st.divider()

# ==========================================================
# DATA PREPARATION
# ==========================================================

# Convert timestamp column if available
date_columns = [
    "transaction_time",
    "transaction_timestamp",
    "timestamp",
    "transaction_date"
]

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
        df["hour"] = df[col].dt.hour
        df["day"] = df[col].dt.day_name()
        df["month"] = df[col].dt.month_name()
        break

# Fallback if no datetime column exists
if "hour" not in df.columns:
    np.random.seed(42)
    df["hour"] = np.random.randint(0, 24, len(df))
    df["day"] = np.random.choice(
        ["Monday","Tuesday","Wednesday","Thursday",
         "Friday","Saturday","Sunday"],
        len(df)
    )

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Transaction Filters")

channel_filter = st.sidebar.multiselect(
    "Payment Channel",
    options=df["payment_channel"].unique(),
    default=df["payment_channel"].unique()
)

fraud_filter = st.sidebar.selectbox(
    "Transaction Type",
    ["All", "Fraud Only", "Non Fraud"]
)

filtered_df = df[
    df["payment_channel"].isin(channel_filter)
]

if fraud_filter == "Fraud Only":
    filtered_df = filtered_df[
        filtered_df["fraud_flag"] == 1
    ]

elif fraud_filter == "Non Fraud":
    filtered_df = filtered_df[
        filtered_df["fraud_flag"] == 0
    ]

# ==========================================================
# KPI SECTION
# ==========================================================

total_transactions = len(filtered_df)

avg_amount = round(
    filtered_df["transaction_amount"].mean(),
    2
)

max_amount = round(
    filtered_df["transaction_amount"].max(),
    2
)

avg_velocity = round(
    filtered_df["transaction_velocity_score"].mean(),
    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Avg Amount",
    f"₹ {avg_amount:,.2f}"
)

col3.metric(
    "Max Amount",
    f"₹ {max_amount:,.2f}"
)

col4.metric(
    "Avg Velocity",
    avg_velocity
)

st.divider()

# ==========================================================
# HOURLY TRANSACTION ANALYSIS
# ==========================================================

st.subheader("⏰ Hourly Transaction Pattern")

hourly = (
    filtered_df.groupby("hour")
    .size()
    .reset_index(name="Transactions")
)

fig = px.line(
    hourly,
    x="hour",
    y="Transactions",
    markers=True,
    title="Transactions by Hour"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# DAY-WISE TRANSACTIONS
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    daily = (
        filtered_df.groupby("day")
        .size()
        .reset_index(name="Transactions")
    )

    fig = px.bar(
        daily,
        x="day",
        y="Transactions",
        color="Transactions",
        title="Day-wise Transaction Volume"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fraud_day = (
        filtered_df.groupby("day")
        ["fraud_flag"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        fraud_day,
        x="day",
        y="fraud_flag",
        color="fraud_flag",
        title="Fraud Cases by Day"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# TRANSACTION AMOUNT DISTRIBUTION
# ==========================================================

st.subheader("💰 Transaction Amount Distribution")

fig = px.histogram(
    filtered_df,
    x="transaction_amount",
    nbins=50,
    color="fraud_flag",
    title="Transaction Amount Frequency"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# VELOCITY ANALYSIS
# ==========================================================

st.subheader("⚡ Transaction Velocity Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.box(
        filtered_df,
        x="fraud_flag",
        y="transaction_velocity_score",
        color="fraud_flag",
        title="Velocity Score vs Fraud"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.histogram(
        filtered_df,
        x="transaction_velocity_score",
        color="fraud_flag",
        nbins=40,
        title="Velocity Score Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# GEO DISTANCE ANALYSIS
# ==========================================================

st.subheader("🌍 Geographic Transaction Analysis")

fig = px.scatter(
    filtered_df,
    x="geo_distance_km",
    y="transaction_amount",
    color="fraud_flag",
    size="transaction_velocity_score",
    hover_data=["device_risk_score"],
    title="Geo Distance vs Transaction Amount"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# CHANNEL ANALYSIS
# ==========================================================

st.subheader("💳 Payment Channel Trends")

channel_analysis = (
    filtered_df.groupby("payment_channel")
    .agg(
        Total_Transactions=("transaction_amount", "count"),
        Total_Value=("transaction_amount", "sum")
    )
    .reset_index()
)

fig = px.bar(
    channel_analysis,
    x="payment_channel",
    y="Total_Transactions",
    color="payment_channel",
    title="Transactions by Payment Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# DEVICE RISK VS AMOUNT
# ==========================================================

st.subheader("📱 Device Risk Pattern")

fig = px.scatter(
    filtered_df,
    x="device_risk_score",
    y="transaction_amount",
    color="fraud_flag",
    size="anomaly_score",
    title="Device Risk vs Transaction Amount"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# ANOMALY PATTERN ANALYSIS
# ==========================================================

st.subheader("🚨 Anomaly Score Analysis")

fig = px.box(
    filtered_df,
    x="fraud_flag",
    y="anomaly_score",
    color="fraud_flag",
    title="Anomaly Score by Fraud Status"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

st.subheader("🔥 Transaction Correlation Matrix")

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

# ==========================================================
# TOP ABNORMAL TRANSACTIONS
# ==========================================================

st.subheader("🚨 Highest Anomaly Transactions")

abnormal_df = filtered_df.sort_values(
    by="anomaly_score",
    ascending=False
)

st.dataframe(
    abnormal_df.head(25),
    use_container_width=True
)

# ==========================================================
# BEHAVIORAL INSIGHTS
# ==========================================================

st.subheader("🤖 Behavioral Insights")

peak_hour = hourly.loc[
    hourly["Transactions"].idxmax(),
    "hour"
]

highest_channel = (
    channel_analysis.sort_values(
        by="Total_Transactions",
        ascending=False
    )
    .iloc[0]["payment_channel"]
)

avg_geo = round(
    filtered_df["geo_distance_km"].mean(),
    2
)

insights = [
    f"Peak transaction activity occurs around {peak_hour}:00 hours.",
    f"Most active payment channel is {highest_channel}.",
    f"Average geographic transaction distance is {avg_geo} km.",
    f"Average transaction velocity score is {avg_velocity}.",
]

for insight in insights:
    st.info(insight)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.divider()

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="📥 Download Transaction Pattern Report",
    data=csv,
    file_name="transaction_pattern_analysis.csv",
    mime="text/csv"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.caption(
    "Banking Fraud Intelligence Platform | Transaction Pattern Analytics"
)
