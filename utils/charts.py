# utils/charts.py

import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# FRAUD DISTRIBUTION PIE CHART
# ==========================================================

def fraud_distribution_chart(df):

    fraud_data = (
        df["fraud_flag"]
        .value_counts()
        .reset_index()
    )

    fraud_data.columns = [
        "Fraud Status",
        "Count"
    ]

    fraud_data["Fraud Status"] = fraud_data[
        "Fraud Status"
    ].replace({
        0: "Legitimate",
        1: "Fraud"
    })

    fig = px.pie(
        fraud_data,
        names="Fraud Status",
        values="Count",
        hole=0.5,
        title="Fraud Distribution"
    )

    return fig


# ==========================================================
# PAYMENT CHANNEL ANALYSIS
# ==========================================================

def payment_channel_chart(df):

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

    return fig


# ==========================================================
# FRAUD BY CHANNEL
# ==========================================================

def fraud_by_channel_chart(df):

    fraud_df = (
        df.groupby("payment_channel")
        ["fraud_flag"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        fraud_df,
        x="payment_channel",
        y="fraud_flag",
        color="payment_channel",
        title="Fraud Cases by Payment Channel"
    )

    return fig


# ==========================================================
# AUTHENTICATION ANALYSIS
# ==========================================================

def authentication_chart(df):

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
        title="Authentication Distribution"
    )

    return fig


# ==========================================================
# TRANSACTION AMOUNT HISTOGRAM
# ==========================================================

def transaction_amount_chart(df):

    fig = px.histogram(
        df,
        x="transaction_amount",
        nbins=50,
        title="Transaction Amount Distribution"
    )

    return fig


# ==========================================================
# ANOMALY SCORE DISTRIBUTION
# ==========================================================

def anomaly_score_chart(df):

    fig = px.histogram(
        df,
        x="anomaly_score",
        color="fraud_flag",
        nbins=40,
        title="Anomaly Score Distribution"
    )

    return fig


# ==========================================================
# DEVICE RISK DISTRIBUTION
# ==========================================================

def device_risk_chart(df):

    fig = px.box(
        df,
        x="fraud_flag",
        y="device_risk_score",
        color="fraud_flag",
        title="Device Risk Score vs Fraud"
    )

    return fig


# ==========================================================
# GEO DISTANCE ANALYSIS
# ==========================================================

def geo_distance_chart(df):

    fig = px.scatter(
        df,
        x="geo_distance_km",
        y="transaction_amount",
        color="fraud_flag",
        size="device_risk_score",
        title="Geo Distance vs Transaction Amount"
    )

    return fig


# ==========================================================
# TRANSACTION VELOCITY
# ==========================================================

def velocity_chart(df):

    fig = px.box(
        df,
        x="fraud_flag",
        y="transaction_velocity_score",
        color="fraud_flag",
        title="Velocity Score Analysis"
    )

    return fig


# ==========================================================
# DEVICE RISK VS AMOUNT
# ==========================================================

def risk_vs_amount_chart(df):

    fig = px.scatter(
        df,
        x="device_risk_score",
        y="transaction_amount",
        color="fraud_flag",
        size="anomaly_score",
        title="Device Risk vs Transaction Amount"
    )

    return fig


# ==========================================================
# HOURLY TRANSACTION TREND
# ==========================================================

def hourly_transaction_chart(hourly_df):

    fig = px.line(
        hourly_df,
        x="hour",
        y="Transactions",
        markers=True,
        title="Hourly Transaction Trend"
    )

    return fig


# ==========================================================
# DAILY TRANSACTION ANALYSIS
# ==========================================================

def daily_transaction_chart(daily_df):

    fig = px.bar(
        daily_df,
        x="day",
        y="Transactions",
        color="Transactions",
        title="Daily Transaction Activity"
    )

    return fig


# ==========================================================
# FRAUD RATE BY DAY
# ==========================================================

def fraud_day_chart(df):

    fraud_day = (
        df.groupby("day")
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

    return fig


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance_chart(feature_df):

    fig = px.bar(
        feature_df.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )

    return fig


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def confusion_matrix_chart(cm_df):

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    return fig


# ==========================================================
# RISK SEGMENTATION PIE
# ==========================================================

def risk_segmentation_chart(risk_df):

    fig = px.pie(
        risk_df,
        names="Risk Category",
        values="Count",
        hole=0.5,
        title="Risk Segmentation"
    )

    return fig


# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def correlation_heatmap(corr_matrix):

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    return fig


# ==========================================================
# FRAUD PROBABILITY
# ==========================================================

def fraud_probability_chart(prob_df):

    fig = px.histogram(
        prob_df,
        x="Fraud_Probability",
        nbins=50,
        title="Fraud Probability Distribution"
    )

    return fig


# ==========================================================
# EXECUTIVE SECURITY SCORE
# ==========================================================

def security_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Security Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.4},
                "steps": [
                    {"range": [0, 40]},
                    {"range": [40, 70]},
                    {"range": [70, 100]}
                ]
            }
        )
    )

    fig.update_layout(
        height=350
    )

    return fig


# ==========================================================
# TOP RISK TRANSACTIONS
# ==========================================================

def risk_score_scatter(df):

    fig = px.scatter(
        df,
        x="risk_score",
        y="transaction_amount",
        color="fraud_flag",
        size="device_risk_score",
        title="Risk Score vs Transaction Amount"
    )

    return fig
