# utils/insights.py

import pandas as pd
import numpy as np

# ==========================================================
# OVERVIEW INSIGHTS
# ==========================================================

def generate_overview_insights(df):

    insights = []

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

    insights.append(
        f"Total transactions processed: {total_transactions:,}"
    )

    insights.append(
        f"Total transaction volume: ₹ {total_amount:,.2f}"
    )

    insights.append(
        f"Average transaction amount: ₹ {avg_amount:,.2f}"
    )

    insights.append(
        f"Fraud rate observed: {fraud_rate}%"
    )

    return insights


# ==========================================================
# FRAUD INSIGHTS
# ==========================================================

def generate_fraud_insights(df):

    insights = []

    fraud_cases = int(
        df["fraud_flag"].sum()
    )

    fraud_rate = round(
        (fraud_cases / len(df)) * 100,
        2
    )

    fraud_amount = df[
        df["fraud_flag"] == 1
    ]["transaction_amount"].sum()

    insights.append(
        f"{fraud_cases:,} fraudulent transactions detected."
    )

    insights.append(
        f"Fraud exposure value: ₹ {fraud_amount:,.2f}"
    )

    if fraud_rate > 5:
        insights.append(
            "Fraud rate exceeds recommended banking threshold."
        )

    elif fraud_rate > 2:
        insights.append(
            "Fraud activity is moderately elevated."
        )

    else:
        insights.append(
            "Fraud activity remains within expected limits."
        )

    return insights


# ==========================================================
# PAYMENT CHANNEL INSIGHTS
# ==========================================================

def generate_channel_insights(df):

    insights = []

    channel_fraud = (

        df.groupby(
            "payment_channel"
        )["fraud_flag"]

        .sum()

        .reset_index()

    )

    highest_channel = channel_fraud.loc[
        channel_fraud["fraud_flag"].idxmax()
    ]

    insights.append(
        f"Highest fraud concentration observed in {highest_channel['payment_channel']} channel."
    )

    channel_volume = (

        df.groupby(
            "payment_channel"
        )

        .size()

        .reset_index(name="count")

    )

    busiest_channel = channel_volume.loc[
        channel_volume["count"].idxmax()
    ]

    insights.append(
        f"Most utilized payment channel is {busiest_channel['payment_channel']}."
    )

    return insights


# ==========================================================
# AUTHENTICATION INSIGHTS
# ==========================================================

def generate_authentication_insights(df):

    insights = []

    auth_fraud = (

        df.groupby(
            "authentication_type"
        )["fraud_flag"]

        .sum()

        .reset_index()

    )

    highest_auth = auth_fraud.loc[
        auth_fraud["fraud_flag"].idxmax()
    ]

    insights.append(
        f"Highest fraud occurrences detected under {highest_auth['authentication_type']} authentication."
    )

    return insights


# ==========================================================
# DEVICE RISK INSIGHTS
# ==========================================================

def generate_device_risk_insights(df):

    insights = []

    avg_risk = round(
        df["device_risk_score"].mean(),
        2
    )

    high_risk = len(
        df[
            df["device_risk_score"] > 75
        ]
    )

    insights.append(
        f"Average device risk score: {avg_risk}"
    )

    insights.append(
        f"{high_risk:,} transactions originate from high-risk devices."
    )

    if avg_risk > 60:

        insights.append(
            "Device risk exposure requires additional monitoring."
        )

    return insights


# ==========================================================
# ANOMALY INSIGHTS
# ==========================================================

def generate_anomaly_insights(df):

    insights = []

    avg_anomaly = round(
        df["anomaly_score"].mean(),
        2
    )

    high_anomaly = len(
        df[
            df["anomaly_score"] > 80
        ]
    )

    insights.append(
        f"Average anomaly score: {avg_anomaly}"
    )

    insights.append(
        f"{high_anomaly:,} transactions exhibit extreme anomaly behavior."
    )

    return insights


# ==========================================================
# VELOCITY INSIGHTS
# ==========================================================

def generate_velocity_insights(df):

    insights = []

    avg_velocity = round(
        df["transaction_velocity_score"].mean(),
        2
    )

    insights.append(
        f"Average transaction velocity score: {avg_velocity}"
    )

    if avg_velocity > 60:

        insights.append(
            "Transaction velocity patterns indicate elevated activity."
        )

    return insights


# ==========================================================
# GEO INSIGHTS
# ==========================================================

def generate_geo_insights(df):

    insights = []

    avg_distance = round(
        df["geo_distance_km"].mean(),
        2
    )

    max_distance = round(
        df["geo_distance_km"].max(),
        2
    )

    insights.append(
        f"Average geographic distance: {avg_distance} km."
    )

    insights.append(
        f"Maximum transaction distance observed: {max_distance} km."
    )

    return insights


# ==========================================================
# RISK SEGMENTATION
# ==========================================================

def generate_risk_insights(df):

    insights = []

    if "risk_score" not in df.columns:

        df["risk_score"] = (

            df["device_risk_score"] * 0.40 +

            df["anomaly_score"] * 0.40 +

            df["transaction_velocity_score"] * 0.20

        )

    high_risk = len(
        df[
            df["risk_score"] >= 70
        ]
    )

    medium_risk = len(
        df[
            (df["risk_score"] >= 40)
            &
            (df["risk_score"] < 70)
        ]
    )

    insights.append(
        f"{high_risk:,} transactions fall under High Risk category."
    )

    insights.append(
        f"{medium_risk:,} transactions fall under Medium Risk category."
    )

    return insights


# ==========================================================
# EXECUTIVE INSIGHTS
# ==========================================================

def generate_executive_insights(df):

    insights = []

    fraud_rate = round(
        (
            df["fraud_flag"].sum()
            / len(df)
        ) * 100,
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

    security_score = round(

        100 -

        (
            fraud_rate * 2 +

            avg_risk * 0.30 +

            avg_anomaly * 0.20
        ),

        2

    )

    insights.append(
        f"Banking Security Score: {security_score}/100"
    )

    if security_score > 80:

        insights.append(
            "Overall security posture is strong."
        )

    elif security_score > 60:

        insights.append(
            "Security posture is moderate."
        )

    else:

        insights.append(
            "Security posture requires immediate attention."
        )

    return insights


# ==========================================================
# ML MODEL INSIGHTS
# ==========================================================

def generate_ml_insights(
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    feature_importance_df
):

    insights = []

    top_feature = (
        feature_importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .iloc[0]["Feature"]
    )

    insights.append(
        f"Model Accuracy: {accuracy:.2%}"
    )

    insights.append(
        f"Model Precision: {precision:.2%}"
    )

    insights.append(
        f"Model Recall: {recall:.2%}"
    )

    insights.append(
        f"ROC-AUC Score: {roc_auc:.3f}"
    )

    insights.append(
        f"Most influential feature: {top_feature}"
    )

    return insights


# ==========================================================
# RECOMMENDATIONS ENGINE
# ==========================================================

def generate_recommendations(df):

    recommendations = []

    recommendations.append(
        "Implement real-time fraud monitoring."
    )

    recommendations.append(
        "Strengthen multi-factor authentication."
    )

    recommendations.append(
        "Monitor high-risk devices continuously."
    )

    recommendations.append(
        "Deploy machine learning fraud detection."
    )

    recommendations.append(
        "Track transaction velocity anomalies."
    )

    recommendations.append(
        "Review transactions with anomaly score above 80."
    )

    recommendations.append(
        "Perform quarterly fraud risk assessments."
    )

    recommendations.append(
        "Enhance geo-location verification controls."
    )

    return recommendations


# ==========================================================
# COMPLETE AI REPORT
# ==========================================================

def generate_complete_report(df):

    report = {}

    report["overview"] = generate_overview_insights(df)

    report["fraud"] = generate_fraud_insights(df)

    report["channel"] = generate_channel_insights(df)

    report["authentication"] = generate_authentication_insights(df)

    report["device_risk"] = generate_device_risk_insights(df)

    report["anomaly"] = generate_anomaly_insights(df)

    report["velocity"] = generate_velocity_insights(df)

    report["geo"] = generate_geo_insights(df)

    report["risk"] = generate_risk_insights(df)

    report["executive"] = generate_executive_insights(df)

    report["recommendations"] = generate_recommendations(df)

    return report
