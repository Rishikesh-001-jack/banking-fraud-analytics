# utils/data_loader.py

import pandas as pd
import numpy as np
import streamlit as st

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data(
    file_path="data/banking_transactions.csv"
):
    """
    Load banking transaction dataset
    """

    df = pd.read_csv(file_path)

    return df


# ==========================================================
# DATA CLEANING
# ==========================================================

def clean_data(df):
    """
    Basic cleaning
    """

    df = df.copy()

    # Remove duplicate rows

    df.drop_duplicates(inplace=True)

    # Handle missing values

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical_cols:

        df[col] = df[col].fillna(
            "Unknown"
        )

    return df


# ==========================================================
# DATETIME FEATURES
# ==========================================================

def create_datetime_features(df):
    """
    Extract date-time components
    """

    df = df.copy()

    datetime_columns = [

        "transaction_time",
        "transaction_timestamp",
        "timestamp",
        "transaction_date"

    ]

    for col in datetime_columns:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            df["hour"] = df[col].dt.hour

            df["day"] = df[col].dt.day_name()

            df["month"] = df[col].dt.month_name()

            df["weekday"] = df[col].dt.weekday

            df["week"] = (
                df[col]
                .dt.isocalendar()
                .week
            )

            return df

    return df


# ==========================================================
# CREATE RISK SCORE
# ==========================================================

def create_risk_score(df):
    """
    Composite Risk Score
    """

    df = df.copy()

    required_cols = [

        "device_risk_score",
        "anomaly_score",
        "transaction_velocity_score"

    ]

    if all(
        col in df.columns
        for col in required_cols
    ):

        df["risk_score"] = (

            df["device_risk_score"] * 0.40 +

            df["anomaly_score"] * 0.40 +

            df["transaction_velocity_score"] * 0.20

        )

    return df


# ==========================================================
# CREATE RISK SEGMENTS
# ==========================================================

def create_risk_segments(df):
    """
    Categorize risk
    """

    df = df.copy()

    if "risk_score" not in df.columns:

        df = create_risk_score(df)

    df["risk_category"] = pd.cut(

        df["risk_score"],

        bins=[0, 40, 70, 100],

        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ]

    )

    return df


# ==========================================================
# FULL PIPELINE
# ==========================================================

@st.cache_data
def load_processed_data(
    file_path="data/banking_transactions.csv"
):
    """
    Complete processing pipeline
    """

    df = load_data(file_path)

    df = clean_data(df)

    df = create_datetime_features(df)

    df = create_risk_score(df)

    df = create_risk_segments(df)

    return df


# ==========================================================
# KPI SUMMARY
# ==========================================================

def get_kpi_summary(df):
    """
    Executive KPI summary
    """

    total_transactions = len(df)

    total_fraud = int(
        df["fraud_flag"].sum()
    )

    fraud_rate = round(

        (
            total_fraud
            / total_transactions
        ) * 100,

        2

    )

    total_amount = round(

        df["transaction_amount"].sum(),

        2

    )

    avg_transaction = round(

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

    return {

        "total_transactions":
        total_transactions,

        "total_fraud":
        total_fraud,

        "fraud_rate":
        fraud_rate,

        "total_amount":
        total_amount,

        "avg_transaction":
        avg_transaction,

        "avg_risk":
        avg_risk,

        "avg_anomaly":
        avg_anomaly

    }


# ==========================================================
# FRAUD DATA
# ==========================================================

def get_fraud_data(df):
    """
    Fraud only transactions
    """

    return df[
        df["fraud_flag"] == 1
    ]


# ==========================================================
# HIGH RISK DATA
# ==========================================================

def get_high_risk_data(df):
    """
    High risk transactions
    """

    if "risk_category" not in df.columns:

        df = create_risk_segments(df)

    return df[
        df["risk_category"] == "High Risk"
    ]


# ==========================================================
# CHANNEL SUMMARY
# ==========================================================

def get_channel_summary(df):
    """
    Payment channel metrics
    """

    summary = (

        df.groupby(
            "payment_channel"
        )

        .agg(

            transactions=(
                "transaction_amount",
                "count"
            ),

            total_amount=(
                "transaction_amount",
                "sum"
            ),

            fraud_cases=(
                "fraud_flag",
                "sum"
            )

        )

        .reset_index()

    )

    return summary


# ==========================================================
# AUTHENTICATION SUMMARY
# ==========================================================

def get_authentication_summary(df):
    """
    Authentication metrics
    """

    summary = (

        df.groupby(
            "authentication_type"
        )

        .agg(

            transactions=(
                "transaction_amount",
                "count"
            ),

            fraud_cases=(
                "fraud_flag",
                "sum"
            )

        )

        .reset_index()

    )

    return summary


# ==========================================================
# MODEL FEATURES
# ==========================================================

def get_ml_features(df):
    """
    Features used by ML model
    """

    features = [

        "transaction_amount",

        "device_risk_score",

        "anomaly_score",

        "geo_distance_km",

        "transaction_velocity_score"

    ]

    available_features = [

        col for col in features

        if col in df.columns

    ]

    return df[
        available_features
    ]


# ==========================================================
# DATASET INFORMATION
# ==========================================================

def dataset_info(df):
    """
    Dataset metadata
    """

    info = {

        "rows": df.shape[0],

        "columns": df.shape[1],

        "missing_values":
        int(
            df.isnull()
            .sum()
            .sum()
        ),

        "duplicates":
        int(
            df.duplicated()
            .sum()
        )

    }

    return info
