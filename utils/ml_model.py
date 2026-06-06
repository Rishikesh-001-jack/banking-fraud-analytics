# models/ml_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# DATA PREPROCESSING
# ==========================================================

def preprocess_data(df, target_column="fraud_flag"):
    """
    Prepare data for ML training
    """

    data = df.copy()

    # Remove common ID columns

    id_columns = [
        "transaction_id",
        "customer_id",
        "account_id"
    ]

    for col in id_columns:

        if col in data.columns:
            data.drop(
                columns=[col],
                inplace=True
            )

    # Encode categorical columns

    encoders = {}

    categorical_cols = data.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:

        if col != target_column:

            encoder = LabelEncoder()

            data[col] = encoder.fit_transform(
                data[col].astype(str)
            )

            encoders[col] = encoder

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    return X, y, encoders


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_data(
    X,
    y,
    test_size=0.20,
    random_state=42
):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    X_train,
    y_train
):

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=12,

        min_samples_split=5,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    results = {

        "accuracy":
        accuracy_score(
            y_test,
            y_pred
        ),

        "precision":
        precision_score(
            y_test,
            y_pred
        ),

        "recall":
        recall_score(
            y_test,
            y_pred
        ),

        "f1":
        f1_score(
            y_test,
            y_pred
        ),

        "roc_auc":
        roc_auc_score(
            y_test,
            y_prob
        ),

        "confusion_matrix":
        confusion_matrix(
            y_test,
            y_pred
        ),

        "classification_report":
        classification_report(
            y_test,
            y_pred,
            output_dict=True
        ),

        "predictions":
        y_pred,

        "probabilities":
        y_prob

    }

    return results


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def get_feature_importance(
    model,
    feature_names
):

    importance_df = pd.DataFrame({

        "Feature":
        feature_names,

        "Importance":
        model.feature_importances_

    })

    importance_df = (

        importance_df

        .sort_values(
            by="Importance",
            ascending=False
        )

        .reset_index(drop=True)

    )

    return importance_df


# ==========================================================
# SINGLE PREDICTION
# ==========================================================

def predict_transaction(
    model,
    input_df
):

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    return {

        "prediction":
        prediction,

        "probability":
        probability

    }


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(
    model,
    filename="models/fraud_model.pkl"
):

    joblib.dump(
        model,
        filename
    )

    return filename


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(
    filename="models/fraud_model.pkl"
):

    model = joblib.load(
        filename
    )

    return model


# ==========================================================
# SAVE ENCODERS
# ==========================================================

def save_encoders(
    encoders,
    filename="models/encoders.pkl"
):

    joblib.dump(
        encoders,
        filename
    )

    return filename


# ==========================================================
# LOAD ENCODERS
# ==========================================================

def load_encoders(
    filename="models/encoders.pkl"
):

    return joblib.load(
        filename
    )


# ==========================================================
# COMPLETE TRAINING PIPELINE
# ==========================================================

def train_complete_pipeline(
    df,
    target_column="fraud_flag"
):
    """
    Complete ML pipeline
    """

    X, y, encoders = preprocess_data(
        df,
        target_column
    )

    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = split_data(X, y)

    model = train_model(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    feature_importance = (
        get_feature_importance(
            model,
            X.columns
        )
    )

    results = {

        "model":
        model,

        "encoders":
        encoders,

        "X_train":
        X_train,

        "X_test":
        X_test,

        "y_train":
        y_train,

        "y_test":
        y_test,

        "metrics":
        metrics,

        "feature_importance":
        feature_importance

    }

    return results


# ==========================================================
# FRAUD RISK SCORING
# ==========================================================

def calculate_fraud_risk_level(
    probability
):

    if probability >= 0.80:
        return "Critical Risk"

    elif probability >= 0.60:
        return "High Risk"

    elif probability >= 0.40:
        return "Medium Risk"

    elif probability >= 0.20:
        return "Low Risk"

    return "Minimal Risk"


# ==========================================================
# TOP HIGH-RISK PREDICTIONS
# ==========================================================

def get_high_risk_predictions(
    model,
    X_test,
    threshold=0.80
):

    probs = model.predict_proba(
        X_test
    )[:, 1]

    results = X_test.copy()

    results["Fraud_Probability"] = probs

    high_risk = results[
        results["Fraud_Probability"] >= threshold
    ]

    return high_risk.sort_values(
        by="Fraud_Probability",
        ascending=False
    )
