# pages/6_ML_Predictions.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="ML Fraud Prediction",
    page_icon="🧠",
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

st.title("🧠 Machine Learning Fraud Prediction")
st.markdown(
    "Predict Fraudulent Transactions Using Machine Learning"
)

st.divider()

# ==========================================================
# DATA PREPROCESSING
# ==========================================================

data = df.copy()

# Remove IDs if available

drop_cols = [
    "transaction_id",
    "customer_id",
    "account_id"
]

for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Encode categorical columns

label_encoders = {}

for col in data.select_dtypes(include="object").columns:

    if col != "fraud_flag":

        le = LabelEncoder()

        data[col] = le.fit_transform(
            data[col].astype(str)
        )

        label_encoders[col] = le

# ==========================================================
# TARGET
# ==========================================================

target_column = "fraud_flag"

if target_column not in data.columns:

    st.error(
        "fraud_flag column not found in dataset."
    )
    st.stop()

# ==========================================================
# FEATURES / TARGET
# ==========================================================

X = data.drop(target_column, axis=1)

y = data[target_column]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# TRAIN MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

# ==========================================================
# KPI SECTION
# ==========================================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Accuracy",
    f"{accuracy:.3f}"
)

col2.metric(
    "Precision",
    f"{precision:.3f}"
)

col3.metric(
    "Recall",
    f"{recall:.3f}"
)

col4.metric(
    "F1 Score",
    f"{f1:.3f}"
)

col5.metric(
    "ROC AUC",
    f"{roc_auc:.3f}"
)

st.divider()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.subheader("🎯 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    importance_df.head(15),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 15 Important Features"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Legit", "Actual Fraud"],
    columns=["Pred Legit", "Pred Fraud"]
)

fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FRAUD PROBABILITY DISTRIBUTION
# ==========================================================

st.subheader("📈 Fraud Probability Distribution")

prob_df = pd.DataFrame({
    "Fraud Probability": y_prob
})

fig = px.histogram(
    prob_df,
    x="Fraud Probability",
    nbins=50,
    title="Predicted Fraud Probability"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

st.subheader("📉 Actual vs Predicted")

compare_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

result = (
    compare_df.groupby(
        ["Actual", "Predicted"]
    )
    .size()
    .reset_index(name="Count")
)

fig = px.bar(
    result,
    x="Actual",
    y="Count",
    color="Predicted",
    barmode="group"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# ==========================================================
# HIGH RISK PREDICTIONS
# ==========================================================

st.subheader("🚨 High-Risk Predicted Transactions")

prediction_df = X_test.copy()

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = y_pred

prediction_df["Fraud_Probability"] = y_prob

high_risk = prediction_df.sort_values(
    by="Fraud_Probability",
    ascending=False
)

st.dataframe(
    high_risk.head(25),
    use_container_width=True
)

# ==========================================================
# MANUAL FRAUD PREDICTION
# ==========================================================

st.subheader("🔮 Predict New Transaction")

numeric_cols = X.select_dtypes(
    include=np.number
).columns

input_data = {}

cols = st.columns(3)

for i, col in enumerate(numeric_cols):

    with cols[i % 3]:

        input_data[col] = st.number_input(
            col,
            value=float(
                X[col].median()
            )
        )

if st.button("Predict Fraud Risk"):

    input_df = pd.DataFrame(
        [input_data]
    )

    # add missing columns

    for c in X.columns:

        if c not in input_df.columns:
            input_df[c] = 0

    input_df = input_df[X.columns]

    pred = model.predict(
        input_df
    )[0]

    prob = model.predict_proba(
        input_df
    )[0][1]

    if pred == 1:

        st.error(
            f"⚠️ Fraud Detected | Probability: {prob:.2%}"
        )

    else:

        st.success(
            f"✅ Legitimate Transaction | Probability: {prob:.2%}"
        )

# ==========================================================
# AI INSIGHTS
# ==========================================================

st.subheader("🤖 ML Insights")

top_feature = (
    importance_df.iloc[0]["Feature"]
)

insights = [

    f"Most influential feature is '{top_feature}'.",

    f"Model achieved {accuracy:.2%} accuracy.",

    f"ROC-AUC score of {roc_auc:.2f} indicates fraud detection capability.",

    "Transactions with higher fraud probability should be reviewed manually.",

    "Feature importance can be used for real-time fraud monitoring."
]

for insight in insights:
    st.info(insight)

# ==========================================================
# DOWNLOAD RESULTS
# ==========================================================

st.divider()

csv = high_risk.to_csv(
    index=False
)

st.download_button(
    label="📥 Download ML Prediction Report",
    data=csv,
    file_name="ML_Fraud_Predictions.csv",
    mime="text/csv"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Banking Fraud Intelligence Platform | Machine Learning Prediction Engine"
)
