# utils/report_generator.py

import pandas as pd
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================================
# CREATE EXECUTIVE DATAFRAME
# ==========================================================

def create_executive_summary(df):

    total_transactions = len(df)

    total_fraud = int(
        df["fraud_flag"].sum()
    )

    fraud_rate = round(
        (
            total_fraud /
            total_transactions
        ) * 100,
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

    avg_risk = round(
        df["device_risk_score"].mean(),
        2
    )

    avg_anomaly = round(
        df["anomaly_score"].mean(),
        2
    )

    summary = pd.DataFrame({

        "Metric": [

            "Total Transactions",
            "Fraud Cases",
            "Fraud Rate (%)",
            "Total Transaction Value",
            "Average Transaction",
            "Average Device Risk",
            "Average Anomaly Score"

        ],

        "Value": [

            total_transactions,
            total_fraud,
            fraud_rate,
            total_amount,
            avg_amount,
            avg_risk,
            avg_anomaly

        ]

    })

    return summary


# ==========================================================
# CREATE SECURITY SCORE
# ==========================================================

def calculate_security_score(df):

    fraud_rate = round(
        (
            df["fraud_flag"].sum()
            / len(df)
        ) * 100,
        2
    )

    avg_risk = df[
        "device_risk_score"
    ].mean()

    avg_anomaly = df[
        "anomaly_score"
    ].mean()

    score = round(

        100 -

        (
            fraud_rate * 2 +
            avg_risk * 0.30 +
            avg_anomaly * 0.20
        ),

        2

    )

    return max(
        0,
        min(100, score)
    )


# ==========================================================
# GENERATE PDF REPORT
# ==========================================================

def generate_pdf_report(
    df,
    output_path="Executive_Fraud_Report.pdf"
):

    doc = SimpleDocTemplate(
        output_path
    )

    styles = getSampleStyleSheet()

    elements = []

    # ======================================================
    # TITLE
    # ======================================================

    title = Paragraph(
        "Banking Fraud Intelligence Report",
        styles["Title"]
    )

    elements.append(title)

    elements.append(
        Spacer(1, 12)
    )

    date_text = Paragraph(
        f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        styles["Normal"]
    )

    elements.append(date_text)

    elements.append(
        Spacer(1, 20)
    )

    # ======================================================
    # EXECUTIVE SUMMARY
    # ======================================================

    elements.append(
        Paragraph(
            "Executive Summary",
            styles["Heading1"]
        )
    )

    summary_df = create_executive_summary(df)

    table_data = [
        summary_df.columns.tolist()
    ] + summary_df.values.tolist()

    table = Table(table_data)

    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.darkblue
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.black
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            )

        ])

    )

    elements.append(table)

    elements.append(
        Spacer(1, 20)
    )

    # ======================================================
    # SECURITY SCORE
    # ======================================================

    security_score = calculate_security_score(df)

    elements.append(
        Paragraph(
            "Security Assessment",
            styles["Heading1"]
        )
    )

    elements.append(

        Paragraph(
            f"Security Score: {security_score}/100",
            styles["BodyText"]
        )

    )

    elements.append(
        Spacer(1, 12)
    )

    # ======================================================
    # FRAUD ANALYSIS
    # ======================================================

    total_fraud = int(
        df["fraud_flag"].sum()
    )

    fraud_rate = round(
        (
            total_fraud / len(df)
        ) * 100,
        2
    )

    fraud_text = f"""

    Total Fraud Cases: {total_fraud}

    Fraud Rate: {fraud_rate}%

    Fraud monitoring should focus on
    high-risk transactions and
    anomalous behavior patterns.

    """

    elements.append(
        Paragraph(
            "Fraud Analysis",
            styles["Heading1"]
        )
    )

    elements.append(
        Paragraph(
            fraud_text,
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 12)
    )

    # ======================================================
    # CHANNEL ANALYSIS
    # ======================================================

    if "payment_channel" in df.columns:

        channel_df = (

            df.groupby(
                "payment_channel"
            )["fraud_flag"]

            .sum()

            .reset_index()

        )

        top_channel = (

            channel_df.sort_values(
                by="fraud_flag",
                ascending=False
            )

            .iloc[0][
                "payment_channel"
            ]

        )

        elements.append(
            Paragraph(
                "Payment Channel Analysis",
                styles["Heading1"]
            )
        )

        elements.append(

            Paragraph(
                f"Highest fraud activity observed in {top_channel}.",
                styles["BodyText"]
            )

        )

    elements.append(
        Spacer(1, 12)
    )

    # ======================================================
    # RECOMMENDATIONS
    # ======================================================

    elements.append(
        Paragraph(
            "Recommendations",
            styles["Heading1"]
        )
    )

    recommendations = [

        "Implement real-time fraud monitoring.",

        "Strengthen multi-factor authentication.",

        "Monitor high-risk devices.",

        "Deploy machine learning fraud detection.",

        "Review anomaly scores regularly.",

        "Conduct quarterly fraud audits."

    ]

    for item in recommendations:

        elements.append(

            Paragraph(
                f"• {item}",
                styles["BodyText"]
            )

        )

    elements.append(PageBreak())

    # ======================================================
    # TOP HIGH-RISK TRANSACTIONS
    # ======================================================

    elements.append(
        Paragraph(
            "High Risk Transactions",
            styles["Heading1"]
        )
    )

    if all(
        col in df.columns
        for col in [
            "device_risk_score",
            "anomaly_score",
            "transaction_velocity_score"
        ]
    ):

        df["risk_score"] = (

            df["device_risk_score"] * 0.40 +

            df["anomaly_score"] * 0.40 +

            df["transaction_velocity_score"] * 0.20

        )

        high_risk = df.sort_values(
            by="risk_score",
            ascending=False
        ).head(20)

        table_data = [
            high_risk.head(20)
            .astype(str)
            .columns
            .tolist()
        ]

        table_data += (
            high_risk.head(20)
            .astype(str)
            .values
            .tolist()
        )

        risk_table = Table(
            table_data
        )

        risk_table.setStyle(

            TableStyle([

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.black
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                )

            ])

        )

        elements.append(
            risk_table
        )

    # ======================================================
    # BUILD PDF
    # ======================================================

    doc.build(elements)

    return output_path


# ==========================================================
# CSV EXPORT
# ==========================================================

def export_csv_report(
    df,
    output_path="Fraud_Report.csv"
):

    df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ==========================================================
# HIGH RISK EXPORT
# ==========================================================

def export_high_risk_transactions(
    df,
    output_path="High_Risk_Transactions.csv"
):

    if all(
        col in df.columns
        for col in [
            "device_risk_score",
            "anomaly_score",
            "transaction_velocity_score"
        ]
    ):

        df["risk_score"] = (

            df["device_risk_score"] * 0.40 +

            df["anomaly_score"] * 0.40 +

            df["transaction_velocity_score"] * 0.20

        )

        high_risk = df.sort_values(
            by="risk_score",
            ascending=False
        )

        high_risk.to_csv(
            output_path,
            index=False
        )

    return output_path


# ==========================================================
# GENERATE COMPLETE REPORT PACKAGE
# ==========================================================

def generate_report_package(df):

    results = {

        "pdf_report":
        generate_pdf_report(df),

        "csv_report":
        export_csv_report(df),

        "high_risk_report":
        export_high_risk_transactions(df)

    }

    return results
