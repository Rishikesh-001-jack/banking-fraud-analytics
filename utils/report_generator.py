# utils/report_generator.py

import pandas as pd
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# ==========================================================
# KPI SUMMARY
# ==========================================================

def generate_kpi_summary(df):

    return {
        "Total Transactions": len(df),
        "Fraud Cases": int(df["fraud_flag"].sum()),
        "Fraud Rate (%)": round(
            (df["fraud_flag"].sum() / len(df)) * 100,
            2
        ),
        "Transaction Volume": round(
            df["transaction_amount"].sum(),
            2
        ),
        "Average Transaction": round(
            df["transaction_amount"].mean(),
            2
        ),
        "Average Device Risk": round(
            df["device_risk_score"].mean(),
            2
        ),
        "Average Anomaly Score": round(
            df["anomaly_score"].mean(),
            2
        )
    }


# ==========================================================
# SECURITY SCORE
# ==========================================================

def calculate_security_score(df):

    fraud_rate = (
        df["fraud_flag"].sum() / len(df)
    ) * 100

    avg_risk = df["device_risk_score"].mean()

    avg_anomaly = df["anomaly_score"].mean()

    score = 100 - (
        fraud_rate * 2 +
        avg_risk * 0.3 +
        avg_anomaly * 0.2
    )

    return round(
        max(0, min(score, 100)),
        2
    )


# ==========================================================
# CREATE RISK SCORE
# ==========================================================

def add_risk_score(df):

    data = df.copy()

    data["risk_score"] = (

        data["device_risk_score"] * 0.40 +

        data["anomaly_score"] * 0.40 +

        data["transaction_velocity_score"] * 0.20

    )

    return data


# ==========================================================
# HIGH RISK TRANSACTIONS
# ==========================================================

def get_high_risk_transactions(
    df,
    top_n=100
):

    df = add_risk_score(df)

    return (

        df.sort_values(
            "risk_score",
            ascending=False
        )

        .head(top_n)

    )


# ==========================================================
# EXECUTIVE REPORT DATAFRAME
# ==========================================================

def generate_executive_report(df):

    kpis = generate_kpi_summary(df)

    security_score = calculate_security_score(df)

    report = pd.DataFrame({

        "Metric": list(kpis.keys()) + [
            "Security Score"
        ],

        "Value": list(kpis.values()) + [
            security_score
        ]

    })

    return report


# ==========================================================
# PDF REPORT
# ==========================================================

def generate_pdf_report(
    df,
    output_file="Executive_Report.pdf"
):

    doc = SimpleDocTemplate(
        output_file
    )

    styles = getSampleStyleSheet()

    story = []

    # TITLE

    story.append(

        Paragraph(
            "Banking Fraud Intelligence Report",
            styles["Title"]
        )

    )

    story.append(
        Spacer(1, 12)
    )

    story.append(

        Paragraph(
            f"Generated On: {datetime.now()}",
            styles["Normal"]
        )

    )

    story.append(
        Spacer(1, 20)
    )

    # KPI SECTION

    story.append(

        Paragraph(
            "Executive Summary",
            styles["Heading1"]
        )

    )

    report_df = generate_executive_report(df)

    table_data = [
        report_df.columns.tolist()
    ] + report_df.values.tolist()

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

    story.append(table)

    story.append(
        Spacer(1, 20)
    )

    # FRAUD ANALYSIS

    story.append(

        Paragraph(
            "Fraud Analysis",
            styles["Heading1"]
        )

    )

    fraud_cases = int(
        df["fraud_flag"].sum()
    )

    fraud_rate = round(
        fraud_cases / len(df) * 100,
        2
    )

    fraud_text = f"""
    Total Fraud Cases: {fraud_cases}<br/>
    Fraud Rate: {fraud_rate}%<br/>
    Continuous monitoring is recommended.
    """

    story.append(

        Paragraph(
            fraud_text,
            styles["BodyText"]
        )

    )

    story.append(
        Spacer(1, 20)
    )

    # CHANNEL ANALYSIS

    if "payment_channel" in df.columns:

        channel_df = (

            df.groupby(
                "payment_channel"
            )["fraud_flag"]

            .sum()

            .reset_index()

        )

        highest_channel = (

            channel_df.sort_values(
                by="fraud_flag",
                ascending=False
            )

            .iloc[0]["payment_channel"]

        )

        story.append(

            Paragraph(
                "Channel Analysis",
                styles["Heading1"]
            )

        )

        story.append(

            Paragraph(
                f"Highest fraud activity observed in {highest_channel}.",
                styles["BodyText"]
            )

        )

    story.append(PageBreak())

    # HIGH RISK TABLE

    story.append(

        Paragraph(
            "Top High-Risk Transactions",
            styles["Heading1"]
        )

    )

    high_risk = get_high_risk_transactions(
        df,
        top_n=20
    )

    risk_table = [
        high_risk.astype(str)
        .columns.tolist()
    ]

    risk_table += (
        high_risk.astype(str)
        .values.tolist()
    )

    table = Table(risk_table)

    table.setStyle(

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

    story.append(table)

    doc.build(story)

    return output_file


# ==========================================================
# CSV EXPORT
# ==========================================================

def export_csv_report(
    df,
    filename="Fraud_Report.csv"
):

    df.to_csv(
        filename,
        index=False
    )

    return filename


# ==========================================================
# HIGH RISK CSV
# ==========================================================

def export_high_risk_report(
    df,
    filename="High_Risk_Transactions.csv"
):

    high_risk = get_high_risk_transactions(df)

    high_risk.to_csv(
        filename,
        index=False
    )

    return filename


# ==========================================================
# EXCEL REPORT
# ==========================================================

def export_excel_report(
    df,
    filename="Fraud_Report.xlsx"
):

    high_risk = get_high_risk_transactions(df)

    executive = generate_executive_report(df)

    with pd.ExcelWriter(
        filename,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Transactions",
            index=False
        )

        executive.to_excel(
            writer,
            sheet_name="Executive_Summary",
            index=False
        )

        high_risk.to_excel(
            writer,
            sheet_name="High_Risk",
            index=False
        )

    return filename


# ==========================================================
# ML REPORT
# ==========================================================

def generate_ml_report(
    metrics_dict
):

    report = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ],

        "Value": [

            metrics_dict["accuracy"],

            metrics_dict["precision"],

            metrics_dict["recall"],

            metrics_dict["f1"],

            metrics_dict["roc_auc"]

        ]

    })

    return report


# ==========================================================
# COMPLETE REPORT PACKAGE
# ==========================================================

def generate_report_package(df):

    files = {

        "pdf":
        generate_pdf_report(df),

        "csv":
        export_csv_report(df),

        "excel":
        export_excel_report(df),

        "high_risk":
        export_high_risk_report(df)

    }

    return files
