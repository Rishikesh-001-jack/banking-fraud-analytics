import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = "data/banking_transactions.csv"
OUTPUT_FILE = "data/banking_transactions_fixed.csv"

EXPECTED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
    "transaction_amount",
    "payment_channel",
    "authentication_type",
    "device_risk_score",
    "anomaly_score",
    "transaction_velocity_score",
    "geo_distance_km",
    "merchant_category",
    "customer_age",
    "account_tenure_months",
    "fraud_flag"
]

EXPECTED_COL_COUNT = len(EXPECTED_COLUMNS)

# ==========================================================
# FIX CSV
# ==========================================================

fixed_rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:

    lines = file.readlines()

    header = lines[0].strip().split(",")

    print("=" * 60)
    print("CSV VALIDATION STARTED")
    print("=" * 60)

    for line_no, line in enumerate(lines[1:], start=2):

        row = line.strip().split(",")

        current_cols = len(row)

        if current_cols == EXPECTED_COL_COUNT:

            fixed_rows.append(row)

        elif current_cols < EXPECTED_COL_COUNT:

            print(
                f"Line {line_no}: Missing "
                f"{EXPECTED_COL_COUNT-current_cols} column(s)"
            )

            while len(row) < EXPECTED_COL_COUNT:
                row.append("0")

            fixed_rows.append(row)

        else:

            print(
                f"Line {line_no}: Extra "
                f"{current_cols-EXPECTED_COL_COUNT} column(s)"
            )

            row = row[:EXPECTED_COL_COUNT]

            fixed_rows.append(row)

# ==========================================================
# CREATE CLEAN DATAFRAME
# ==========================================================

df = pd.DataFrame(
    fixed_rows,
    columns=EXPECTED_COLUMNS
)

# ==========================================================
# DATA TYPE CLEANUP
# ==========================================================

numeric_columns = [

    "transaction_amount",

    "device_risk_score",

    "anomaly_score",

    "transaction_velocity_score",

    "geo_distance_km",

    "customer_age",

    "account_tenure_months",

    "fraud_flag"

]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Fill missing numeric values

df[numeric_columns] = (
    df[numeric_columns]
    .fillna(0)
)

# Remove duplicates

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

duplicates_removed = before - after

# ==========================================================
# SAVE FIXED FILE
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================================
# REPORT
# ==========================================================

print("\n")
print("=" * 60)
print("CSV VALIDATION COMPLETED")
print("=" * 60)

print(f"Rows             : {df.shape[0]}")
print(f"Columns          : {df.shape[1]}")
print(f"Duplicates Removed : {duplicates_removed}")

print("\nColumn Names:")

for col in df.columns:
    print(f"✓ {col}")

print("\nFixed file saved as:")
print(OUTPUT_FILE)

print("\nDataset is now ready for Streamlit.")
