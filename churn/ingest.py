"""ETL step: CSV -> cleaned -> MySQL `customers` table.

Run:  python -m churn.ingest
"""
import pandas as pd

from churn.config import RAW_CSV
from churn.db import create_database_if_missing, customers, get_engine, metadata
from churn.generate_data import generate
from churn.logger import get_logger

log = get_logger("ingest")

COLUMN_MAP = {
    "customerID": "customer_id", "gender": "gender", "SeniorCitizen": "senior_citizen",
    "Partner": "partner", "Dependents": "dependents", "tenure": "tenure",
    "PhoneService": "phone_service", "MultipleLines": "multiple_lines",
    "InternetService": "internet_service", "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup", "DeviceProtection": "device_protection",
    "TechSupport": "tech_support", "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies", "Contract": "contract",
    "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges", "TotalCharges": "total_charges", "Churn": "churn",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["senior_citizen"] = df["senior_citizen"].astype(int)
    df = df.drop_duplicates(subset="customer_id")
    return df


def run() -> int:
    if RAW_CSV.exists():
        log.info("Reading %s", RAW_CSV)
        raw = pd.read_csv(RAW_CSV)
    else:
        log.warning("%s not found -> using synthetic data", RAW_CSV)
        raw = generate()

    df = clean(raw)
    create_database_if_missing()
    engine = get_engine()
    metadata.drop_all(engine, tables=[customers])
    metadata.create_all(engine)
    df.to_sql("customers", engine, if_exists="append", index=False,
              chunksize=500, method="multi")
    log.info("Inserted %d rows into customers", len(df))
    return len(df)


if __name__ == "__main__":
    run()
