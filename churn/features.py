"""Feature engineering shared by training, batch scoring and the API.

Keeping ONE add_features() used everywhere avoids train/serve skew.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.db import get_engine

NUM_FEATURES = ["tenure", "monthly_charges", "total_charges", "avg_monthly_spend",
                "charge_gap", "is_month_to_month", "senior_citizen"]
CAT_FEATURES = ["contract", "internet_service", "payment_method", "paperless_billing",
                "partner", "dependents", "tech_support", "online_security"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "churn"


def load_customers(engine=None) -> pd.DataFrame:
    """Read the customers table from MySQL."""
    engine = engine or get_engine()
    return pd.read_sql("SELECT * FROM customers", engine)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure"].clip(lower=1)
    df["charge_gap"] = df["monthly_charges"] - df["avg_monthly_spend"]
    df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)
    return df


def get_xy(df: pd.DataFrame):
    df = add_features(df)
    y = (df[TARGET] == "Yes").astype(int)
    return df[ALL_FEATURES], y


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)])
