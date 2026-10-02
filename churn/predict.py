"""Model loading, scoring and batch scoring into MySQL.

Run batch scoring:  python -m churn.predict
"""
import joblib
import pandas as pd

from churn.config import MODEL_PATH
from churn.db import get_engine
from churn.features import ALL_FEATURES, add_features, load_customers
from churn.logger import get_logger

log = get_logger("predict")


def risk_band(p: float) -> str:
    return "High" if p >= 0.70 else "Medium" if p >= 0.40 else "Low"


class ChurnModel:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["threshold"]
        self.version = bundle["version"]
        self.metrics = bundle["metrics"]

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        X = add_features(df)[ALL_FEATURES]
        proba = self.pipeline.predict_proba(X)[:, 1]
        out = pd.DataFrame({
            "churn_probability": proba.round(4),
            "will_churn": (proba >= self.threshold).astype(int),
            "risk_band": [risk_band(p) for p in proba],
        }, index=df.index)
        return out


def score_all() -> int:
    """Score every customer in MySQL and store results in `churn_scores`."""
    model = ChurnModel()
    df = load_customers()
    scores = model.predict_df(df)
    result = pd.concat([df[["customer_id", "contract", "monthly_charges", "tenure"]], scores], axis=1)
    result["model_version"] = model.version
    result.to_sql("churn_scores", get_engine(), if_exists="replace", index=False,
                  chunksize=500, method="multi")
    log.info("Scored %d customers -> table churn_scores (%d high risk)",
             len(result), (result["risk_band"] == "High").sum())
    return len(result)


if __name__ == "__main__":
    score_all()
