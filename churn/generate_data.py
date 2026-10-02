"""Synthetic data with the SAME schema as the Kaggle 'Telco Customer Churn' CSV.

Use it when you don't have the Kaggle file yet. For the real dataset just put the
CSV at data/raw/telco_churn.csv and ingest will use it instead.
"""
import numpy as np
import pandas as pd

from churn.config import RAW_CSV
from churn.logger import get_logger

log = get_logger("generate")


def generate(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure = rng.integers(0, 73, n)
    tenure = np.where(contract == "Month-to-month", (tenure * 0.5).astype(int), tenure)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    base = np.select([internet == "No", internet == "DSL"], [20, 45], default=80)
    monthly = np.clip(base + rng.normal(0, 8, n), 18, 120).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, n)).round(2)

    def svc():
        v = rng.choice(["Yes", "No"], n, p=[0.4, 0.6]).astype(object)
        v[internet == "No"] = "No internet service"
        return v

    online_security, tech_support = svc(), svc()
    payment = rng.choice(["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"],
                         n, p=[0.34, 0.23, 0.22, 0.21])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])

    logit = (-1.4 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
             - 0.03 * tenure + 0.012 * (monthly - 65) + 0.6 * (internet == "Fiber optic")
             + 0.5 * (payment == "Electronic check") - 0.5 * (tech_support == "Yes")
             - 0.4 * (online_security == "Yes") + 0.2 * senior + 0.2 * (paperless == "Yes"))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    yn = lambda p: rng.choice(["Yes", "No"], n, p=[p, 1 - p])
    df = pd.DataFrame({
        "customerID": [f"C{100000 + i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": senior,
        "Partner": yn(0.48), "Dependents": yn(0.30),
        "tenure": tenure,
        "PhoneService": yn(0.90),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": svc(), "DeviceProtection": svc(),
        "TechSupport": tech_support,
        "StreamingTV": svc(), "StreamingMovies": svc(),
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(str),
        "Churn": churn,
    })
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "   # same quirk as the real dataset
    return df


if __name__ == "__main__":
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(RAW_CSV, index=False)
    log.info("Synthetic data written to %s", RAW_CSV)
