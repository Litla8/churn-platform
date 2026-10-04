"""Unsupervised customer segmentation with KMeans -> table `customer_segments`.

Run:  python -m churn.segment
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from churn.config import REPORTS_DIR
from churn.db import customer_segments, replace_table
from churn.features import load_customers
from churn.logger import get_logger

log = get_logger("segment")
COLS = ["tenure", "monthly_charges", "total_charges"]


def run(k_range: range = range(2, 7)) -> pd.DataFrame:
    df = load_customers()
    data = df[COLS].fillna(0)
    X = StandardScaler().fit_transform(data)

    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        scores[k] = silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42)
        log.info("k=%d silhouette=%.3f", k, scores[k])
    best_k = max(scores, key=scores.get)
    log.info("Chosen k = %d", best_k)

    df["segment_id"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
    prof = df.groupby("segment_id").agg(
        customers=("customer_id", "count"), avg_tenure=("tenure", "mean"),
        avg_monthly=("monthly_charges", "mean"),
        churn_rate=("churn", lambda s: (s == "Yes").mean())).round(2)

    med_t, med_m = df["tenure"].median(), df["monthly_charges"].median()
    labels = {i: f"{'Long' if r.avg_tenure >= med_t else 'Short'}-tenure / "
                 f"{'High' if r.avg_monthly >= med_m else 'Low'}-bill (#{i})"
              for i, r in prof.iterrows()}
    df["segment_label"] = df["segment_id"].map(labels)
    prof["segment_label"] = prof.index.map(labels)

    replace_table(df[["customer_id", "segment_id", "segment_label"]], customer_segments)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prof.to_csv(REPORTS_DIR / "segment_profile.csv")
    print(prof.to_string())
    return prof


if __name__ == "__main__":
    run()