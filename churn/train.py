"""Train + compare models, pick the best, tune threshold, save artifacts.

Run:  python -m churn.train
Then: mlflow ui --backend-store-uri sqlite:///mlflow.db   (open http://127.0.0.1:5000)
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import MLFLOW_URI, MODEL_PATH, REPORTS_DIR
from churn.features import build_preprocessor, get_xy, load_customers
from churn.logger import get_logger

log = get_logger("train")
SEED = 42


def build_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED),
    }


def best_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1 (validation data only)."""
    grid = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    values = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    return (pd.DataFrame({"feature": names, "importance": values})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def train(df: pd.DataFrame, model_path: Path = MODEL_PATH,
          reports_dir: Path = REPORTS_DIR, track: bool = True) -> dict:
    X, y = get_xy(df)
    # 70 / 15 / 15  (train / validation / test), stratified
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEED)
    log.info("Rows -> train %d | val %d | test %d | churn rate %.1f%%",
             len(X_train), len(X_val), len(X_test), 100 * y.mean())

    mlflow = None
    if track:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("telco-churn")

    results, fitted = [], {}
    for name, clf in build_models().items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        fitted[name] = pipe
        results.append({"model": name, "val_roc_auc": round(val_auc, 4)})
        log.info("%-20s val ROC-AUC = %.4f", name, val_auc)
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_params({f"hp_{k}": v for k, v in clf.get_params().items()
                                   if isinstance(v, (int, float, str, bool)) and v is not None})
                mlflow.log_metric("val_roc_auc", val_auc)

    best_name = max(results, key=lambda r: r["val_roc_auc"])["model"]
    best = fitted[best_name]
    threshold = best_threshold(y_val, best.predict_proba(X_val)[:, 1])

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "best_model": best_name, "threshold": round(threshold, 2),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "test_pr_auc": round(average_precision_score(y_test, proba), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "comparison": results,
    }
    log.info("BEST=%s | test AUC=%.4f | precision=%.3f recall=%.3f",
             best_name, metrics["test_roc_auc"], metrics["test_precision"], metrics["test_recall"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fi = feature_importance(best)
    fi.to_csv(reports_dir / "feature_importance.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(fpr, tpr, label=f"{best_name} (AUC={metrics['test_roc_auc']})")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend()
    top = fi.head(10).iloc[::-1]
    ax[1].barh(top["feature"], top["importance"])
    ax[1].set_title("Top 10 features")
    fig.tight_layout()
    fig.savefig(reports_dir / "evaluation.png", dpi=120)
    plt.close(fig)

    bundle = {"pipeline": best, "threshold": threshold, "metrics": metrics,
              "version": f"{best_name}-{datetime.now():%Y%m%d%H%M}"}
    joblib.dump(bundle, model_path)
    log.info("Model saved -> %s", model_path)

    if mlflow:
        with mlflow.start_run(run_name=f"BEST-{best_name}"):
            mlflow.log_metrics({k: v for k, v in metrics.items() if k.startswith("test_")})
            mlflow.log_artifact(str(reports_dir / "metrics.json"))
            mlflow.log_artifact(str(reports_dir / "evaluation.png"))
    return metrics


if __name__ == "__main__":
    train(load_customers())
