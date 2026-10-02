import pandas as pd

from churn.features import ALL_FEATURES, add_features, get_xy
from churn.generate_data import generate
from churn.ingest import clean


def test_clean_converts_total_charges():
    df = clean(generate(n=300))
    assert pd.api.types.is_float_dtype(df["total_charges"])
    assert df["customer_id"].is_unique


def test_add_features_no_crash_on_zero_tenure():
    df = clean(generate(n=300))
    out = add_features(df)
    assert "avg_monthly_spend" in out.columns
    assert not (out["avg_monthly_spend"] == float("inf")).any()


def test_get_xy_shapes():
    X, y = get_xy(clean(generate(n=300)))
    assert list(X.columns) == ALL_FEATURES
    assert set(y.unique()) <= {0, 1}
