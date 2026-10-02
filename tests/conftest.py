import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["MODEL_PATH"] = f"{_tmp}/model.joblib"
os.environ["REPORTS_DIR"] = f"{_tmp}/reports"

import pytest  # noqa: E402

from churn.generate_data import generate  # noqa: E402
from churn.ingest import clean  # noqa: E402
from churn.train import train  # noqa: E402


@pytest.fixture(scope="session")
def trained_metrics():
    df = clean(generate(n=1500))
    return train(df, track=False)
