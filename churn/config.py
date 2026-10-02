"""Central configuration. Values come from environment / .env file."""
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW_CSV = Path(os.getenv("RAW_CSV", ROOT / "data" / "raw" / "telco_churn.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "churn_model.joblib"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
MLFLOW_URI = os.getenv("MLFLOW_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def get_database_url() -> str:
    """Build the SQLAlchemy URL. DATABASE_URL wins, otherwise MySQL from MYSQL_* vars."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = quote_plus(os.getenv("MYSQL_USER", "root"))
    pwd = quote_plus(os.getenv("MYSQL_PASSWORD", ""))
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    db = os.getenv("MYSQL_DATABASE", "churn_db")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"
