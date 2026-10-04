"""Database layer (MySQL in production, SQLite allowed for quick tests)."""
from functools import lru_cache

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, Index, Integer, MetaData, String,
                        Table, create_engine, text)
from sqlalchemy.engine import Engine, make_url

from churn.config import get_database_url
from churn.logger import get_logger

log = get_logger("db")
metadata = MetaData()

customers = Table(
    "customers", metadata,
    Column("customer_id", String(20), primary_key=True),
    Column("gender", String(10)),
    Column("senior_citizen", Integer),
    Column("partner", String(3)),
    Column("dependents", String(3)),
    Column("tenure", Integer),
    Column("phone_service", String(3)),
    Column("multiple_lines", String(30)),
    Column("internet_service", String(20)),
    Column("online_security", String(30)),
    Column("online_backup", String(30)),
    Column("device_protection", String(30)),
    Column("tech_support", String(30)),
    Column("streaming_tv", String(30)),
    Column("streaming_movies", String(30)),
    Column("contract", String(20)),
    Column("paperless_billing", String(3)),
    Column("payment_method", String(40)),
    Column("monthly_charges", Float),
    Column("total_charges", Float),
    Column("churn", String(3)),
    Index("idx_customers_contract", "contract"),
    Index("idx_customers_churn", "churn"),
)

# Batch-scoring output. Defined here (with a primary key) because managed MySQL
# services such as Aiven set sql_require_primary_key, which rejects the
# key-less tables that pandas.to_sql would otherwise create.
churn_scores = Table(
    "churn_scores", metadata,
    Column("customer_id", String(20), primary_key=True),
    Column("contract", String(20)),
    Column("monthly_charges", Float),
    Column("tenure", Integer),
    Column("churn_probability", Float),
    Column("will_churn", Integer),
    Column("risk_band", String(10)),
    Column("model_version", String(50)),
)

# KMeans segmentation output (same primary-key reasoning as churn_scores).
customer_segments = Table(
    "customer_segments", metadata,
    Column("customer_id", String(20), primary_key=True),
    Column("segment_id", Integer),
    Column("segment_label", String(60)),
)

prediction_log = Table(
    "prediction_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
    Column("customer_id", String(20)),
    Column("churn_probability", Float),
    Column("risk_band", String(10)),
    Column("model_version", String(50)),
)


def create_database_if_missing() -> None:
    """MySQL only: create the database itself (tables are created separately)."""
    url = make_url(get_database_url())
    if not url.get_backend_name().startswith("mysql"):
        return
    server_engine = create_engine(url.set(database=None))
    with server_engine.begin() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{url.database}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    server_engine.dispose()
    log.info("Database '%s' ready", url.database)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_database_url(), pool_pre_ping=True, pool_recycle=3600)


def init_db() -> None:
    create_database_if_missing()
    metadata.create_all(get_engine())


def replace_table(df: pd.DataFrame, table: Table, chunksize: int = 500) -> None:
    """Drop and recreate `table` (keeping its primary key), then bulk-insert `df`.

    Keeps batch jobs idempotent: re-running always leaves exactly the latest rows.
    """
    engine = get_engine()
    metadata.drop_all(engine, tables=[table])
    metadata.create_all(engine, tables=[table])
    df.to_sql(table.name, engine, if_exists="append", index=False,
              chunksize=chunksize, method="multi")
