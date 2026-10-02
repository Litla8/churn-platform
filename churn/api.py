"""FastAPI service.  Run:  uvicorn churn.api:app --reload   ->  http://127.0.0.1:8000/docs"""
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import insert

from churn.config import MODEL_PATH
from churn.db import get_engine, init_db, prediction_log
from churn.logger import get_logger
from churn.predict import ChurnModel

log = get_logger("api")
YN = Literal["Yes", "No"]
SVC = Literal["Yes", "No", "No internet service"]


class CustomerIn(BaseModel):
    customer_id: Optional[str] = None
    tenure: int = Field(ge=0, le=100, description="Months with company")
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    senior_citizen: Literal[0, 1] = 0
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"]
    paperless_billing: YN = "Yes"
    partner: YN = "No"
    dependents: YN = "No"
    tech_support: SVC = "No"
    online_security: SVC = "No"


class PredictionOut(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    will_churn: bool
    risk_band: str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    try:
        app.state.model = ChurnModel(MODEL_PATH)
        log.info("Model loaded: %s", app.state.model.version)
    except FileNotFoundError:
        log.error("Model file not found at %s. Run: python -m churn.train", MODEL_PATH)
    try:
        init_db()
    except Exception as exc:  # API must still start if DB is down
        log.warning("DB unavailable, prediction logging disabled: %s", exc)
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


def _get_model() -> ChurnModel:
    if app.state.model is None:
        raise HTTPException(503, "Model not loaded. Train the model first.")
    return app.state.model


def _log_predictions(rows: List[dict]) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(insert(prediction_log), rows)
    except Exception as exc:
        log.warning("Could not log predictions: %s", exc)


@app.get("/health")
def health():
    model = app.state.model
    return {"status": "ok", "model_loaded": model is not None,
            "model_version": model.version if model else None}


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerIn):
    return predict_batch([customer])[0]


@app.post("/predict/batch", response_model=List[PredictionOut])
def predict_batch(customers: List[CustomerIn]):
    model = _get_model()
    df = pd.DataFrame([c.model_dump() for c in customers])
    scores = model.predict_df(df)
    out = [PredictionOut(customer_id=c.customer_id,
                         churn_probability=float(s.churn_probability),
                         will_churn=bool(s.will_churn), risk_band=s.risk_band,
                         model_version=model.version)
           for c, (_, s) in zip(customers, scores.iterrows())]
    _log_predictions([{"customer_id": o.customer_id, "churn_probability": o.churn_probability,
                       "risk_band": o.risk_band, "model_version": o.model_version} for o in out])
    return out
