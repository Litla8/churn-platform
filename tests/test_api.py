from fastapi.testclient import TestClient

PAYLOAD = {"tenure": 3, "monthly_charges": 95.5, "total_charges": 286.5,
           "contract": "Month-to-month", "internet_service": "Fiber optic",
           "payment_method": "Electronic check", "customer_id": "TEST-1"}


def test_model_quality(trained_metrics):
    assert trained_metrics["test_roc_auc"] > 0.70


def test_health_and_predict(trained_metrics):
    from churn.api import app
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/predict", json=PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_band"] in {"Low", "Medium", "High"}


def test_validation_error(trained_metrics):
    from churn.api import app
    with TestClient(app) as client:
        bad = dict(PAYLOAD, contract="Weekly")
        assert client.post("/predict", json=bad).status_code == 422


def test_risky_customer_scores_higher_than_loyal(trained_metrics):
    from churn.api import app
    loyal = dict(PAYLOAD, tenure=60, total_charges=3000, contract="Two year",
                 internet_service="DSL", payment_method="Credit card (automatic)",
                 monthly_charges=50, tech_support="Yes")
    with TestClient(app) as client:
        risky_p = client.post("/predict", json=PAYLOAD).json()["churn_probability"]
        loyal_p = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky_p > loyal_p
