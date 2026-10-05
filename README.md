# Telecom Customer Intelligence Platform

![CI](https://github.com/Litla8/churn-platform/actions/workflows/ci.yml/badge.svg)

End-to-end data science project: **MySQL -> ETL -> feature engineering -> model comparison (MLflow) -> customer segmentation -> batch scoring -> FastAPI -> Streamlit dashboard -> Docker -> CI -> cloud deployment**.

**Live demo:** [Streamlit dashboard](https://churn-platform-h9ypys68boc5n5zeempy2l.streamlit.app/)

> The demo reads from a free-tier cloud MySQL (Aiven) that powers off when idle. If the dashboard shows a database error, the database is asleep. The screenshots below show every tab.

## Business problem

A telecom company wants to know which customers are likely to leave (churn), how much recurring revenue is at risk, and which customer groups exist, so the retention team can contact the right people first.

## Results

Dataset: Kaggle "Telco Customer Churn", 7,043 customers, 26.5% churn rate. Split 70/15/15 (train/validation/test), stratified.

| Model | Validation ROC-AUC |
|---|---|
| Logistic Regression | 0.8351 |
| Random Forest | 0.8355 |
| XGBoost (selected) | 0.8378 |

Test set: ROC-AUC 0.8321 | Precision 0.517 | Recall 0.746 (decision threshold tuned on the validation set to maximise F1).

| Business finding | Result |
|---|---|
| Churn by contract | Month-to-month 42.7%, one-year 11.3%, two-year 2.8% |
| Customer segments (KMeans, k=2) | Short-tenure / low-bill: 31% churn. Long-tenure / high-bill: 17% churn |
| High-risk customers | 676 of 7,043 (9.6%) scored 0.70 or higher |
| Revenue already lost | 139,130.85 in monthly recurring revenue from churned customers (no currency in the dataset) |

## Architecture

```
CSV / Kaggle  --ingest-->  MySQL (customers)
                              |
             features.py -> train.py (LogReg / RF / XGBoost, MLflow) -> models/churn_model.joblib
                              |                                            |
                 segment.py (KMeans) -> customer_segments        predict.py -> churn_scores
                              |                                            |
                              +------------- Streamlit dashboard ----------+
                                         FastAPI  (/predict, logs to prediction_log)
```

## Tech stack

Python, Pandas, NumPy, SQLAlchemy, **MySQL**, scikit-learn, XGBoost, KMeans, MLflow, FastAPI, Pydantic, Streamlit, Plotly, pytest, Docker, GitHub Actions, Aiven (managed MySQL), Streamlit Community Cloud.

## Screenshots

### Dashboard (Streamlit)

![Dashboard overview](docs/screenshots/streamlit-app-overview-tab.png)
![Dashboard charts and model evaluation](docs/screenshots/streamlit-app-overview-tab1.png)
![High-risk customers tab](docs/screenshots/Telecom-Intelligence-Customer-Platform-high-risk-customers-tab.png)
![Segments tab](docs/screenshots/Telecom-Intelligence-Customer-Platform-segments-tab.png)
![Single-customer prediction tab](docs/screenshots/streamlit-app-predict-tab.png)

### Experiment tracking (MLflow)

![MLflow runs](docs/screenshots/runs-page.png)
![MLflow best model run](docs/screenshots/BEST-xgboost.png)
![MLflow experiment overview](docs/screenshots/overview1.png)
![MLflow experiment comparison](docs/screenshots/overview2.png)

### API (FastAPI)

![API health check](docs/screenshots/model-version.png)

### MySQL tables

![customers table](docs/screenshots/customers.png)
![churn_scores table](docs/screenshots/churn-scores.png)
![customer_segments table](docs/screenshots/customer-segments.png)
![prediction_log table](docs/screenshots/prediction-log.png)

## SQL analysis (`sql/analysis_queries.sql`)

1. **Churn by contract:** month-to-month customers churn at 42.7% (3,875 customers), versus 11.3% on one-year and 2.8% on two-year contracts, so contract length is the strongest churn signal.

   ![Churn by contract](docs/screenshots/1-Churn-percentage-by-contract.png)

2. **Revenue at risk:** customers who churned represent 139,130.85 in lost monthly recurring revenue (about 1.67M annualised).

   ![Monthly revenue of churned customers](docs/screenshots/2-Monthly-revenue-of-churned-customers.png)

3. **Retention call list:** the 20 highest-value high-risk customers are all month-to-month, pay 104.65 to 110.10 per month, and have churn probabilities of 0.71 to 0.91.

   ![Top 20 high-risk, high-value customers](docs/screenshots/3-Top-20-high-risk-high-value-customers.png)

4. **Bill ranking (window function):** `RANK() OVER (PARTITION BY contract ORDER BY monthly_charges DESC)` ranks all 7,043 customers within their contract type; equal bills share a rank.

   ![RANK window function by contract](docs/screenshots/4-rank-window-function-by-contract.png)

5. **API monitoring:** `prediction_log` stores every API prediction (date, count, average churn probability) for monitoring.

   ![API usage by day](docs/screenshots/5-API-usage-by-day.png)

## Quick start (Windows)

1. Install Python 3.12, MySQL Server 8, VS Code and Git.
2. In the project folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then set MYSQL_PASSWORD in .env
```

3. Dataset: download "Telco Customer Churn" from Kaggle and save it as `data/raw/telco_churn.csv` (not committed to Git). Without it, the code generates synthetic data with the same schema.

Run in order:

```bash
python -m churn.ingest      # CSV -> MySQL
python -m churn.train       # compare models, save the best, write reports/
python -m churn.segment     # KMeans segments -> MySQL
python -m churn.predict     # score all customers -> MySQL
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
uvicorn churn.api:app --reload                        # http://127.0.0.1:8000/docs
streamlit run dashboard/app.py                        # http://localhost:8501
pytest                                                # 7 tests
```

Test the API from PowerShell:

```powershell
$body = '{"tenure":3,"monthly_charges":95.5,"total_charges":286.5,"contract":"Month-to-month","internet_service":"Fiber optic","payment_method":"Electronic check"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

## Docker (MySQL + API + dashboard)

```bash
python -m churn.train           # the model file in models/ is copied into the image
docker compose up --build -d
docker compose run --rm api python -m churn.ingest
docker compose run --rm api python -m churn.segment
docker compose run --rm api python -m churn.predict
```

API: http://localhost:8000/docs | Dashboard: http://localhost:8501. Docker's MySQL is published on host port 3307 so it does not clash with a local MySQL on port 3306.

## Cloud deployment

- **Database:** Aiven for MySQL (free plan, MySQL 8.4). SSL is required, and the service enforces `sql_require_primary_key`, so every table (including the scoring and segment tables) is defined in `churn/db.py` with a primary key.
- **Dashboard:** Streamlit Community Cloud, deployed from this repository (`dashboard/app.py`). The connection string is stored as the `DATABASE_URL` secret and never committed. The Aiven CA certificate (public) is in `certs/`.
- **Connection:** `DATABASE_URL` overrides the local `MYSQL_*` settings:

```
mysql+pymysql://USER:PASSWORD@HOST:PORT/churn_db?ssl_ca=certs/aiven-ca.pem
```

- **Loading the cloud database** from a local terminal, with `DATABASE_URL` set for that window only:

```bash
set "DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:PORT/churn_db?ssl_ca=C:/path/to/aiven-ca.pem"
python -m churn.ingest
python -m churn.segment
python -m churn.predict
```

The FastAPI service runs locally and in Docker Compose; it is not hosted publicly.

## Key design decisions

- **Train / validation / test split (70/15/15):** the model and the decision threshold were chosen on validation data; the test set is used only for the final report, which avoids leakage.
- **Pipeline + ColumnTransformer:** preprocessing is saved with the model, so serving cannot drift from training.
- **One `add_features()`** is used by training, batch scoring and the API.
- **Threshold tuning:** 0.5 is not assumed; the threshold maximises F1 on validation data, and can be moved toward recall depending on business cost.
- **MySQL as the source of truth;** API predictions are stored in `prediction_log` for monitoring.
- **Tests and CI:** API contract, validation errors, and a sanity check (a risky customer scores higher than a loyal one) run on every push.

## Limitations and next steps

- The three models score within 0.003 ROC-AUC of each other, so the choice of XGBoost is a small margin.
- `is_month_to_month` and `contract_Month-to-month` encode the same information and split the importance; one should be dropped.
- KMeans with k=2 gives coarse segments (silhouette 0.480, versus 0.470 for k=4); k=4 may be more actionable.
- The dashboard's Predict tab fills fields it does not ask for with defaults, so its score can differ slightly from the batch score.
- Next: SHAP explanations, drift monitoring from `prediction_log`, hyperparameter tuning (Optuna), scheduled batch scoring, a limited-privilege database user for the hosted app.

## Project structure

```
churn/        config, db, ingest, features, train, predict, segment, api
dashboard/    Streamlit app
sql/          schema.sql, analysis_queries.sql
tests/        pytest suite (features, API)
notebooks/    EDA notebook
docs/         screenshots
certs/        Aiven CA certificate (public)
models/       trained model
reports/      metrics, charts
```






