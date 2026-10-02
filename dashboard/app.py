"""Streamlit dashboard.  Run:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from churn.config import MODEL_PATH, REPORTS_DIR
from churn.db import get_engine
from churn.predict import ChurnModel

st.set_page_config(page_title="Churn Intelligence", layout="wide")
st.title("Telecom Customer Intelligence Platform")


@st.cache_data(ttl=300)
def q(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, get_engine())


@st.cache_resource
def load_model():
    return ChurnModel(MODEL_PATH)


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "High-risk customers", "Segments", "Predict"])

with tab1:
    kpi = q("""SELECT COUNT(*) AS customers,
                      ROUND(100*AVG(CASE WHEN churn='Yes' THEN 1 ELSE 0 END),1) AS churn_pct,
                      ROUND(AVG(monthly_charges),2) AS avg_bill FROM customers""").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{int(kpi.customers):,}")
    c2.metric("Churn rate", f"{kpi.churn_pct}%")
    c3.metric("Avg monthly bill", f"{kpi.avg_bill}")
    by_contract = q("""SELECT contract, COUNT(*) AS customers,
                       ROUND(100*AVG(CASE WHEN churn='Yes' THEN 1 ELSE 0 END),1) AS churn_pct
                       FROM customers GROUP BY contract ORDER BY churn_pct DESC""")
    st.plotly_chart(px.bar(by_contract, x="contract", y="churn_pct", text="churn_pct",
                           title="Churn % by contract type"), width="stretch")
    by_tenure = q("""SELECT CASE WHEN tenure<12 THEN '0-11' WHEN tenure<24 THEN '12-23'
                     WHEN tenure<48 THEN '24-47' ELSE '48+' END AS tenure_band,
                     ROUND(100*AVG(CASE WHEN churn='Yes' THEN 1 ELSE 0 END),1) AS churn_pct
                     FROM customers GROUP BY tenure_band""")
    st.plotly_chart(px.bar(by_tenure, x="tenure_band", y="churn_pct",
                           title="Churn % by tenure (months)",
                           category_orders={"tenure_band": ["0-11", "12-23", "24-47", "48+"]}),
                    width="stretch")                           
    metrics_img = REPORTS_DIR / "evaluation.png"
    if metrics_img.exists():
        st.image(str(metrics_img), caption="Model evaluation (test set)")

with tab2:
    try:
        risk = q("""SELECT customer_id, contract, tenure, monthly_charges,
                    churn_probability, risk_band FROM churn_scores
                    ORDER BY churn_probability DESC LIMIT 200""")
        band = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High"])
        st.dataframe(risk[risk.risk_band.isin(band)], width="stretch")
        st.download_button("Download CSV", risk.to_csv(index=False), "high_risk.csv")
    except Exception:
        st.warning("Run `python -m churn.predict` first to create the churn_scores table.")

with tab3:
    try:
        seg = q("""SELECT s.segment_label, COUNT(*) AS customers,
                   ROUND(AVG(c.monthly_charges),1) AS avg_bill, ROUND(AVG(c.tenure),1) AS avg_tenure,
                   ROUND(100*AVG(CASE WHEN c.churn='Yes' THEN 1 ELSE 0 END),1) AS churn_pct
                   FROM customer_segments s JOIN customers c USING (customer_id)
                   GROUP BY s.segment_label""")
        st.dataframe(seg, width="stretch")
        st.plotly_chart(px.scatter(seg, x="avg_tenure", y="avg_bill", size="customers",
                                   color="churn_pct", text="segment_label"), width="stretch")
    except Exception:
        st.warning("Run `python -m churn.segment` first.")

with tab4:
    st.subheader("Single customer prediction")
    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", 0, 72, 6)
    monthly = c2.number_input("Monthly charges", 0.0, 200.0, 85.0)
    contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = c1.selectbox("Internet", ["Fiber optic", "DSL", "No"])
    payment = c2.selectbox("Payment", ["Electronic check", "Mailed check",
                                       "Bank transfer (automatic)", "Credit card (automatic)"])
    support = c3.selectbox("Tech support", ["No", "Yes", "No internet service"])
    if st.button("Predict churn"):
        row = pd.DataFrame([{
            "tenure": tenure, "monthly_charges": monthly, "total_charges": monthly * tenure,
            "senior_citizen": 0, "contract": contract, "internet_service": internet,
            "payment_method": payment, "paperless_billing": "Yes", "partner": "No",
            "dependents": "No", "tech_support": support, "online_security": "No"}])
        try:
            r = load_model().predict_df(row).iloc[0]
            st.metric("Churn probability", f"{r.churn_probability:.1%}")
            (st.error if r.risk_band == "High" else st.warning if r.risk_band == "Medium"
             else st.success)(f"Risk: {r.risk_band}")
        except FileNotFoundError:
            st.error("Model not found. Run `python -m churn.train`.")
