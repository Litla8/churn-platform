USE churn_db;
-- 1. Churn % by contract
SELECT contract, COUNT(*) AS customers,
       ROUND(100*AVG(churn='Yes'),1) AS churn_pct
FROM customers GROUP BY contract ORDER BY churn_pct DESC;

-- 2. Revenue at risk: monthly revenue of customers who churned
SELECT ROUND(SUM(monthly_charges),2) AS monthly_revenue_lost
FROM customers WHERE churn='Yes';

-- 3. Top 20 high-risk, high-value customers (after python -m churn.predict)
SELECT customer_id, contract, monthly_charges, churn_probability
FROM churn_scores
WHERE risk_band='High'
ORDER BY monthly_charges DESC, churn_probability DESC
LIMIT 20;

-- 4. Window function: rank customers by bill inside each contract type
SELECT customer_id, contract, monthly_charges,
       RANK() OVER (PARTITION BY contract ORDER BY monthly_charges DESC) AS bill_rank
FROM customers;

-- 5. API usage monitoring
SELECT DATE(created_at) AS day, COUNT(*) AS predictions,
       ROUND(AVG(churn_probability),3) AS avg_prob
FROM prediction_log GROUP BY DATE(created_at);
