-- Reference schema for MySQL (the Python code creates the same tables automatically).
CREATE DATABASE IF NOT EXISTS churn_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE churn_db;

CREATE TABLE IF NOT EXISTS customers (
    customer_id       VARCHAR(20) PRIMARY KEY,
    gender            VARCHAR(10),
    senior_citizen    INT,
    partner           VARCHAR(3),
    dependents        VARCHAR(3),
    tenure            INT,
    phone_service     VARCHAR(3),
    multiple_lines    VARCHAR(30),
    internet_service  VARCHAR(20),
    online_security   VARCHAR(30),
    online_backup     VARCHAR(30),
    device_protection VARCHAR(30),
    tech_support      VARCHAR(30),
    streaming_tv      VARCHAR(30),
    streaming_movies  VARCHAR(30),
    contract          VARCHAR(20),
    paperless_billing VARCHAR(3),
    payment_method    VARCHAR(40),
    monthly_charges   DOUBLE,
    total_charges     DOUBLE,
    churn             VARCHAR(3),
    INDEX idx_customers_contract (contract),
    INDEX idx_customers_churn (churn)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS prediction_log (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    customer_id       VARCHAR(20),
    churn_probability DOUBLE,
    risk_band         VARCHAR(10),
    model_version     VARCHAR(50)
) ENGINE=InnoDB;
