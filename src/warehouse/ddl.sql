-- ============================================================================
-- Data Warehouse DDL: Unified Medallion Architecture (PostgreSQL & SQLite)
-- ============================================================================

-- 1. BASE TABLES (Directly queryable across both PostgreSQL and SQLite)

CREATE TABLE IF NOT EXISTS raw_contracts (
    customer_id VARCHAR(50),
    contract_type VARCHAR(50),
    tenure_months INT,
    internet_service VARCHAR(50),
    tech_support VARCHAR(50),
    online_security VARCHAR(50),
    streaming_tv VARCHAR(50),
    payment_method VARCHAR(100),
    paperless_billing VARCHAR(10),
    monthly_charges NUMERIC(10, 2),
    total_charges NUMERIC(12, 2),
    signup_date VARCHAR(20),
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_invoices (
    invoice_id VARCHAR(50),
    customer_id VARCHAR(50),
    invoice_date VARCHAR(20),
    amount NUMERIC(10, 2),
    payment_status VARCHAR(50),
    payment_delay_days INT,
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_support_tickets (
    ticket_id VARCHAR(50),
    customer_id VARCHAR(50),
    created_at VARCHAR(30),
    issue_type VARCHAR(100),
    priority VARCHAR(20),
    status VARCHAR(50),
    resolution_time_hrs NUMERIC(8, 2),
    sentiment_score NUMERIC(4, 2),
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_usage_events (
    event_id VARCHAR(50),
    customer_id VARCHAR(50),
    event_date VARCHAR(20),
    period_window_days INT,
    data_usage_gb NUMERIC(10, 2),
    login_count INT,
    streaming_hours NUMERIC(10, 2),
    app_crashes INT,
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging_dim_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    contract_type VARCHAR(50) NOT NULL,
    tenure_months INT NOT NULL,
    internet_service VARCHAR(50),
    has_tech_support INT,
    has_online_security INT,
    has_streaming_tv INT,
    payment_method VARCHAR(100),
    is_paperless_billing INT,
    monthly_charges NUMERIC(10, 2) NOT NULL,
    total_charges NUMERIC(12, 2),
    signup_date VARCHAR(20),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging_fct_invoices (
    invoice_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    invoice_date VARCHAR(20) NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    payment_status VARCHAR(50) NOT NULL,
    payment_delay_days INT DEFAULT 0,
    is_failed_payment INT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging_fct_support_tickets (
    ticket_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    created_at VARCHAR(30) NOT NULL,
    issue_type VARCHAR(100),
    priority VARCHAR(20),
    status VARCHAR(50),
    resolution_time_hrs NUMERIC(8, 2),
    sentiment_score NUMERIC(4, 2),
    is_unresolved INT,
    is_escalated INT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging_fct_usage_events (
    event_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    event_date VARCHAR(20) NOT NULL,
    data_usage_gb NUMERIC(10, 2) NOT NULL,
    login_count INT NOT NULL,
    streaming_hours NUMERIC(10, 2) NOT NULL,
    app_crashes INT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics_mart_churn_features (
    customer_id VARCHAR(50) PRIMARY KEY,
    contract_type VARCHAR(50),
    is_month_to_month INT,
    tenure_months INT,
    internet_service VARCHAR(50),
    has_tech_support INT,
    has_online_security INT,
    has_streaming_tv INT,
    monthly_charges NUMERIC(10, 2),
    total_charges NUMERIC(12, 2),
    is_electronic_check INT,
    is_paperless_billing INT,
    total_invoices INT,
    failed_invoices_count INT,
    failed_invoices_ratio NUMERIC(5, 4),
    avg_payment_delay_days NUMERIC(6, 2),
    max_payment_delay_days INT,
    total_tickets INT,
    unresolved_tickets_count INT,
    escalated_tickets_count INT,
    critical_tickets_count INT,
    avg_ticket_sentiment NUMERIC(4, 2),
    has_cancellation_inquiry INT,
    total_data_usage_gb NUMERIC(10, 2),
    avg_biweekly_usage_gb NUMERIC(10, 2),
    total_logins INT,
    recent_usage_30d_gb NUMERIC(10, 2),
    prior_usage_60d_gb NUMERIC(10, 2),
    usage_drop_velocity NUMERIC(6, 4),
    recent_logins_30d INT,
    total_crashes INT,
    recent_crashes_30d INT,
    feature_generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. MEDALLION SCHEMAS & VIEWS (For PostgreSQL Architecture Presentation)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

DROP TABLE IF EXISTS raw.contracts CASCADE;
DROP VIEW IF EXISTS raw.contracts CASCADE;
CREATE VIEW raw.contracts AS SELECT * FROM raw_contracts;

DROP TABLE IF EXISTS raw.invoices CASCADE;
DROP VIEW IF EXISTS raw.invoices CASCADE;
CREATE VIEW raw.invoices AS SELECT * FROM raw_invoices;

DROP TABLE IF EXISTS raw.support_tickets CASCADE;
DROP VIEW IF EXISTS raw.support_tickets CASCADE;
CREATE VIEW raw.support_tickets AS SELECT * FROM raw_support_tickets;

DROP TABLE IF EXISTS raw.usage_events CASCADE;
DROP VIEW IF EXISTS raw.usage_events CASCADE;
CREATE VIEW raw.usage_events AS SELECT * FROM raw_usage_events;

DROP TABLE IF EXISTS staging.dim_customers CASCADE;
DROP VIEW IF EXISTS staging.dim_customers CASCADE;
CREATE VIEW staging.dim_customers AS SELECT * FROM staging_dim_customers;

DROP TABLE IF EXISTS staging.fct_invoices CASCADE;
DROP VIEW IF EXISTS staging.fct_invoices CASCADE;
CREATE VIEW staging.fct_invoices AS SELECT * FROM staging_fct_invoices;

DROP TABLE IF EXISTS staging.fct_support_tickets CASCADE;
DROP VIEW IF EXISTS staging.fct_support_tickets CASCADE;
CREATE VIEW staging.fct_support_tickets AS SELECT * FROM staging_fct_support_tickets;

DROP TABLE IF EXISTS staging.fct_usage_events CASCADE;
DROP VIEW IF EXISTS staging.fct_usage_events CASCADE;
CREATE VIEW staging.fct_usage_events AS SELECT * FROM staging_fct_usage_events;

DROP TABLE IF EXISTS analytics.mart_churn_features CASCADE;
DROP VIEW IF EXISTS analytics.mart_churn_features CASCADE;
CREATE VIEW analytics.mart_churn_features AS SELECT * FROM analytics_mart_churn_features;
