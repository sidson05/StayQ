-- ============================================================================
-- Data Warehouse DDL: Medallion Architecture (Raw / Staging / Analytics)
-- ============================================================================

-- 1. SCHEMAS
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

-- 2. RAW LAYER (Bronze)
CREATE TABLE IF NOT EXISTS raw.contracts (
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

CREATE TABLE IF NOT EXISTS raw.invoices (
    invoice_id VARCHAR(50),
    customer_id VARCHAR(50),
    invoice_date VARCHAR(20),
    amount NUMERIC(10, 2),
    payment_status VARCHAR(50),
    payment_delay_days INT,
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw.support_tickets (
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

CREATE TABLE IF NOT EXISTS raw.usage_events (
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

-- 3. STAGING LAYER (Silver - Typed, Cleansed, Deduplicated)
CREATE TABLE IF NOT EXISTS staging.dim_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    contract_type VARCHAR(50) NOT NULL,
    tenure_months INT NOT NULL,
    internet_service VARCHAR(50),
    has_tech_support BOOLEAN,
    has_online_security BOOLEAN,
    has_streaming_tv BOOLEAN,
    payment_method VARCHAR(100),
    is_paperless_billing BOOLEAN,
    monthly_charges NUMERIC(10, 2) NOT NULL,
    total_charges NUMERIC(12, 2),
    signup_date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging.fct_invoices (
    invoice_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    invoice_date DATE NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    payment_status VARCHAR(50) NOT NULL,
    payment_delay_days INT DEFAULT 0,
    is_failed_payment BOOLEAN GENERATED ALWAYS AS (payment_status IN ('FAILED', 'DISPUTED')) STORED,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging.fct_support_tickets (
    ticket_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    issue_type VARCHAR(100),
    priority VARCHAR(20),
    status VARCHAR(50),
    resolution_time_hrs NUMERIC(8, 2),
    sentiment_score NUMERIC(4, 2),
    is_unresolved BOOLEAN GENERATED ALWAYS AS (status IN ('OPEN', 'ESCALATED')) STORED,
    is_escalated BOOLEAN GENERATED ALWAYS AS (status = 'ESCALATED') STORED,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staging.fct_usage_events (
    event_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    event_date DATE NOT NULL,
    data_usage_gb NUMERIC(10, 2) NOT NULL,
    login_count INT NOT NULL,
    streaming_hours NUMERIC(10, 2) NOT NULL,
    app_crashes INT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. ANALYTICS LAYER (Gold Marts for ML & BI)
CREATE TABLE IF NOT EXISTS analytics.mart_churn_features (
    customer_id VARCHAR(50) PRIMARY KEY,
    -- Customer Contract & Demographic Signals
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
    
    -- Invoicing & Billing Behavioral Signals (Last 6 Months)
    total_invoices INT,
    failed_invoices_count INT,
    failed_invoices_ratio NUMERIC(5, 4),
    avg_payment_delay_days NUMERIC(6, 2),
    max_payment_delay_days INT,
    
    -- Support Ticket Friction Signals (Last 90 Days)
    total_tickets INT,
    unresolved_tickets_count INT,
    escalated_tickets_count INT,
    critical_tickets_count INT,
    avg_ticket_sentiment NUMERIC(4, 2),
    has_cancellation_inquiry INT,
    
    -- App Usage & Telemetry Velocity Signals
    total_data_usage_gb NUMERIC(10, 2),
    avg_biweekly_usage_gb NUMERIC(10, 2),
    total_logins INT,
    recent_usage_30d_gb NUMERIC(10, 2),
    prior_usage_60d_gb NUMERIC(10, 2),
    usage_drop_velocity NUMERIC(6, 4), -- (Recent 30d / Prior 60d ratio)
    recent_logins_30d INT,
    total_crashes INT,
    recent_crashes_30d INT,
    
    -- Metadata
    feature_generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
