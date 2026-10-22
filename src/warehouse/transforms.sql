-- ============================================================================
-- SQL Transformations: Silver (Staging) & Gold (Analytics Mart)
-- ============================================================================

-- 1. STAGING TRANSFORMATIONS (Deduplication, Type Casting, Value Normalization)

-- Staging Customers
INSERT OR IGNORE INTO staging_dim_customers (
    customer_id, contract_type, tenure_months, internet_service,
    has_tech_support, has_online_security, has_streaming_tv,
    payment_method, is_paperless_billing, monthly_charges,
    total_charges, signup_date
)
SELECT 
    customer_id,
    contract_type,
    CAST(tenure_months AS INT),
    internet_service,
    CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END,
    CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END,
    CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END,
    payment_method,
    CASE WHEN paperless_billing = 'Yes' THEN 1 ELSE 0 END,
    CAST(monthly_charges AS NUMERIC(10, 2)),
    CAST(total_charges AS NUMERIC(12, 2)),
    signup_date
FROM raw_contracts
WHERE customer_id IS NOT NULL
GROUP BY customer_id;

-- Staging Invoices
INSERT OR IGNORE INTO staging_fct_invoices (
    invoice_id, customer_id, invoice_date, amount,
    payment_status, payment_delay_days, is_failed_payment
)
SELECT 
    invoice_id,
    customer_id,
    invoice_date,
    CAST(amount AS NUMERIC(10, 2)),
    payment_status,
    CAST(COALESCE(payment_delay_days, 0) AS INT),
    CASE WHEN payment_status IN ('FAILED', 'DISPUTED') THEN 1 ELSE 0 END
FROM raw_invoices
WHERE invoice_id IS NOT NULL
GROUP BY invoice_id;

-- Staging Support Tickets
INSERT OR IGNORE INTO staging_fct_support_tickets (
    ticket_id, customer_id, created_at, issue_type,
    priority, status, resolution_time_hrs, sentiment_score,
    is_unresolved, is_escalated
)
SELECT 
    ticket_id,
    customer_id,
    created_at,
    issue_type,
    priority,
    status,
    CAST(resolution_time_hrs AS NUMERIC(8, 2)),
    CAST(sentiment_score AS NUMERIC(4, 2)),
    CASE WHEN status IN ('OPEN', 'ESCALATED') THEN 1 ELSE 0 END,
    CASE WHEN status = 'ESCALATED' THEN 1 ELSE 0 END
FROM raw_support_tickets
WHERE ticket_id IS NOT NULL
GROUP BY ticket_id;

-- Staging Usage Events
INSERT OR IGNORE INTO staging_fct_usage_events (
    event_id, customer_id, event_date,
    data_usage_gb, login_count, streaming_hours, app_crashes
)
SELECT 
    event_id,
    customer_id,
    event_date,
    CAST(data_usage_gb AS NUMERIC(10, 2)),
    CAST(login_count AS INT),
    CAST(streaming_hours AS NUMERIC(10, 2)),
    CAST(COALESCE(app_crashes, 0) AS INT)
FROM raw_usage_events
WHERE event_id IS NOT NULL
GROUP BY event_id;

-- 2. GOLD ANALYTICS CHURN FEATURE MART
-- Aggregates customer profile, 6-month billing history, 90-day ticket friction, and app usage drop velocity
INSERT INTO analytics_mart_churn_features (
    customer_id,
    contract_type,
    is_month_to_month,
    tenure_months,
    internet_service,
    has_tech_support,
    has_online_security,
    has_streaming_tv,
    monthly_charges,
    total_charges,
    is_electronic_check,
    is_paperless_billing,
    total_invoices,
    failed_invoices_count,
    failed_invoices_ratio,
    avg_payment_delay_days,
    max_payment_delay_days,
    total_tickets,
    unresolved_tickets_count,
    escalated_tickets_count,
    critical_tickets_count,
    avg_ticket_sentiment,
    has_cancellation_inquiry,
    total_data_usage_gb,
    avg_biweekly_usage_gb,
    total_logins,
    recent_usage_30d_gb,
    prior_usage_60d_gb,
    usage_drop_velocity,
    recent_logins_30d,
    total_crashes,
    recent_crashes_30d
)
WITH invoice_agg AS (
    SELECT 
        customer_id,
        COUNT(invoice_id) AS total_invoices,
        SUM(CASE WHEN is_failed_payment = 1 THEN 1 ELSE 0 END) AS failed_invoices_count,
        ROUND(CAST(SUM(CASE WHEN is_failed_payment = 1 THEN 1 ELSE 0 END) AS NUMERIC) / NULLIF(COUNT(invoice_id), 0), 4) AS failed_invoices_ratio,
        ROUND(AVG(payment_delay_days), 2) AS avg_payment_delay_days,
        MAX(payment_delay_days) AS max_payment_delay_days
    FROM staging_fct_invoices
    GROUP BY customer_id
),
ticket_agg AS (
    SELECT 
        customer_id,
        COUNT(ticket_id) AS total_tickets,
        SUM(CASE WHEN is_unresolved = 1 THEN 1 ELSE 0 END) AS unresolved_tickets_count,
        SUM(CASE WHEN is_escalated = 1 THEN 1 ELSE 0 END) AS escalated_tickets_count,
        SUM(CASE WHEN priority = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_tickets_count,
        ROUND(AVG(sentiment_score), 2) AS avg_ticket_sentiment,
        MAX(CASE WHEN issue_type = 'Cancellation Request' THEN 1 ELSE 0 END) AS has_cancellation_inquiry
    FROM staging_fct_support_tickets
    GROUP BY customer_id
),
usage_agg AS (
    SELECT 
        customer_id,
        ROUND(SUM(data_usage_gb), 2) AS total_data_usage_gb,
        ROUND(AVG(data_usage_gb), 2) AS avg_biweekly_usage_gb,
        SUM(login_count) AS total_logins,
        -- Last 30 days is recent 2 event periods
        ROUND(SUM(CASE WHEN event_date >= '2026-09-01' THEN data_usage_gb ELSE 0 END), 2) AS recent_usage_30d_gb,
        ROUND(SUM(CASE WHEN event_date < '2026-09-01' THEN data_usage_gb ELSE 0 END), 2) AS prior_usage_60d_gb,
        SUM(CASE WHEN event_date >= '2026-09-01' THEN login_count ELSE 0 END) AS recent_logins_30d,
        SUM(app_crashes) AS total_crashes,
        SUM(CASE WHEN event_date >= '2026-09-01' THEN app_crashes ELSE 0 END) AS recent_crashes_30d
    FROM staging_fct_usage_events
    GROUP BY customer_id
)
SELECT 
    c.customer_id,
    c.contract_type,
    CASE WHEN c.contract_type = 'Month-to-month' THEN 1 ELSE 0 END AS is_month_to_month,
    c.tenure_months,
    c.internet_service,
    c.has_tech_support,
    c.has_online_security,
    c.has_streaming_tv,
    c.monthly_charges,
    c.total_charges,
    CASE WHEN c.payment_method = 'Electronic check' THEN 1 ELSE 0 END AS is_electronic_check,
    c.is_paperless_billing,
    COALESCE(i.total_invoices, 0) AS total_invoices,
    COALESCE(i.failed_invoices_count, 0) AS failed_invoices_count,
    COALESCE(i.failed_invoices_ratio, 0.0) AS failed_invoices_ratio,
    COALESCE(i.avg_payment_delay_days, 0.0) AS avg_payment_delay_days,
    COALESCE(i.max_payment_delay_days, 0) AS max_payment_delay_days,
    COALESCE(t.total_tickets, 0) AS total_tickets,
    COALESCE(t.unresolved_tickets_count, 0) AS unresolved_tickets_count,
    COALESCE(t.escalated_tickets_count, 0) AS escalated_tickets_count,
    COALESCE(t.critical_tickets_count, 0) AS critical_tickets_count,
    COALESCE(t.avg_ticket_sentiment, 0.0) AS avg_ticket_sentiment,
    COALESCE(t.has_cancellation_inquiry, 0) AS has_cancellation_inquiry,
    COALESCE(u.total_data_usage_gb, 0.0) AS total_data_usage_gb,
    COALESCE(u.avg_biweekly_usage_gb, 0.0) AS avg_biweekly_usage_gb,
    COALESCE(u.total_logins, 0) AS total_logins,
    COALESCE(u.recent_usage_30d_gb, 0.0) AS recent_usage_30d_gb,
    COALESCE(u.prior_usage_60d_gb, 0.0) AS prior_usage_60d_gb,
    -- Behavioral drop velocity: (Recent 30d usage / (Prior 60d usage / 2 + 0.1))
    ROUND(
        COALESCE(u.recent_usage_30d_gb, 0.0) / 
        NULLIF(COALESCE(u.prior_usage_60d_gb / 2.0, 1.0), 0.0), 
        4
    ) AS usage_drop_velocity,
    COALESCE(u.recent_logins_30d, 0) AS recent_logins_30d,
    COALESCE(u.total_crashes, 0) AS total_crashes,
    COALESCE(u.recent_crashes_30d, 0) AS recent_crashes_30d
FROM staging_dim_customers c
LEFT JOIN invoice_agg i ON c.customer_id = i.customer_id
LEFT JOIN ticket_agg t ON c.customer_id = t.customer_id
LEFT JOIN usage_agg u ON c.customer_id = u.customer_id;
