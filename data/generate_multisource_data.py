"""
Realistic Multi-Source Data Generator for Subscription Churn Pipeline.
Splits Telco customer churn domain data into three independent data sources:
1. Billing System (CSV files: contracts.csv, invoices.csv)
2. Customer Support Ticketing API (JSON store for Mock REST API)
3. App Usage & Telemetry Logs (JSONL event stream with usage drop-off dynamics)
"""

import os
import json
import uuid
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set deterministic random seed
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR_BILLING = os.path.join("data", "raw", "billing")
OUTPUT_DIR_USAGE = os.path.join("data", "raw", "usage")
OUTPUT_DIR_TICKETS = os.path.join("services", "mock_ticket_api")

os.makedirs(OUTPUT_DIR_BILLING, exist_ok=True)
os.makedirs(OUTPUT_DIR_USAGE, exist_ok=True)
os.makedirs(OUTPUT_DIR_TICKETS, exist_ok=True)

N_CUSTOMERS = 7043
REFERENCE_DATE = datetime(2026, 10, 1)


def generate_customer_profiles(n: int = N_CUSTOMERS):
    """Generate customer demographics, base subscription, and churn ground truth."""
    customer_ids = [f"CUST-{10000 + i:05d}" for i in range(n)]
    
    # Contract distributions
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.24, 0.21]
    )
    
    # Tenure in months (skewed toward new customers, long-tail loyal)
    tenure_months = np.random.exponential(scale=24, size=n).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)
    
    # Internet Service
    internet_services = np.random.choice(
        ["Fiber optic", "DSL", "No"],
        size=n,
        p=[0.44, 0.34, 0.22]
    )
    
    # Pricing dynamics
    base_charges = []
    for internet in internet_services:
        if internet == "Fiber optic":
            base = np.random.normal(75, 12)
        elif internet == "DSL":
            base = np.random.normal(55, 8)
        else:
            base = np.random.normal(25, 4)
        base_charges.append(round(max(18.5, base), 2))
    base_charges = np.array(base_charges)
    
    # Payment method & billing
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
        p=[0.34, 0.22, 0.22, 0.22]
    )
    paperless_billing = np.random.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    
    # Services add-ons
    online_security = np.where(internet_services == "No", "No internet service", np.random.choice(["Yes", "No"], size=n, p=[0.35, 0.65]))
    tech_support = np.where(internet_services == "No", "No internet service", np.random.choice(["Yes", "No"], size=n, p=[0.33, 0.67]))
    streaming_tv = np.where(internet_services == "No", "No internet service", np.random.choice(["Yes", "No"], size=n, p=[0.48, 0.52]))
    
    # Calculate churn probability based on genuine churn risk factors
    # Risk factors: Month-to-month, Fiber optic without tech support, Electronic check, short tenure
    logit = -1.8
    logit += np.where(contract_types == "Month-to-month", 1.25, -0.9)
    logit += np.where(internet_services == "Fiber optic", 0.65, -0.3)
    logit += np.where(payment_methods == "Electronic check", 0.70, -0.2)
    logit += np.where(tech_support == "No", 0.55, -0.4)
    logit += np.where(online_security == "No", 0.45, -0.3)
    logit += -0.045 * tenure_months  # longer tenure strongly decreases churn
    logit += 0.015 * (base_charges - 60) # price pressure
    
    churn_prob = 1 / (1 + np.exp(-logit))
    churn_30d = (np.random.rand(n) < churn_prob).astype(int)
    
    # Calculate total charges
    total_charges = np.round(base_charges * tenure_months * np.random.uniform(0.95, 1.05, size=n), 2)
    
    # Calculate signup date
    signup_dates = [
        (REFERENCE_DATE - timedelta(days=int(t * 30.4))).strftime("%Y-%m-%d")
        for t in tenure_months
    ]
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "contract_type": contract_types,
        "tenure_months": tenure_months,
        "internet_service": internet_services,
        "tech_support": tech_support,
        "online_security": online_security,
        "streaming_tv": streaming_tv,
        "payment_method": payment_methods,
        "paperless_billing": paperless_billing,
        "monthly_charges": base_charges,
        "total_charges": total_charges,
        "signup_date": signup_dates,
        "churn_30d": churn_30d
    })
    
    return df


def generate_billing_system(df_cust: pd.DataFrame):
    """Generate contracts.csv and invoices.csv (Billing System)."""
    contracts_file = os.path.join(OUTPUT_DIR_BILLING, "contracts.csv")
    invoices_file = os.path.join(OUTPUT_DIR_BILLING, "invoices.csv")
    
    # Contracts file (excludes direct target for realistic raw warehouse separation)
    contracts_df = df_cust[[
        "customer_id", "contract_type", "tenure_months", "internet_service",
        "tech_support", "online_security", "streaming_tv", "payment_method",
        "paperless_billing", "monthly_charges", "total_charges", "signup_date"
    ]].copy()
    
    contracts_df.to_csv(contracts_file, index=False)
    print(f"[Billing] Saved {len(contracts_df)} contracts to {contracts_file}")
    
    # Generate 6 months of invoice history per customer
    invoices = []
    for _, row in df_cust.iterrows():
        c_id = row["customer_id"]
        monthly = row["monthly_charges"]
        is_churn = row["churn_30d"]
        
        # 6 monthly invoices back from REFERENCE_DATE
        for m in range(6):
            inv_date = REFERENCE_DATE - timedelta(days=30 * (5 - m))
            inv_id = f"INV-{uuid.uuid4().hex[:10].upper()}"
            
            # Realistic probabilities with realistic overlap
            if is_churn and m >= 4:
                # 45% of churners experience billing friction
                status_probs = [0.55, 0.25, 0.12, 0.08]
            elif is_churn:
                status_probs = [0.75, 0.14, 0.07, 0.04]
            else:
                # 8% of healthy customers also have occasional payment glitches
                status_probs = [0.92, 0.05, 0.02, 0.01]
                
            status = np.random.choice(["PAID", "FAILED", "PENDING", "DISPUTED"], p=status_probs)
            delay_days = 0
            if status == "PAID":
                delay_days = int(np.random.exponential(scale=1.5 if not is_churn else 3.5))
            elif status in ["FAILED", "PENDING", "DISPUTED"]:
                delay_days = random.randint(5, 45)
                
            invoices.append({
                "invoice_id": inv_id,
                "customer_id": c_id,
                "invoice_date": inv_date.strftime("%Y-%m-%d"),
                "amount": round(monthly * random.uniform(0.98, 1.05), 2),
                "payment_status": status,
                "payment_delay_days": delay_days
            })
            
    invoices_df = pd.DataFrame(invoices)
    invoices_df.to_csv(invoices_file, index=False)
    print(f"[Billing] Saved {len(invoices_df)} invoices to {invoices_file}")
    return contracts_df, invoices_df


def generate_support_tickets_api(df_cust: pd.DataFrame):
    """Generate customer support tickets with realistic noise and sentiment."""
    tickets_file = os.path.join(OUTPUT_DIR_TICKETS, "tickets.json")
    
    tickets = []
    issue_types = ["Billing", "Technical Connection", "Slow Speeds", "Hardware/Router", "Cancellation Request", "General Inquiry"]
    
    for _, row in df_cust.iterrows():
        c_id = row["customer_id"]
        is_churn = row["churn_30d"]
        
        # 30% of churners are 'silent churners' (no tickets)
        # 70% submit tickets; healthy customers also submit tickets
        if is_churn and random.random() > 0.30:
            n_tickets = np.random.poisson(lam=2.1)
        else:
            n_tickets = np.random.poisson(lam=0.9)
            
        n_tickets = min(n_tickets, 6)
        
        for _ in range(n_tickets):
            t_id = f"TCK-{uuid.uuid4().hex[:12].upper()}"
            days_ago = random.randint(1, 90)
            created_at = (REFERENCE_DATE - timedelta(days=days_ago, hours=random.randint(1, 23))).strftime("%Y-%m-%d %H:%M:%S")
            
            issue = np.random.choice(issue_types)
            if is_churn:
                priority = np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], p=[0.2, 0.4, 0.25, 0.15])
                status = np.random.choice(["RESOLVED", "CLOSED", "OPEN", "ESCALATED"], p=[0.45, 0.25, 0.18, 0.12])
                sentiment = round(np.random.normal(loc=-0.25, scale=0.45), 2)
                resolution_hrs = round(float(np.random.exponential(scale=32)), 1)
            else:
                priority = np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], p=[0.45, 0.35, 0.15, 0.05])
                status = np.random.choice(["RESOLVED", "CLOSED", "OPEN"], p=[0.70, 0.22, 0.08])
                sentiment = round(np.random.normal(loc=0.20, scale=0.40), 2)
                resolution_hrs = round(float(np.random.exponential(scale=16)), 1)
                
            sentiment = max(-1.0, min(1.0, sentiment))
            if status in ["OPEN", "ESCALATED"]:
                resolution_hrs = None
                
            tickets.append({
                "ticket_id": t_id,
                "customer_id": c_id,
                "created_at": created_at,
                "issue_type": issue,
                "priority": priority,
                "status": status,
                "resolution_time_hrs": resolution_hrs,
                "sentiment_score": sentiment
            })
            
    with open(tickets_file, "w") as f:
        json.dump(tickets, f, indent=2)
        
    print(f"[Support API] Saved {len(tickets)} support tickets to {tickets_file}")
    return tickets


def generate_usage_logs(df_cust: pd.DataFrame):
    """
    Generate telemetry usage events stream (JSONL) with realistic behavioral variance.
    """
    usage_file = os.path.join(OUTPUT_DIR_USAGE, "usage_events.jsonl")
    
    records_written = 0
    with open(usage_file, "w") as f:
        for _, row in df_cust.iterrows():
            c_id = row["customer_id"]
            is_churn = row["churn_30d"]
            internet = row["internet_service"]
            
            # Baseline usage
            if internet == "Fiber optic":
                base_gb = random.uniform(180, 420)
            elif internet == "DSL":
                base_gb = random.uniform(60, 160)
            else:
                base_gb = random.uniform(10, 35)
                
            base_logins = random.randint(15, 40)
            base_streaming = random.uniform(15, 60) if row["streaming_tv"] == "Yes" else random.uniform(2, 8)
            
            # Behavioral pattern type:
            # 65% of churners show gradual drop-off
            # 35% churn abruptly with minimal usage drop
            # 12% of non-churners experience a temporary dip (e.g., travel)
            has_drop = (is_churn and random.random() < 0.65) or (not is_churn and random.random() < 0.12)
            
            for period in range(6):  # 0 is 75-90 days ago, 5 is 0-15 days ago
                p_date = REFERENCE_DATE - timedelta(days=15 * (5 - period))
                
                if has_drop and period >= 4:
                    decay = np.random.uniform(0.35, 0.70)
                    crashes = np.random.poisson(lam=0.9 if is_churn else 0.4)
                elif has_drop and period == 3:
                    decay = np.random.uniform(0.65, 0.85)
                    crashes = np.random.poisson(lam=0.6 if is_churn else 0.3)
                else:
                    decay = np.random.uniform(0.85, 1.15)
                    crashes = np.random.poisson(lam=0.3)
                    
                event = {
                    "event_id": f"EVT-{uuid.uuid4().hex[:10]}",
                    "customer_id": c_id,
                    "event_date": p_date.strftime("%Y-%m-%d"),
                    "period_window_days": 15,
                    "data_usage_gb": round(max(1.0, base_gb * decay * random.uniform(0.9, 1.1)), 2),
                    "login_count": int(max(1, round(base_logins * decay * random.uniform(0.9, 1.1)))),
                    "streaming_hours": round(max(0.0, base_streaming * decay * random.uniform(0.9, 1.1)), 1),
                    "app_crashes": int(crashes)
                }
                f.write(json.dumps(event) + "\n")
                records_written += 1
                
    print(f"[Usage Logs] Saved {records_written} usage event records to {usage_file}")
    
    # Also save ground truth targets separately for ML evaluation
    ground_truth_path = os.path.join("data", "ground_truth_churn.csv")
    df_cust[["customer_id", "churn_30d"]].to_csv(ground_truth_path, index=False)
    print(f"[Ground Truth] Saved churn labels to {ground_truth_path} (Churn rate: {df_cust['churn_30d'].mean():.2%})")


if __name__ == "__main__":
    print("Generating synthetic multi-source Telco churn dataset...")
    cust_df = generate_customer_profiles(N_CUSTOMERS)
    generate_billing_system(cust_df)
    generate_support_tickets_api(cust_df)
    generate_usage_logs(cust_df)
    print("Multi-source data generation completed successfully!")
