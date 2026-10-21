"""
Mock Customer Support Ticket REST API.
Simulates a real-world Customer Success & Support ticketing system (like Zendesk/Freshdesk).
Exposes JSON endpoints for ticket retrieval, pagination, filtering, and customer summary.
"""

import os
import json
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Customer Support Ticketing Mock API",
    description="Simulates an enterprise customer support ticketing service",
    version="1.0.0"
)

TICKETS_DB_PATH = os.path.join(os.path.dirname(__file__), "tickets.json")

# In-memory store
_tickets_cache = []


def load_tickets():
    global _tickets_cache
    if not _tickets_cache and os.path.exists(TICKETS_DB_PATH):
        with open(TICKETS_DB_PATH, "r") as f:
            _tickets_cache = json.load(f)
    return _tickets_cache


class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    created_at: str
    issue_type: str
    priority: str
    status: str
    resolution_time_hrs: Optional[float] = None
    sentiment_score: float


@app.on_event("startup")
def startup_event():
    load_tickets()
    print(f"Mock Ticket API started. Loaded {len(_tickets_cache)} tickets.")


@app.get("/health")
def health_check():
    tickets = load_tickets()
    return {
        "status": "healthy",
        "service": "mock_ticket_api",
        "total_tickets": len(tickets)
    }


@app.get("/api/v1/tickets", response_model=List[Ticket])
def get_tickets(
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    status: Optional[str] = Query(None, description="Filter by ticket status"),
    priority: Optional[str] = Query(None, description="Filter by ticket priority"),
    limit: int = Query(500, ge=1, le=5000, description="Max tickets to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Retrieve customer support tickets with pagination and filtering."""
    tickets = load_tickets()
    results = tickets
    
    if customer_id:
        results = [t for t in results if t["customer_id"] == customer_id]
    if status:
        results = [t for t in results if t["status"] == status]
    if priority:
        results = [t for t in results if t["priority"] == priority]
        
    return results[offset:offset + limit]


@app.get("/api/v1/tickets/summary/{customer_id}")
def get_customer_ticket_summary(customer_id: str):
    """Get aggregated support ticket friction metrics for a specific customer."""
    tickets = load_tickets()
    cust_tickets = [t for t in tickets if t["customer_id"] == customer_id]
    
    if not cust_tickets:
        return {
            "customer_id": customer_id,
            "total_tickets": 0,
            "open_tickets": 0,
            "escalated_tickets": 0,
            "avg_sentiment": 0.0,
            "critical_tickets": 0
        }
        
    open_count = sum(1 for t in cust_tickets if t["status"] in ["OPEN", "ESCALATED"])
    escalated_count = sum(1 for t in cust_tickets if t["status"] == "ESCALATED")
    critical_count = sum(1 for t in cust_tickets if t["priority"] == "CRITICAL")
    avg_sentiment = sum(t["sentiment_score"] for t in cust_tickets) / len(cust_tickets)
    
    return {
        "customer_id": customer_id,
        "total_tickets": len(cust_tickets),
        "open_tickets": open_count,
        "escalated_tickets": escalated_count,
        "avg_sentiment": round(avg_sentiment, 2),
        "critical_tickets": critical_count
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
