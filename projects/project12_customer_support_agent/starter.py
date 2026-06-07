"""
Project 12 — Multi-Tier Customer Support Agent (Starter)

Build a production customer support agent that:
  1. Classifies incoming message intent (order/billing/technical/general/escalate)
  2. Routes to specialised sub-agents that call mock CRM tools
  3. Escapes escalation triggers to a human queue (fast path, no LLM needed)
  4. Redacts PII from all outgoing responses (output guard)
  5. Tracks SLA compliance (first response time + exchange count)
  6. Produces a structured JSON session report

Usage:
  python starter.py

Runs 5 test scenarios covering all intent categories and escalation.
"""

import os, sys, json, time, re, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import achat, get_text, MODEL

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Mock CRM Database
# ══════════════════════════════════════════════════════════════════════════════

CUSTOMERS = {
    "C001": {"name": "Alice Johnson", "email": "alice@example.com", "tier": "premium", "joined": "2022-03-15"},
    "C002": {"name": "Bob Smith",     "email": "bob@example.com",   "tier": "basic",   "joined": "2023-07-01"},
    "C003": {"name": "Carol Davis",   "email": "carol@example.com", "tier": "premium", "joined": "2021-11-20"},
}

ORDERS = {
    "ORD-1001": {"customer_id": "C001", "product": "Pro Plan",    "amount": 199.99, "status": "delivered",  "date": "2026-05-28"},
    "ORD-1002": {"customer_id": "C002", "product": "Basic Plan",  "amount":  49.99, "status": "in_transit", "date": "2026-06-01"},
    "ORD-1003": {"customer_id": "C003", "product": "Enterprise",  "amount": 499.99, "status": "processing", "date": "2026-06-05"},
    "ORD-1004": {"customer_id": "C001", "product": "Add-on Pack", "amount":  29.99, "status": "refunded",   "date": "2026-05-10"},
}

TICKETS: list[dict] = []
_TICKET_COUNTER = {"n": 1000}

KB = {
    "password reset":            "Go to Settings → Security → Reset Password. A link will be emailed to you.",
    "cancel subscription":       "You can cancel anytime from Account → Billing → Cancel. No fees.",
    "data export":               "Go to Settings → Data → Export. Supports CSV, JSON, and PDF formats.",
    "pricing":                   "Basic: $49/mo, Pro: $199/mo, Enterprise: custom. Annual billing saves 20%.",
    "api rate limit":            "Basic: 100 req/min, Pro: 1000 req/min, Enterprise: unlimited.",
    "two factor authentication": "Enable 2FA under Settings → Security → Two-Factor Auth.",
    "refund policy":             "Full refund within 30 days of purchase. Pro-rated refunds for annual plans.",
    "sla":                       "Basic: 24h support, Pro: 4h, Enterprise: 1h with dedicated account manager.",
}


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic CRM Response Models
# ══════════════════════════════════════════════════════════════════════════════

class CustomerRecord(BaseModel):
    found:       bool
    customer_id: Optional[str] = None
    name:        Optional[str] = None
    email:       Optional[str] = None
    tier:        Optional[str] = None
    joined:      Optional[str] = None

class OrderRecord(BaseModel):
    found:    bool
    order_id: Optional[str]   = None
    product:  Optional[str]   = None
    amount:   Optional[float] = None
    status:   Optional[str]   = None
    date:     Optional[str]   = None

class RefundResult(BaseModel):
    success:  bool
    order_id: str
    amount:   Optional[float] = None
    message:  str

class TicketRecord(BaseModel):
    ticket_id:  str
    category:   str
    status:     str
    created_at: str

class KBResult(BaseModel):
    found:      bool
    answer:     Optional[str] = None
    confidence: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CRM Tool Layer
# ══════════════════════════════════════════════════════════════════════════════

def lookup_customer(customer_id: str) -> CustomerRecord:
    """TODO: Look up customer by ID in CUSTOMERS dict.
       Return CustomerRecord(found=True, ...) on hit, CustomerRecord(found=False) on miss."""
    raise NotImplementedError


def get_order_status(order_id: str) -> OrderRecord:
    """TODO: Look up order by ID in ORDERS dict.
       Return OrderRecord(found=True, ...) on hit, OrderRecord(found=False) on miss."""
    raise NotImplementedError


def process_refund(order_id: str) -> RefundResult:
    """TODO: Initiate a refund for an order.
       Rules:
         - If order not found: return success=False, message="Order not found."
         - If status not in ('delivered', 'in_transit'): return success=False with reason
         - Otherwise: update ORDERS[order_id]['status'] = 'refund_initiated'
                      return success=True with amount and confirmation message
    """
    raise NotImplementedError


def create_ticket(customer_id: str, category: str, description: str) -> TicketRecord:
    """TODO: Create a support ticket.
       - Increment _TICKET_COUNTER['n'], assign ticket_id = f'T-{n}'
       - Append to TICKETS list
       - Return TicketRecord with status='open' and current ISO timestamp
    """
    raise NotImplementedError


def search_knowledge_base(query: str) -> KBResult:
    """TODO: Keyword search over KB dict.
       - Normalise query to lowercase
       - For each KB key: compute hit fraction (words in query / words in key)
       - If best score > 0.3: return KBResult(found=True, answer=KB[key], confidence=score)
       - Otherwise: return KBResult(found=False)
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════════════

async def _llm(prompt: str,
               system: str = "You are a helpful customer support agent. Be concise, warm, and professional.") -> str:
    r = await achat([{"role": "user", "content": prompt}], system=system, max_tokens=300)
    return get_text(r)


# ══════════════════════════════════════════════════════════════════════════════
# Triage Agent
# ══════════════════════════════════════════════════════════════════════════════

INTENT_CATEGORIES = ["order", "billing", "technical", "general", "escalate"]


async def classify_intent(message: str) -> str:
    """TODO: Send message to LLM with a prompt that defines all 5 categories.
       Strip whitespace and lowercase the reply.
       Default to 'general' if the reply is not in INTENT_CATEGORIES.
       Return one of: order | billing | technical | general | escalate
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Specialist Sub-Agents
# ══════════════════════════════════════════════════════════════════════════════

async def order_agent(message: str, customer_id: Optional[str] = None) -> str:
    """TODO: Handle order-related queries.
       Steps:
         1. Extract order ID from message: re.findall(r'ORD-\\d+', message)
         2. Call get_order_status() for the first match (if any)
         3. Call lookup_customer() if customer_id is provided
         4. Build a context-rich prompt with both CRM results
         5. Return LLM response
    """
    raise NotImplementedError


async def billing_agent(message: str, customer_id: Optional[str] = None) -> str:
    """TODO: Handle billing/refund queries.
       Steps:
         1. Extract order ID from message
         2. If 'refund' or 'money back' in message: call process_refund(), include result
         3. Otherwise: call get_order_status() for context
         4. Call lookup_customer() if customer_id provided
         5. Return LLM response with CRM context
    """
    raise NotImplementedError


async def technical_agent(message: str) -> str:
    """TODO: Handle technical/bug queries.
       Steps:
         1. Call search_knowledge_base(message)
         2. Include KB result in prompt (if found)
         3. Return LLM response — if KB matched, synthesise it;
            if no match, give troubleshooting steps + offer to create a ticket
    """
    raise NotImplementedError


async def general_agent(message: str) -> str:
    """TODO: Handle general/FAQ queries.
       Steps:
         1. Call search_knowledge_base(message)
         2. If found AND confidence > 0.5: rephrase KB answer via LLM
         3. Otherwise: call LLM for a general helpful response
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Escalation Handler
# ══════════════════════════════════════════════════════════════════════════════

ESCALATION_TRIGGERS = [
    "speak to human", "real person", "manager", "supervisor",
    "unacceptable", "lawsuit", "furious", "cancel account",
]


def should_escalate(message: str) -> bool:
    """TODO: Return True if message contains any ESCALATION_TRIGGERS phrase (case-insensitive)."""
    raise NotImplementedError


async def escalation_handler(message: str, customer_id: Optional[str] = None) -> str:
    """TODO:
       1. Call create_ticket(customer_id or 'anonymous', 'escalation', message[:200])
       2. If customer_id given, call lookup_customer() and use their name in response
       3. Return a warm apology + ticket ID + 'human agent within 1 hour' message
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Output Guard
# ══════════════════════════════════════════════════════════════════════════════

PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL REDACTED]"),
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',             "[CARD REDACTED]"),
    (r'\b\d{3}-\d{2}-\d{4}\b',                                "[SSN REDACTED]"),
]


def output_guard(response: str) -> str:
    """TODO: Apply each (pattern, replacement) in PII_PATTERNS to the response string.
       Return the sanitised response."""
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# SLA Tracker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SLATracker:
    session_start:    float = field(default_factory=time.time)
    first_response_s: Optional[float] = None
    total_exchanges:  int = 0
    sla_targets: dict = field(default_factory=lambda: {
        "first_response_s": 3.0,   # must respond within 3 s
        "max_exchanges":    5,     # resolve within 5 exchanges
    })

    def record_response(self, elapsed_s: float):
        """TODO: Set first_response_s on the FIRST call only. Always increment total_exchanges."""
        raise NotImplementedError

    def check_sla(self) -> dict:
        """TODO: Return dict:
           {
             'first_response_s': float,
             'first_response_ok': bool,   # <= sla_targets['first_response_s']
             'total_exchanges': int,
             'exchanges_ok': bool,        # <= sla_targets['max_exchanges']
             'overall_ok': bool,          # both True
           }
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Session Data Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Interaction:
    user_message:  str
    intent:        str
    response:      str
    response_time: float
    escalated:     bool = False
    tools_used:    list[str] = field(default_factory=list)


@dataclass
class SupportSession:
    session_id:   str
    customer_id:  Optional[str]
    interactions: list[Interaction] = field(default_factory=list)
    sla:          SLATracker = field(default_factory=SLATracker)
    resolved:     bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Main Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

async def handle_message(session: SupportSession, user_message: str) -> str:
    """TODO: Full message handling pipeline.
       1. t0 = time.perf_counter()
       2. If should_escalate(user_message): call escalation_handler(), set escalated=True
       3. Else: classify_intent() → route to matching specialist agent
       4. Apply output_guard() to response
       5. session.sla.record_response(elapsed)
       6. Append Interaction to session.interactions
       7. Return guarded response
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Session Reporter
# ══════════════════════════════════════════════════════════════════════════════

def generate_session_report(session: SupportSession) -> dict:
    """TODO: Return report dict with:
       - session_id, customer_id, total_interactions, resolved
       - intent_breakdown: {intent: count}
       - avg_response_time_s (rounded to 3 dp)
       - escalation_count
       - sla_compliance: session.sla.check_sla()
       - interactions: list of per-interaction dicts
         (intent, response_time_s, escalated, tools_used, response_snippet[:100])
    """
    raise NotImplementedError


def print_report(report: dict):
    sla = report["sla_compliance"]
    icon = "✅" if sla["overall_ok"] else "⚠️"
    print(f"  {icon} SLA: first_response={sla['first_response_s']}s "
          f"({'OK' if sla['first_response_ok'] else 'BREACH'}) | "
          f"exchanges={sla['total_exchanges']} "
          f"({'OK' if sla['exchanges_ok'] else 'BREACH'})")
    print(f"  📋 Intent: {report['intent_breakdown']} | "
          f"avg_rt={report['avg_response_time_s']}s | "
          f"escalations={report['escalation_count']}")


# ══════════════════════════════════════════════════════════════════════════════
# Test Scenarios
# ══════════════════════════════════════════════════════════════════════════════

TEST_SCENARIOS = [
    {
        "name": "Order Inquiry — In Transit",
        "customer_id": "C002",
        "message": "Hi, I ordered something (ORD-1002) last week and it still hasn't arrived. Can you check the status?",
    },
    {
        "name": "Billing — Refund Request",
        "customer_id": "C001",
        "message": "I want a refund for order ORD-1001. I'm not happy with the product.",
    },
    {
        "name": "Technical — Password Reset",
        "customer_id": None,
        "message": "I can't log into my account. How do I reset my password?",
    },
    {
        "name": "General — Pricing FAQ",
        "customer_id": None,
        "message": "What's the difference between the Basic and Pro plans in terms of API rate limits?",
    },
    {
        "name": "Escalation — Frustrated Customer",
        "customer_id": "C003",
        "message": "This is completely unacceptable! I've been waiting 3 weeks. I want to speak to a manager NOW!",
    },
]


async def main():
    from pathlib import Path
    print(f"{'='*65}")
    print(f" Project 12 — Customer Support Agent   [{MODEL}]")
    print(f"{'='*65}\n")

    all_reports = []
    for scenario in TEST_SCENARIOS:
        print(f"\n{'─'*60}")
        print(f"▶ {scenario['name']}")
        print(f"  Customer : {scenario['customer_id'] or 'anonymous'}")
        print(f"  Message  : {scenario['message'][:80]}…")

        session = SupportSession(
            session_id=f"S-{int(time.time()*1000) % 100000:05d}",
            customer_id=scenario["customer_id"],
        )
        response = await handle_message(session, scenario["message"])
        print(f"\n  Response : {response[:200]}")
        session.resolved = True
        report = generate_session_report(session)
        all_reports.append(report)
        print_report(report)

    # Aggregate
    total  = len(all_reports)
    res    = sum(1 for r in all_reports if r["resolved"])
    esc    = sum(r["escalation_count"] for r in all_reports)
    sla_ok = sum(1 for r in all_reports if r["sla_compliance"]["overall_ok"])
    avg_rt = sum(r["avg_response_time_s"] for r in all_reports) / total

    print(f"\n{'='*65}")
    print(f" Summary: {total} sessions | {res} resolved | {esc} escalated")
    print(f" SLA compliance: {sla_ok}/{total} | avg response: {avg_rt:.2f}s")
    print(f"{'='*65}")

    output = {
        "summary": {"total": total, "resolved": res, "escalated": esc,
                    "sla_ok": sla_ok, "avg_rt_s": round(avg_rt, 3)},
        "sessions": all_reports,
    }
    Path("support_report.json").write_text(json.dumps(output, indent=2))
    print(f"✅ Report saved → support_report.json")


if __name__ == "__main__":
    asyncio.run(main())
