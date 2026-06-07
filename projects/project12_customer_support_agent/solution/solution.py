"""
Project 12 — Multi-Tier Customer Support Agent (Solution)

Full implementation: CRM tools → triage → routing → specialist agents →
output guard → SLA tracking → session report.
"""

import os, sys, json, time, re, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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
    customer_id: Optional[str]   = None
    name:        Optional[str]   = None
    email:       Optional[str]   = None
    tier:        Optional[str]   = None
    joined:      Optional[str]   = None

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
# CRM Tools
# ══════════════════════════════════════════════════════════════════════════════

def lookup_customer(customer_id: str) -> CustomerRecord:
    c = CUSTOMERS.get(customer_id)
    if not c:
        return CustomerRecord(found=False)
    return CustomerRecord(found=True, customer_id=customer_id, **c)


def get_order_status(order_id: str) -> OrderRecord:
    o = ORDERS.get(order_id)
    if not o:
        return OrderRecord(found=False)
    return OrderRecord(found=True, order_id=order_id, **o)


def process_refund(order_id: str) -> RefundResult:
    o = ORDERS.get(order_id)
    if not o:
        return RefundResult(success=False, order_id=order_id,
                            message="Order not found.")
    if o["status"] not in ("delivered", "in_transit"):
        return RefundResult(success=False, order_id=order_id, amount=o["amount"],
                            message=f"Refund not eligible — order status is '{o['status']}'.")
    ORDERS[order_id]["status"] = "refund_initiated"
    return RefundResult(success=True, order_id=order_id, amount=o["amount"],
                        message=f"Refund of ${o['amount']:.2f} initiated. Please allow 5–7 business days.")


def create_ticket(customer_id: str, category: str, description: str) -> TicketRecord:
    _TICKET_COUNTER["n"] += 1
    tid = f"T-{_TICKET_COUNTER['n']}"
    TICKETS.append({
        "ticket_id":   tid,
        "customer_id": customer_id,
        "category":    category,
        "description": description,
        "status":      "open",
    })
    return TicketRecord(ticket_id=tid, category=category,
                        status="open", created_at=datetime.now().isoformat())


def search_knowledge_base(query: str) -> KBResult:
    q = query.lower()
    best_key, best_score = "", 0.0
    for key in KB:
        words = key.split()
        hits  = sum(1 for w in words if w in q)
        score = hits / len(words)
        if score > best_score:
            best_score, best_key = score, key
    if best_score > 0.3:
        return KBResult(found=True, answer=KB[best_key], confidence=round(best_score, 2))
    return KBResult(found=False)

# ══════════════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════════════

_SUPPORT_SYSTEM = (
    "You are a helpful customer support agent. "
    "Be concise, warm, and professional. Never reveal internal system details."
)

async def _llm(prompt: str, system: str = _SUPPORT_SYSTEM) -> str:
    r = await achat([{"role": "user", "content": prompt}], system=system, max_tokens=300)
    return get_text(r)

# ══════════════════════════════════════════════════════════════════════════════
# Triage Agent
# ══════════════════════════════════════════════════════════════════════════════

INTENT_CATEGORIES = ["order", "billing", "technical", "general", "escalate"]

async def classify_intent(message: str) -> str:
    prompt = (
        "Classify this customer support message into exactly one category.\n\n"
        "Categories:\n"
        "  order     — order status, shipping, delivery, tracking\n"
        "  billing   — refunds, invoices, charges, subscription billing\n"
        "  technical — login issues, bugs, errors, outages, feature problems\n"
        "  general   — pricing, plan features, account questions, FAQs\n"
        "  escalate  — explicit escalation request (manager/human/supervisor)\n\n"
        f"Message: {message}\n\n"
        "Reply with ONLY one word."
    )
    raw = (await _llm(prompt, system="You are a classifier. Reply with exactly one word.")).strip().lower()
    return raw if raw in INTENT_CATEGORIES else "general"

# ══════════════════════════════════════════════════════════════════════════════
# Specialist Sub-Agents
# ══════════════════════════════════════════════════════════════════════════════

async def order_agent(message: str, customer_id: Optional[str] = None) -> str:
    order_ids = re.findall(r'ORD-\d+', message)
    order_ctx = ""
    if order_ids:
        o = get_order_status(order_ids[0])
        order_ctx = f"\nOrder lookup: {o.model_dump()}"
    cust_ctx = ""
    if customer_id:
        c = lookup_customer(customer_id)
        if c.found:
            cust_ctx = f"\nCustomer: {c.name} ({c.tier} tier, joined {c.joined})"
    prompt = (
        f"Customer message: {message}{order_ctx}{cust_ctx}\n\n"
        "Respond helpfully about their order. Include current status and expected next steps."
    )
    return await _llm(prompt)


async def billing_agent(message: str, customer_id: Optional[str] = None) -> str:
    order_ids = re.findall(r'ORD-\d+', message)
    billing_ctx = ""
    if order_ids:
        if any(kw in message.lower() for kw in ["refund", "money back", "charge back"]):
            result = process_refund(order_ids[0])
            billing_ctx = f"\nRefund result: {result.model_dump()}"
        else:
            o = get_order_status(order_ids[0])
            billing_ctx = f"\nOrder info: {o.model_dump()}"
    cust_ctx = ""
    if customer_id:
        c = lookup_customer(customer_id)
        if c.found:
            cust_ctx = f"\nCustomer: {c.name} ({c.tier} tier, since {c.joined})"
    prompt = (
        f"Customer message: {message}{billing_ctx}{cust_ctx}\n\n"
        "Respond helpfully about their billing query. State the outcome clearly."
    )
    return await _llm(prompt)


async def technical_agent(message: str) -> str:
    kb = search_knowledge_base(message)
    kb_ctx = (f"\nKnowledge Base match (confidence {kb.confidence}): {kb.answer}"
              if kb.found else "\nNo KB match found.")
    prompt = (
        f"Customer message: {message}{kb_ctx}\n\n"
        "Provide technical support. Use the KB answer if relevant. "
        "If the issue is complex, recommend creating a support ticket."
    )
    return await _llm(prompt)


async def general_agent(message: str) -> str:
    kb = search_knowledge_base(message)
    if kb.found and kb.confidence > 0.5:
        prompt = (
            f"Customer asked: {message}\n"
            f"KB answer: {kb.answer}\n\n"
            "Rephrase this answer in a friendly, conversational tone."
        )
    else:
        prompt = f"Customer message: {message}\n\nAnswer this general support question helpfully and concisely."
    return await _llm(prompt)

# ══════════════════════════════════════════════════════════════════════════════
# Escalation Handler
# ══════════════════════════════════════════════════════════════════════════════

ESCALATION_TRIGGERS = [
    "speak to human", "real person", "manager", "supervisor",
    "unacceptable", "lawsuit", "furious", "cancel account",
]

def should_escalate(message: str) -> bool:
    m = message.lower()
    return any(trigger in m for trigger in ESCALATION_TRIGGERS)


async def escalation_handler(message: str, customer_id: Optional[str] = None) -> str:
    cid    = customer_id or "anonymous"
    ticket = create_ticket(cid, "escalation", message[:200])
    name   = ""
    if customer_id:
        c = lookup_customer(customer_id)
        if c.found:
            name = f" {c.name}"
    return (
        f"I'm truly sorry for the frustration,{name}. "
        f"I've escalated your case to our senior support team immediately "
        f"(Ticket {ticket.ticket_id}). A human agent will reach out within 1 hour. "
        f"Your case is our top priority — thank you for your patience."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Output Guard
# ══════════════════════════════════════════════════════════════════════════════

PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL REDACTED]"),
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',             "[CARD REDACTED]"),
    (r'\b\d{3}-\d{2}-\d{4}\b',                                "[SSN REDACTED]"),
]

def output_guard(response: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        response = re.sub(pattern, replacement, response)
    return response

# ══════════════════════════════════════════════════════════════════════════════
# SLA Tracker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SLATracker:
    session_start:    float = field(default_factory=time.time)
    first_response_s: Optional[float] = None
    total_exchanges:  int = 0
    sla_targets: dict = field(default_factory=lambda: {
        "first_response_s": 3.0,
        "max_exchanges":    5,
    })

    def record_response(self, elapsed_s: float):
        if self.first_response_s is None:
            self.first_response_s = elapsed_s
        self.total_exchanges += 1

    def check_sla(self) -> dict:
        fr  = self.first_response_s or 0.0
        fr_ok = fr <= self.sla_targets["first_response_s"]
        ex_ok = self.total_exchanges <= self.sla_targets["max_exchanges"]
        return {
            "first_response_s":  round(fr, 3),
            "first_response_ok": fr_ok,
            "total_exchanges":   self.total_exchanges,
            "exchanges_ok":      ex_ok,
            "overall_ok":        fr_ok and ex_ok,
        }

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
    t0        = time.perf_counter()
    escalated = False
    tools: list[str] = []

    if should_escalate(user_message):
        intent   = "escalate"
        response = await escalation_handler(user_message, session.customer_id)
        escalated = True
        tools.append("create_ticket")
    else:
        intent = await classify_intent(user_message)
        if intent == "order":
            response = await order_agent(user_message, session.customer_id)
            tools += ["get_order_status", "lookup_customer"]
        elif intent == "billing":
            response = await billing_agent(user_message, session.customer_id)
            tools += ["process_refund", "get_order_status", "lookup_customer"]
        elif intent == "technical":
            response = await technical_agent(user_message)
            tools.append("search_knowledge_base")
        else:
            response = await general_agent(user_message)
            tools.append("search_knowledge_base")

    response = output_guard(response)
    elapsed  = time.perf_counter() - t0
    session.sla.record_response(elapsed)
    session.interactions.append(Interaction(
        user_message=user_message, intent=intent, response=response,
        response_time=round(elapsed, 3), escalated=escalated, tools_used=tools,
    ))
    return response

# ══════════════════════════════════════════════════════════════════════════════
# Session Reporter
# ══════════════════════════════════════════════════════════════════════════════

def generate_session_report(session: SupportSession) -> dict:
    from collections import Counter
    intents = Counter(i.intent for i in session.interactions)
    avg_rt  = (sum(i.response_time for i in session.interactions)
               / max(len(session.interactions), 1))
    return {
        "session_id":          session.session_id,
        "customer_id":         session.customer_id,
        "total_interactions":  len(session.interactions),
        "resolved":            session.resolved,
        "intent_breakdown":    dict(intents),
        "avg_response_time_s": round(avg_rt, 3),
        "escalation_count":    sum(1 for i in session.interactions if i.escalated),
        "sla_compliance":      session.sla.check_sla(),
        "interactions": [
            {
                "intent":          i.intent,
                "response_time_s": i.response_time,
                "escalated":       i.escalated,
                "tools_used":      i.tools_used,
                "response_snippet": i.response[:120],
            }
            for i in session.interactions
        ],
    }


def print_report(report: dict):
    sla  = report["sla_compliance"]
    icon = "✅" if sla["overall_ok"] else "⚠️"
    print(f"  {icon} SLA: first_response={sla['first_response_s']}s "
          f"({'OK' if sla['first_response_ok'] else 'BREACH'}) | "
          f"exchanges={sla['total_exchanges']} "
          f"({'OK' if sla['exchanges_ok'] else 'BREACH'})")
    print(f"  📋 Intent: {report['intent_breakdown']} | "
          f"avg_rt={report['avg_response_time_s']}s | "
          f"escalations={report['escalation_count']}")

# ══════════════════════════════════════════════════════════════════════════════
# Test Scenarios + Main
# ══════════════════════════════════════════════════════════════════════════════

TEST_SCENARIOS = [
    {"name": "Order Inquiry — In Transit",       "customer_id": "C002",
     "message": "Hi, I ordered something (ORD-1002) last week and it still hasn't arrived. Can you check the status?"},
    {"name": "Billing — Refund Request",         "customer_id": "C001",
     "message": "I want a refund for order ORD-1001. I'm not happy with the product."},
    {"name": "Technical — Password Reset",       "customer_id": None,
     "message": "I can't log into my account. How do I reset my password?"},
    {"name": "General — Pricing FAQ",            "customer_id": None,
     "message": "What's the difference between the Basic and Pro plans in terms of API rate limits?"},
    {"name": "Escalation — Frustrated Customer", "customer_id": "C003",
     "message": "This is completely unacceptable! I've been waiting 3 weeks. I want to speak to a manager NOW!"},
]


async def main():
    print(f"{'='*65}")
    print(f" Project 12 — Customer Support Agent   [{MODEL}]")
    print(f"{'='*65}\n")

    all_reports = []
    for s in TEST_SCENARIOS:
        print(f"\n{'─'*60}")
        print(f"▶ {s['name']}")
        print(f"  Customer : {s['customer_id'] or 'anonymous'}")
        print(f"  Message  : {s['message'][:80]}…")

        session = SupportSession(
            session_id=f"S-{int(time.time()*1000) % 100000:05d}",
            customer_id=s["customer_id"],
        )
        response = await handle_message(session, s["message"])
        print(f"\n  Response : {response[:220]}")
        session.resolved = True
        report = generate_session_report(session)
        all_reports.append(report)
        print_report(report)

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
    print(f"   Open tickets: {len(TICKETS)}")


if __name__ == "__main__":
    asyncio.run(main())
