# Project 12 — Multi-Tier Customer Support Agent

## What You Build

A production-grade customer support system that triages incoming messages by
intent, routes them to specialised sub-agents, uses mock CRM tools to look up
and update records, escalates when necessary, enforces SLA targets, guards
outgoing responses for PII, and produces a per-session JSON report.

---

## Architecture

```
User Message
      │
      ▼
┌─────────────────────┐
│   should_escalate() │ ← keyword guard (fast path)
└──────┬──────────────┘
       │ No
       ▼
┌─────────────────────┐
│   classify_intent() │ ← LLM triage: order|billing|technical|general
└──────┬──────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              Specialist Sub-Agent                 │
│  ┌────────────┐  ┌───────────────┐               │
│  │ OrderAgent │  │ BillingAgent  │               │
│  │ (status,   │  │ (refunds,     │               │
│  │  shipping) │  │  invoices)    │               │
│  └────────────┘  └───────────────┘               │
│  ┌────────────┐  ┌───────────────┐               │
│  │ TechAgent  │  │ GeneralAgent  │               │
│  │ (bugs, KB) │  │ (FAQ, docs)   │               │
│  └────────────┘  └───────────────┘               │
└──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│    output_guard()   │ ← PII redaction before sending
└──────┬──────────────┘
       │
       ▼
  SLATracker.record()
       │
       ▼
  SupportSession.interactions.append(Interaction)
       │
       ▼
  generate_session_report() → JSON
```

---

## Production Patterns Covered

| Pattern | Where |
|---------|-------|
| Intent classification + routing | `classify_intent()` + `handle_message()` |
| Specialised sub-agents | `order_agent`, `billing_agent`, `technical_agent`, `general_agent` |
| CRM tool use with Pydantic models | `lookup_customer`, `get_order_status`, `process_refund`, `create_ticket` |
| Keyword knowledge-base search | `search_knowledge_base()` |
| Forced escalation (keyword fast-path) | `should_escalate()` + `escalation_handler()` |
| Output PII guardrail | `output_guard()` — regex redaction |
| SLA tracking (first response + exchanges) | `SLATracker` dataclass |
| Session state management | `SupportSession` + `Interaction` dataclasses |
| Structured session report | `generate_session_report()` → JSON |
| Guide reference | §4 (tool use), §7 (production), §12 (eval/quality) |

---

## Milestones

### Milestone 1 — CRM Tool Layer
Implement the 5 mock CRM tools. Each should return a Pydantic model even when
the record does not exist (use `found=False`). `process_refund()` must reject
orders whose status is not `delivered` or `in_transit`.

### Milestone 2 — Triage Agent
Implement `classify_intent()`. Send the user message to the LLM with a prompt
that defines all 5 categories. Strip whitespace and lowercase the reply; default
to `"general"` if the LLM returns something unrecognised.

### Milestone 3 — Specialist Agents
Implement all four specialist handlers. Each must:
1. Call the relevant CRM tool(s) to gather context
2. Build a concise context-rich prompt
3. Call the LLM for a warm, professional reply

### Milestone 4 — Escalation, Guards, SLA
Implement `should_escalate()` (O(1) keyword scan), `escalation_handler()`
(create ticket → confirmation message), `output_guard()` (regex PII redaction),
and both `SLATracker` methods.

### Milestone 5 — Session Pipeline + Report
Implement `handle_message()` — the main dispatcher — and `generate_session_report()`.
The report must include: intent breakdown, avg response time, escalation count, and
per-interaction details.

---

## Expected Output

```
═══════════════════════════════════════════════════════════════════
 Project 12 — Customer Support Agent
═══════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────
▶ Order Inquiry — In Transit
  Customer: C002
  Response: Hi Bob! I checked your order ORD-1002 (Basic Plan) and it is
            currently in transit ...
  ✅ SLA: first_response=1.23s (OK) | exchanges=1 (OK)
  📋 Intent: {'order': 1} | avg_rt=1.23s | escalations=0

────────────────────────────────────────────────────────────────
▶ Escalation — Frustrated Customer
  ✅ SLA: first_response=0.12s (OK) | exchanges=1 (OK)
  📋 Intent: {'escalate': 1} | escalations=1
…

 Summary: 5 sessions | 5 resolved | 1 escalated
 SLA compliance: 5/5 | avg response: 1.45s
✅ Report saved → support_report.json
```

---

## Setup

```bash
# from repo root
pip install litellm python-dotenv pydantic
python projects/project12_customer_support_agent/starter.py
# or
python projects/project12_customer_support_agent/solution/solution.py
```
