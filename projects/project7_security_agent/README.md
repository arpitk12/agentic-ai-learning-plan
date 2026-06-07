# Project 7 — Secure Customer Support Agent

## What You Build

A production-hardened ReAct agent that handles customer support requests safely. Every layer of the agent is security-hardened: input is sanitized, LLM outputs are scanned, tool calls are validated, and risky actions require human approval.

## Production Skills Practised

| Skill | Guide Section |
|-------|--------------|
| Prompt injection detection + defense | §8.2 |
| PII detection and redaction | §8.4 |
| Pydantic tool argument validation | §8.3 |
| Human-in-the-Loop for risky actions | §4.7 |
| Output scanning (API keys, PII leakage) | §8.4 |
| Secure system prompt design | §8.2 |
| Per-user cost limits + daily budget | §6.2 |

## Architecture

```
User Input
    │
    ▼
[detect_injection()]      ← reject if injection pattern found
    │
    ▼
[scan_pii()]              ← log PII types (redacted), alert if sensitive
    │
    ▼
[Secure System Prompt]    ← hardened with IMMUTABLE SECURITY RULES
    │
    ▼
[ReAct Agent Loop]
    ├── LLM decides → tool call
    │       │
    │       ▼
    │   [validate_and_dispatch()]
    │       ├── ALLOWED_TOOLS whitelist check
    │       ├── Pydantic argument validation
    │       └── HITL prompt for HIGH risk tools
    │
    └── LLM decides → stop
            │
            ▼
        [safe_response_scan()]
            ├── redact PII in output
            └── redact API keys / secrets
```

## Tools Provided (already implemented)

| Tool | Risk Level | Requires Approval? |
|------|-----------|-------------------|
| `check_order_status` | LOW | Auto-approved |
| `get_account_info` | MEDIUM | Auto-approved with warning |
| `process_refund` | HIGH | Requires human y/n |
| `escalate_to_human` | LOW | Auto-approved |

## Setup

```bash
pip install litellm python-dotenv pydantic
cp ../../.env .env          # or set MODEL + API key env vars
```

## Usage

```bash
python starter.py "I need a refund for order ORD-123456"
python starter.py "Check the status of my order ORD-789012"
python solution.py "My card number 4111-1111-1111-1111 was charged twice"
```

## What To Implement (5 TODOs)

1. **`detect_injection(text)`** — check INJECTION_PATTERNS list
2. **`scan_pii(text)`** — check PII_PATTERNS, return list of types found
3. **`validate_and_dispatch(tool_name, args)`** — whitelist + Pydantic validation + dispatch
4. **`safe_response_scan(text)`** — redact PII and API key patterns in output
5. **`safe_agent_loop(task)`** — full ReAct with injection check, HITL routing, output scan

## Learning Goals

After completing this project you will be able to:
- Defend against prompt injection at the input layer
- Detect and redact PII before it hits logs or output
- Validate tool arguments strictly (preventing SSRF, path traversal, injection)
- Implement risk-tiered human approval for dangerous tools
- Scan and sanitize agent output before returning to users

## Key Insight

Security is not one check — it's a layered defence:
**Input → Prompt → Tool Dispatch → Output → Logging**
Each layer must independently validate/sanitize.
