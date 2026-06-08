# Project 29 — Advanced Guardrails (NeMo + Llama Guard + Guardrails AI)

> **Stack**: NeMo Guardrails · Llama Guard · Guardrails AI · LiteLLM · FastAPI  
> **Phase 7 — Advanced Production** | Priority: P1 🟠

---

## What You'll Build

A 4-layer production safety pipeline that screens every LLM input and output through independent safety layers — each catching different failure modes with different speed/accuracy tradeoffs.

```
User Input
    │
    ▼ Layer 1: Pattern Injection Detector   (<1ms, regex, zero cost)
    │  Catches: "ignore instructions", jailbreaks, DAN mode
    ▼ Layer 2: PII Scanner + Anonymizer     (<5ms, regex)
    │  Catches: SSN, email, phone, credit card — replaces before LLM
    ▼ Layer 3: Llama Guard                  (~50ms, local model, zero API cost)
    │  Catches: 14 hazard categories (violence, CSAM, hate, self-harm...)
    ▼ Layer 4: NeMo Guardrails             (~200ms, programmable conversation rails)
    │  Catches: off-topic, confidentiality, custom business rules
    ▼
  LLM Call
    │
    ▼ Output Layer: PII + citation + hallucination check
    ▼
  Safe Response
```

---

## Milestones

### Milestone 1 — Layer 1: Injection Patterns
Build a regex-based injection detector with 8+ patterns covering: jailbreaks, instruction override, DAN mode, role-play bypasses, system prompt extraction. Test with 20 adversarial inputs.

### Milestone 2 — Layer 2: PII Anonymizer
Detect and replace 5 PII types (email, phone, SSN, credit card, IP) using regex. Pass-through (never block) — sanitize and continue. Log what was redacted.

### Milestone 3 — Layer 3: Llama Guard
Load `Meta-Llama-Guard-2-8B` (or stub with GPT-4o-mini for teams without GPU). Classify input (and output) across 14 hazard categories. Block unsafe; pass safe.

### Milestone 4 — Layer 4: NeMo Guardrails
Write Colang rules for: compliance topic enforcement (redirect off-topic questions), jailbreak attempt refusal, confidentiality protection. Test each rail independently.

### Milestone 5 — Output Safety
After the LLM responds: check for PII in output, hallucinated citations (URLs/document IDs mentioned in response but not in input), internal prompt leakage.

### Milestone 6 — Adversarial Test Suite
Run 30 test cases: 10 injection attempts, 5 PII inputs, 5 off-topic, 5 unsafe content, 5 normal compliance queries. Measure: block rate per category, false positive rate on normal queries.

### Milestone 7 — FastAPI Middleware
Wrap the pipeline as FastAPI middleware that applies to all endpoints. Add: per-layer latency metrics (Prometheus), block reason logging, bypass option for admin users (with audit log).

---

## Setup

```bash
pip install guardrails-ai nemoguardrails transformers torch \
            litellm fastapi uvicorn prometheus-client python-dotenv

# Install Guardrails Hub validators:
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/detect_pii
```

---

## Expected Output

```
=== Safety Pipeline Test Results ===

Running 30 adversarial test cases...

Results by layer:
  Layer 1 (injection):    8/10 blocked (2 false negatives — obfuscated)
  Layer 2 (PII):          5/5 redacted (0 missed)
  Layer 3 (Llama Guard):  4/5 unsafe blocked (1 false negative)
  Layer 4 (NeMo):         4/5 off-topic redirected

False positive rate (normal queries): 0/5 (0%) ✅

Per-layer latency:
  Layer 1: 0.3ms | Layer 2: 2.1ms | Layer 3: 47ms | Layer 4: 218ms
  Total safety overhead: ~267ms

Recommendation: Use layers 1+2 for all traffic (fast),
                Layer 3 for user-facing endpoints,
                Layer 4 for high-stakes workflows only.
```
