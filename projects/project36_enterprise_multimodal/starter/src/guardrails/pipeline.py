"""
src/guardrails/pipeline.py — Orchestrate all 4 guardrail layers.

Pipeline order:
  L1 (sync):  injection_checker.check()        — fast regex, blocks immediately
  L2 (sync):  pii_scanner.scan_and_anonymize() — sanitize text before sending to LLM
  L3 (async): safety_checker.check_safety()    — LlamaGuard / GPT-4o
  L4 (async): topic_relevance_check()          — is question relevant to compliance?

Key insight: L3 + L4 run in asyncio.gather (parallel) to halve latency.

TODO:
  1. implement topic_relevance_check() — L4 topic gate
  2. implement run_pipeline() — chain all 4 layers, return GuardrailResult
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    safe: bool
    sanitized_text: str
    issues: list[str] = field(default_factory=list)
    pii_types_found: list[str] = field(default_factory=list)
    blocked_layer: str | None = None   # "L1" | "L2" | "L3" | "L4" | None


# ── TODO 1: Topic relevance check (Layer 4) ───────────────────────────────────
async def topic_relevance_check(
    text: str,
    model: str = "openai/gpt-4o-mini",
    allowed_topics: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Gate off-topic requests (Layer 4).

    Default allowed topics: compliance, legal, regulatory, GDPR, contracts,
    risk assessment, auditing, data privacy, SOC2, ISO 27001, HIPAA.

    Steps:
      1a. Build a short prompt listing allowed topics
      1b. litellm.acompletion with response_format={"type": "json_object"}
          Ask: {"relevant": bool, "topic": str, "reason": str}
      1c. Return (result["relevant"], result.get("reason", ""))
      1d. On exception: return (True, "") — fail open

    Returns:
        tuple[bool, str] — (is_relevant, reason)
    """
    # import litellm, json
    # allowed = allowed_topics or ["compliance", "legal", "GDPR", ...]
    raise NotImplementedError


# ── TODO 2: Run full 4-layer pipeline ────────────────────────────────────────
async def run_pipeline(
    text: str,
    check_topic: bool = True,
) -> GuardrailResult:
    """
    Execute all 4 guardrail layers and return a GuardrailResult.

    Layer execution order:
      L1 (sync):  injection_checker.check(text)
                  → if blocked: return immediately (GuardrailResult(safe=False, blocked_layer="L1"))
      L2 (sync):  pii_scanner.scan_and_anonymize(text)
                  → sanitized_text replaces text for all subsequent layers
                  → always continues (PII found is not a block, just anonymized)
      L3+L4 (async, parallel):
                  safe_l3, reason_l3 = await safety_checker.check_safety(sanitized_text)
                  safe_l4, reason_l4 = await topic_relevance_check(sanitized_text)
                  → asyncio.gather both
                  → if L3 blocked: GuardrailResult(safe=False, blocked_layer="L3")
                  → if L4 blocked: GuardrailResult(safe=False, blocked_layer="L4")
      If all pass:
                  Return GuardrailResult(safe=True, sanitized_text=sanitized_text, ...)

    Steps:
      2a. L1 check
      2b. L2 scan + anonymize → update sanitized_text
      2c. asyncio.gather(L3, L4) if check_topic else just L3
      2d. Evaluate results, build and return GuardrailResult

    Returns:
        GuardrailResult — with all issues, PII types, and blocked_layer populated
    """
    # from src.guardrails import injection_checker, pii_scanner, safety_checker
    raise NotImplementedError
