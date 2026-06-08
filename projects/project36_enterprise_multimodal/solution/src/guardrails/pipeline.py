"""
solution/src/guardrails/pipeline.py — Full implementation (4-layer).
"""
from __future__ import annotations
import asyncio
import json
import litellm  # type: ignore
from dataclasses import dataclass, field
from src.guardrails.injection_checker import check as injection_check  # type: ignore
from src.guardrails.pii_scanner import scan_and_anonymize  # type: ignore

_LLAMA_GUARD_MODEL = None
_LLAMA_GUARD_TOKENIZER = None


@dataclass
class GuardrailResult:
    safe: bool
    sanitized_text: str
    issues: list[str] = field(default_factory=list)
    pii_types_found: list[str] = field(default_factory=list)
    blocked_layer: str | None = None


async def _safety_check(text: str, model: str = "openai/gpt-4o-mini") -> tuple[bool, str]:
    if _LLAMA_GUARD_MODEL is not None:
        # Use loaded LlamaGuard model
        import torch  # type: ignore
        inputs = _LLAMA_GUARD_TOKENIZER([text], return_tensors="pt").to(_LLAMA_GUARD_MODEL.device)
        with torch.no_grad():
            output = _LLAMA_GUARD_MODEL.generate(**inputs, max_new_tokens=20)
        decoded = _LLAMA_GUARD_TOKENIZER.decode(output[0], skip_special_tokens=True).strip().lower()
        safe = "safe" in decoded and "unsafe" not in decoded
        category = decoded.replace("safe", "").replace("unsafe", "").strip()
        return safe, category

    # GPT-4o-mini fallback
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content":
                f"Is this text safe for a compliance AI assistant?\nText: {text[:500]}\n"
                'Return JSON: {"safe": true/false, "reason": "...", "category": "..."}'}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("safe", True), data.get("reason", "")
    except Exception:
        return True, ""   # fail open


async def _topic_check(text: str, model: str = "openai/gpt-4o-mini") -> tuple[bool, str]:
    allowed = ["compliance", "legal", "regulatory", "GDPR", "contract", "risk",
               "audit", "data privacy", "SOC2", "ISO 27001", "HIPAA", "PCI DSS"]
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content":
                f"Is this question about: {', '.join(allowed)}?\n"
                f"Question: {text[:300]}\n"
                'Return JSON: {"relevant": true/false, "reason": "..."}'}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("relevant", True), data.get("reason", "")
    except Exception:
        return True, ""   # fail open


async def run_pipeline(text: str, check_topic: bool = True) -> GuardrailResult:
    issues: list[str] = []

    # L1: Injection check (sync, fast)
    safe_l1, reason_l1 = injection_check(text)
    if not safe_l1:
        return GuardrailResult(safe=False, sanitized_text=text,
                               issues=[reason_l1], blocked_layer="L1")

    # L2: PII scan + anonymize (sync — sanitized_text used for L3+L4)
    sanitized_text, pii_found = scan_and_anonymize(text)
    if pii_found:
        issues.append(f"pii_found: {','.join(pii_found)}")

    # L3 + L4: run in parallel (both async, independent)
    if check_topic:
        (safe_l3, reason_l3), (safe_l4, reason_l4) = await asyncio.gather(
            _safety_check(sanitized_text),
            _topic_check(sanitized_text),
        )
    else:
        safe_l3, reason_l3 = await _safety_check(sanitized_text)
        safe_l4, reason_l4 = True, ""

    if not safe_l3:
        issues.append(f"safety: {reason_l3}")
        return GuardrailResult(safe=False, sanitized_text=sanitized_text,
                               issues=issues, pii_types_found=pii_found, blocked_layer="L3")

    if not safe_l4:
        issues.append(f"off_topic: {reason_l4}")
        return GuardrailResult(safe=False, sanitized_text=sanitized_text,
                               issues=issues, pii_types_found=pii_found, blocked_layer="L4")

    return GuardrailResult(safe=True, sanitized_text=sanitized_text,
                           issues=issues, pii_types_found=pii_found)
