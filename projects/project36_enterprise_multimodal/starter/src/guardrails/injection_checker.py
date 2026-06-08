"""
src/guardrails/injection_checker.py — Layer 1: Prompt injection detection.

TODO:
  1. Define INJECTION_PATTERNS — at least 8 compiled regex patterns
  2. implement check() — scan text against all patterns, return (safe, reason)
"""
from __future__ import annotations
import re

# ── TODO 1: Define injection patterns (compile once for O(1) per call) ────────
# INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
#     (re.compile(r"ignore\s+(previous|prior|above)\s+instructions?", re.I), "ignore_instructions"),
#     (re.compile(r"you\s+are\s+now\s+(?:a|an)\s+\w+", re.I), "persona_override"),
#     (re.compile(r"disregard\s+(all|your|the)\s+\w+", re.I), "disregard_command"),
#     (re.compile(r"jailbreak|DAN\b|do\s+anything\s+now", re.I), "jailbreak_attempt"),
#     (re.compile(r"system\s*prompt\s*[:=]", re.I), "system_prompt_override"),
#     (re.compile(r"</?(system|instruction|prompt)>", re.I), "xml_injection"),
#     (re.compile(r"print\s+(your\s+)?(system\s+)?prompt", re.I), "prompt_extraction"),
#     (re.compile(r"repeat\s+(after|the\s+following)", re.I), "repeat_injection"),
# ]

INJECTION_PATTERNS: list[tuple] = []  # TODO: fill this in


# ── TODO 2: Check text for injection ─────────────────────────────────────────
def check(text: str) -> tuple[bool, str]:
    """
    Scan `text` against all injection patterns.

    Steps:
      2a. For each (pattern, label) in INJECTION_PATTERNS:
              if pattern.search(text): return (False, f"injection_{label}")
      2b. If no pattern matched: return (True, "")

    Returns:
        (safe: bool, reason: str) — reason is empty string when safe=True
    """
    raise NotImplementedError
