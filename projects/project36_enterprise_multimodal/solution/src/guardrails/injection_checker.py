"""
solution/src/guardrails/injection_checker.py — Full implementation.
"""
from __future__ import annotations
import re

INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(previous|prior|above)\s+instructions?", re.I), "ignore_instructions"),
    (re.compile(r"you\s+are\s+now\s+(?:a|an)\s+\w+", re.I), "persona_override"),
    (re.compile(r"disregard\s+(all|your|the)\s+\w+", re.I), "disregard_command"),
    (re.compile(r"jailbreak|DAN\b|do\s+anything\s+now", re.I), "jailbreak"),
    (re.compile(r"system\s*prompt\s*[:=]", re.I), "system_prompt_override"),
    (re.compile(r"</?(system|instruction|prompt)>", re.I), "xml_injection"),
    (re.compile(r"print\s+(your\s+)?(system\s+)?prompt", re.I), "prompt_extraction"),
    (re.compile(r"repeat\s+(after|the\s+following)", re.I), "repeat_injection"),
]


def check(text: str) -> tuple[bool, str]:
    for pattern, label in INJECTION_PATTERNS:
        if pattern.search(text):
            return False, f"injection_{label}"
    return True, ""
