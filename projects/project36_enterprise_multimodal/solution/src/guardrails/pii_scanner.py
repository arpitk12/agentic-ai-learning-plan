"""
solution/src/guardrails/pii_scanner.py — Full implementation.
"""
from __future__ import annotations
import re

PII_PATTERNS: dict[str, re.Pattern] = {
    "email":       re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone":       re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "ip_address":  re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "passport":    re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    "iban":        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"),
}


def scan_and_anonymize(text: str) -> tuple[str, list[str]]:
    found_types: list[str] = []
    for pii_type, pattern in PII_PATTERNS.items():
        new_text, n = pattern.subn(f"[{pii_type.upper()}_REDACTED]", text)
        if n > 0:
            found_types.append(pii_type)
            text = new_text
    return text, found_types


def scan_only(text: str) -> list[dict]:
    finds: list[dict] = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            val = match.group()
            finds.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "value": val[:4] + "****" if len(val) > 4 else "****",
            })
    return finds
