"""
src/guardrails/pii_scanner.py — Layer 2: PII detection and anonymization.

TODO:
  1. Define PII_PATTERNS — regex patterns for email, phone, SSN, credit card, IP
  2. implement scan_and_anonymize() — replace all PII with [TYPE_REDACTED] tokens
  3. implement scan_only() — detect without modifying (for logging)
"""
from __future__ import annotations
import re

# ── TODO 1: Define PII patterns (compile once) ────────────────────────────────
# PII_PATTERNS: dict[str, re.Pattern] = {
#     "email":        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
#     "phone":        re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
#     "ssn":          re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
#     "credit_card":  re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
#     "ip_address":   re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
#     "passport":     re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
#     "iban":         re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"),
# }

PII_PATTERNS: dict[str, object] = {}  # TODO: fill this in


# ── TODO 2: Scan and anonymize ────────────────────────────────────────────────
def scan_and_anonymize(text: str) -> tuple[str, list[str]]:
    """
    Replace all PII matches with [TYPE_REDACTED] tokens.

    Steps:
      2a. For each (pii_type, pattern) in PII_PATTERNS.items():
              text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
              if any match was replaced: track pii_type in found_types
      2b. Return (sanitized_text, found_types)

    Tip: Use pattern.sub(..., text) which replaces ALL occurrences (not just first).

    Returns:
        tuple[str, list[str]] — (sanitized text, list of PII types found)
    """
    raise NotImplementedError


# ── TODO 3: Scan only (no modification) ──────────────────────────────────────
def scan_only(text: str) -> list[dict]:
    """
    Detect PII without modifying the text — useful for logging/alerting.

    Steps:
      3a. For each (pii_type, pattern) in PII_PATTERNS.items():
              for match in pattern.finditer(text):
                  record {"type": pii_type, "start": match.start(), "end": match.end(),
                          "value": match.group()[:4] + "****"}  # partial for logging
      3b. Return list of all finds

    Returns:
        list[dict] — one dict per PII match found
    """
    raise NotImplementedError
