"""
Project 14 — Document Intelligence Pipeline (Starter)

Build an end-to-end document processing pipeline that:
  1. Classifies each document type (invoice/contract/report/unknown)
  2. Extracts structured data via LLM → Pydantic validation
  3. Applies business-rule validation (amounts, dates, required fields)
  4. Deduplicates by SHA-256 content hash
  5. Detects anomalies (amount > threshold, date inversions, etc.)
  6. Stores valid documents in a mock in-memory DB
  7. Generates an aggregated JSON + HTML report

Usage:
  python starter.py

Processes 8 embedded sample documents concurrently with asyncio.gather().
"""

import os, sys, json, asyncio, hashlib, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from llm import achat, get_text, MODEL

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Embedded Sample Documents
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENTS = [
    {
        "id": "doc001", "filename": "invoice_acme_may.txt",
        "content": """
INVOICE
Vendor: Acme Corp  |  Invoice #: INV-2026-0512  |  Date: 2026-05-12  |  Due: 2026-06-12
Bill To: TechStart Inc

Items:
  Cloud Server Hosting (3 months) x 1  @ $850.00
  SSL Certificate                 x 2  @  $75.00
Subtotal: $1,000.00   Tax (10%): $100.00   Total: $1,100.00
""",
    },
    {
        "id": "doc002", "filename": "invoice_bigcorp_high.txt",
        "content": """
INVOICE
Vendor: BigCorp Ltd  |  Invoice #: INV-2026-0099  |  Date: 2026-05-20  |  Due: 2026-06-20
Bill To: Enterprise Client

Items:
  Enterprise Software License x 1  @ $250,000.00
Total: $250,000.00
""",
    },
    {
        "id": "doc003", "filename": "invoice_acme_may_copy.txt",
        "content": """
INVOICE
Vendor: Acme Corp  |  Invoice #: INV-2026-0512  |  Date: 2026-05-12  |  Due: 2026-06-12
Bill To: TechStart Inc

Items:
  Cloud Server Hosting (3 months) x 1  @ $850.00
  SSL Certificate                 x 2  @  $75.00
Subtotal: $1,000.00   Tax (10%): $100.00   Total: $1,100.00
""",
    },
    {
        "id": "doc004", "filename": "contract_cloudco.txt",
        "content": """
SERVICE AGREEMENT
Contract ID: CTR-2026-001
Between: TechStart Inc  AND  CloudCo Services
Start Date: 2026-06-01   End Date: 2027-05-31
Total Contract Value: $36,000.00  (12 payments of $3,000/month)
Governing Law: State of California

Key Obligations:
- CloudCo to provide 99.9% uptime SLA
- TechStart to pay invoices within 30 days
- 90-day termination notice required
""",
    },
    {
        "id": "doc005", "filename": "contract_bad_dates.txt",
        "content": """
SERVICE AGREEMENT
Contract ID: CTR-2026-002
Between: Alpha Inc  AND  Beta Corp
Start Date: 2026-12-01   End Date: 2026-03-01
Total Contract Value: $12,000.00
Governing Law: New York

Key Obligations:
- Alpha to deliver software by end date
- Beta to pay 50% upfront
""",
    },
    {
        "id": "doc006", "filename": "contract_missing_fields.txt",
        "content": """
SERVICE AGREEMENT
Contract ID: CTR-2026-003

Start Date: 2026-07-01   End Date: 2027-06-30
Total Contract Value: $5,000.00

Key Obligations:
- Deliver monthly reports
""",
    },
    {
        "id": "doc007", "filename": "quarterly_report_q1.txt",
        "content": """
Q1 2026 Business Performance Report
Prepared by: Finance Team  |  Date: 2026-04-05

Executive Summary:
Total revenue reached $2.4M in Q1, representing a 15% YoY growth.
Operating costs rose 8% due to infrastructure investments.
Net margin: 22%.

Key Highlights:
- North region outperformed by 12% vs target
- New product Widget C launched successfully
- Customer churn reduced to 2.1% (from 3.4% last year)
""",
    },
    {
        "id": "doc008", "filename": "invoice_zeta_june.txt",
        "content": """
INVOICE
Vendor: Zeta Technologies  |  Invoice #: INV-2026-0601  |  Date: 2026-06-01  |  Due: 2026-07-01
Bill To: TechStart Inc

Items:
  API Integration Services  x 5  @ $750.00
  Technical Documentation   x 1  @ $1,000.00
Subtotal: $4,750.00   Tax: $0.00 (B2B exempt)   Total: $4,750.00
""",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Extraction Models
# ══════════════════════════════════════════════════════════════════════════════

class InvoiceData(BaseModel):
    invoice_number: str
    vendor:         str
    billed_to:      str
    amount_total:   float
    currency:       str = "USD"
    invoice_date:   str        # YYYY-MM-DD
    due_date:       str        # YYYY-MM-DD
    line_items:     list[str] = []

class ContractData(BaseModel):
    contract_id:     str
    parties:         list[str]     # must have >= 2 entries
    start_date:      str           # YYYY-MM-DD
    end_date:        str           # YYYY-MM-DD
    total_value:     Optional[float] = None
    governing_law:   Optional[str] = None
    key_obligations: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════════════

async def _llm(prompt: str,
               system: str = "You are a document extraction expert. Return only valid JSON.") -> str:
    r = await achat([{"role": "user", "content": prompt}], system=system, max_tokens=600)
    return get_text(r)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Document Classifier
# ══════════════════════════════════════════════════════════════════════════════

DOC_TYPES = ["invoice", "contract", "report", "unknown"]


async def classify_document(content: str) -> str:
    """TODO: Send the first 400 characters of content to LLM with a prompt
       that defines all 4 document types. Return one of DOC_TYPES.
       Default to 'unknown' if the reply is not in DOC_TYPES.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Structured Extractor
# ══════════════════════════════════════════════════════════════════════════════

async def extract_invoice(content: str) -> Optional[InvoiceData]:
    """TODO: Send document content to LLM. Ask for JSON matching InvoiceData schema.
       Parse JSON → InvoiceData.model_validate(parsed).
       Return None on JSONDecodeError or ValidationError.
       Strip markdown fences before parsing.
    """
    raise NotImplementedError


async def extract_contract(content: str) -> Optional[ContractData]:
    """TODO: Send document content to LLM. Ask for JSON matching ContractData schema.
       Parse JSON → ContractData.model_validate(parsed).
       Return None on JSONDecodeError or ValidationError.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Business Rule Validator
# ══════════════════════════════════════════════════════════════════════════════

INVOICE_MAX_AMOUNT = 100_000.0    # flag invoices above this

def validate_invoice(inv: InvoiceData) -> list[str]:
    """TODO: Return list of violation strings. Check:
       - vendor is non-empty
       - billed_to is non-empty
       - amount_total > 0
       - invoice_date and due_date are parseable (YYYY-MM-DD format)
       Return [] if all checks pass.
    """
    raise NotImplementedError


def validate_contract(con: ContractData) -> list[str]:
    """TODO: Return list of violation strings. Check:
       - len(parties) >= 2
       - start_date and end_date are parseable (YYYY-MM-DD)
       - end_date > start_date
       Return [] if all checks pass.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Deduplicator
# ══════════════════════════════════════════════════════════════════════════════

class Deduplicator:
    def __init__(self):
        self._seen: set[str] = set()

    def _normalise(self, text: str) -> str:
        """TODO: Lowercase, strip leading/trailing whitespace, collapse internal whitespace."""
        raise NotImplementedError

    def _hash(self, text: str) -> str:
        """TODO: SHA-256 hex digest of normalised text."""
        raise NotImplementedError

    def is_duplicate(self, text: str) -> bool:
        """TODO: Return True if hash of normalised text is already in _seen."""
        raise NotImplementedError

    def add(self, text: str):
        """TODO: Add hash of normalised text to _seen."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Anomaly Detector
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(doc_type: str, extracted) -> list[str]:
    """TODO: Return list of anomaly description strings.
       For invoices: flag if amount_total > INVOICE_MAX_AMOUNT
       For contracts: flag if end_date <= start_date (belt-and-suspenders after validation)
       Return [] if no anomalies.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Mock Database
# ══════════════════════════════════════════════════════════════════════════════

class MockDB:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def store(self, content_hash: str, doc_type: str, extracted_data) -> bool:
        """TODO: Store {doc_type, data: extracted_data.model_dump()} under content_hash.
           Return True on new insert, False if key already exists (upsert = False).
        """
        raise NotImplementedError

    def count(self) -> int:
        return len(self._store)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocResult:
    doc_id:        str
    filename:      str
    doc_type:      str = "unknown"
    status:        str = "pending"   # extracted | invalid | duplicate | anomaly | unknown
    extracted:     Optional[dict] = None
    violations:    list[str] = field(default_factory=list)
    anomalies:     list[str] = field(default_factory=list)
    content_hash:  str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Per-Document Pipeline
# ══════════════════════════════════════════════════════════════════════════════

async def process_document(doc: dict, dedup: Deduplicator, db: MockDB) -> DocResult:
    """TODO: Run the full pipeline for one document.
       Steps:
         1. classify_document(content) → doc_type
         2. dedup.is_duplicate(content)?  → status='duplicate', return early
         3. dedup.add(content)
         4. Branch on doc_type:
              invoice  → extract_invoice() → validate_invoice()
              contract → extract_contract() → validate_contract()
              _        → status='unknown', return
         5. If extraction failed (None): status='invalid', return
         6. If violations: status='invalid', return
         7. detect_anomalies() → if anomalies: status='anomaly'
            else: status='extracted'
         8. db.store(hash, doc_type, extracted)
         9. Return DocResult with all fields populated
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Report Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(results: list[DocResult]) -> tuple[dict, str]:
    """TODO: Build:
       report_dict = {
           summary: {total, extracted, invalid, duplicate, anomaly, unknown},
           documents: [per-doc dicts from result]
       }
       html = minimal HTML page with a <table> of document statuses
       Return (report_dict, html_str)
    """
    raise NotImplementedError


def _status_icon(status: str) -> str:
    return {"extracted": "✅", "invalid": "❌", "duplicate": "♻️",
            "anomaly": "⚠️", "unknown": "❓"}.get(status, "•")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print(f"{'='*65}")
    print(f" Project 14 — Document Intelligence Pipeline   [{MODEL}]")
    print(f"{'='*65}")
    print(f"Processing {len(DOCUMENTS)} documents concurrently…\n")

    dedup = Deduplicator()
    db    = MockDB()

    # Process all documents concurrently
    results: list[DocResult] = await asyncio.gather(
        *[process_document(doc, dedup, db) for doc in DOCUMENTS]
    )

    # Print per-doc summary
    for r in results:
        icon = _status_icon(r.status)
        ext_info = ""
        if r.extracted:
            if r.doc_type == "invoice":
                ext_info = f"  {r.extracted.get('invoice_number', '')}  ${r.extracted.get('amount_total', 0):,.2f}"
            elif r.doc_type == "contract":
                ext_info = f"  {r.extracted.get('contract_id', '')}"
        anom = f"  [{', '.join(r.anomalies)}]" if r.anomalies else ""
        viol = f"  [{', '.join(r.violations)}]" if r.violations else ""
        print(f"  {icon} {r.doc_id:<8} {r.doc_type:<9} {r.status:<12}{ext_info}{anom}{viol}")

    # Generate report
    report_dict, html = generate_report(results)
    summary = report_dict["summary"]
    print(
        f"\nSummary: {summary['total']} docs | "
        f"{summary['extracted']} extracted | {summary['invalid']} invalid | "
        f"{summary['duplicate']} duplicate | {summary['anomaly']} anomalies | "
        f"{summary['unknown']} unknown"
    )

    Path("doc_pipeline_report.json").write_text(json.dumps(report_dict, indent=2))
    Path("doc_pipeline_report.html").write_text(html)
    print("✅ Report saved → doc_pipeline_report.json | doc_pipeline_report.html")
    print(f"   DB records stored: {db.count()}")


if __name__ == "__main__":
    asyncio.run(main())
