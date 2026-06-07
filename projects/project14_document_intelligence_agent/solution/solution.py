"""
Project 14 — Document Intelligence Pipeline (Solution)

Full pipeline: classify → extract (LLM+Pydantic) → validate → deduplicate
→ detect anomalies → store (MockDB) → JSON + HTML report.
"""

import os, sys, json, asyncio, hashlib, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

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
    {"id": "doc001", "filename": "invoice_acme_may.txt", "content": """
INVOICE
Vendor: Acme Corp  |  Invoice #: INV-2026-0512  |  Date: 2026-05-12  |  Due: 2026-06-12
Bill To: TechStart Inc
Items:
  Cloud Server Hosting (3 months) x 1  @ $850.00
  SSL Certificate                 x 2  @  $75.00
Subtotal: $1,000.00   Tax (10%): $100.00   Total: $1,100.00
"""},
    {"id": "doc002", "filename": "invoice_bigcorp_high.txt", "content": """
INVOICE
Vendor: BigCorp Ltd  |  Invoice #: INV-2026-0099  |  Date: 2026-05-20  |  Due: 2026-06-20
Bill To: Enterprise Client
Items:
  Enterprise Software License x 1  @ $250,000.00
Total: $250,000.00
"""},
    {"id": "doc003", "filename": "invoice_acme_may_copy.txt", "content": """
INVOICE
Vendor: Acme Corp  |  Invoice #: INV-2026-0512  |  Date: 2026-05-12  |  Due: 2026-06-12
Bill To: TechStart Inc
Items:
  Cloud Server Hosting (3 months) x 1  @ $850.00
  SSL Certificate                 x 2  @  $75.00
Subtotal: $1,000.00   Tax (10%): $100.00   Total: $1,100.00
"""},
    {"id": "doc004", "filename": "contract_cloudco.txt", "content": """
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
"""},
    {"id": "doc005", "filename": "contract_bad_dates.txt", "content": """
SERVICE AGREEMENT
Contract ID: CTR-2026-002
Between: Alpha Inc  AND  Beta Corp
Start Date: 2026-12-01   End Date: 2026-03-01
Total Contract Value: $12,000.00
Governing Law: New York
Key Obligations:
- Alpha to deliver software by end date
- Beta to pay 50% upfront
"""},
    {"id": "doc006", "filename": "contract_missing_fields.txt", "content": """
SERVICE AGREEMENT
Contract ID: CTR-2026-003
Start Date: 2026-07-01   End Date: 2027-06-30
Total Contract Value: $5,000.00
Key Obligations:
- Deliver monthly reports
"""},
    {"id": "doc007", "filename": "quarterly_report_q1.txt", "content": """
Q1 2026 Business Performance Report
Prepared by: Finance Team  |  Date: 2026-04-05
Executive Summary:
Total revenue reached $2.4M in Q1, representing a 15% YoY growth.
Net margin: 22%.
Key Highlights:
- North region outperformed by 12% vs target
- Customer churn reduced to 2.1%
"""},
    {"id": "doc008", "filename": "invoice_zeta_june.txt", "content": """
INVOICE
Vendor: Zeta Technologies  |  Invoice #: INV-2026-0601  |  Date: 2026-06-01  |  Due: 2026-07-01
Bill To: TechStart Inc
Items:
  API Integration Services  x 5  @ $750.00
  Technical Documentation   x 1  @ $1,000.00
Subtotal: $4,750.00   Tax: $0.00 (B2B exempt)   Total: $4,750.00
"""},
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
    invoice_date:   str
    due_date:       str
    line_items:     list[str] = []

class ContractData(BaseModel):
    contract_id:     str
    parties:         list[str]
    start_date:      str
    end_date:        str
    total_value:     Optional[float] = None
    governing_law:   Optional[str] = None
    key_obligations: list[str] = []

# ══════════════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════════════

async def _llm(prompt: str,
               system: str = "You are a document extraction expert. Return only valid JSON.") -> str:
    r = await achat([{"role": "user", "content": prompt}], system=system, max_tokens=700)
    return get_text(r)

def _strip_fences(text: str) -> str:
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Classifier
# ══════════════════════════════════════════════════════════════════════════════

DOC_TYPES = ["invoice", "contract", "report", "unknown"]

async def classify_document(content: str) -> str:
    prompt = (
        "Classify this document into exactly one category:\n"
        "  invoice  — vendor billing document with amounts and line items\n"
        "  contract — service/legal agreement between named parties\n"
        "  report   — periodic business or financial report\n"
        "  unknown  — anything else\n\n"
        f"Document (first 400 chars):\n{content[:400]}\n\n"
        "Reply with ONLY one word."
    )
    raw = (await _llm(prompt, system="You are a document classifier. Reply with one word.")).strip().lower()
    return raw if raw in DOC_TYPES else "unknown"

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Extractor
# ══════════════════════════════════════════════════════════════════════════════

async def extract_invoice(content: str) -> Optional[InvoiceData]:
    schema = (
        '{"invoice_number":"str","vendor":"str","billed_to":"str",'
        '"amount_total":float,"currency":"str","invoice_date":"YYYY-MM-DD",'
        '"due_date":"YYYY-MM-DD","line_items":["str"]}'
    )
    prompt = (
        f"Extract invoice data from this document.\n\n"
        f"Document:\n{content}\n\n"
        f"Return ONLY valid JSON matching this schema:\n{schema}"
    )
    raw = _strip_fences(await _llm(prompt))
    try:
        return InvoiceData.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None


async def extract_contract(content: str) -> Optional[ContractData]:
    schema = (
        '{"contract_id":"str","parties":["str","str"],'
        '"start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD",'
        '"total_value":float_or_null,"governing_law":"str_or_null",'
        '"key_obligations":["str"]}'
    )
    prompt = (
        f"Extract contract data from this document.\n\n"
        f"Document:\n{content}\n\n"
        f"Return ONLY valid JSON matching this schema:\n{schema}"
    )
    raw = _strip_fences(await _llm(prompt))
    try:
        return ContractData.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Validator
# ══════════════════════════════════════════════════════════════════════════════

INVOICE_MAX_AMOUNT = 100_000.0

def _parse_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None

def validate_invoice(inv: InvoiceData) -> list[str]:
    violations: list[str] = []
    if not inv.vendor.strip():
        violations.append("missing: vendor")
    if not inv.billed_to.strip():
        violations.append("missing: billed_to")
    if inv.amount_total <= 0:
        violations.append(f"invalid amount: {inv.amount_total}")
    if not _parse_date(inv.invoice_date):
        violations.append(f"invalid invoice_date: {inv.invoice_date}")
    if not _parse_date(inv.due_date):
        violations.append(f"invalid due_date: {inv.due_date}")
    return violations


def validate_contract(con: ContractData) -> list[str]:
    violations: list[str] = []
    if len(con.parties) < 2:
        violations.append(f"missing: parties (got {len(con.parties)}, need ≥2)")
    sd = _parse_date(con.start_date)
    ed = _parse_date(con.end_date)
    if not sd:
        violations.append(f"invalid start_date: {con.start_date}")
    if not ed:
        violations.append(f"invalid end_date: {con.end_date}")
    if sd and ed and ed <= sd:
        violations.append(f"end_date ({con.end_date}) is not after start_date ({con.start_date})")
    return violations

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Deduplicator
# ══════════════════════════════════════════════════════════════════════════════

class Deduplicator:
    def __init__(self):
        self._seen: set[str] = set()

    def _normalise(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())

    def _hash(self, text: str) -> str:
        return hashlib.sha256(self._normalise(text).encode()).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        return self._hash(text) in self._seen

    def add(self, text: str):
        self._seen.add(self._hash(text))

    def get_hash(self, text: str) -> str:
        return self._hash(text)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Anomaly Detector
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(doc_type: str, extracted) -> list[str]:
    anomalies: list[str] = []
    if doc_type == "invoice" and isinstance(extracted, InvoiceData):
        if extracted.amount_total > INVOICE_MAX_AMOUNT:
            anomalies.append(f"amount ${extracted.amount_total:,.2f} exceeds ${INVOICE_MAX_AMOUNT:,.0f} threshold")
    elif doc_type == "contract" and isinstance(extracted, ContractData):
        sd = _parse_date(extracted.start_date)
        ed = _parse_date(extracted.end_date)
        if sd and ed and ed <= sd:
            anomalies.append(f"end_date before start_date ({extracted.end_date} ≤ {extracted.start_date})")
    return anomalies

# ══════════════════════════════════════════════════════════════════════════════
# Mock Database
# ══════════════════════════════════════════════════════════════════════════════

class MockDB:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def store(self, content_hash: str, doc_type: str, extracted_data) -> bool:
        if content_hash in self._store:
            return False
        self._store[content_hash] = {
            "doc_type": doc_type,
            "data":     extracted_data.model_dump(),
        }
        return True

    def count(self) -> int:
        return len(self._store)

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocResult:
    doc_id:       str
    filename:     str
    doc_type:     str = "unknown"
    status:       str = "pending"
    extracted:    Optional[dict] = None
    violations:   list[str] = field(default_factory=list)
    anomalies:    list[str] = field(default_factory=list)
    content_hash: str = ""

# ══════════════════════════════════════════════════════════════════════════════
# Per-Document Pipeline
# ══════════════════════════════════════════════════════════════════════════════

async def process_document(doc: dict, dedup: Deduplicator, db: MockDB) -> DocResult:
    content  = doc["content"]
    result   = DocResult(doc_id=doc["id"], filename=doc["filename"])

    # Stage 1: Classify
    result.doc_type    = await classify_document(content)
    result.content_hash = dedup.get_hash(content)

    # Stage 2: Dedup check
    if dedup.is_duplicate(content):
        result.status = "duplicate"
        return result
    dedup.add(content)

    # Stage 3: Extract + Validate
    extracted = None
    if result.doc_type == "invoice":
        extracted = await extract_invoice(content)
        if extracted is None:
            result.status = "invalid"
            result.violations = ["extraction failed"]
            return result
        result.violations = validate_invoice(extracted)

    elif result.doc_type == "contract":
        extracted = await extract_contract(content)
        if extracted is None:
            result.status = "invalid"
            result.violations = ["extraction failed"]
            return result
        result.violations = validate_contract(extracted)

    else:
        result.status = "unknown"
        return result

    if result.violations:
        result.status = "invalid"
        return result

    # Stage 4: Anomaly detection
    result.anomalies = detect_anomalies(result.doc_type, extracted)
    result.extracted = extracted.model_dump()

    if result.anomalies:
        result.status = "anomaly"
    else:
        result.status = "extracted"
        db.store(result.content_hash, result.doc_type, extracted)

    return result

# ══════════════════════════════════════════════════════════════════════════════
# Report Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(results: list[DocResult]) -> tuple[dict, str]:
    from collections import Counter
    status_counts = Counter(r.status for r in results)
    report_dict = {
        "summary": {
            "total":     len(results),
            "extracted": status_counts.get("extracted", 0),
            "invalid":   status_counts.get("invalid", 0),
            "duplicate": status_counts.get("duplicate", 0),
            "anomaly":   status_counts.get("anomaly", 0),
            "unknown":   status_counts.get("unknown", 0),
        },
        "documents": [
            {
                "doc_id":       r.doc_id,
                "filename":     r.filename,
                "doc_type":     r.doc_type,
                "status":       r.status,
                "content_hash": r.content_hash[:16] + "…",
                "violations":   r.violations,
                "anomalies":    r.anomalies,
                "extracted":    r.extracted,
            }
            for r in results
        ],
    }

    ICON = {"extracted": "✅", "invalid": "❌", "duplicate": "♻️", "anomaly": "⚠️", "unknown": "❓"}
    rows = "".join(
        f"<tr><td>{ICON.get(r.status,'•')}</td><td>{r.doc_id}</td>"
        f"<td>{r.filename}</td><td>{r.doc_type}</td>"
        f"<td><b>{r.status}</b></td>"
        f"<td>{'<br>'.join(r.violations + r.anomalies) or '—'}</td></tr>"
        for r in results
    )
    s = report_dict["summary"]
    html = f"""<!DOCTYPE html><html><head>
<title>Document Intelligence Report</title>
<style>body{{font-family:sans-serif;padding:20px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#f0f0f0}}</style></head><body>
<h1>Document Intelligence Pipeline Report</h1>
<p><b>Total:</b> {s['total']} &nbsp;|&nbsp;
   <b>Extracted:</b> {s['extracted']} &nbsp;|&nbsp;
   <b>Invalid:</b> {s['invalid']} &nbsp;|&nbsp;
   <b>Duplicate:</b> {s['duplicate']} &nbsp;|&nbsp;
   <b>Anomaly:</b> {s['anomaly']} &nbsp;|&nbsp;
   <b>Unknown:</b> {s['unknown']}</p>
<table><tr><th></th><th>ID</th><th>Filename</th><th>Type</th>
<th>Status</th><th>Issues</th></tr>
{rows}
</table></body></html>"""
    return report_dict, html


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

    results: list[DocResult] = await asyncio.gather(
        *[process_document(doc, dedup, db) for doc in DOCUMENTS]
    )

    for r in results:
        icon = _status_icon(r.status)
        ext_info = ""
        if r.extracted:
            if r.doc_type == "invoice":
                ext_info = f"  {r.extracted.get('invoice_number','')}  ${r.extracted.get('amount_total',0):,.2f}"
            elif r.doc_type == "contract":
                ext_info = f"  {r.extracted.get('contract_id','')}"
        issues = ""
        if r.anomalies:
            issues += f"  [{', '.join(r.anomalies)}]"
        if r.violations:
            issues += f"  [{', '.join(r.violations)}]"
        print(f"  {icon} {r.doc_id:<8} {r.doc_type:<9} {r.status:<12}{ext_info}{issues}")

    report_dict, html = generate_report(results)
    s = report_dict["summary"]
    print(
        f"\nSummary: {s['total']} docs | "
        f"{s['extracted']} extracted | {s['invalid']} invalid | "
        f"{s['duplicate']} duplicate | {s['anomaly']} anomalies | "
        f"{s['unknown']} unknown"
    )

    Path("doc_pipeline_report.json").write_text(json.dumps(report_dict, indent=2, ensure_ascii=False))
    Path("doc_pipeline_report.html").write_text(html)
    print("✅ Report saved → doc_pipeline_report.json | doc_pipeline_report.html")
    print(f"   DB records stored: {db.count()}")


if __name__ == "__main__":
    asyncio.run(main())
