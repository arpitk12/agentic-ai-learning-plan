# Project 14 — Document Intelligence Pipeline

## What You Build

An autonomous document processing pipeline that ingests a batch of heterogeneous
documents (invoices and contracts), classifies each document type, extracts
structured data via LLM + Pydantic validation, deduplicates by content hash,
detects business-rule anomalies, stores to a mock database, and generates an
aggregated JSON + HTML report — all without any external file I/O.

---

## Architecture

```
Sample Documents (8 embedded strings)
          │
          ▼
┌──────────────────────────┐
│   classify_document()    │  ← LLM: invoice | contract | report | unknown
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
extract_       extract_
invoice()      contract()   ← LLM → JSON → Pydantic
     │           │
     └─────┬─────┘
           ▼
┌──────────────────────────┐
│   validate_*()           │  ← business rules (amounts, dates, required fields)
└──────────┬───────────────┘
           │
┌──────────────────────────┐
│   Deduplicator           │  ← SHA-256 of normalised content
└──────────┬───────────────┘
           │
┌──────────────────────────┐
│   detect_anomalies()     │  ← amount > threshold, date inversions, etc.
└──────────┬───────────────┘
           │
┌──────────────────────────┐
│   MockDB.store()         │  ← in-memory dict, keyed by content hash
└──────────┬───────────────┘
           │
┌──────────────────────────┐
│   generate_report()      │  ← JSON + HTML with per-doc status + aggregates
└──────────────────────────┘
```

---

## Production Patterns Covered

| Pattern | Implementation |
|---------|---------------|
| Batch concurrent processing | `asyncio.gather()` across all documents |
| LLM document classification | `classify_document()` — structured prompt |
| Structured extraction (LLM → Pydantic) | `extract_invoice()`, `extract_contract()` |
| Two-pass validation | Pydantic schema check + `validate_*()` business rules |
| SHA-256 content deduplication | `Deduplicator` — normalise → hash → check seen set |
| Business-rule anomaly detection | `detect_anomalies()` — amount/date/field rules |
| Mock DB with upsert | `MockDB.store()` — in-memory dict keyed by hash |
| Per-document status tracking | `DocResult` dataclass — extracted/invalid/duplicate/anomaly |
| Aggregated report (JSON + HTML) | `generate_report()` — counts + per-doc details |
| Guide reference | §4 (tool use + Pydantic), §7 (production), §12 (quality) |

---

## Document Set (Embedded)

| ID | Type | Status |
|----|------|--------|
| doc001 | Invoice | Valid |
| doc002 | Invoice | Anomaly (amount > $100k) |
| doc003 | Invoice | Duplicate of doc001 |
| doc004 | Contract | Valid |
| doc005 | Contract | Anomaly (end_date before start_date) |
| doc006 | Contract | Validation error (missing parties) |
| doc007 | Report | Unknown type — stored as-is |
| doc008 | Invoice | Valid, different vendor |

---

## Milestones

### Milestone 1 — Document Classifier
Implement `classify_document()`: send document text + examples to LLM, return one
of `invoice | contract | report | unknown`. Default to `unknown` on ambiguous input.

### Milestone 2 — Structured Extractor
Implement `extract_invoice()` and `extract_contract()`:
- Call LLM with the document text + Pydantic schema description
- Parse response as JSON
- Validate with Pydantic (`InvoiceData.model_validate(parsed)`)
- Return the model instance or `None` on failure

### Milestone 3 — Business Rule Validator
Implement `validate_invoice()` and `validate_contract()`:
- Check required fields are non-empty
- Check `amount_total > 0` and `amount_total < 500_000`
- Check `end_date > start_date` for contracts
- Return `list[str]` of violation messages (empty = valid)

### Milestone 4 — Deduplicator
Implement the `Deduplicator` class:
- `_normalise(text)`: lowercase + strip whitespace
- `is_duplicate(text)`: compute `sha256(normalised).hexdigest()`, check `seen` set
- `add(text)`: add hash to `seen`

### Milestone 5 — Anomaly Detector + Report
Implement `detect_anomalies()` (list of anomaly strings from extracted data) and
`generate_report()` (JSON dict + HTML string, written to disk).

---

## Expected Output

```
═══════════════════════════════════════════════════════════════════
 Project 14 — Document Intelligence Pipeline
═══════════════════════════════════════════════════════════════════
Processing 8 documents concurrently…

  ✅ doc001  invoice   extracted   INV-2026-0512  $1,100.00
  ⚠️  doc002  invoice   anomaly     INV-2026-0099  $250,000.00  [amount > $100k]
  ♻️  doc003  invoice   duplicate   (same as doc001)
  ✅ doc004  contract  extracted   CTR-2026-001
  ⚠️  doc005  contract  anomaly     CTR-2026-002   [end_date before start_date]
  ❌ doc006  contract  invalid     CTR-2026-003   [missing: parties]
  ❓ doc007  report    unknown
  ✅ doc008  invoice   extracted   INV-2026-0601  $4,750.00

Summary: 8 docs | 3 extracted | 1 invalid | 1 duplicate | 2 anomalies | 1 unknown
✅ Report saved → doc_pipeline_report.json | doc_pipeline_report.html
```

---

## Setup

```bash
pip install litellm python-dotenv pydantic
python projects/project14_document_intelligence_agent/starter.py
# or
python projects/project14_document_intelligence_agent/solution/solution.py
```
