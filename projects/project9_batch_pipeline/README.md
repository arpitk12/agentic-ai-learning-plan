# Project 9 — Batch Intelligence Pipeline

## What You Build

Process hundreds of customer reviews in parallel, extract structured insights from each, then hierarchically reduce all results into an executive summary report. This is the **Fan-Out / Fan-In + Map-Reduce** pattern for large-scale LLM processing.

## Production Skills Practised

| Skill | Guide Section |
|-------|--------------|
| Fan-Out with `asyncio.gather()` | §4.4 |
| Concurrency control with `asyncio.Semaphore` | §4.4 |
| Per-item timeout with `asyncio.wait_for()` | §4.4 |
| Map phase: extract structured data in parallel | §4.5 |
| Reduce phase: hierarchical summarisation | §4.5 |
| Error handling: graceful partial failures | §4.4 |
| Pydantic models for structured LLM output | §2.8 |

## Architecture

```
Input CSV (500 reviews)
         │
         ▼
   [fan_out_analyze()]
         │  asyncio.Semaphore(max_concurrent=10)
         │  asyncio.wait_for(timeout=30s per item)
         │
    ┌────┴────┐────────────┐────────────┐ ...  (parallel)
    ▼         ▼            ▼            ▼
analyze() analyze()  analyze()   analyze()
    │         │            │            │
    └────┬────┘────────────┘────────────┘
         │  asyncio.gather() → list[ReviewAnalysis]
         ▼
  [hierarchical_reduce()]
    ┌─ batch 1 (5 items) → reduce_batch() → intermediate summary
    ├─ batch 2 (5 items) → reduce_batch() → intermediate summary
    ├─ ...
    └─ final reduce_batch() → synthesis
         │
         ▼
  [generate_executive_report()]
    + aggregate stats (sentiment counts, avg score)
         │
         ▼
    Executive Report (Markdown)
```

## Input Format

A CSV file with at least a `review` column:
```csv
review,product,rating
"Great product, fast shipping!",Widget A,5
"Broke after one week, terrible quality.",Widget B,1
```

A sample CSV (`sample_reviews.csv`) is included for quick testing.

## Setup

```bash
pip install litellm python-dotenv pydantic aiofiles
```

## Usage

```bash
# Use included sample data
python starter.py sample_reviews.csv

# Use your own CSV
python starter.py /path/to/reviews.csv --max-concurrent 10 --output report.md

# Solution with all features
python solution.py sample_reviews.csv
```

## What To Implement (5 TODOs)

1. **`analyze_review(review, idx)`** — single async LLM call → `ReviewAnalysis`
2. **`fan_out_analyze(reviews)`** — `asyncio.gather()` with `Semaphore` + timeout
3. **`reduce_batch(analyses)`** — combine a batch of results into a summary string
4. **`hierarchical_reduce(analyses)`** — multi-level reduce until ≤5 items remain
5. **`generate_executive_report(analyses, synthesis)`** — final markdown report

## Learning Goals

After completing this project you will:
- Process 100–1000 items in parallel without hitting rate limits
- Handle partial failures gracefully (some items fail, pipeline continues)
- Implement two-level Map-Reduce for datasets that don't fit in one context window
- Know how to tune `max_concurrent` and `timeout` for cost vs throughput
