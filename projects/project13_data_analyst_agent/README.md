# Project 13 — Data Analyst Agent (Code Generation + Self-Correction)

## What You Build

An agent that accepts natural language analysis requests about a sales dataset,
decomposes the task into steps, **generates runnable Python code**, executes it
in a sandboxed subprocess, **self-corrects on errors** (up to 3 retries), extracts
results from stdout, and synthesises a narrative Markdown report with key findings.

---

## Architecture

```
Natural Language Question
          │
          ▼
┌─────────────────────┐
│   plan_analysis()   │  ← LLM: decompose into 3-5 concrete steps
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  generate_code()    │  ← LLM: write self-contained Python (stdlib only)
│  (dataset embedded) │     dataset injected as `data = [...]` literal
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│              run_with_retry()               │
│                                             │
│  attempt = 0                                │
│  while attempt < 3:                         │
│    execute_code(code)  ← subprocess + 10s   │
│    if success: break                        │
│    code = debug_code(code, stderr, question)│
│    attempt += 1                             │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│ synthesise_report() │  ← LLM: turn stdout results → narrative + JSON
└──────┬──────────────┘
       │
       ▼
   Export (Markdown + JSON)
```

---

## Production Patterns Covered

| Pattern | Implementation |
|---------|---------------|
| Task decomposition (planner) | `plan_analysis()` — structured step list |
| LLM code generation | `generate_code()` — stdlib-only Python with embedded data |
| Safe code execution | `execute_code()` — `subprocess.run()` + `tempfile` + 10 s timeout |
| Self-correction loop | `run_with_retry()` — feeds `stderr` + old code back to LLM |
| Max-retry guard | Hard cap at 3 attempts — never loops forever |
| Result extraction | Parse stdout for structured output |
| Narrative synthesis | `synthesise_report()` — findings + recommendations |
| Multi-format export | Markdown report + machine-readable JSON |
| Guide reference | §2 (tool use), §5 (planning), §9 (self-reflection) |

---

## Dataset

Embedded 20-row sales dataset (`SALES_DATA`) covering:
- Columns: `date`, `region`, `product`, `quantity`, `unit_price`, `salesperson`, `category`
- 4 regions (North/South/East/West), 5 products, 4 salespeople

---

## Milestones

### Milestone 1 — Plan Decomposer
Implement `plan_analysis()`: given a dataset description + question, return a
`list[str]` of 3–5 concrete analysis steps. Each step should map to one
computation (e.g. "Compute total revenue per region").

### Milestone 2 — Code Generator
Implement `generate_code()`: produce a single Python script that:
- Starts with `data = [...]` containing the full dataset as a literal
- Uses only Python stdlib (`statistics`, `collections`, `datetime`, etc.)
- Prints all results clearly labelled to stdout
- Follows the plan steps

### Milestone 3 — Safe Executor
Implement `execute_code()`: write code to a `tempfile`, run it with
`subprocess.run([sys.executable, path], capture_output=True, timeout=10)`,
return `{"stdout", "stderr", "success", "return_code", "attempts_used"}`.

### Milestone 4 — Debug & Retry Loop
Implement `debug_code()`: given the failing code + stderr, ask the LLM to fix it.
Then implement `run_with_retry()`: loop up to 3 times, passing errors back to
`debug_code()` on each failure.

### Milestone 5 — Narrative Report
Implement `synthesise_report()`: send stdout + question to LLM, get back a
Markdown report with sections: Executive Summary, Key Findings, Recommendations.
Export both `.md` and `.json` files.

---

## Expected Output

```
═══════════════════════════════════════════════════════════════════
 Project 13 — Data Analyst Agent   [gemini/gemini-2.0-flash]
═══════════════════════════════════════════════════════════════════

Q1: Which region had the highest total revenue last quarter?
────────────────────────────────────────────────────────────────
Plan (3 steps):
  1. Group sales records by region
  2. Sum quantity * unit_price per region
  3. Find the region with the maximum total

Attempt 1/3 ... ✅ executed (0 errors)

Stdout preview:
  North: $12,450.00
  South: $9,870.00
  East:  $11,230.00
  West:  $8,190.00
  → Highest revenue region: North ($12,450.00)

📝 Report saved → analysis_q1.md  |  analysis_q1.json
…
```

---

## Setup

```bash
pip install litellm python-dotenv pydantic
python projects/project13_data_analyst_agent/starter.py
# or
python projects/project13_data_analyst_agent/solution/solution.py
```
