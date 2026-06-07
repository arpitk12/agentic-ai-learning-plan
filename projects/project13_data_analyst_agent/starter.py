"""
Project 13 — Data Analyst Agent: Code Generation + Self-Correction (Starter)

Build an agent that:
  1. Decomposes a natural language question into analysis steps (planner)
  2. Generates runnable stdlib-only Python code with the dataset embedded
  3. Executes it safely in a subprocess (10 s timeout)
  4. Self-corrects on errors — feeds stderr back to LLM, retry up to 3×
  5. Synthesises a narrative Markdown report from stdout results

Usage:
  python starter.py

Runs 3 different analysis questions against the embedded sales dataset.
"""

import os, sys, json, time, re, asyncio, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pathlib import Path
from dotenv import load_dotenv
from llm import achat, get_text, MODEL

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Embedded Sales Dataset (20 rows — no external file needed)
# ══════════════════════════════════════════════════════════════════════════════

SALES_DATA = [
    {"date":"2026-01-05","region":"North","product":"Widget A","category":"Hardware","quantity":12,"unit_price":49.99,"salesperson":"Alice"},
    {"date":"2026-01-07","region":"South","product":"Widget B","category":"Hardware","quantity": 5,"unit_price":89.99,"salesperson":"Bob"},
    {"date":"2026-01-10","region":"East", "product":"Service X","category":"Services","quantity": 3,"unit_price":299.99,"salesperson":"Carol"},
    {"date":"2026-01-14","region":"West", "product":"Widget A","category":"Hardware","quantity": 8,"unit_price":49.99,"salesperson":"Dave"},
    {"date":"2026-01-18","region":"North","product":"Service Y","category":"Services","quantity": 2,"unit_price":499.99,"salesperson":"Alice"},
    {"date":"2026-01-22","region":"South","product":"Widget C","category":"Hardware","quantity":15,"unit_price":29.99,"salesperson":"Bob"},
    {"date":"2026-01-25","region":"East", "product":"Widget B","category":"Hardware","quantity": 7,"unit_price":89.99,"salesperson":"Carol"},
    {"date":"2026-02-01","region":"West", "product":"Service X","category":"Services","quantity": 1,"unit_price":299.99,"salesperson":"Dave"},
    {"date":"2026-02-04","region":"North","product":"Widget C","category":"Hardware","quantity":20,"unit_price":29.99,"salesperson":"Alice"},
    {"date":"2026-02-08","region":"South","product":"Service Y","category":"Services","quantity": 4,"unit_price":499.99,"salesperson":"Bob"},
    {"date":"2026-02-12","region":"East", "product":"Widget A","category":"Hardware","quantity": 9,"unit_price":49.99,"salesperson":"Carol"},
    {"date":"2026-02-15","region":"West", "product":"Widget B","category":"Hardware","quantity":11,"unit_price":89.99,"salesperson":"Dave"},
    {"date":"2026-02-19","region":"North","product":"Service X","category":"Services","quantity": 2,"unit_price":299.99,"salesperson":"Alice"},
    {"date":"2026-02-22","region":"South","product":"Widget A","category":"Hardware","quantity":18,"unit_price":49.99,"salesperson":"Bob"},
    {"date":"2026-03-01","region":"East", "product":"Widget C","category":"Hardware","quantity": 6,"unit_price":29.99,"salesperson":"Carol"},
    {"date":"2026-03-05","region":"West", "product":"Service Y","category":"Services","quantity": 3,"unit_price":499.99,"salesperson":"Dave"},
    {"date":"2026-03-09","region":"North","product":"Widget B","category":"Hardware","quantity":14,"unit_price":89.99,"salesperson":"Alice"},
    {"date":"2026-03-12","region":"South","product":"Service X","category":"Services","quantity": 5,"unit_price":299.99,"salesperson":"Bob"},
    {"date":"2026-03-16","region":"East", "product":"Service Y","category":"Services","quantity": 2,"unit_price":499.99,"salesperson":"Carol"},
    {"date":"2026-03-20","region":"West", "product":"Widget C","category":"Hardware","quantity":10,"unit_price":29.99,"salesperson":"Dave"},
]

DATASET_DESCRIPTION = (
    "A Q1 2026 sales dataset with 20 records. "
    "Columns: date (YYYY-MM-DD), region (North/South/East/West), "
    "product (Widget A/B/C, Service X/Y), category (Hardware/Services), "
    "quantity (int), unit_price (float USD), salesperson (Alice/Bob/Carol/Dave). "
    "Revenue = quantity * unit_price."
)


# ══════════════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════════════

async def _llm(prompt: str, system: str = "You are an expert Python data analyst.") -> str:
    r = await achat([{"role": "user", "content": prompt}], system=system, max_tokens=1200)
    return get_text(r)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Plan Decomposer
# ══════════════════════════════════════════════════════════════════════════════

async def plan_analysis(question: str) -> list[str]:
    """TODO: Send question + DATASET_DESCRIPTION to LLM.
       Ask for a numbered list of 3–5 concrete analysis steps.
       Parse the response into a list[str] — one step per item.
       Strip numbering (e.g. '1. ', '2. ') from each item.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Code Generator
# ══════════════════════════════════════════════════════════════════════════════

def _dataset_as_python_literal() -> str:
    """Return SALES_DATA as a Python list literal string for embedding in generated code."""
    return "data = " + repr(SALES_DATA)


async def generate_code(question: str, plan: list[str]) -> str:
    """TODO: Ask the LLM to write Python code that:
       - Starts with the dataset literal: call _dataset_as_python_literal()
         and include it at the top of the code
       - Uses ONLY Python stdlib (statistics, collections, datetime — no pip installs)
       - Follows the plan steps
       - Prints all results clearly labelled to stdout
       - Is self-contained and runnable standalone

       Strip any markdown code fences (```python ... ```) from the response.
       Return the raw Python code string.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Safe Executor
# ══════════════════════════════════════════════════════════════════════════════

def execute_code(code: str, timeout: int = 10) -> dict:
    """TODO: Execute Python code safely.
       Steps:
         1. Write code to a NamedTemporaryFile (suffix='.py', delete=False)
         2. subprocess.run([sys.executable, tmp_path],
                          capture_output=True, text=True, timeout=timeout)
         3. Always delete the temp file in a finally block
         4. On TimeoutExpired: return success=False, stderr='Timeout: exceeded Ns'
         5. Return dict: {stdout, stderr, success (returncode==0), return_code}
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Debug & Retry Loop
# ══════════════════════════════════════════════════════════════════════════════

async def debug_code(broken_code: str, error: str, question: str) -> str:
    """TODO: Ask the LLM to fix broken_code given the error message.
       Include: original question, the failing code, the stderr error.
       Strip markdown fences from response.
       Return fixed Python code string.
    """
    raise NotImplementedError


async def run_with_retry(question: str, code: str, max_attempts: int = 3) -> dict:
    """TODO: Execute code with up to max_attempts retries.
       On each failure: call debug_code() → update code → retry.
       Return the result dict from execute_code() plus:
         - 'attempts_used': int (1-3)
         - 'final_code': str (the code that finally ran, or last attempt)
       If all attempts fail: still return the last result with success=False.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Narrative Report Synthesis
# ══════════════════════════════════════════════════════════════════════════════

async def synthesise_report(question: str, plan: list[str], stdout: str) -> str:
    """TODO: Send question + plan + stdout results to LLM.
       Ask for a Markdown report with sections:
         ## Executive Summary
         ## Key Findings
         ## Recommendations
       Return the Markdown string.
    """
    raise NotImplementedError


def export_results(slug: str, question: str, plan: list[str],
                   stdout: str, report_md: str, attempts: int):
    """TODO: Save two files:
       - f'analysis_{slug}.md'  → report_md
       - f'analysis_{slug}.json' → JSON with question, plan, stdout, attempts, report
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Questions
# ══════════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    {
        "slug": "q1_revenue_by_region",
        "question": "Which region generated the highest total revenue in Q1 2026, and by how much did it beat second place?",
    },
    {
        "slug": "q2_top_salesperson",
        "question": "Who was the top salesperson by total revenue? Show a ranked leaderboard for all four salespeople.",
    },
    {
        "slug": "q3_monthly_trend",
        "question": "How did total monthly revenue trend across January, February, and March? Is there growth or decline?",
    },
]


async def main():
    print(f"{'='*65}")
    print(f" Project 13 — Data Analyst Agent   [{MODEL}]")
    print(f"{'='*65}\n")

    for item in QUESTIONS:
        q    = item["question"]
        slug = item["slug"]
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print()

        # Stage 1 — Plan
        plan = await plan_analysis(q)
        print(f"Plan ({len(plan)} steps):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")

        # Stage 2 — Generate code
        code = await generate_code(q, plan)

        # Stage 3+4 — Execute with retry
        result = await run_with_retry(q, code)
        status = "✅" if result["success"] else "❌"
        print(f"\n{status} Executed in {result['attempts_used']} attempt(s)")

        if result["stdout"]:
            preview = result["stdout"].strip()[:400]
            print(f"\nStdout preview:\n{preview}")

        # Stage 5 — Report
        if result["success"]:
            report_md = await synthesise_report(q, plan, result["stdout"])
            export_results(slug, q, plan, result["stdout"], report_md, result["attempts_used"])
            print(f"\n📝 Report saved → analysis_{slug}.md | analysis_{slug}.json")
        else:
            print(f"\n⚠️  All {result['attempts_used']} attempts failed. Last error:")
            print(f"   {result['stderr'][:200]}")

    print(f"\n{'='*65}")
    print(" Done. Check analysis_*.md files for narrative reports.")
    print(f"{'='*65}")


if __name__ == "__main__":
    asyncio.run(main())
