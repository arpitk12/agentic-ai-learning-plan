"""
Project 13 — Data Analyst Agent: Code Generation + Self-Correction (Solution)

Full implementation: plan → generate → execute (subprocess) → retry on error
→ synthesise narrative Markdown report → export JSON + Markdown.
"""

import os, sys, json, time, re, asyncio, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from pathlib import Path
from dotenv import load_dotenv
from llm import achat, get_text, MODEL

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Embedded Sales Dataset
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
    r = await achat([{"role": "user", "content": prompt}], system=system, max_tokens=1500)
    return get_text(r)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Plan Decomposer
# ══════════════════════════════════════════════════════════════════════════════

async def plan_analysis(question: str) -> list[str]:
    prompt = (
        f"Dataset: {DATASET_DESCRIPTION}\n\n"
        f"Analysis question: {question}\n\n"
        "List 3-5 concrete analysis steps to answer this question. "
        "Number each step. Keep each step to one sentence."
    )
    raw  = await _llm(prompt, system="You are a data analysis planner.")
    steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering: "1. ", "1) ", "- " etc.
        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•]\s*', '', cleaned).strip()
        if cleaned:
            steps.append(cleaned)
    return steps[:5] if steps else ["Compute total revenue per group", "Rank groups", "Report top result"]

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Code Generator
# ══════════════════════════════════════════════════════════════════════════════

def _dataset_as_python_literal() -> str:
    return "data = " + repr(SALES_DATA)


async def generate_code(question: str, plan: list[str]) -> str:
    plan_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
    data_lit = _dataset_as_python_literal()
    prompt = (
        f"Write Python code to answer: \"{question}\"\n\n"
        f"Follow this plan:\n{plan_str}\n\n"
        "RULES:\n"
        "- Start the code with the data literal below (copy it verbatim)\n"
        "- Use ONLY Python stdlib: statistics, collections, datetime, etc. — NO pip installs\n"
        "- Print all results clearly to stdout with descriptive labels\n"
        "- Make the code self-contained and runnable as-is\n\n"
        f"Data literal to include at top:\n{data_lit[:300]}...  (use the full literal)\n\n"
        "Write ONLY the Python code — no markdown fences."
    )
    code = await _llm(prompt)
    # Strip markdown fences
    code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*', '',    code, flags=re.MULTILINE)
    # If the LLM forgot to embed the data, prepend it
    if "data = " not in code:
        code = _dataset_as_python_literal() + "\n\n" + code
    return code.strip()

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Safe Executor
# ══════════════════════════════════════════════════════════════════════════════

def execute_code(code: str, timeout: int = 10) -> dict:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                        delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout":      result.stdout,
            "stderr":      result.stderr,
            "success":     result.returncode == 0,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timeout: code exceeded {timeout}s",
                "success": False, "return_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False, "return_code": -2}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Debug & Retry Loop
# ══════════════════════════════════════════════════════════════════════════════

async def debug_code(broken_code: str, error: str, question: str) -> str:
    prompt = (
        f"This Python code was written to answer: \"{question}\"\n\n"
        f"It failed with this error:\n{error}\n\n"
        f"Broken code:\n```python\n{broken_code}\n```\n\n"
        "Fix the code. Return ONLY the corrected Python code — no markdown fences."
    )
    fixed = await _llm(prompt)
    fixed = re.sub(r'^```python\s*', '', fixed, flags=re.MULTILINE)
    fixed = re.sub(r'^```\s*', '',    fixed, flags=re.MULTILINE)
    if "data = " not in fixed:
        fixed = _dataset_as_python_literal() + "\n\n" + fixed
    return fixed.strip()


async def run_with_retry(question: str, code: str, max_attempts: int = 3) -> dict:
    last_result: dict = {}
    for attempt in range(1, max_attempts + 1):
        print(f"  Attempt {attempt}/{max_attempts} … ", end="", flush=True)
        last_result = execute_code(code)
        if last_result["success"]:
            print("✅")
            last_result["attempts_used"] = attempt
            last_result["final_code"]    = code
            return last_result
        print(f"❌  stderr: {last_result['stderr'][:80]}")
        if attempt < max_attempts:
            code = await debug_code(code, last_result["stderr"], question)

    last_result["attempts_used"] = max_attempts
    last_result["final_code"]    = code
    return last_result

# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Narrative Report
# ══════════════════════════════════════════════════════════════════════════════

async def synthesise_report(question: str, plan: list[str], stdout: str) -> str:
    plan_str = "\n".join(f"- {s}" for s in plan)
    prompt = (
        f"Analysis question: {question}\n\n"
        f"Analysis plan followed:\n{plan_str}\n\n"
        f"Raw analysis results (Python stdout):\n{stdout}\n\n"
        "Write a professional Markdown report with these sections:\n"
        "## Executive Summary\n"
        "## Key Findings\n"
        "## Recommendations\n\n"
        "Be specific — use the actual numbers from the results."
    )
    return await _llm(prompt, system="You are a business analyst writing clear data reports.")


def export_results(slug: str, question: str, plan: list[str],
                   stdout: str, report_md: str, attempts: int):
    Path(f"analysis_{slug}.md").write_text(
        f"# Analysis Report: {question}\n\n{report_md}", encoding="utf-8"
    )
    Path(f"analysis_{slug}.json").write_text(
        json.dumps({
            "question":    question,
            "plan":        plan,
            "stdout":      stdout,
            "attempts":    attempts,
            "report_md":   report_md,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Analysis Questions + Main
# ══════════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    {
        "slug":     "q1_revenue_by_region",
        "question": "Which region generated the highest total revenue in Q1 2026, and by how much did it beat second place?",
    },
    {
        "slug":     "q2_top_salesperson",
        "question": "Who was the top salesperson by total revenue? Show a ranked leaderboard for all four salespeople.",
    },
    {
        "slug":     "q3_monthly_trend",
        "question": "How did total monthly revenue trend across January, February, and March? Is there growth or decline?",
    },
]


async def main():
    print(f"{'='*65}")
    print(f" Project 13 — Data Analyst Agent   [{MODEL}]")
    print(f"{'='*65}\n")
    t_total = time.perf_counter()

    for item in QUESTIONS:
        q    = item["question"]
        slug = item["slug"]
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print()

        # Stage 1: Plan
        plan = await plan_analysis(q)
        print(f"Plan ({len(plan)} steps):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        print()

        # Stage 2: Generate code
        code = await generate_code(q, plan)

        # Stages 3+4: Execute with self-correction
        result = await run_with_retry(q, code)

        if result["stdout"]:
            preview = result["stdout"].strip()[:400]
            print(f"\nStdout preview:\n{preview}")

        if result["success"]:
            # Stage 5: Narrative report
            report_md = await synthesise_report(q, plan, result["stdout"])
            export_results(slug, q, plan, result["stdout"], report_md, result["attempts_used"])
            print(f"\n📝 Saved → analysis_{slug}.md  |  analysis_{slug}.json")
        else:
            print(f"\n⚠️  All {result['attempts_used']} attempts failed.")
            print(f"   Last error: {result['stderr'][:200]}")

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*65}")
    print(f" Done in {elapsed:.1f}s. Check analysis_*.md for reports.")
    print(f"{'='*65}")


if __name__ == "__main__":
    asyncio.run(main())
