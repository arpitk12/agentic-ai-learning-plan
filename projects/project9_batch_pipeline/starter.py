"""
Project 9 Starter — Batch Intelligence Pipeline (Fan-Out + Map-Reduce)

Process hundreds of customer reviews in parallel:
  MAP   → classify each review (sentiment, score, topics) concurrently
  REDUCE → hierarchically synthesise results → executive report

Usage:
    python starter.py sample_reviews.csv
    python starter.py sample_reviews.csv --max-concurrent 10

Key patterns:
  - asyncio.Semaphore  → cap concurrency to avoid rate limits
  - asyncio.wait_for() → per-item timeout so one slow call never blocks all
  - asyncio.gather()   → run all items in parallel (fan-out)
  - Hierarchical reduce → handle datasets too large for one LLM context

What you need to implement (TODOs 1-5):
  1. analyze_review(review, idx)       — single async LLM call → ReviewAnalysis dict
  2. fan_out_analyze(reviews)          — gather + Semaphore + timeout
  3. reduce_batch(analyses, question)  — combine a small batch into text summary
  4. hierarchical_reduce(analyses)     — multi-level reduce for large result sets
  5. generate_executive_report(...)    — final markdown report with stats
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import asyncio
import csv
import json
import re
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()


# ── Data Models ────────────────────────────────────────────────────────────────

class ReviewAnalysis(BaseModel):
    sentiment:   str         # "positive" | "neutral" | "negative"
    score:       int         # 1-10
    key_topics:  list[str]  # up to 3 topics e.g. ["shipping", "quality", "price"]
    summary:     str         # one sentence


# ── LLM Helper (already complete) ─────────────────────────────────────────────

async def call_llm(system: str, user: str, max_tokens: int = 400) -> str:
    """Thin async wrapper around achat."""
    r = await achat([{"role": "user", "content": user}], system=system, max_tokens=max_tokens)
    return get_text(r)


# ── Data Loading (already complete) ───────────────────────────────────────────

def load_reviews(csv_path: str) -> list[str]:
    """Load reviews from a CSV file. Returns list of review text strings."""
    reviews = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Accept 'review' or first column as review text
            text = row.get("review") or next(iter(row.values()), "")
            text = text.strip().strip('"')
            if text:
                reviews.append(text)
    return reviews


def print_stats(analyses: list[dict]):
    """Print a quick summary table of completed analyses."""
    ok  = [a for a in analyses if a["status"] == "ok"]
    err = [a for a in analyses if a["status"] != "ok"]
    if not ok:
        print("  No successful analyses.")
        return
    sentiments = {}
    for a in ok:
        s = a.get("sentiment", "?")
        sentiments[s] = sentiments.get(s, 0) + 1
    scores = [a.get("score", 5) for a in ok]
    print(f"  ✅ {len(ok)} analysed, {len(err)} failed")
    print(f"  📊 Avg score: {sum(scores)/len(scores):.1f}/10")
    print(f"  💬 Sentiment: {sentiments}")


# ── MAP Phase ─────────────────────────────────────────────────────────────────

async def analyze_review(review: str, idx: int) -> dict:
    """
    Classify a single review using the LLM.

    TODO 1:
      a. Call call_llm() with:
           system = (
               "You are a review analyst. Analyse the customer review and return ONLY valid JSON "
               "matching this schema (no markdown, no explanation):\\n"
               '{"sentiment": "positive|neutral|negative", "score": 1-10, '
               '"key_topics": ["topic1", "topic2"], "summary": "one sentence"}'
           )
           user = review
           max_tokens = 300
      b. Extract JSON: s = raw.find("{"); e = raw.rfind("}") + 1; data = json.loads(raw[s:e])
      c. Validate with ReviewAnalysis(**data) to ensure the schema is correct.
      d. Return:
             {"idx": idx, "review": review[:80], "status": "ok",
              "sentiment": data["sentiment"], "score": data["score"],
              "key_topics": data["key_topics"], "summary": data["summary"]}
      e. On ANY exception, return:
             {"idx": idx, "review": review[:80], "status": "error", "error": str(e)}

    Tip: The JSON extraction pattern (find first { to last }) handles markdown fences.
    """
    # TODO 1: implement single review analysis
    raise NotImplementedError("analyze_review() not implemented yet")


async def fan_out_analyze(
    reviews:        list[str],
    max_concurrent: int   = 5,
    timeout_sec:    float = 30.0,
) -> list[dict]:
    """
    Process ALL reviews in parallel, respecting a concurrency cap.

    TODO 2:
      a. Create a Semaphore: sem = asyncio.Semaphore(max_concurrent)
      b. Define a bounded coroutine:
             async def bounded(review, idx):
                 async with sem:
                     return await asyncio.wait_for(
                         analyze_review(review, idx), timeout=timeout_sec)
         On asyncio.TimeoutError in the wait_for, catch it inside bounded and return:
             {"idx": idx, "review": review[:80], "status": "timeout"}
      c. Build tasks = [bounded(r, i) for i, r in enumerate(reviews)]
      d. Return list(await asyncio.gather(*tasks))

    Why Semaphore? asyncio.gather() would launch ALL coroutines simultaneously.
    For 500 reviews that would blast 500 concurrent API calls — instant rate-limit.
    The Semaphore caps active calls to max_concurrent at any one time.
    """
    # TODO 2: implement fan-out with Semaphore + timeout
    raise NotImplementedError("fan_out_analyze() not implemented yet")


# ── REDUCE Phase ──────────────────────────────────────────────────────────────

async def reduce_batch(analyses: list[dict]) -> str:
    """
    Synthesise a small batch (≤5) of analysis results into a text summary.

    TODO 3:
      a. Filter to successful analyses: ok = [a for a in analyses if a["status"] == "ok"]
         If no successful results, return "No successful analyses in this batch."
      b. Build a summary string from each successful analysis:
             lines = [f"[{a['idx']}] {a['sentiment']} ({a['score']}/10): {a['summary']}"
                      for a in ok]
             formatted = "\\n".join(lines)
      c. Call call_llm() with:
           system = "You are a data analyst. Identify the key patterns, common themes, and notable findings from these review analyses. Be concise."
           user   = formatted
           max_tokens = 500
      d. Return the result.
    """
    # TODO 3: implement batch reduction
    raise NotImplementedError("reduce_batch() not implemented yet")


async def hierarchical_reduce(analyses: list[dict], batch_size: int = 5) -> str:
    """
    Reduce any number of analyses to a final synthesis using multi-level reduction.

    TODO 4 — implement the reduction loop:
      a. Filter to successful analyses: current = [a for a in analyses if a["status"] == "ok"]
         If empty, return "No successful analyses to reduce."
      b. While len(current) > batch_size:
           i.  Split into batches: batches = [current[i:i+batch_size] for i in range(0, len(current), batch_size)]
           ii. Reduce each batch IN PARALLEL:
                   summaries = await asyncio.gather(*[reduce_batch(b) for b in batches])
           iii. Convert summaries back to analysis-like dicts for the next level:
                   current = [{"idx": i, "status": "ok", "sentiment": "neutral", "score": 5,
                               "summary": s, "key_topics": [], "review": ""}
                              for i, s in enumerate(summaries)]
      c. After the while loop, call reduce_batch(current) one final time and return the result.

    Example: 20 items → 4 batches of 5 → 4 summaries → 1 final reduction.
    """
    # TODO 4: implement hierarchical reduction
    raise NotImplementedError("hierarchical_reduce() not implemented yet")


# ── Report Generation ─────────────────────────────────────────────────────────

async def generate_executive_report(
    analyses:  list[dict],
    synthesis: str,
    source:    str = "customer_reviews",
) -> str:
    """
    Generate a polished markdown executive summary report.

    TODO 5:
      a. Compute aggregate statistics from the successful analyses:
           ok     = [a for a in analyses if a["status"] == "ok"]
           total  = len(analyses)
           failed = total - len(ok)
           sentiments = count positive/neutral/negative across ok
           avg_score  = average of a["score"] for a in ok  (or 0 if empty)
           all_topics = flatten all a["key_topics"] lists, pick top 5 most common
      b. Build a stats_block string that includes: total reviews, successful count,
         failed/timeout count, average score, sentiment breakdown, and top topics.
         Use an f-string with the variables computed in step (a).
      c. Call call_llm() with:
           system = "You are a business analyst. Write a professional markdown executive report."
           user   = f"Source: {source}\\n\\nStatistics:\\n{stats_block}\\n\\nKey findings:\\n{synthesis}"
           max_tokens = 1000
      d. Return the result.
    """
    # TODO 5: implement executive report generation
    raise NotImplementedError("generate_executive_report() not implemented yet")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(
    csv_path:       str,
    max_concurrent: int = 5,
    output_path:    str = None,
):
    print(f"\n🔄 Batch Intelligence Pipeline")
    print(f"📄 Input: {csv_path}")

    # Load reviews
    reviews = load_reviews(csv_path)
    print(f"📊 Loaded {len(reviews)} reviews\n")

    # MAP phase
    print("⚡ MAP: Analysing reviews in parallel...")
    analyses = await fan_out_analyze(reviews, max_concurrent=max_concurrent)
    print_stats(analyses)

    # REDUCE phase
    print("\n🔀 REDUCE: Synthesising results...")
    synthesis = await hierarchical_reduce(analyses)

    # Report
    print("\n📝 Generating executive report...")
    report = await generate_executive_report(analyses, synthesis, source=Path(csv_path).stem)

    # Output
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    if output_path:
        Path(output_path).write_text(report)
        print(f"\n💾 Report saved to: {output_path}")

    return report


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch review analysis pipeline")
    parser.add_argument("csv_file", nargs="?", default="sample_reviews.csv")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.csv_file, args.max_concurrent, args.output))
