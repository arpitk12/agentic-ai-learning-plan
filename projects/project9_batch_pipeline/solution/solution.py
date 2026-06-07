"""
SOLUTION — Project 9: Batch Intelligence Pipeline (Fan-Out + Map-Reduce)

MAP  → classify every review concurrently with asyncio.Semaphore + wait_for
REDUCE → hierarchical batch reduction → executive report

Run:
    python solution.py sample_reviews.csv
    python solution.py sample_reviews.csv --max-concurrent 10 --output report.md
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import asyncio
import csv
import json
import re
from collections import Counter as FreqCounter
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()


# ── Data Models ────────────────────────────────────────────────────────────────

class ReviewAnalysis(BaseModel):
    sentiment:  str
    score:      int
    key_topics: list[str]
    summary:    str


# ── LLM Helper ────────────────────────────────────────────────────────────────

async def call_llm(system: str, user: str, max_tokens: int = 400) -> str:
    r = await achat([{"role": "user", "content": user}], system=system, max_tokens=max_tokens)
    return get_text(r)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_reviews(csv_path: str) -> list[str]:
    reviews = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("review") or next(iter(row.values()), "")
            text = text.strip().strip('"')
            if text:
                reviews.append(text)
    return reviews


def print_stats(analyses: list[dict]):
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
    print(f"  ✅ {len(ok)} analysed, {len(err)} failed/timed out")
    print(f"  📊 Avg score: {sum(scores)/len(scores):.1f}/10")
    print(f"  💬 Sentiment: {sentiments}")


# ── MAP Phase ─────────────────────────────────────────────────────────────────

ANALYZE_SYSTEM = (
    "You are a review analyst. Analyse the customer review and return ONLY valid JSON "
    "matching this schema (no markdown, no extra text):\n"
    '{"sentiment": "positive|neutral|negative", "score": 1, "key_topics": ["topic1"], '
    '"summary": "one sentence"}'
)


async def analyze_review(review: str, idx: int) -> dict:
    try:
        raw = await call_llm(ANALYZE_SYSTEM, review, max_tokens=300)
        s = raw.find("{"); e = raw.rfind("}") + 1
        data = json.loads(raw[s:e])
        parsed = ReviewAnalysis(**data)
        return {
            "idx":        idx,
            "review":     review[:80],
            "status":     "ok",
            "sentiment":  parsed.sentiment,
            "score":      parsed.score,
            "key_topics": parsed.key_topics,
            "summary":    parsed.summary,
        }
    except Exception as exc:
        return {"idx": idx, "review": review[:80], "status": "error", "error": str(exc)}


async def fan_out_analyze(
    reviews:        list[str],
    max_concurrent: int   = 5,
    timeout_sec:    float = 30.0,
) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(review: str, idx: int) -> dict:
        async with sem:
            try:
                return await asyncio.wait_for(
                    analyze_review(review, idx), timeout=timeout_sec)
            except asyncio.TimeoutError:
                return {"idx": idx, "review": review[:80], "status": "timeout"}

    tasks = [bounded(r, i) for i, r in enumerate(reviews)]
    results = await asyncio.gather(*tasks)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"  Fan-out complete: {ok_count}/{len(reviews)} succeeded.")
    return list(results)


# ── REDUCE Phase ──────────────────────────────────────────────────────────────

REDUCE_SYSTEM = (
    "You are a data analyst. Identify the key patterns, common themes, "
    "and notable findings from these review analyses. Be concise and specific."
)


async def reduce_batch(analyses: list[dict]) -> str:
    ok = [a for a in analyses if a["status"] == "ok"]
    if not ok:
        return "No successful analyses in this batch."
    lines = [
        f"[{a['idx']}] {a['sentiment']} ({a['score']}/10): {a['summary']}"
        for a in ok
    ]
    return await call_llm(REDUCE_SYSTEM, "\n".join(lines), max_tokens=500)


async def hierarchical_reduce(analyses: list[dict], batch_size: int = 5) -> str:
    current = [a for a in analyses if a["status"] == "ok"]
    if not current:
        return "No successful analyses to reduce."

    level = 0
    while len(current) > batch_size:
        level += 1
        batches = [current[i:i+batch_size] for i in range(0, len(current), batch_size)]
        print(f"  Reduce level {level}: {len(batches)} batches of ≤{batch_size}...")
        summaries = await asyncio.gather(*[reduce_batch(b) for b in batches])
        # Wrap each summary as a minimal analysis dict for the next level
        current = [
            {"idx": i, "status": "ok", "sentiment": "neutral",
             "score": 5, "summary": s, "key_topics": [], "review": ""}
            for i, s in enumerate(summaries)
        ]

    print(f"  Final reduce: combining {len(current)} summaries...")
    return await reduce_batch(current)


# ── Report Generation ─────────────────────────────────────────────────────────

async def generate_executive_report(
    analyses:  list[dict],
    synthesis: str,
    source:    str = "customer_reviews",
) -> str:
    ok     = [a for a in analyses if a["status"] == "ok"]
    total  = len(analyses)
    failed = total - len(ok)

    sentiments: dict[str, int] = {}
    scores = []
    all_topics: list[str] = []
    for a in ok:
        s = a.get("sentiment", "neutral")
        sentiments[s] = sentiments.get(s, 0) + 1
        scores.append(a.get("score", 5))
        all_topics.extend(a.get("key_topics", []))

    avg_score  = sum(scores) / len(scores) if scores else 0
    top_topics = [t for t, _ in FreqCounter(all_topics).most_common(5)]

    stats_block = (
        f"Total reviews: {total}\n"
        f"Successful: {len(ok)} | Failed/timeout: {failed}\n"
        f"Average score: {avg_score:.1f}/10\n"
        f"Sentiment breakdown: {sentiments}\n"
        f"Top topics: {', '.join(top_topics)}"
    )

    return await call_llm(
        "You are a business analyst. Write a professional markdown executive report "
        "with sections: Overview, Key Findings, Sentiment Analysis, Top Issues, "
        "and Recommendations.",
        f"Source: {source}\n\nStatistics:\n{stats_block}\n\nKey findings:\n{synthesis}",
        max_tokens=1000,
    )


# ── Main Pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(
    csv_path:       str,
    max_concurrent: int = 5,
    output_path:    str = None,
):
    print(f"\n🔄 Batch Intelligence Pipeline")
    print(f"📄 Input: {csv_path}")

    reviews = load_reviews(csv_path)
    print(f"📊 Loaded {len(reviews)} reviews\n")

    print("⚡ MAP: Analysing reviews in parallel...")
    analyses = await fan_out_analyze(reviews, max_concurrent=max_concurrent)
    print_stats(analyses)

    print("\n🔀 REDUCE: Synthesising results...")
    synthesis = await hierarchical_reduce(analyses)

    print("\n📝 Generating executive report...")
    report = await generate_executive_report(analyses, synthesis, source=Path(csv_path).stem)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"\n💾 Report saved to: {output_path}")

    return report


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", nargs="?", default="sample_reviews.csv")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.csv_file, args.max_concurrent, args.output))
