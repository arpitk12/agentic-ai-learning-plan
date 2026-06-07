#!/usr/bin/env python
"""
CLI script — run the evaluation suite and optionally gate on a threshold.

Usage
-----
  python scripts/evaluate.py
  python scripts/evaluate.py --fail-below 0.80
  python scripts/evaluate.py --output results/eval_report.json

Exit codes
----------
  0 — gate passed (or --no-gate)
  1 — gate failed / unexpected error
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, ".")

from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.evaluation.evaluator import run_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RAG evaluation suite with golden QA pairs",
    )
    p.add_argument(
        "--fail-below",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Exit with code 1 if aggregate_overall < THRESHOLD. "
            "Overrides EVAL_PASS_THRESHOLD from .env."
        ),
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Write JSON report to this file (default: stdout only)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show DEBUG log output",
    )
    p.add_argument(
        "--no-gate",
        action="store_true",
        default=False,
        help="Print results but always exit 0 (useful for inspection only)",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    log = logging.getLogger("evaluate")

    store     = VectorStore()
    retriever = HybridRetriever(store)
    n = store.count()
    log.info("ChromaDB ready — %d chunks indexed", n)

    if n == 0:
        log.warning(
            "Vector store is empty — run `python scripts/ingest.py --source data/sample_docs/` first"
        )

    report = await run_eval(retriever, request_id="cli-eval")
    report_dict = report.model_dump()

    # ── print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    for i, case in enumerate(report.cases):
        print(f"  [{i+1}] {case.question[:60]}")
        print(f"       faith={case.faithfulness:.2f}  rel={case.relevancy:.2f}  overall={case.overall:.2f}")
    print("-" * 60)
    print(f"  Aggregate faithfulness : {report.aggregate_faithfulness:.3f}")
    print(f"  Aggregate relevancy    : {report.aggregate_relevancy:.3f}")
    print(f"  Aggregate overall      : {report.aggregate_overall:.3f}")
    print(f"  Threshold              : {report.threshold:.2f}")
    print(f"  Gate                   : {'✅ PASS' if report.gate_passed else '❌ FAIL'}")
    print("=" * 60)

    # ── write JSON if requested ───────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report_dict, indent=2))
        log.info("Report written to %s", out)

    # ── gate logic ────────────────────────────────────────────────────────────
    if args.no_gate:
        return

    threshold = args.fail_below if args.fail_below is not None else report.threshold
    if report.aggregate_overall < threshold:
        log.error(
            "Eval gate FAILED: %.3f < %.2f",
            report.aggregate_overall, threshold,
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
