#!/usr/bin/env python
"""
TODO — Implement the evaluation CLI script.

This script is the CI evaluation gate:
  - Runs the golden QA evaluation suite
  - Prints per-case and aggregate scores
  - sys.exit(1) if aggregate_overall < threshold  ← this is how GitHub Actions fails

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --fail-below 0.80
  python scripts/evaluate.py --output results/report.json --no-gate

Exit codes:
  0 — gate passed (or --no-gate)
  1 — gate failed / unexpected error
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.evaluation.evaluator import run_eval


def parse_args() -> argparse.Namespace:
    """
    TODO 1: Create ArgumentParser with:
      --fail-below FLOAT   — override eval threshold (default: use cfg value)
      --output / -o PATH   — write JSON report to file
      --verbose / -v       — DEBUG logs
      --no-gate            — always exit 0 (inspection mode)
    """
    raise NotImplementedError


async def _run(args: argparse.Namespace) -> None:
    """
    TODO 2: store     = VectorStore()
    TODO 3: retriever = HybridRetriever(store)
    TODO 4: Warn if store.count() == 0 (not ingested yet)
    TODO 5: report = await run_eval(retriever, request_id="cli-eval")
    TODO 6: Print per-case table (question, faithfulness, relevancy, overall)
    TODO 7: Print aggregate row + gate result (PASS / FAIL)
    TODO 8: If args.output: write report.model_dump() as JSON to file
    TODO 9: If not args.no_gate and aggregate < threshold: sys.exit(1)
    """
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
