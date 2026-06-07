#!/usr/bin/env python
"""
CLI — run the evaluation suite and optionally gate on a quality threshold.

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
    TODO 1: Build an argument parser with:
            --fail-below (float) — override the pass threshold
            --output / -o (path) — write the JSON report to a file
            --verbose (flag)     — show DEBUG logs
            --no-gate (flag)     — print results but always exit 0
    """
    raise NotImplementedError


async def _run(args: argparse.Namespace) -> None:
    """
    TODO 2: Create a VectorStore and a HybridRetriever
    TODO 3: Warn the user if the store is empty (documents not yet ingested)
    TODO 4: Run the evaluation suite
    TODO 5: Print a per-case table and aggregate scores
    TODO 6: Optionally write the report to a JSON file
    TODO 7: Exit with code 1 if the aggregate score is below the threshold
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
