#!/usr/bin/env python
"""
TODO — Implement the ingestion CLI script.

This script is the entry point for the OFFLINE embedding pipeline.
Run it once before starting the API server to populate ChromaDB.

Usage:
  python scripts/ingest.py --source data/sample_docs/
  python scripts/ingest.py --source data/sample_docs/ --verbose
  python scripts/ingest.py --source data/sample_docs/ --replace

Steps to implement:
  1. Parse CLI arguments with argparse
  2. Configure logging based on --verbose flag
  3. Create VectorStore()
  4. Call ingest_directory(source, store, replace_existing)
  5. Print a summary table (docs processed, chunks created, errors)
  6. sys.exit(1) if there were errors
"""
from __future__ import annotations
import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

from src.store.chroma_store import VectorStore
from src.ingestion.pipeline import ingest_directory


def parse_args() -> argparse.Namespace:
    """
    TODO 1: Create ArgumentParser with:
      --source   (required) — directory to ingest
      --replace  (flag)     — re-embed existing documents
      --verbose / -v (flag) — show DEBUG logs
    """
    raise NotImplementedError


def main() -> None:
    """
    TODO 2: args = parse_args()
    TODO 3: Configure logging (DEBUG if args.verbose else INFO)
    TODO 4: store = VectorStore()
    TODO 5: t0 = time.perf_counter()
    TODO 6: result = ingest_directory(args.source, store, replace_existing=args.replace)
    TODO 7: elapsed = round(time.perf_counter() - t0, 2)
    TODO 8: Print summary:
              Documents processed : N
              Documents skipped   : N
              Chunks created      : N
              Total in store      : store.count()
              Time elapsed        : Xs
              Errors (if any)
    TODO 9: sys.exit(1) if result.errors
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
