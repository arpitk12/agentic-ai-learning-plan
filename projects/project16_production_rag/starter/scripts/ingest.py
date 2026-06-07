#!/usr/bin/env python
"""
CLI — ingest a directory of documents into ChromaDB.

Usage:
  python scripts/ingest.py --source data/sample_docs/
  python scripts/ingest.py --source data/sample_docs/ --replace
  python scripts/ingest.py --source data/sample_docs/ --verbose
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
    TODO 1: Build an argument parser with:
            --source (required) — directory to ingest
            --replace (flag)    — re-embed existing documents
            --verbose (flag)    — show DEBUG level logs
    """
    raise NotImplementedError


def main() -> None:
    """
    TODO 2: Parse arguments and configure logging
    TODO 3: Create a VectorStore and run ingest_directory
    TODO 4: Print a summary of documents processed, chunks created, and any errors
    TODO 5: Exit with code 1 if there were errors
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
