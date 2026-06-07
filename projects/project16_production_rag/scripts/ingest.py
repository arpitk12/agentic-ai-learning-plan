#!/usr/bin/env python
"""
CLI script — ingest a directory of documents into ChromaDB.

Usage
-----
  python scripts/ingest.py --source data/sample_docs/
  python scripts/ingest.py --source data/sample_docs/ --verbose
  python scripts/ingest.py --source data/sample_docs/ --replace
"""
from __future__ import annotations
import argparse
import logging
import sys
import time

# Make imports work from the project root
sys.path.insert(0, ".")

from src.store.chroma_store import VectorStore
from src.ingestion.pipeline import ingest_directory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB (offline embedding pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        required=True,
        help="Directory containing .md / .txt documents to ingest",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Re-embed and replace existing documents (default: skip duplicates)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show DEBUG log output",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    log = logging.getLogger("ingest")

    log.info("Source directory : %s", args.source)
    log.info("Replace existing : %s", args.replace)

    store = VectorStore()
    t0    = time.perf_counter()

    result = ingest_directory(
        source=args.source,
        store=store,
        replace_existing=args.replace,
    )

    elapsed = round(time.perf_counter() - t0, 2)

    print()
    print("=" * 50)
    print(f"  Documents processed : {result.documents_processed}")
    print(f"  Documents skipped   : {result.documents_skipped}")
    print(f"  Chunks created      : {result.chunks_created}")
    print(f"  Total in store      : {store.count()}")
    print(f"  Time elapsed        : {elapsed}s")
    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for e in result.errors:
            print(f"    • {e}")
    print("=" * 50)

    if result.errors:
        log.warning("%d error(s) during ingestion", len(result.errors))
        sys.exit(1)

    log.info("Ingestion complete ✓")


if __name__ == "__main__":
    main()
