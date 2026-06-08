"""
Offline CLI ingestion — publishes all documents in a directory to Kafka.
Workers (chunk/embed/index consumers) handle the rest asynchronously.

Usage:
  python scripts/ingest.py --source data/sample_docs/ --verbose
  python scripts/ingest.py --source /data/corpus/ --pattern "*.txt"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import click


@click.command()
@click.option("--source", required=True, type=click.Path(exists=True), help="Directory of documents")
@click.option("--pattern", default="*.md", help="File glob pattern (default: *.md)")
@click.option("--source-tag", default="offline_ingest", help="Source tag for metadata")
@click.option("--verbose", is_flag=True)
def main(source: str, pattern: str, source_tag: str, verbose: bool) -> None:
    """
    TODO 1: Set up logging (DEBUG if verbose else INFO).

    TODO 2: Import DocumentProducer and cfg.
            Create producer = DocumentProducer(cfg.kafka_bootstrap_servers).

    TODO 3: Find all files matching pattern in source directory recursively
            (Path(source).rglob(pattern)).

    TODO 4: For each file:
              - Read text content
              - Call producer.publish(text=content, title=file.stem, source=source_tag,
                  metadata={"file_path": str(file), "size_bytes": file.stat().st_size})
              - Print progress

    TODO 5: Print final count.
            Call producer.close().
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
