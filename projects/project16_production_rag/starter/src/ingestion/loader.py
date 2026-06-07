"""
TODO — Implement file and directory loaders.

These functions read .md and .txt files from disk and return RawDocument objects.
A RawDocument's doc_id must be *stable* (same file → same id on repeated runs)
so the ingestion pipeline can detect duplicates.

Stable ID recipe:
    import hashlib
    doc_id = hashlib.md5(str(path).encode()).hexdigest()[:16]
"""
from __future__ import annotations
from pathlib import Path
from src.models import RawDocument


def load_file(path: str) -> list[RawDocument]:
    """
    Load a single file and return a list with one RawDocument.

    TODO 1: Resolve path with pathlib.Path(path).resolve()
    TODO 2: Raise FileNotFoundError if the path doesn't exist
    TODO 3: Return [] for unsupported extensions (only accept .md and .txt)
    TODO 4: Read file text with p.read_text(encoding="utf-8")
    TODO 5: Compute a stable doc_id from the file path (MD5 hex first 16 chars)
    TODO 6: Return [RawDocument(doc_id=..., title=p.stem, source=str(p), content=text)]
    """
    raise NotImplementedError


def load_directory(path: str) -> list[RawDocument]:
    """
    Recursively load all .md and .txt files under a directory.

    TODO 7: Use Path(path).rglob("*") to find all files
    TODO 8: Filter to suffix in (".md", ".txt")
    TODO 9: Call load_file() on each and concatenate results
    TODO 10: Sort by source path for reproducible ordering
    """
    raise NotImplementedError
