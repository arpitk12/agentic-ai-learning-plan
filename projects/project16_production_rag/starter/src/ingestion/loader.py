"""
Load .md and .txt files from disk and return RawDocument objects.

Each RawDocument needs a stable doc_id so the ingestion pipeline can detect
duplicates across runs. Use a hash of the resolved absolute file path.
"""
from __future__ import annotations
from pathlib import Path
from src.models import RawDocument


def load_file(path: str) -> list[RawDocument]:
    """
    Load a single file and return a list with one RawDocument.

    TODO 1: Resolve the path to an absolute path
    TODO 2: Raise FileNotFoundError if the file does not exist
    TODO 3: Return an empty list for unsupported file types (accept .md and .txt only)
    TODO 4: Read the file contents as UTF-8 text
    TODO 5: Generate a stable doc_id by hashing the resolved file path
    TODO 6: Return a list containing one RawDocument
    """
    raise NotImplementedError


def load_directory(path: str) -> list[RawDocument]:
    """
    Recursively load all .md and .txt files under a directory.

    TODO 7: Recursively find all files in the directory
    TODO 8: Keep only .md and .txt files
    TODO 9: Load each file using load_file() and combine the results
    TODO 10: Sort the results by source path for reproducible ordering
    """
    raise NotImplementedError
