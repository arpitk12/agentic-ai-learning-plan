"""
TODO — Implement a recursive character text splitter with overlap.

Algorithm (recursive character splitting):
  Separators tried in order: ["\n\n", "\n", ". ", " ", ""]
  1. Try splitting text on the first separator
  2. Merge small pieces back together until chunk_size is reached
  3. Add `overlap` characters of the previous chunk's tail to the next chunk
  4. Recurse on any piece that is still too large using the next separator

Each Chunk gets a unique chunk_id built from doc_id + sequential index.
"""
from __future__ import annotations
import hashlib
from src.models import RawDocument, Chunk
from src.config import cfg

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_document(
    doc: RawDocument,
    chunk_size: int = cfg.CHUNK_SIZE,
    overlap: int = cfg.CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split a RawDocument into overlapping Chunk objects.

    TODO 1: Call _split_text(doc.content, chunk_size, overlap) to get raw text pieces
    TODO 2: For each piece, build chunk_id = f"{doc.doc_id}-{i:04d}"
    TODO 3: Return list of Chunk(chunk_id=..., doc_id=..., title=...,
                                 source=..., text=..., index=i)
    """
    raise NotImplementedError


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursively split text into pieces of at most chunk_size characters.

    TODO 4: Iterate through _SEPARATORS
    TODO 5: For the current separator, split text and recursively handle
            pieces that are still > chunk_size
    TODO 6: Merge short pieces together (greedy accumulation up to chunk_size)
    TODO 7: Prepend overlap chars from previous chunk to the current chunk
    TODO 8: Return the final list of string pieces
    """
    raise NotImplementedError
