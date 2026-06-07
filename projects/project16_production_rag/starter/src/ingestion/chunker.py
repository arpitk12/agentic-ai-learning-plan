"""
Recursive character text splitter with overlap.

Splits a RawDocument into fixed-size Chunk objects by trying different
separators in priority order (paragraph → line → sentence → word → character).
When a piece is still too long after splitting, it recurses with the next separator.

Overlap causes the tail of the previous chunk to be prepended to the next,
preserving context across chunk boundaries.
"""
from __future__ import annotations
from src.models import RawDocument, Chunk
from src.config import cfg


def chunk_document(
    doc: RawDocument,
    chunk_size: int = cfg.CHUNK_SIZE,
    overlap: int = cfg.CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split a RawDocument into overlapping Chunk objects.

    TODO 1: Split the document content into text pieces using a recursive splitter
    TODO 2: Build a unique chunk_id for each piece using the doc_id and its position index
    TODO 3: Return a list of Chunk objects with all metadata fields populated
    """
    raise NotImplementedError


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursively split text into pieces no larger than chunk_size characters.

    TODO 4: Try separators in priority order: paragraph break, line break, sentence end, space, empty string
    TODO 5: Split on the current separator; for any piece still larger than chunk_size, recurse with the next separator
    TODO 6: Greedily merge short pieces back together up to chunk_size
    TODO 7: Prepend the last overlap characters of the previous chunk to each new chunk
    TODO 8: Return the final list of text pieces
    """
    raise NotImplementedError
