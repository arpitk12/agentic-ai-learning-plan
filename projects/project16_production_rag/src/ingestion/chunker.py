"""
Recursive character chunker.
Splits on paragraphs → sentences → words — preserving semantic coherence.
"""
from __future__ import annotations
import re
import logging
from src.config import cfg
from src.models import RawDocument, Chunk

logger = logging.getLogger(__name__)

# Ordered list of separators — try each in turn, fall back to next
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Recursively split text into chunks of ≤ size chars with overlap."""
    if len(text) <= size:
        return [text]

    # Try each separator in order
    for sep in _SEPARATORS:
        if sep not in text:
            continue
        parts  = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the part itself is too long, recurse on it
                if len(part) > size:
                    chunks.extend(_split_text(part, size, overlap))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)

        # Apply overlap: prepend tail of previous chunk
        if overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                tail  = chunks[i-1][-overlap:]
                chunk = (tail + " " + chunks[i]).strip()
                overlapped.append(chunk)
            return overlapped

        return chunks

    # Fallback: hard character split
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]


def chunk_document(
    doc: RawDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_len: int | None = None,
) -> list[Chunk]:
    """Split one RawDocument into Chunk objects."""
    size    = chunk_size    or cfg.CHUNK_SIZE
    overlap = chunk_overlap or cfg.CHUNK_OVERLAP
    min_l   = min_len       or cfg.MIN_CHUNK_LEN

    raw_chunks = _split_text(doc.content, size, overlap)
    chunks: list[Chunk] = []
    for i, text in enumerate(raw_chunks):
        if len(text.strip()) < min_l:
            continue
        chunks.append(Chunk(
            chunk_id  = f"{doc.doc_id}::{i:04d}",
            doc_id    = doc.doc_id,
            source    = doc.source,
            title     = doc.title,
            content   = text.strip(),
            chunk_idx = i,
            metadata  = {**doc.metadata, "chunk_idx": i, "doc_id": doc.doc_id},
        ))

    logger.debug("'%s' → %d chunks (raw=%d)", doc.title, len(chunks), len(raw_chunks))
    return chunks
