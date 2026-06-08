"""Recursive character text splitter."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    char_start: int
    char_end: int


class Chunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        # Split on paragraph boundaries first, then sentences, then words
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split(text, separators, 0)

    def _split(self, text: str, separators: List[str], start: int) -> List[Chunk]:
        sep = separators[0]
        chunks: List[Chunk] = []
        current = ""
        current_start = 0

        parts = re.split(re.escape(sep), text) if sep else list(text)

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self._size:
                current = candidate
            else:
                if current:
                    chunks.append(Chunk(text=current, char_start=start + current_start, char_end=start + current_start + len(current)))
                    current_start += len(current) - self._overlap
                    current = current[-self._overlap:] + sep + part if self._overlap else part
                else:
                    # Part itself is too long — recurse with next separator
                    if len(separators) > 1:
                        chunks.extend(self._split(part, separators[1:], start + current_start))
                    else:
                        chunks.append(Chunk(text=part[:self._size], char_start=start + current_start, char_end=start + current_start + self._size))
                    current = ""
                    current_start += len(part)

        if current.strip():
            chunks.append(Chunk(text=current.strip(), char_start=start + current_start, char_end=start + current_start + len(current)))

        return [c for c in chunks if c.text.strip()]
