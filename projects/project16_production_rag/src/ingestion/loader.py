"""
Document loader — reads .md and .txt files from a directory.
Returns RawDocument objects; source of truth is the file system.
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from src.models import RawDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def _doc_id(path: Path) -> str:
    """Stable document ID derived from the file path."""
    return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]


def load_file(path: Path) -> RawDocument:
    """Load a single .md or .txt file."""
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Empty file: {path}")
    # Use first non-empty line as title (works for markdown headings)
    first_line = content.splitlines()[0].lstrip("#").strip()
    title = first_line or path.stem
    return RawDocument(
        doc_id=_doc_id(path),
        source=str(path),
        title=title,
        content=content,
        metadata={
            "filename":  path.name,
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
        },
    )


def load_directory(directory: Path) -> list[RawDocument]:
    """Load all supported documents from a directory (non-recursive)."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: list[RawDocument] = []
    errors: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        try:
            docs.append(load_file(path))
            logger.debug("Loaded: %s (%d chars)", path.name, len(docs[-1].content))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            logger.warning("Skipped %s — %s", path.name, exc)

    logger.info("Loaded %d docs from %s (%d errors)", len(docs), directory, len(errors))
    return docs
