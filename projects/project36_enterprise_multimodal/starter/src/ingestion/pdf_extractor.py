"""
src/ingestion/pdf_extractor.py
Extract text chunks and embedded images from a PDF file using PyMuPDF (fitz).

TODOs:
  1. implement extract_text_chunks() — open PDF, get text per page, split into
     overlapping chunks, return list of dicts
  2. implement extract_images() — iterate pages, get embedded image refs,
     extract raw bytes + extension, deduplicate by xref
"""
from __future__ import annotations


# ── TODO 1: Extract text chunks ───────────────────────────────────────────────
def extract_text_chunks(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Open `pdf_path` with fitz and extract overlapping text chunks.

    Steps:
      1a. fitz.open(pdf_path)
      1b. For each page: page.get_text() → strip whitespace
      1c. Concatenate all page text, tracking page boundaries
      1d. Split into chunks of `chunk_size` chars with `overlap` char overlap
          (slide a window: start at 0, step by chunk_size - overlap)
      1e. Each chunk dict: {"text": str, "page": int, "chunk_idx": int, "source": pdf_path}

    Returns:
        list[dict] — one dict per chunk
    """
    # import fitz
    # doc = fitz.open(pdf_path)
    # ...
    raise NotImplementedError


# ── TODO 2: Extract embedded images ──────────────────────────────────────────
def extract_images(pdf_path: str) -> list[dict]:
    """
    Extract all embedded images from a PDF.

    Steps:
      2a. fitz.open(pdf_path)
      2b. For each page: page.get_images(full=True) → list of image refs
          Each ref is (xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter, referencer)
      2c. doc.extract_image(xref) → {"image": bytes, "ext": str, "width": int, "height": int}
      2d. Deduplicate: track seen xrefs (same image can appear on multiple pages)
      2e. Skip tiny images (width < 50 or height < 50 — likely icons/bullets)

    Returns:
        list[dict] — [{"bytes": bytes, "ext": str, "page": int, "xref": int,
                        "width": int, "height": int, "source": pdf_path}]
    """
    # import fitz
    # doc = fitz.open(pdf_path)
    # seen_xrefs = set()
    # ...
    raise NotImplementedError
