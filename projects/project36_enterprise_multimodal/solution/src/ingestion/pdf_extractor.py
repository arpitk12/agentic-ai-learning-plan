"""
solution/src/ingestion/pdf_extractor.py — Full implementation.
"""
from __future__ import annotations

def extract_text_chunks(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    import fitz  # type: ignore
    doc = fitz.open(pdf_path)
    all_text_pages: list[tuple[str, int]] = []  # (text, page_num)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            all_text_pages.append((text, page_num))

    chunks: list[dict] = []
    chunk_idx = 0
    for text, page_num in all_text_pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_idx": chunk_idx,
                    "source": pdf_path,
                })
                chunk_idx += 1
            start += chunk_size - overlap
    return chunks


def extract_images(pdf_path: str) -> list[dict]:
    import fitz  # type: ignore
    doc = fitz.open(pdf_path)
    seen_xrefs: set[int] = set()
    images: list[dict] = []
    for page_num, page in enumerate(doc, start=1):
        for img_ref in page.get_images(full=True):
            xref = img_ref[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                img_info = doc.extract_image(xref)
                w, h = img_info.get("width", 0), img_info.get("height", 0)
                if w < 50 or h < 50:
                    continue  # skip icons / decorations
                images.append({
                    "bytes": img_info["image"],
                    "ext": img_info.get("ext", "png"),
                    "page": page_num,
                    "xref": xref,
                    "width": w,
                    "height": h,
                    "source": pdf_path,
                })
            except Exception:
                continue
    return images
