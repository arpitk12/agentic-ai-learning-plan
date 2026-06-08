"""
SOLUTION — Exercise 1: Multi-Modal PDF Processing + Vision Agent Pipeline
Phase 7 / Week 14

How this solution works:
  TODO 1: pymupdf4llm.to_markdown extracts text with layout preservation and saves
           embedded images to disk. We split on "---" page separators and chunk text.
  TODO 2: Images are base64-encoded and sent to a vision LLM which returns structured
           JSON describing content type, description, and any extracted data.
  TODO 3: Two ChromaDB collections — one for text chunks, one for image descriptions.
  TODO 4/5: Each chunk/image gets embedded by the default sentence-transformer and stored.
  TODO 6: Multi-modal search queries both collections and merges results by relevance.
  TODO 7: QA agent calls both retrievers, builds a combined context, calls GPT-4o.
  BONUS:  Scanned pages are sent to vision model for full-page OCR.
"""
from __future__ import annotations
import os, json, base64, asyncio, textwrap
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TextChunk:
    doc_id: str
    page: int
    chunk_idx: int
    text: str
    heading: str | None = None

@dataclass
class ImageContext:
    doc_id: str
    page: int
    image_idx: int
    image_path: str
    description: str
    data_extracted: dict | None = None


# ── TODO 1 SOLUTION: PDF Text + Image Extraction ──────────────────────────────

def extract_pdf_content(
    pdf_path: str, output_dir: str = "./extracted"
) -> tuple[list[TextChunk], list[str]]:
    import pymupdf4llm  # type: ignore

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc_id = Path(pdf_path).stem

    # to_markdown saves images and returns markdown with "---" page separators
    md_text: str = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path=output_dir,
        image_format="png",
    )

    # Split on "---" page separators
    pages = md_text.split("\n---\n")

    chunks: list[TextChunk] = []
    current_heading: str | None = None

    for page_num, page_text in enumerate(pages, start=1):
        # Track headings (lines starting with #)
        for line in page_text.split("\n"):
            if line.startswith("#"):
                current_heading = line.lstrip("#").strip()

        # Split page into ~500-char chunks on paragraph boundaries
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        buf = ""
        chunk_idx = 0
        for para in paragraphs:
            buf += para + "\n\n"
            if len(buf) >= 500:
                chunks.append(TextChunk(
                    doc_id=doc_id,
                    page=page_num,
                    chunk_idx=chunk_idx,
                    text=buf.strip(),
                    heading=current_heading,
                ))
                chunk_idx += 1
                buf = ""
        if buf.strip():
            chunks.append(TextChunk(
                doc_id=doc_id, page=page_num, chunk_idx=chunk_idx,
                text=buf.strip(), heading=current_heading,
            ))

    # Find all images saved by pymupdf4llm
    image_paths: list[str] = sorted([
        str(p) for p in Path(output_dir).glob("*.png")
    ] + [
        str(p) for p in Path(output_dir).glob("*.jpg")
    ])

    print(f"  Extracted {len(chunks)} text chunks and {len(image_paths)} images from {doc_id}")
    return chunks, image_paths


# ── TODO 2 SOLUTION: Vision LLM Analysis ─────────────────────────────────────

async def analyze_image_with_vision(
    image_path: str, context: str = "", model: str = "openai/gpt-4o"
) -> ImageContext:
    img_data = base64.b64encode(Path(image_path).read_bytes()).decode()
    ext = Path(image_path).suffix.lstrip(".")
    page_num = 0
    image_idx = 0
    # pymupdf4llm names images like: extracted/1-1.png (page-imageIdx)
    stem = Path(image_path).stem
    parts = stem.split("-")
    if len(parts) >= 2:
        try:
            page_num = int(parts[0])
            image_idx = int(parts[1])
        except ValueError:
            pass

    prompt = f"""Analyze this image extracted from a business document.
Context: {context or 'Business compliance document'}

Return ONLY valid JSON with these fields:
{{
  "description": "What does this image show? (2-3 sentences)",
  "type": "chart|table|diagram|signature|photo|text|other",
  "data": null or extracted data (for charts: {{"labels": [], "values": []}}, for tables: [{{"col": "val"}}])
}}"""

    resp = await litellm.acompletion(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_data}"}},
            ],
        }],
        response_format={"type": "json_object"},
    )

    result = json.loads(resp.choices[0].message.content)
    return ImageContext(
        doc_id=Path(image_path).parent.name,
        page=page_num,
        image_idx=image_idx,
        image_path=image_path,
        description=result["description"],
        data_extracted=result.get("data"),
    )


# ── TODO 3 SOLUTION: Setup ChromaDB Collections ───────────────────────────────

def setup_collections():
    import chromadb  # type: ignore
    client = chromadb.Client()
    # Delete collections if they exist (idempotent setup)
    for name in ["text_chunks", "image_contexts"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    text_col = client.create_collection("text_chunks")
    image_col = client.create_collection("image_contexts")
    print("  ChromaDB collections created: text_chunks, image_contexts")
    return text_col, image_col


# ── TODO 4 SOLUTION: Store text chunks ───────────────────────────────────────

def store_text_chunks(collection, chunks: list[TextChunk]) -> None:
    if not chunks:
        return
    collection.add(
        ids=[f"{c.doc_id}_p{c.page}_c{c.chunk_idx}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{
            "doc_id": c.doc_id,
            "page": c.page,
            "chunk_idx": c.chunk_idx,
            "heading": c.heading or "",
        } for c in chunks],
    )
    print(f"  Stored {len(chunks)} text chunks in ChromaDB")


# ── TODO 5 SOLUTION: Store image contexts ────────────────────────────────────

def store_image_contexts(collection, contexts: list[ImageContext]) -> None:
    if not contexts:
        return
    collection.add(
        ids=[f"{c.doc_id}_p{c.page}_img{c.image_idx}" for c in contexts],
        documents=[c.description for c in contexts],
        metadatas=[{
            "doc_id": c.doc_id,
            "page": c.page,
            "image_path": c.image_path,
            "type": c.data_extracted.get("type", "unknown") if c.data_extracted else "unknown",
        } for c in contexts],
    )
    print(f"  Stored {len(contexts)} image contexts in ChromaDB")


# ── TODO 6 SOLUTION: Multi-modal search ──────────────────────────────────────

def multimodal_search(
    text_col, image_col, query: str, n_results: int = 3
) -> dict[str, list[dict]]:
    text_results = text_col.query(query_texts=[query], n_results=n_results)
    image_results = image_col.query(query_texts=[query], n_results=n_results)

    def parse_results(results: dict) -> list[dict]:
        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, distances):
            out.append({"content": doc, "metadata": meta, "score": 1 - dist})
        return out

    return {
        "text": parse_results(text_results),
        "images": parse_results(image_results),
    }


# ── TODO 7 SOLUTION: Multi-modal QA agent ────────────────────────────────────

async def multimodal_qa(text_col, image_col, question: str, model: str = "openai/gpt-4o-mini") -> str:
    # Retrieve from both modalities
    search = multimodal_search(text_col, image_col, question)

    text_context = "\n\n".join(
        f"[Text p{r['metadata'].get('page', '?')}]: {r['content']}"
        for r in search["text"]
    )
    image_context = "\n\n".join(
        f"[Image p{r['metadata'].get('page', '?')} — {r['metadata'].get('type', 'image')}]: {r['content']}"
        for r in search["images"]
    )

    system = """You are a document analyst. Answer questions using the provided text and image context.
If the answer comes from an image/chart, say so. Be precise and cite page numbers."""

    user = f"""Question: {question}

Text context:
{text_context or "(none)"}

Image/chart context:
{image_context or "(none)"}

Answer:"""

    resp = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── BONUS: OCR scanned PDFs ───────────────────────────────────────────────────

async def ocr_scanned_pdf(pdf_path: str, model: str = "openai/gpt-4o") -> list[str]:
    """
    For PDFs where all content is images (scanned), extract each page as image
    and send to vision model for full-page OCR.
    """
    import fitz  # PyMuPDF  # type: ignore
    doc = fitz.open(pdf_path)
    transcripts: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        resp = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe all visible text from this scanned document page. Preserve formatting. Return only the text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
        )
        text = resp.choices[0].message.content.strip()
        transcripts.append(text)
        print(f"  OCR page {page_num + 1}: {len(text)} chars extracted")

    return transcripts


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Multi-Modal Pipeline — SOLUTION ===\n")
    print("NOTE: This exercise needs a real PDF. Creating a demo with a minimal PDF.\n")

    # Create a minimal test PDF (requires reportlab)
    test_pdf = "test_document.pdf"
    try:
        from reportlab.pdfgen import canvas  # type: ignore
        c = canvas.Canvas(test_pdf)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Q3 2025 Compliance Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, 720, "Revenue grew 23% YoY to $4.2M.")
        c.drawString(100, 700, "GDPR incidents: 0. SOX audit: passed.")
        c.drawString(100, 680, "Vendor risk: DataVendor Ltd flagged for ISO 27001 gap.")
        c.save()
        print(f"  Created minimal test PDF: {test_pdf}\n")
    except ImportError:
        print("  reportlab not installed. Use your own PDF as test_document.pdf\n")
        if not Path(test_pdf).exists():
            return

    print("1. Extracting text and images from PDF...")
    chunks, image_paths = extract_pdf_content(test_pdf)

    print("\n2. Setting up ChromaDB collections...")
    text_col, image_col = setup_collections()

    print("\n3. Storing text chunks...")
    store_text_chunks(text_col, chunks)

    if image_paths:
        print(f"\n4. Analyzing {len(image_paths)} images with vision LLM...")
        contexts = await asyncio.gather(*[
            analyze_image_with_vision(p) for p in image_paths
        ])
        print("\n5. Storing image contexts...")
        store_image_contexts(image_col, list(contexts))
    else:
        print("\n4-5. No images found in PDF (text-only document)")

    print("\n6. Multi-modal QA...")
    questions = [
        "What was the revenue growth?",
        "Were there any GDPR incidents?",
        "Which vendor was flagged for a security gap?",
    ]
    for q in questions:
        print(f"\n  Q: {q}")
        answer = await multimodal_qa(text_col, image_col, q)
        print(f"  A: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
