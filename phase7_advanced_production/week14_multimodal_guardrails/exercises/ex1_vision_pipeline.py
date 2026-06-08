"""
Exercise 1: Multi-Modal PDF Processing + Vision Agent Pipeline
Phase 7 / Week 14 — Multi-Modal Agents + Advanced Guardrails

Goal: Build a pipeline that extracts text AND images from PDFs, embeds both,
      and runs a vision-capable agent that can answer questions about charts,
      tables, and scanned content in business documents.

Stack: pymupdf4llm · chromadb · sentence-transformers · litellm · pillow

pip install pymupdf4llm chromadb sentence-transformers litellm pillow python-dotenv

TODOs:
  1. Extract text + images from a PDF using pymupdf4llm
  2. Analyze each extracted image with a vision LLM to get a text description
  3. Create separate ChromaDB collections for text chunks and image descriptions
  4. Embed and store text chunks + image metadata
  5. Build a multi-modal search function that queries both collections
  6. Build a QA agent that uses multi-modal retrieval for context
  7. BONUS: Handle scanned (image-only) PDFs by OCR-ing all pages with vision LLM
"""
from __future__ import annotations
import os, json, base64, asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    doc_id: str
    page: int
    chunk_idx: int
    text: str
    heading: str | None = None  # nearest heading above this chunk

@dataclass
class ImageContext:
    doc_id: str
    page: int
    image_idx: int
    image_path: str
    description: str  # LLM-generated description
    data_extracted: dict | None = None  # structured data from charts/tables

# ── TODO 1: PDF Text + Image Extraction ───────────────────────────────────────

def extract_pdf_content(pdf_path: str, output_dir: str = "./extracted") -> tuple[list[TextChunk], list[str]]:
    """
    TODO 1: Extract text chunks and save images from a PDF.

    Steps:
    a) import pymupdf4llm
       md_text = pymupdf4llm.to_markdown(pdf_path, write_images=True, image_path=output_dir)

    b) Split the markdown into chunks by section (split on "## " or "\n\n"):
       - Each chunk: ~500 characters
       - Try to split on paragraph boundaries
       - Track page numbers by counting "-----" horizontal rules in pymupdf4llm output

    c) Find all images saved to output_dir (*.png, *.jpg, *.jpeg)

    d) Return (list[TextChunk], list[image_paths])

    Hint: pymupdf4llm.to_markdown returns markdown with "---" page separators
    and saves images as {output_dir}/{page_number}-{image_index}.png
    """
    Path(output_dir).mkdir(exist_ok=True)
    doc_id = Path(pdf_path).stem

    # TODO 1: implement here
    raise NotImplementedError

# ── TODO 2: Vision LLM Analysis ───────────────────────────────────────────────

async def analyze_image_with_vision(
    image_path: str, context: str = "", model: str = "openai/gpt-4o"
) -> ImageContext:
    """
    TODO 2: Send an image to a vision LLM and get a structured description.

    Steps:
    a) Load and base64-encode the image:
       img_data = base64.b64encode(Path(image_path).read_bytes()).decode()
       ext = Path(image_path).suffix.lstrip(".")

    b) Build a prompt asking the model to:
       - Describe what the image shows (chart, table, diagram, signature, etc.)
       - If it's a chart: extract data points as JSON {"labels": [...], "values": [...]}
       - If it's a table: convert to JSON array of row dicts
       - If it's text/scanned: transcribe the visible text
       - Always return: {"description": str, "type": str, "data": dict|null}

    c) Call litellm.acompletion with model and the image_url content block.
       (See week14/notes.md for the message format.)

    d) Parse the JSON response.

    e) Return an ImageContext with doc_id from image_path filename,
       description, and data_extracted from the parsed JSON.
    """
    # TODO 2: implement here
    raise NotImplementedError

# ── TODO 3: Initialize ChromaDB Collections ───────────────────────────────────

def setup_collections(db_path: str = "./multimodal_db") -> tuple[Any, Any]:
    """
    TODO 3: Create two ChromaDB collections.

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    Create a PersistentClient at db_path.
    Use SentenceTransformerEmbeddingFunction("BAAI/bge-base-en-v1.5") as embedding_function.
    Get or create collections:
      - "text_chunks" with the embedding function
      - "image_descriptions" with the embedding function

    Return (text_collection, image_collection).
    """
    # TODO 3: implement here
    raise NotImplementedError

# ── TODO 4: Embed and Store Content ───────────────────────────────────────────

def store_text_chunks(collection, chunks: list[TextChunk]) -> int:
    """
    TODO 4a: Embed and store text chunks in the text collection.

    Call collection.add() with:
      ids=[f"{c.doc_id}_text_{c.chunk_idx}" for c in chunks]
      documents=[c.text for c in chunks]
      metadatas=[{"doc_id": c.doc_id, "page": c.page, "heading": c.heading or ""} for c in chunks]

    ChromaDB auto-embeds using the collection's embedding function.
    Return the count of stored chunks.
    """
    # TODO 4a: implement here
    raise NotImplementedError

def store_image_contexts(collection, images: list[ImageContext]) -> int:
    """
    TODO 4b: Store image descriptions (the TEXT description, not the pixel data).

    Call collection.add() with:
      ids=[f"{img.doc_id}_img_{img.image_idx}" for img in images]
      documents=[img.description for img in images]   ← embed the description text
      metadatas=[{"doc_id": img.doc_id, "page": img.page,
                  "image_path": img.image_path,
                  "data": json.dumps(img.data_extracted or {})} for img in images]

    Return the count of stored images.
    """
    # TODO 4b: implement here
    raise NotImplementedError

# ── TODO 5: Multi-Modal Search ────────────────────────────────────────────────

def multimodal_search(
    text_col, image_col, query: str, n_results: int = 3
) -> list[dict]:
    """
    TODO 5: Search both collections and merge results.

    a) Query text_col: text_hits = text_col.query(query_texts=[query], n_results=n_results)
    b) Query image_col: img_hits = image_col.query(query_texts=[query], n_results=n_results)

    c) Merge into a unified list of dicts:
       [{"type": "text", "content": doc, "metadata": meta, "distance": dist}, ...]

    d) Sort by distance (ascending = more similar).
    e) Return top n_results from the merged list.

    Note: distances from ChromaDB are L2 distances — lower = more similar.
    """
    # TODO 5: implement here
    raise NotImplementedError

# ── TODO 6: Multi-Modal QA Agent ──────────────────────────────────────────────

async def multimodal_qa(
    text_col, image_col, question: str, model: str = "openai/gpt-4o"
) -> dict:
    """
    TODO 6: Answer a question using text + image context.

    Steps:
    a) Search both collections: hits = multimodal_search(text_col, image_col, question)

    b) Build context string from hits:
       For text hits: f"[TEXT - Page {meta['page']}]: {content}"
       For image hits: f"[IMAGE - Page {meta['page']}]: {content}"
                       If meta["data"] has data, append it as JSON summary.

    c) Build a message with the context and question.
       If any image hit has an actual image file, include it in the message as
       a base64 image_url content block (like TODO 2, but in the messages).

    d) Call litellm.acompletion and return:
       {"answer": str, "sources": [{"type": ..., "page": ..., "excerpt": ...}]}
    """
    # TODO 6: implement here
    raise NotImplementedError

# ── TODO 7 (BONUS): OCR for scanned PDFs ─────────────────────────────────────

async def ocr_scanned_pdf(pdf_path: str, model: str = "openai/gpt-4o") -> list[dict]:
    """
    TODO 7: Handle image-only (scanned) PDFs where pymupdf4llm returns no text.

    Steps:
    a) import fitz (PyMuPDF)
       doc = fitz.open(pdf_path)

    b) For each page, render to PNG:
       page = doc[i]
       mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for clarity
       pix = page.get_pixmap(matrix=mat)
       pix.save(f"./extracted/page_{i}.png")

    c) Call analyze_image_with_vision() on each page PNG with context="Transcribe all text visible on this page. Return {\"text\": \"...\"}".

    d) Return list of {"page": i, "text": transcribed_text} dicts.

    e) These page texts can then be chunked and stored in the text collection.
    """
    # TODO 7: implement here
    raise NotImplementedError

# ── Demo helpers ──────────────────────────────────────────────────────────────

def create_sample_pdf(output_path: str = "./sample_doc.pdf"):
    """Create a simple sample PDF for testing (requires reportlab)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(output_path, pagesize=letter)
        c.drawString(100, 750, "VENDOR COMPLIANCE REPORT Q1 2026")
        c.drawString(100, 720, "Executive Summary")
        c.drawString(100, 700, "This report covers compliance status for 47 active vendors.")
        c.drawString(100, 680, "Critical findings: 3 vendors missing DPA clauses.")
        c.drawString(100, 660, "Estimated remediation cost: $45,000")
        c.save()
        print(f"Sample PDF created: {output_path}")
    except ImportError:
        print("pip install reportlab to create sample PDFs")
        print("Or provide your own PDF path.")

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Multi-Modal PDF Agent Exercise ===\n")

    pdf_path = "./sample_doc.pdf"
    if not Path(pdf_path).exists():
        create_sample_pdf(pdf_path)

    # Step 1: Extract
    print("1. Extracting PDF content...")
    chunks, image_paths = extract_pdf_content(pdf_path)
    print(f"   Text chunks: {len(chunks)} | Images found: {len(image_paths)}")

    # Step 2: Analyze images
    print("2. Analyzing images with vision LLM...")
    image_contexts = await asyncio.gather(*[
        analyze_image_with_vision(img_path) for img_path in image_paths
    ]) if image_paths else []
    print(f"   Analyzed {len(image_contexts)} images")

    # Step 3-4: Store in ChromaDB
    print("3. Storing in multi-modal vector DB...")
    text_col, image_col = setup_collections()
    n_text = store_text_chunks(text_col, chunks)
    n_img = store_image_contexts(image_col, list(image_contexts))
    print(f"   Stored {n_text} text chunks + {n_img} image descriptions")

    # Step 5-6: QA
    print("4. Running multi-modal Q&A...")
    questions = [
        "How many vendors are missing DPA clauses?",
        "What is the estimated remediation cost?",
    ]
    for q in questions:
        print(f"\n   Q: {q}")
        result = await multimodal_qa(text_col, image_col, q)
        print(f"   A: {result.get('answer', 'N/A')[:200]}")

    print("\n✅ Multi-modal pipeline exercise complete!")

if __name__ == "__main__":
    asyncio.run(main())
