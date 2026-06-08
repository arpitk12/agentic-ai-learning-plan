"""
Project 26 SOLUTION — Multi-Modal Document Intelligence Agent
PDF + Vision + Audio pipeline with cross-modality ChromaDB RAG.
"""
from __future__ import annotations
import os, json, base64, asyncio
from pathlib import Path
from dataclasses import dataclass
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── PDF Text Extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str, output_dir: str = "./extracted") -> list[dict]:
    import pymupdf4llm  # type: ignore
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    md_text = pymupdf4llm.to_markdown(pdf_path, write_images=True, image_path=output_dir)
    pages = md_text.split("\n---\n")
    chunks = []
    for page_num, page_text in enumerate(pages, 1):
        heading = None
        for line in page_text.split("\n"):
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
        for chunk_idx, chunk in enumerate(
            [p.strip() for p in page_text.split("\n\n") if len(p.strip()) > 50]
        ):
            chunks.append({"page": page_num, "chunk_idx": chunk_idx, "text": chunk, "heading": heading})
    return chunks

def extract_pdf_images(pdf_path: str, output_dir: str = "./extracted") -> list[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return sorted([
        str(p) for p in Path(output_dir).glob("*.png")
    ] + [str(p) for p in Path(output_dir).glob("*.jpg")])


# ── Vision Analysis ───────────────────────────────────────────────────────────

async def analyze_image(image_path: str, model: str = "openai/gpt-4o") -> dict:
    img_data = base64.b64encode(Path(image_path).read_bytes()).decode()
    ext = Path(image_path).suffix.lstrip(".")
    resp = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": 'Analyze this image. Return JSON: {"type": "chart|table|diagram|photo|text|other", "description": "2-3 sentences", "data": null_or_extracted_data}'},
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_data}"}},
        ]}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ── Audio Transcription ───────────────────────────────────────────────────────

def transcribe_audio(audio_path: str, chunk_seconds: int = 60) -> list[dict]:
    import whisper  # type: ignore
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, verbose=False)
    segments = result.get("segments", [])
    # Group into chunk_seconds windows
    chunks = []
    buf_text, buf_start, buf_end = "", None, None
    for seg in segments:
        if buf_start is None:
            buf_start = seg["start"]
        buf_text += " " + seg["text"]
        buf_end = seg["end"]
        if buf_end - buf_start >= chunk_seconds:
            chunks.append({"start_sec": buf_start, "end_sec": buf_end, "text": buf_text.strip()})
            buf_text, buf_start, buf_end = "", None, None
    if buf_text:
        chunks.append({"start_sec": buf_start or 0, "end_sec": buf_end or 0, "text": buf_text.strip()})
    return chunks


# ── ChromaDB Multi-Modal Store ────────────────────────────────────────────────

def setup_store():
    import chromadb  # type: ignore
    client = chromadb.Client()
    for name in ["text_chunks", "image_contexts", "audio_segments"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return {
        "text": client.create_collection("text_chunks"),
        "image": client.create_collection("image_contexts"),
        "audio": client.create_collection("audio_segments"),
    }

def index_text_chunks(store: dict, chunks: list[dict], doc_id: str):
    if not chunks:
        return
    store["text"].add(
        ids=[f"{doc_id}_p{c['page']}_c{c['chunk_idx']}" for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": doc_id, "page": c["page"], "source": "text"} for c in chunks],
    )

def index_image(store: dict, image_path: str, description: str, doc_id: str, page: int):
    store["image"].add(
        ids=[f"{doc_id}_img_{Path(image_path).stem}"],
        documents=[description],
        metadatas=[{"doc_id": doc_id, "page": page, "image_path": image_path, "source": "image"}],
    )

def index_audio_chunks(store: dict, chunks: list[dict], audio_id: str):
    if not chunks:
        return
    store["audio"].add(
        ids=[f"{audio_id}_seg_{i}" for i, _ in enumerate(chunks)],
        documents=[c["text"] for c in chunks],
        metadatas=[{"audio_id": audio_id, "start_sec": c["start_sec"], "source": "audio"} for c in chunks],
    )


# ── Multi-Modal Search ────────────────────────────────────────────────────────

def multimodal_search(store: dict, query: str, n: int = 3) -> dict:
    def _query(col, q, n):
        try:
            r = col.query(query_texts=[q], n_results=n)
            docs = r.get("documents", [[]])[0]
            metas = r.get("metadatas", [[]])[0]
            dists = r.get("distances", [[]])[0]
            return [{"content": d, "metadata": m, "score": 1 - dist}
                    for d, m, dist in zip(docs, metas, dists)]
        except Exception:
            return []
    return {
        "text": _query(store["text"], query, n),
        "image": _query(store["image"], query, n),
        "audio": _query(store["audio"], query, n),
    }


# ── Multi-Modal QA Agent ──────────────────────────────────────────────────────

async def multimodal_agent(store: dict, question: str) -> str:
    results = multimodal_search(store, question)

    context_parts = []
    for source, items in results.items():
        for r in items:
            label = f"[{source.upper()} p{r['metadata'].get('page', r['metadata'].get('start_sec', '?'))}]"
            context_parts.append(f"{label} {r['content']}")

    context = "\n\n".join(context_parts) or "(no relevant content found)"

    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer questions using provided text, image, and audio context. Cite which source type you used."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 26: Multi-Modal Agent SOLUTION ===\n")
    print("This solution requires a PDF (and optionally audio) file.")
    print("Set PDF_PATH and optionally AUDIO_PATH environment variables.\n")

    pdf_path = os.getenv("PDF_PATH", "sample.pdf")
    audio_path = os.getenv("AUDIO_PATH", None)

    store = setup_store()

    if Path(pdf_path).exists():
        print(f"1. Processing PDF: {pdf_path}")
        chunks = extract_pdf_text(pdf_path)
        image_paths = extract_pdf_images(pdf_path)
        index_text_chunks(store, chunks, doc_id="doc1")
        print(f"   Indexed {len(chunks)} text chunks, {len(image_paths)} images")

        if image_paths:
            print("2. Analyzing images...")
            for img_path in image_paths[:3]:  # limit for demo
                analysis = await analyze_image(img_path)
                index_image(store, img_path, analysis["description"], "doc1", 1)
    else:
        print(f"PDF not found at {pdf_path}. Add sample content to store for demo...")
        store["text"].add(
            ids=["demo_1"],
            documents=["Q3 2025 revenue grew 23% to $4.2M. GDPR audit passed. Two vendors flagged."],
            metadatas=[{"doc_id": "demo", "page": 1, "source": "text"}],
        )

    if audio_path and Path(audio_path).exists():
        print(f"3. Transcribing audio: {audio_path}")
        audio_chunks = transcribe_audio(audio_path)
        index_audio_chunks(store, audio_chunks, "audio1")
        print(f"   Transcribed {len(audio_chunks)} segments")

    print("\n4. Multi-modal QA:")
    questions = [
        "What was the revenue performance?",
        "Were there any compliance issues?",
        "Which vendors were flagged?",
    ]
    for q in questions:
        print(f"\n  Q: {q}")
        answer = await multimodal_agent(store, q)
        print(f"  A: {answer[:150]}...")

if __name__ == "__main__":
    asyncio.run(main())
