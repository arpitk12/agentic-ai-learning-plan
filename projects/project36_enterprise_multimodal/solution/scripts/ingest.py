"""
solution/scripts/ingest.py — Full implementation.
"""
from __future__ import annotations
import argparse, asyncio, os, time, uuid
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
PDF_EXTENSIONS   = {".pdf"}


async def ingest_pdf(pdf_path: str, deps: dict, sem: asyncio.Semaphore) -> dict:
    from src.ingestion.pdf_extractor import extract_text_chunks, extract_images  # type: ignore
    from src.ingestion.vision_analyzer import analyze_images_batch  # type: ignore
    from src.graph.entity_extractor import extract_all  # type: ignore
    from src.retrieval.vector_store import upsert_text_chunks, upsert_image_contexts  # type: ignore
    from src.graph.neo4j_store import load_document  # type: ignore

    async with sem:
        doc_id = str(uuid.uuid4())
        chunks = extract_text_chunks(pdf_path)
        raw_images = extract_images(pdf_path)
        annotated_images = await analyze_images_batch(raw_images) if raw_images else []
        entities, relations = await extract_all(chunks)
        upsert_text_chunks(deps["collections"]["text"], chunks, doc_id)
        upsert_image_contexts(deps["collections"]["images"], annotated_images, doc_id)
        if deps.get("driver"):
            load_document(deps["driver"], doc_id, pdf_path, entities, relations)
        return {
            "path": pdf_path, "doc_id": doc_id,
            "chunks": len(chunks), "images": len(annotated_images),
            "entities": len(entities),
        }


async def ingest_audio(audio_path: str, deps: dict, sem: asyncio.Semaphore) -> dict:
    from src.ingestion.audio_transcriber import transcribe, chunk_transcript  # type: ignore
    from src.retrieval.vector_store import upsert_audio_segments  # type: ignore

    async with sem:
        doc_id = str(uuid.uuid4())
        transcript = await asyncio.to_thread(transcribe, audio_path)
        segments = chunk_transcript(transcript)
        upsert_audio_segments(deps["collections"]["audio"], segments, doc_id)
        return {"path": audio_path, "doc_id": doc_id, "audio_segments": len(segments)}


async def main(input_dir: str, workers: int = 4):
    from src.config import get_config  # type: ignore
    from src.retrieval.vector_store import setup_store  # type: ignore

    cfg = get_config()
    collections = setup_store(cfg.chroma_persist_dir)
    driver = None
    try:
        from src.graph.neo4j_store import connect  # type: ignore
        driver = connect(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
    except Exception:
        print("[ingest] Neo4j unavailable — graph indexing disabled")

    deps = {"collections": collections, "driver": driver}
    sem = asyncio.Semaphore(workers)

    files = list(Path(input_dir).rglob("*"))
    pdf_files   = [str(f) for f in files if f.suffix.lower() in PDF_EXTENSIONS]
    audio_files = [str(f) for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]

    print(f"Found {len(pdf_files)} PDFs and {len(audio_files)} audio files in {input_dir}")

    t0 = time.time()
    tasks = (
        [ingest_pdf(p, deps, sem) for p in pdf_files] +
        [ingest_audio(a, deps, sem) for a in audio_files]
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]

    print(f"\nIngestion complete in {time.time()-t0:.1f}s")
    print(f"  Succeeded: {len(ok)} files")
    print(f"  Errors:    {len(errors)}")
    total_chunks   = sum(r.get("chunks", 0) for r in ok)
    total_images   = sum(r.get("images", 0) for r in ok)
    total_entities = sum(r.get("entities", 0) for r in ok)
    total_audio    = sum(r.get("audio_segments", 0) for r in ok)
    print(f"  Text chunks:    {total_chunks}")
    print(f"  Image contexts: {total_images}")
    print(f"  Entities:       {total_entities}")
    print(f"  Audio segments: {total_audio}")

    if driver:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main(args.input, args.workers))
