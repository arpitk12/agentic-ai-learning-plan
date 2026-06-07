"""
Ingestion pipeline — orchestrates: load → chunk → embed → store.

This pipeline runs OFFLINE (not at serve time).
Run it when documents are added or updated:
    python scripts/ingest.py --source data/sample_docs

In production: trigger via webhook, cron job, or CI step on repo changes.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from src.config import cfg
from src.models import IngestionResult
from src.ingestion.loader import load_directory, load_file
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_texts
from src.store.chroma_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_directory(
    source: Path,
    store: VectorStore | None = None,
    replace_existing: bool = True,
) -> IngestionResult:
    """
    Full pipeline: load all docs from `source` directory → chunk → embed → store.
    If replace_existing=True, deletes old chunks for each doc before reinserting.
    """
    source = Path(source)
    store  = store or VectorStore()
    t0     = time.perf_counter()
    errors: list[str] = []

    docs = load_directory(source)
    if not docs:
        return IngestionResult(
            source=str(source), docs_loaded=0, chunks_created=0,
            chunks_stored=0, skipped=0, duration_s=0, errors=["No documents found."]
        )

    all_chunks, skipped = [], 0
    for doc in docs:
        if replace_existing:
            store.delete_by_doc(doc.doc_id)
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    # Batch embed all chunks in one pass (efficient)
    texts      = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)

    # Upsert into vector store
    ids        = [c.chunk_id  for c in all_chunks]
    documents  = [c.content   for c in all_chunks]
    metadatas  = [c.metadata  for c in all_chunks]

    stored = store.upsert(ids, embeddings, documents, metadatas)

    result = IngestionResult(
        source        = str(source),
        docs_loaded   = len(docs),
        chunks_created= len(all_chunks),
        chunks_stored = stored,
        skipped       = skipped,
        duration_s    = round(time.perf_counter() - t0, 2),
        errors        = errors,
    )
    logger.info(
        "Ingestion complete: %d docs → %d chunks stored in %.2fs",
        result.docs_loaded, result.chunks_stored, result.duration_s,
    )
    return result


def ingest_text(
    text: str,
    title: str = "dynamic",
    source: str = "api",
    store: VectorStore | None = None,
) -> IngestionResult:
    """
    Ingest a raw text string at runtime (e.g. via API or MCP tool).
    Writes a temp RawDocument, chunks it, embeds, and stores.
    """
    import hashlib
    from src.models import RawDocument
    store = store or VectorStore()
    t0    = time.perf_counter()

    doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
    from src.models import RawDocument
    raw    = RawDocument(doc_id=doc_id, source=source, title=title,
                         content=text, metadata={"source": source})
    chunks = chunk_document(raw)
    if not chunks:
        return IngestionResult(source=source, docs_loaded=0, chunks_created=0,
                               chunks_stored=0, skipped=1, duration_s=0)

    embeddings = embed_texts([c.content for c in chunks])
    stored = store.upsert(
        [c.chunk_id for c in chunks],
        embeddings,
        [c.content  for c in chunks],
        [c.metadata for c in chunks],
    )
    return IngestionResult(
        source=source, docs_loaded=1, chunks_created=len(chunks),
        chunks_stored=stored, skipped=0,
        duration_s=round(time.perf_counter() - t0, 2),
    )
