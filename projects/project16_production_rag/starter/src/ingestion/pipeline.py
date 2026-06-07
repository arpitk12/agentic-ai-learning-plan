"""
TODO — Implement the offline ingestion pipeline and runtime ingest helper.

Two public functions:

1. ingest_directory(source, store, replace_existing=False) → IngestionResult
   Offline batch pipeline:
     load_directory → for each doc → chunk → embed batch → store.upsert
   Called by: scripts/ingest.py (before the API starts)

2. ingest_text(text, title, source, store) → IngestionResult
   Runtime single-doc ingestion:
     create RawDocument → chunk → embed → store.upsert
   Called by: POST /ingest API route (while API is running)
"""
from __future__ import annotations
import hashlib
import logging
from src.models import RawDocument, IngestionResult
from src.store.chroma_store import VectorStore
from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)


def ingest_directory(
    source: str,
    store: VectorStore,
    replace_existing: bool = False,
) -> IngestionResult:
    """
    Ingest all .md/.txt files in `source` directory.

    TODO 1: Call load_directory(source) to get list[RawDocument]
    TODO 2: For each doc, check if it already exists:
              existing = {d["metadata"]["doc_id"] for d in store.get_all_documents()}
              if doc.doc_id in existing and not replace_existing → skip (increment skipped)
    TODO 3: If replacing, call store.delete_by_doc(doc.doc_id) first
    TODO 4: Call chunk_document(doc) to get list[Chunk]
    TODO 5: Call embed_texts([c.text for c in chunks]) to get embeddings
    TODO 6: Call store.upsert(ids, embeddings, documents, metadatas)
              metadatas = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id,
                            "title": c.title, "source": c.source, "index": c.index}
                           for c in chunks]
    TODO 7: Accumulate counts, catch per-doc errors, return IngestionResult
    """
    raise NotImplementedError


def ingest_text(
    text: str,
    title: str = "Untitled",
    source: str = "api",
    store: VectorStore | None = None,
) -> IngestionResult:
    """
    Ingest a single text snippet at runtime (called from POST /ingest).

    TODO 8: Generate a stable doc_id from title+source:
              doc_id = hashlib.md5(f"{title}:{source}".encode()).hexdigest()[:16]
    TODO 9: Create RawDocument(doc_id=doc_id, title=title, source=source, content=text)
    TODO 10: chunk_document → embed_texts → store.upsert (same flow as ingest_directory)
    TODO 11: Return IngestionResult(documents_processed=1, documents_skipped=0,
                                   chunks_created=len(chunks), errors=[])
    """
    raise NotImplementedError
