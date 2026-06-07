"""
Ingestion pipeline connecting loaders, chunker, embedder, and vector store.

Two entry points:

ingest_directory — offline batch pipeline, run before the API starts.
  Loads all documents from disk, chunks, embeds, and stores them.
  Skips documents already in the store unless replace_existing is True.

ingest_text — runtime ingestion called by the POST /ingest API route.
  Ingests a single text snippet while the server is live.
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
    Ingest all .md/.txt files in the source directory.

    TODO 1: Load all documents from the source directory
    TODO 2: For each document, check whether its doc_id already exists in the store
    TODO 3: Skip existing documents unless replace_existing is True; if replacing, delete existing chunks first
    TODO 4: Chunk each document into text pieces
    TODO 5: Embed all chunks for a document in a single batch call
    TODO 6: Upsert chunks into the vector store with full metadata
    TODO 7: Track counts and errors; return an IngestionResult when complete
    """
    raise NotImplementedError


def ingest_text(
    text: str,
    title: str = "Untitled",
    source: str = "api",
    store: VectorStore | None = None,
) -> IngestionResult:
    """
    Ingest a single text snippet at runtime.

    TODO 8: Generate a stable doc_id by hashing the title and source together
    TODO 9: Create a RawDocument from the text, title, source, and doc_id
    TODO 10: Chunk, embed, and upsert following the same steps as ingest_directory
    TODO 11: Return an IngestionResult reporting what was created
    """
    raise NotImplementedError
