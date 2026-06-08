"""
src/retrieval/vector_store.py
Three ChromaDB collections — one per modality — with upsert and query logic.

TODOs:
  1. implement setup_store() — create/get all 3 collections
  2. implement upsert_text_chunks() — store text chunks with metadata
  3. implement upsert_image_contexts() — store image descriptions with metadata
  4. implement upsert_audio_segments() — store transcript chunks with metadata
  5. implement query_collection() — similarity search with score normalisation
"""
from __future__ import annotations

# Collection names — one per modality
COLLECTION_TEXT   = "text_chunks"
COLLECTION_IMAGES = "image_contexts"
COLLECTION_AUDIO  = "audio_segments"


# ── TODO 1: Setup ChromaDB store ──────────────────────────────────────────────
def setup_store(persist_dir: str = "./chroma_db") -> dict:
    """
    Initialize ChromaDB with 3 persistent collections.

    Steps:
      1a. import chromadb
      1b. client = chromadb.PersistentClient(path=persist_dir)
      1c. For each collection name: client.get_or_create_collection(name)
      1d. Return {"text": col_text, "images": col_images, "audio": col_audio}
    """
    raise NotImplementedError


# ── TODO 2: Upsert text chunks ────────────────────────────────────────────────
def upsert_text_chunks(
    collection,          # ChromaDB collection object
    chunks: list[dict],  # [{"text", "page", "chunk_idx", "source"}]
    doc_id: str,
) -> int:
    """
    Upsert text chunks into the text_chunks collection.

    Steps:
      2a. Generate deterministic IDs: f"{doc_id}_chunk_{chunk['chunk_idx']}"
      2b. Separate documents (text) from metadatas (everything else)
      2c. collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
      2d. Return number of chunks upserted

    Note: Use upsert (not add) so re-ingestion doesn't create duplicates.
    """
    raise NotImplementedError


# ── TODO 3: Upsert image contexts ─────────────────────────────────────────────
def upsert_image_contexts(
    collection,
    images: list[dict],  # [{"description", "type", "key_data", "page", "xref", "source"}]
    doc_id: str,
) -> int:
    """
    Upsert image descriptions into the image_contexts collection.

    Steps:
      3a. Use image description as the document text for vector embedding
      3b. Include type, key_data, page, source in metadata
      3c. ID: f"{doc_id}_img_{img['xref']}"
      3d. collection.upsert(...)
    """
    raise NotImplementedError


# ── TODO 4: Upsert audio segments ─────────────────────────────────────────────
def upsert_audio_segments(
    collection,
    segments: list[dict],  # [{"text", "start_time", "end_time", "source"}]
    doc_id: str,
) -> int:
    """
    Upsert audio transcript chunks into the audio_segments collection.

    Steps:
      4a. ID: f"{doc_id}_audio_{i}" for each segment index i
      4b. Document text = segment["text"]
      4c. Metadata: start_time, end_time, source
    """
    raise NotImplementedError


# ── TODO 5: Query a collection ────────────────────────────────────────────────
def query_collection(
    collection,
    query_text: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Search a ChromaDB collection and return normalised results.

    Steps:
      5a. collection.query(query_texts=[query_text], n_results=n_results)
          → returns {"documents": [[...]], "distances": [[...]], "metadatas": [[...]]}
      5b. ChromaDB returns L2 distances (lower = closer).
          Convert to similarity score: score = 1 / (1 + distance)
      5c. Return list of {"text": str, "score": float, "metadata": dict}
          sorted by score descending

    Returns:
        list[dict] — ranked results with normalised similarity scores
    """
    raise NotImplementedError
