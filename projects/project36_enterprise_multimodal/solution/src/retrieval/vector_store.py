"""
solution/src/retrieval/vector_store.py — Full implementation.
"""
from __future__ import annotations
import chromadb  # type: ignore

COLLECTION_TEXT   = "text_chunks"
COLLECTION_IMAGES = "image_contexts"
COLLECTION_AUDIO  = "audio_segments"


def setup_store(persist_dir: str = "./chroma_db") -> dict:
    client = chromadb.PersistentClient(path=persist_dir)
    return {
        "text":   client.get_or_create_collection(COLLECTION_TEXT),
        "images": client.get_or_create_collection(COLLECTION_IMAGES),
        "audio":  client.get_or_create_collection(COLLECTION_AUDIO),
    }


def upsert_text_chunks(collection, chunks: list[dict], doc_id: str) -> int:
    if not chunks:
        return 0
    ids = [f"{doc_id}_chunk_{c['chunk_idx']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def upsert_image_contexts(collection, images: list[dict], doc_id: str) -> int:
    valid = [img for img in images if img.get("description")]
    if not valid:
        return 0
    ids = [f"{doc_id}_img_{img['xref']}" for img in valid]
    documents = [img.get("description", "") for img in valid]
    metadatas = [{k: v for k, v in img.items()
                  if k not in ("bytes", "description") and isinstance(v, (str, int, float, bool))}
                 for img in valid]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(valid)


def upsert_audio_segments(collection, segments: list[dict], doc_id: str) -> int:
    if not segments:
        return 0
    ids = [f"{doc_id}_audio_{i}" for i in range(len(segments))]
    documents = [s["text"] for s in segments]
    metadatas = [{k: v for k, v in s.items() if k != "text"} for s in segments]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(segments)


def query_collection(collection, query_text: str, n_results: int = 5) -> list[dict]:
    try:
        result = collection.query(query_texts=[query_text], n_results=n_results)
        docs = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result.get("metadatas", [[{}] * len(docs)])[0]
        hits = []
        for doc, dist, meta in zip(docs, distances, metadatas):
            hits.append({
                "text": doc,
                "score": 1.0 / (1.0 + dist),   # L2 distance → similarity
                "metadata": meta or {},
            })
        return sorted(hits, key=lambda h: h["score"], reverse=True)
    except Exception:
        return []
