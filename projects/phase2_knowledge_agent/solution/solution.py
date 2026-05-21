"""
SOLUTION — Project 2: Personal Knowledge Agent (RAG + LangGraph)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import hashlib
import sqlite3
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()

DB_PATH = "knowledge_agent_memory.db"


# ── Database for conversation memory ──────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_message(conn: sqlite3.Connection, session_id: str, role: str, content: str):
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
                 (session_id, role, content))
    conn.commit()


def load_history(conn: sqlite3.Connection, session_id: str, last_n: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, last_n)
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


# ── Document Ingestion ─────────────────────────────────────────────────────────

def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    elif suffix in {".md", ".txt", ".py", ".js"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 30]


def ingest_docs(docs_dir: str, collection_name: str = "knowledge") -> chromadb.Collection:
    collection = chroma.get_or_create_collection(collection_name)
    doc_path = Path(docs_dir)
    all_chunks, all_ids, all_metas = [], [], []

    for f in doc_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".txt", ".py"}:
            print(f"  📄 Ingesting: {f.name}")
            text = load_file(f)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{f.name}:{i}:{chunk[:30]}".encode()).hexdigest()
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metas.append({"source": f.name, "chunk": i})

    if all_chunks:
        embeddings = embedder.encode(all_chunks).tolist()
        collection.add(embeddings=embeddings, documents=all_chunks, ids=all_ids, metadatas=all_metas)
        print(f"  ✅ Indexed {len(all_chunks)} chunks from {doc_path}")
    return collection


# ── Router + RAG ───────────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """Route to: RAG | DIRECT"""
    r = chat(
        [{"role": "user", "content": query}],
        system="Reply with exactly one word: RAG or DIRECT. RAG = answer needs documents. DIRECT = general knowledge.",
        max_tokens=10,
    )
    return "RAG" if "RAG" in get_text(r).upper() else "DIRECT"


def rag_answer(query: str, collection: chromadb.Collection, history: list[dict]) -> str:
    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=4)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    for chunk, meta in zip(chunks, metas):
        context_parts.append(f"[source: {meta['source']}, chunk {meta['chunk']}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    messages = history[-6:] + [{"role": "user", "content":
        f"Context from documents:\n{context}\n\nQuestion: {query}\n\n"
        "Answer using only the context. Include source citations like [source: filename.ext]."}]

    r = chat(
        messages,
        system="You are a helpful assistant that answers questions using only provided document context.",
        max_tokens=1024,
    )
    return get_text(r)


def direct_answer(query: str, history: list[dict]) -> str:
    messages = history[-6:] + [{"role": "user", "content": query}]
    r = chat(messages, system="You are a helpful assistant.", max_tokens=1024)
    return get_text(r)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    session_id = sys.argv[2] if len(sys.argv) > 2 else "default"

    print(f"📚 Knowledge Agent — Session: {session_id}")
    print(f"📁 Docs directory: {docs_dir}")

    conn = init_db()
    collection = ingest_docs(docs_dir)
    history = load_history(conn, session_id)

    print(f"\nLoaded {len(history)} previous messages. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break

        save_message(conn, session_id, "user", query)

        route = classify_query(query)
        print(f"  [Route: {route}]")

        if route == "RAG":
            answer = rag_answer(query, collection, history)
        else:
            answer = direct_answer(query, history)

        save_message(conn, session_id, "assistant", answer)
        history = load_history(conn, session_id)
        print(f"\nAssistant: {answer}\n")

    conn.close()


if __name__ == "__main__":
    main()
