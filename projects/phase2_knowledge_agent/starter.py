"""
Project 2 Starter — Personal Knowledge Agent (RAG + Memory)

Build a CLI chatbot that:
  1. Ingests a folder of documents (PDF/MD/TXT/PY) into a ChromaDB vector store.
  2. Maintains multi-turn conversation memory in SQLite.
  3. Routes each query: RAG (search documents) or DIRECT (LLM general knowledge).
  4. Returns cited answers that reference the source documents.

Usage:
    python starter.py ./docs my_session
    python starter.py ./docs          (uses "default" session)
    python starter.py                 (uses ./docs + "default" session)

What you need to implement (TODOs 1-5):
  1. ingest_docs()    — embed chunks and add them to ChromaDB
  2. classify_query() — LLM router returning "RAG" or "DIRECT"
  3. rag_answer()     — embed query → ChromaDB search → LLM with context
  4. direct_answer()  — plain LLM call with conversation history
  5. main() loop      — route each query, save to DB, print response
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import hashlib
import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()


# ── Embedder + Vector Store ────────────────────────────────────────────────────

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma   = chromadb.Client()

DB_PATH = "knowledge_agent_memory.db"


# ── SQLite Conversation Memory ─────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """Create messages table if it doesn't exist. Returns open connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            ts         DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_message(conn: sqlite3.Connection, session_id: str, role: str, content: str):
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
        (session_id, role, content),
    )
    conn.commit()


def load_history(conn: sqlite3.Connection, session_id: str, last_n: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, last_n),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


# ── Document Ingestion ─────────────────────────────────────────────────────────

def load_file(path: Path) -> str:
    """Read text from PDF, MD, TXT, or PY files."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    elif suffix in {".md", ".txt", ".py", ".js"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of ~`size` characters."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 30]


def ingest_docs(docs_dir: str, collection_name: str = "knowledge") -> chromadb.Collection:
    """
    Walk docs_dir, chunk every supported file, embed the chunks, and
    upsert them into a ChromaDB collection.

    This function is partially implemented — the file-walking and chunking
    are done for you.  You need to add the embedding + ChromaDB upsert step.

    TODO 1 — inside the `if all_chunks:` block:
      a. Encode all chunks into float vectors:
             embeddings = embedder.encode(all_chunks).tolist()
      b. Add them to the collection:
             collection.add(
                 embeddings=embeddings,
                 documents=all_chunks,
                 ids=all_ids,
                 metadatas=all_metas,
             )

    Without this, classify_query() will route everything to DIRECT and
    rag_answer() will have nothing to retrieve from.
    """
    collection = chroma.get_or_create_collection(collection_name)
    doc_path = Path(docs_dir)
    all_chunks, all_ids, all_metas = [], [], []

    for f in doc_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".txt", ".py"}:
            print(f"  📄 Ingesting: {f.name}")
            text   = load_file(f)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{f.name}:{i}:{chunk[:30]}".encode()).hexdigest()
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metas.append({"source": f.name, "chunk": i})

    if all_chunks:
        # TODO 1: encode chunks and add to ChromaDB
        pass

    print(f"  ✅ Indexed {len(all_chunks)} chunks from {doc_path}")
    return collection


# ── Router ─────────────────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Use the LLM to decide whether the query needs document retrieval.

    Returns "RAG" if the answer likely lives in the uploaded documents,
    or "DIRECT" if it's general knowledge the LLM already has.

    TODO 2:
      a. Call chat() with the user's query and a system prompt that says:
             "Reply with exactly one word: RAG or DIRECT.
              RAG = answer requires searching the user's uploaded documents.
              DIRECT = general knowledge you already know."
         Use max_tokens=10 so the model doesn't overthink it.
      b. Extract the text with get_text(r).
      c. Return "RAG" if "RAG" appears in the response (uppercase check),
         otherwise return "DIRECT".

    Example:
        r = chat([{"role": "user", "content": query}],
                 system="Reply with exactly one word: RAG or DIRECT. ...",
                 max_tokens=10)
        return "RAG" if "RAG" in get_text(r).upper() else "DIRECT"
    """
    # TODO 2: implement LLM query router
    raise NotImplementedError("classify_query() not implemented yet")


# ── RAG Answer ─────────────────────────────────────────────────────────────────

def rag_answer(query: str, collection: chromadb.Collection, history: list[dict]) -> str:
    """
    Retrieve relevant document chunks and ask the LLM to answer using them.

    TODO 3:
      a. Embed the query into a float vector:
             q_emb = embedder.encode([query])[0].tolist()
      b. Search ChromaDB for the 4 nearest chunks:
             results = collection.query(query_embeddings=[q_emb], n_results=4)
         Chunk text is in results["documents"][0],
         metadata (source, chunk index) is in results["metadatas"][0].
      c. Build a context string by joining each chunk with its citation:
             "[source: filename, chunk N]\n<chunk text>"
         Separate entries with "\n\n---\n\n".
      d. Build the message list: history[-6:] + a new user message that contains
         the context and asks the model to answer with citations.
         User message format:
             "Context from documents:\n{context}\n\nQuestion: {query}\n\n
              Answer using only the context. Cite sources like [source: filename.ext]."
      e. Call chat() with system="You are a helpful assistant that answers
         questions using only provided document context." and max_tokens=1024.
      f. Return get_text(response).
    """
    # TODO 3: implement RAG retrieval + cited LLM answer
    raise NotImplementedError("rag_answer() not implemented yet")


# ── Direct Answer ──────────────────────────────────────────────────────────────

def direct_answer(query: str, history: list[dict]) -> str:
    """
    Answer the query with the LLM's built-in knowledge.

    TODO 4:
      Build messages = history[-6:] + [{"role": "user", "content": query}]
      Call chat(messages, system="You are a helpful assistant.", max_tokens=1024)
      Return get_text(response).
    """
    # TODO 4: implement direct LLM answer with conversation history
    raise NotImplementedError("direct_answer() not implemented yet")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    docs_dir   = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    session_id = sys.argv[2] if len(sys.argv) > 2 else "default"

    print(f"📚 Knowledge Agent — Session: {session_id}")
    print(f"📁 Docs directory: {docs_dir}")

    conn       = init_db()
    collection = ingest_docs(docs_dir)
    history    = load_history(conn, session_id)
    print(f"\nLoaded {len(history)} previous messages. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        save_message(conn, session_id, "user", query)

        # TODO 5: Replace the stub below with real routing logic.
        #   a. Call classify_query(query) → route  (prints "[Route: RAG]" or "[Route: DIRECT]")
        #   b. If route == "RAG":    answer = rag_answer(query, collection, history)
        #      If route == "DIRECT": answer = direct_answer(query, history)
        #   c. Print: f"  [Route: {route}]" before the answer.
        route  = "DIRECT"                    # ← replace with classify_query(query)
        answer = "Not implemented yet."      # ← replace with routing call
        print(f"  [Route: {route}]")

        save_message(conn, session_id, "assistant", answer)
        history = load_history(conn, session_id)
        print(f"\nAssistant: {answer}\n")

    conn.close()


if __name__ == "__main__":
    main()
