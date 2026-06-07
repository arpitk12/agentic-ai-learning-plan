"""
SOLUTION — Exercise 5: LangChain LCEL Chains & Document Loaders

Key concepts demonstrated:
- LCEL pipe syntax: step1 | step2 | step3 (each receives prior output)
- Document loading: plain text, web pages, directories
- RecursiveCharacterTextSplitter: paragraph → sentence → word fallback
- DocumentQAChain: full RAG retrieval + grounded generation
- chain.stream(): yield tokens one by one for real-time UX
- chain.batch(): process multiple questions in one call

pip install langchain langchain-community langchain-core
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
load_dotenv()

import re
import json
import time
import tempfile
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm import chat, get_text

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: LCEL-style 3-step pipeline
# prompt | model | parser  →  we simulate with plain functions
# ─────────────────────────────────────────────────────────────────────────────

def step_explain(topic: str) -> str:
    return get_text(chat([{
        "role": "user",
        "content": f"Explain '{topic}' in 3 concise sentences for a senior engineer.",
    }]))


def step_key_points(explanation: str) -> list[str]:
    raw = get_text(chat([{
        "role": "user",
        "content": (
            "Extract exactly 3 key points from this explanation as a JSON array:\n"
            f"{explanation}\n\n"
            'Output ONLY valid JSON: ["point1", "point2", "point3"]'
        ),
    }]))
    clean = re.sub(r"```json?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return [explanation]


def step_format(points: list[str]) -> str:
    return "\n".join(f"  {i+1}. {p}" for i, p in enumerate(points))


def run_lcel_pipeline(topic: str) -> str:
    """chain = explain | extract_points | format_output"""
    explanation = step_explain(topic)
    points = step_key_points(explanation)
    return step_format(points)


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Document loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_text_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return [{"content": content, "source": path, "type": "text"}]


def load_web_page(url: str) -> list[dict]:
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()
        return [{"content": text, "source": url, "type": "web"}]
    except Exception as e:
        print(f"  ⚠️  Web load failed ({e})")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Text splitting
# RecursiveCharacterTextSplitter tries separators in order:
#   \n\n → \n → ". " → " " → ""
# This preserves semantic boundaries (paragraphs > sentences > words)
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for doc in documents:
        for i, chunk in enumerate(splitter.split_text(doc["content"])):
            if len(chunk.strip()) > 30:
                chunks.append({
                    "content": chunk,
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": -1,  # unknown until full split
                })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: DocumentQAChain — full RAG pipeline
# Equivalent to LangChain's:
#   chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | parser
# ─────────────────────────────────────────────────────────────────────────────

class DocumentQAChain:
    def __init__(self, collection_name: str = "sol_lcel"):
        client = chromadb.Client()
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = client.create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._count = 0

    def add_documents(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        texts = [c["content"] for c in chunks]
        embeddings = EMBEDDER.encode(texts, normalize_embeddings=True).tolist()
        ids = [f"c{self._count + i}" for i in range(len(texts))]
        metadatas = [{"source": c["source"]} for c in chunks]
        self.collection.upsert(documents=texts, embeddings=embeddings,
                               metadatas=metadatas, ids=ids)
        self._count += len(texts)
        return len(texts)

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        q_vec = EMBEDDER.encode(question, normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=[q_vec], n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        return [
            {"content": doc, "score": round(1 - dist, 4), "source": meta.get("source", "?")}
            for doc, dist, meta in zip(
                res["documents"][0], res["distances"][0], res["metadatas"][0]
            )
        ]

    def invoke(self, question: str, k: int = 3) -> dict:
        """Full chain: question → retrieve → build prompt → answer"""
        chunks = self.retrieve(question, k=k)
        context = "\n\n---\n\n".join(
            f"[Source {i+1} | score={c['score']}]\n{c['content']}"
            for i, c in enumerate(chunks)
        )
        answer = get_text(chat(
            messages=[{"role": "user", "content": (
                "Answer using ONLY the sources below. If not found, say so. "
                "Always cite [Source N].\n\n"
                f"Sources:\n{context}\n\nQuestion: {question}"
            )}],
            system="Precise assistant. Answer only from provided sources. Cite [Source N].",
        ))
        return {
            "answer": answer,
            "sources": [c["source"] for c in chunks],
            "scores": [c["score"] for c in chunks],
        }

    def batch(self, questions: list[str], k: int = 3) -> list[dict]:
        """Process multiple questions — equivalent to chain.batch([...])"""
        return [self.invoke(q, k=k) for q in questions]

    def stream(self, question: str, k: int = 3):
        """
        Yield tokens one by one — equivalent to chain.stream(question).
        Used for real-time UI updates (SSE, WebSockets).
        """
        import litellm
        chunks = self.retrieve(question, k=k)
        context = "\n\n".join(c["content"] for c in chunks)
        messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
        for chunk in litellm.completion(
            model=os.getenv("MODEL", "gemini/gemini-2.0-flash"),
            messages=messages, stream=True, max_tokens=500,
        ):
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_KB = """
LangChain is a framework for developing LLM applications using the LangChain Expression
Language (LCEL). LCEL chains steps using the pipe operator |: prompt | model | parser.
Each step receives the previous step's output. This makes pipelines composable and testable.

LangGraph extends LangChain with directed graph execution. Unlike linear chains, LangGraph
supports cycles and conditional routing. State is typed with TypedDict and persists across
nodes. MemorySaver and SqliteSaver enable checkpointing for fault-tolerant agents.

ChromaDB is an open-source vector database for embedding storage and similarity search.
It runs as an in-process library with zero configuration. PersistentClient saves the
index to disk. It supports metadata filtering with where={} conditions.

RAG (Retrieval-Augmented Generation) retrieves relevant documents from a knowledge base
and injects them into the LLM prompt. This grounds answers in factual sources and
reduces hallucination. Faithfulness, answer_relevancy, and context_precision are the
key RAGAS metrics used to evaluate RAG pipeline quality.

The Transformer architecture uses multi-head self-attention to model relationships between
all tokens simultaneously. BERT uses bidirectional attention for understanding; GPT uses
causal (left-to-right) attention for generation. Both are based on the 2017 paper
'Attention Is All You Need' by Vaswani et al. at Google Brain.
"""

if __name__ == "__main__":
    # Part 1: LCEL pipeline
    print("=" * 55)
    print("PART 1: LCEL-Style Pipeline")
    print("=" * 55)
    for topic in ["RAG (Retrieval-Augmented Generation)", "LangGraph StateGraph"]:
        print(f"\nTopic: {topic}")
        print(run_lcel_pipeline(topic))

    # Parts 2-3: Load and split
    print("\n" + "=" * 55)
    print("PART 2-3: Document Loading & Splitting")
    print("=" * 55)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_KB)
        tmp = f.name

    docs = load_text_file(tmp)
    chunks = split_documents(docs, chunk_size=400, chunk_overlap=60)
    print(f"Loaded {len(docs)} doc → {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {len(c['content'])} chars | {c['content'][:70]}…")

    # Part 4: RAG chain
    print("\n" + "=" * 55)
    print("PART 4: Retrieval Chain")
    print("=" * 55)

    chain = DocumentQAChain()
    chain.add_documents(chunks)

    questions = [
        "What is LCEL and how does it compose steps?",
        "How does LangGraph differ from a linear LangChain chain?",
        "Which database is recommended for RAG development?",
    ]

    # batch() — process all questions
    results = chain.batch(questions)
    for q, r in zip(questions, results):
        print(f"\nQ: {q}")
        print(f"A: {r['answer'][:250]}")
        print(f"   Scores: {r['scores']}")

    # stream() — real-time tokens
    print("\nSTREAMING DEMO:")
    print("Q: What is RAG and how does it reduce hallucination?")
    print("A: ", end="", flush=True)
    try:
        for token in chain.stream("What is RAG and how does it reduce hallucination?"):
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"(stream error: {e})")

    os.unlink(tmp)
