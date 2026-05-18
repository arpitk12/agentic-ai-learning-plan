"""
SOLUTION — Exercise 2: Chunking Strategies + Hybrid Search
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Anthropic()

SAMPLE_TEXT = """
Artificial intelligence has transformed many industries. Machine learning, a subset of AI,
enables computers to learn from data. Deep learning uses neural networks with many layers.
Natural language processing allows machines to understand human language.
Computer vision enables machines to interpret images and video.
Reinforcement learning trains agents through trial and error. Transfer learning applies
knowledge from one domain to another. Generative AI creates new content such as text and images.
Large language models are trained on vast amounts of text data. Vector databases store
high-dimensional embeddings for similarity search. Retrieval augmented generation combines
search with language model generation. Agentic AI systems can plan and execute multi-step tasks.
""".strip()


def fixed_chunks(text: str, size: int = 200, overlap: int = 30) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 10]


def sentence_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def semantic_chunks(text: str, threshold: float = 0.6) -> list[str]:
    sentences = sentence_chunks(text)
    if not sentences:
        return []
    embeddings = embedder.encode(sentences)
    chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]) /
                    (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8))
        if sim >= threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]
    if current:
        chunks.append(" ".join(current))
    return chunks


def bm25_search(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i], float(scores[i])) for i in top_indices]


def build_chroma_collection(chunks: list[str], name: str) -> chromadb.Collection:
    c = chromadb.Client()
    collection = c.get_or_create_collection(name)
    embs = embedder.encode(chunks).tolist()
    collection.add(embeddings=embs, documents=chunks, ids=[f"{name}_{i}" for i in range(len(chunks))])
    return collection


def vector_search(query: str, collection, top_k: int = 3) -> list[tuple[str, float]]:
    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results["documents"][0]
    dists = results["distances"][0]
    return list(zip(docs, dists))


def hybrid_search(query: str, corpus: list[str], collection, top_k: int = 3) -> list[str]:
    bm25_results = dict(bm25_search(query, corpus, top_k=len(corpus)))
    vec_results = dict(vector_search(query, collection, top_k=len(corpus)))

    # Normalize scores
    def normalize(d: dict) -> dict:
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn or 1
        return {k: (v - mn) / rng for k, v in d.items()}

    bm25_norm = normalize(bm25_results)
    # Vector distances: lower is better, invert
    vec_norm = normalize({k: -v for k, v in vec_results.items()})

    combined: dict[str, float] = {}
    for doc in set(list(bm25_norm) + list(vec_norm)):
        combined[doc] = 0.3 * bm25_norm.get(doc, 0) + 0.7 * vec_norm.get(doc, 0)

    ranked = sorted(combined, key=combined.__getitem__, reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    questions = ["What is transfer learning?", "How do vector databases work?"]
    strategies = {
        "fixed": fixed_chunks(SAMPLE_TEXT),
        "sentence": sentence_chunks(SAMPLE_TEXT),
        "semantic": semantic_chunks(SAMPLE_TEXT),
    }

    for strategy_name, chunks in strategies.items():
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy_name} — {len(chunks)} chunks")
        for i, q in enumerate(questions):
            bm25_top = bm25_search(q, chunks, top_k=1)
            print(f"  Q{i+1}: {q}")
            if bm25_top:
                print(f"    BM25 top: {bm25_top[0][0][:60]}...")
