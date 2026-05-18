"""
Exercise 2: Compare 3 Chunking Strategies + Hybrid Search
Goal: Measure retrieval quality across fixed, sentence, and semantic chunking.
Also add BM25 keyword search alongside vector search.

Tasks:
  1. Implement three chunking functions:
     a. fixed_chunks(text, size=500, overlap=50)
     b. sentence_chunks(text)  — split on ". "
     c. semantic_chunks(text, embedder)  — group sentences by cosine similarity threshold
  2. For each strategy, build a Chroma collection and run 5 test questions.
  3. Implement BM25 search using `rank_bm25`.
  4. Implement hybrid_search = 0.7 * vector_score + 0.3 * bm25_score, re-rank.
  5. Print a comparison table.

pip install chromadb sentence-transformers rank-bm25 anthropic
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from llm import chat, get_text

embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
    # TODO: Sliding window over characters
    raise NotImplementedError


def sentence_chunks(text: str) -> list[str]:
    # TODO: Split text into sentence-level chunks
    raise NotImplementedError


def semantic_chunks(text: str, threshold: float = 0.7) -> list[str]:
    # TODO: Embed each sentence, merge adjacent ones where cosine_sim > threshold
    raise NotImplementedError


def bm25_search(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    # TODO: Keyword search using BM25. Return top_k (chunk, score) pairs.
    raise NotImplementedError


def vector_search(query: str, collection, top_k: int = 3) -> list[tuple[str, float]]:
    # TODO: Embed the query and return top_k (chunk, distance) pairs from Chroma.
    raise NotImplementedError


def hybrid_search(query: str, corpus: list[str], collection, top_k: int = 3) -> list[str]:
    # TODO: Combine bm25 + vector scores with 0.3/0.7 weights
    raise NotImplementedError


if __name__ == "__main__":
    questions = [
        "What is transfer learning?",
        "How do vector databases work?",
        "What is reinforcement learning?",
    ]
    strategies = {
        "fixed": fixed_chunks(SAMPLE_TEXT),
        "sentence": sentence_chunks(SAMPLE_TEXT),
    }
    for name, chunks in strategies.items():
        print(f"\n=== {name} chunking: {len(chunks)} chunks ===")
        print("First chunk preview:", repr(chunks[0][:80]))
