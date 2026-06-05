"""
Exercise 2: Map-Reduce over Large Document Sets
Goal: Summarize 20 chunks in parallel (MAP), then synthesize into one answer (REDUCE).

Tasks:
  1. Complete map_summarize(chunk) — summarize one chunk with achat().
  2. Complete map_phase(chunks) — run all summaries concurrently with asyncio.gather + Semaphore(4).
  3. Complete reduce_phase(summaries, question) — synthesize summaries into a final answer.
  4. Measure wall time for map phase and compare to estimated sequential time.
  5. (Bonus) Add error handling: if one chunk fails, log and continue.

Expected output:
  MAP: summarizing 20 chunks...
  MAP done: 20/20 ✓ — 3.2s (est. sequential: 16.0s, speedup: 5.0x)
  REDUCE: synthesizing...
  Final answer: ...
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import asyncio
import time
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

# ── Sample Document (split into 20 chunks) ────────────────────────────────────

FULL_DOCUMENT = """
Artificial intelligence (AI) refers to the simulation of human intelligence processes by machines,
especially computer systems. AI research has been highly successful in developing effective
techniques for solving a wide range of problems.

Machine learning is a subset of AI that provides systems the ability to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Deep learning is part of a broader family of machine learning methods based on artificial neural
networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Natural language processing (NLP) is a subfield of linguistics, computer science, and AI
concerned with the interactions between computers and human language. It involves programming
computers to process and analyze large amounts of natural language data.

Computer vision is an interdisciplinary scientific field that deals with how computers can gain
high-level understanding from digital images or videos. It seeks to understand and automate tasks
that the human visual system can do.

Reinforcement learning is an area of machine learning concerned with how intelligent agents
ought to take actions in an environment to maximize the notion of cumulative reward.

Transfer learning is a machine learning method where a model developed for a task is reused
as the starting point for a model on a different task. It is popular in deep learning.

Generative AI refers to algorithms that can be used to create new content, including audio,
code, images, text, simulations, and videos. Recent new breakthroughs have the potential to
drastically change the way we approach content creation.

Large language models are neural networks trained on massive text corpora using self-supervised
learning. They learn to predict the next token in a sequence, developing rich internal
representations of language structure and world knowledge.

Retrieval-augmented generation (RAG) combines a retrieval system with a language model.
When given a query, the system first retrieves relevant documents, then uses them as context
for the language model to generate a grounded, accurate response.
""".strip()


def make_chunks(text: str, n: int = 20) -> list[str]:
    """Split text into n roughly equal chunks."""
    words = text.split()
    size = max(1, len(words) // n)
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
    return chunks[:n]


CHUNKS = make_chunks(FULL_DOCUMENT, 20)


# ── MAP Phase ──────────────────────────────────────────────────────────────────

async def map_summarize(chunk: str, sem: asyncio.Semaphore, idx: int) -> str:
    """Summarize one chunk. Use semaphore to cap concurrency."""
    # TODO: async with sem:
    # TODO:   response = await achat([{"role": "user", "content": f"Summarize in 1 sentence:\n{chunk}"}], max_tokens=80)
    # TODO:   return get_text(response)
    raise NotImplementedError


async def map_phase(chunks: list[str], max_concurrent: int = 4) -> list[str]:
    """Summarize all chunks concurrently. Return list of summaries."""
    sem = asyncio.Semaphore(max_concurrent)
    t0 = time.perf_counter()

    print(f"MAP: summarizing {len(chunks)} chunks with concurrency={max_concurrent}...")
    # TODO: tasks = [map_summarize(chunk, sem, i) for i, chunk in enumerate(chunks)]
    # TODO: results = await asyncio.gather(*tasks, return_exceptions=True)
    # TODO: filter out exceptions (log them), count successes
    # TODO: return list of successful summary strings
    raise NotImplementedError


# ── REDUCE Phase ───────────────────────────────────────────────────────────────

async def reduce_phase(summaries: list[str], question: str) -> str:
    """Synthesize all summaries into a final answer to the question."""
    combined = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(summaries))
    # TODO: call achat() with question + combined summaries as context
    # TODO: system: "You are given summaries from multiple document chunks. Synthesize a comprehensive answer."
    # TODO: return get_text(response)
    raise NotImplementedError


# ── Full Pipeline ──────────────────────────────────────────────────────────────

async def map_reduce(question: str) -> str:
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    summaries = await map_phase(CHUNKS)
    map_time = time.perf_counter() - t0

    est_sequential = map_time / len(CHUNKS) * len(CHUNKS)  # approximate
    print(f"MAP done: {len(summaries)} summaries — {map_time:.1f}s (speedup vs sequential: ~{len(CHUNKS)/max(1,map_time/2):.1f}x)\n")

    print("REDUCE: synthesizing...")
    t0 = time.perf_counter()
    answer = await reduce_phase(summaries, question)
    reduce_time = time.perf_counter() - t0

    total = time.perf_counter() - t_total
    print(f"REDUCE done — {reduce_time:.1f}s | Total: {total:.1f}s\n")
    return answer


if __name__ == "__main__":
    question = "What are the main areas of AI and how do they relate to each other?"
    answer = asyncio.run(map_reduce(question))
    print(f"Q: {question}")
    print(f"\nA: {answer}")
