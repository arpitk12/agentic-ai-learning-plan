"""
Exercise 3: RAGAS — Evaluate RAG Pipeline Quality
Goal: Measure faithfulness, answer relevancy, context precision and recall.

Install: pip install ragas datasets

RAGAS metrics (all 0–1, higher is better):
  - faithfulness:        Is the answer grounded in the retrieved context?
  - answer_relevancy:    Does the answer actually address the question?
  - context_precision:   Are retrieved chunks relevant to the question?
  - context_recall:      Did we retrieve all information needed?

Tasks:
  1. Build a tiny RAG pipeline (reuse ex1_rag_basic.py patterns).
  2. Complete build_ragas_dataset() — create the HuggingFace Dataset RAGAS needs.
  3. Complete run_ragas_eval() — run RAGAS metrics, print scores.
  4. Identify which metric is lowest and explain why.
  5. (Bonus) Try two different chunk sizes and compare RAGAS scores.

Expected output:
  faithfulness:      0.85
  answer_relevancy:  0.91
  context_precision: 0.72
  context_recall:    0.68
  → Weakest: context_recall — try smaller chunks or more top-k results
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import math
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

# ── Sample Knowledge Base ──────────────────────────────────────────────────────

DOCUMENTS = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. Python emphasizes code readability and uses significant whitespace.",
    "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It is dynamically typed and garbage-collected.",
    "pip is the standard package manager for Python. Packages are distributed via PyPI (Python Package Index). Virtual environments isolate project dependencies.",
    "Python's standard library includes modules for file I/O, networking, data structures, and more. The 'os' module provides operating system interfaces.",
    "List comprehensions in Python provide a concise way to create lists: [x**2 for x in range(10)]. They are more Pythonic than equivalent for-loops.",
    "Python decorators are functions that modify other functions. They are applied with the @decorator syntax. Common decorators: @staticmethod, @classmethod, @property.",
    "async/await in Python enables asynchronous programming. asyncio is the standard library for writing async code. Use 'await' to pause coroutines.",
    "Python type hints (PEP 484) allow static type checking. Tools like mypy and Pylance check types. Example: def greet(name: str) -> str: return f'Hello {name}'",
]

QA_PAIRS = [
    {
        "question": "Who created Python?",
        "ground_truth": "Python was created by Guido van Rossum.",
    },
    {
        "question": "What is pip used for?",
        "ground_truth": "pip is the standard package manager for Python used to install packages from PyPI.",
    },
    {
        "question": "How do you write a list comprehension in Python?",
        "ground_truth": "List comprehensions use the syntax [expression for item in iterable], for example [x**2 for x in range(10)].",
    },
    {
        "question": "What are Python decorators?",
        "ground_truth": "Python decorators are functions that modify other functions, applied using the @decorator syntax.",
    },
    {
        "question": "What is asyncio used for?",
        "ground_truth": "asyncio is Python's standard library for writing asynchronous code using async/await syntax.",
    },
]


# ── Simple RAG Pipeline ────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-8)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Simple TF-based embedding (word frequency). Replace with sentence-transformers for quality."""
    vocab: dict[str, int] = {}
    for t in texts:
        for w in t.lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
    vectors = []
    for t in texts:
        vec = [0.0] * len(vocab)
        for w in t.lower().split():
            if w in vocab:
                vec[vocab[w]] += 1.0
        vectors.append(vec)
    return vectors


# Pre-embed documents
DOC_EMBEDDINGS = embed_texts(DOCUMENTS)


def retrieve(question: str, top_k: int = 3) -> list[str]:
    q_emb = embed_texts([question])[0]
    scores = [cosine_sim(q_emb, d) for d in DOC_EMBEDDINGS]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [DOCUMENTS[i] for i in top_idx]


def rag_answer(question: str, contexts: list[str]) -> str:
    context = "\n\n".join(contexts)
    r = chat(
        [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        system="Answer concisely using only the provided context.",
        max_tokens=200,
    )
    return get_text(r)


# ── RAGAS Evaluation ───────────────────────────────────────────────────────────

def build_ragas_dataset(qa_pairs: list[dict], top_k: int = 3) -> dict:
    """
    Build the dict needed by RAGAS:
    {
      "question": [...],
      "answer": [...],
      "contexts": [[...], ...],   # list of lists (retrieved chunks per question)
      "ground_truth": [...]
    }
    TODO:
      For each qa_pair:
        1. contexts = retrieve(qa_pair["question"], top_k)
        2. answer = rag_answer(qa_pair["question"], contexts)
        3. Append to respective lists
    """
    questions, answers, all_contexts, ground_truths = [], [], [], []
    # TODO: implement the loop
    raise NotImplementedError
    return {
        "question": questions,
        "answer": answers,
        "contexts": all_contexts,
        "ground_truth": ground_truths,
    }


def run_ragas_eval(top_k: int = 3) -> dict:
    """Run RAGAS evaluation. Return metric scores."""
    print(f"Building RAG dataset (top_k={top_k})...")
    data = build_ragas_dataset(QA_PAIRS, top_k=top_k)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

        dataset = Dataset.from_dict(data)
        print("Running RAGAS evaluation...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = {k: round(float(v), 4) for k, v in result.items()}
    except ImportError:
        print("[RAGAS not installed] Run: pip install ragas datasets")
        print("Showing simulated results for structure demonstration:")
        scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.91,
            "context_precision": 0.72,
            "context_recall": 0.68,
        }

    print(f"\nRAGAS Results (top_k={top_k}):")
    for metric, score in sorted(scores.items(), key=lambda x: x[1]):
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.2f} {bar}")

    weakest = min(scores, key=scores.get)
    print(f"\n→ Weakest metric: {weakest} ({scores[weakest]:.2f})")
    return scores


if __name__ == "__main__":
    # Compare two configurations
    print("=== Evaluation with top_k=2 ===")
    scores_2 = run_ragas_eval(top_k=2)

    print("\n=== Evaluation with top_k=4 ===")
    scores_4 = run_ragas_eval(top_k=4)

    print("\n=== Comparison ===")
    for metric in scores_2:
        diff = scores_4.get(metric, 0) - scores_2.get(metric, 0)
        print(f"  {metric:<22} top_k=2: {scores_2[metric]:.2f}  top_k=4: {scores_4.get(metric, 0):.2f}  Δ{diff:+.2f}")
