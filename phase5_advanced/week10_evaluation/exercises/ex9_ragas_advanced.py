"""
Exercise 9: RAGAS Advanced — Custom LLM Config, Chunking Comparison & Testset Generation
Goal: Go beyond the basic 4-metric eval to use RAGAS like a production team would.

Install: pip install ragas datasets

What's new vs ex3:
  1. Configure RAGAS to use YOUR LLM (LiteLLM) instead of OpenAI — so it works
     with any model you have (Gemini, Anthropic, etc.)
  2. Compare RAG quality across THREE chunking strategies (chunk_size 128/256/512).
     This answers: "What chunk size gives us the best trade-off between precision
     and recall?"
  3. Generate a synthetic evaluation testset automatically using RAGAS — no need
     to hand-craft QA pairs.
  4. Produce specific improvement recommendations from the metric scores.

Tasks:
  1. Complete configure_ragas_llm()      — wrap our llm.py in a RAGAS-compatible LLM class.
  2. Complete chunk_documents()          — split docs into chunks by size + overlap.
  3. Complete build_eval_data()          — run RAG pipeline and collect (q, a, contexts, gt).
  4. Complete compare_chunking_strategies() — run eval for 3 chunk sizes, collect all scores.
  5. Complete generate_recommendations() — parse scores and emit actionable recommendations.

Run:
  python ex9_ragas_advanced.py

Expected output:
  Comparing chunk strategies: 128 / 256 / 512 tokens
  ──────────────────────────────────────────────────────────────
  Metric              chunk=128  chunk=256  chunk=512
  faithfulness           0.82       0.88       0.91
  answer_relevancy       0.89       0.87       0.84
  context_precision      0.76       0.81       0.78
  context_recall         0.61       0.72       0.79
  ──────────────────────────────────────────────────────────────
  Best overall:  chunk_size=256

  Recommendations:
  ⚠ context_recall=0.72 < 0.80 — try increasing top_k from 3 to 5
  ✅ faithfulness=0.88 — good grounding
  ⚠ context_precision=0.81 — consider similarity threshold > 0.65
"""

import os, sys, math, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
from llm import achat, get_text, MODEL

load_dotenv()

# ── Knowledge base ─────────────────────────────────────────────────────────────

RAW_DOCUMENTS = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability and uses significant indentation.",
    "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It is dynamically typed and garbage-collected.",
    "pip is the standard package manager for Python. Packages are distributed via PyPI (Python Package Index). Virtual environments created with 'python -m venv' isolate project dependencies from the system Python.",
    "Python's standard library is often called 'batteries included'. It includes modules for file I/O (os, pathlib), networking (socket, http), data structures (collections), and more.",
    "List comprehensions in Python provide a concise way to create lists. For example: [x**2 for x in range(10)]. They are generally more Pythonic and faster than equivalent for-loops.",
    "Python decorators are functions that modify the behaviour of other functions. They are applied with the @decorator syntax. Common built-in decorators: @staticmethod, @classmethod, @property, @functools.lru_cache.",
    "async/await in Python enables asynchronous programming without threads. asyncio is the standard library for writing async code. The event loop runs coroutines concurrently without OS thread overhead.",
    "Python type hints (PEP 484) allow optional static type annotations. Tools like mypy and Pylance perform static analysis. Example: def greet(name: str) -> str: return f'Hello {name}'.",
    "Python's context managers (the 'with' statement) ensure resources are properly released. Implement __enter__ and __exit__ to create custom context managers, or use @contextlib.contextmanager.",
    "Python dataclasses (PEP 557) reduce boilerplate for data-holding classes. @dataclass auto-generates __init__, __repr__, __eq__. Use field(default_factory=list) for mutable defaults.",
]

QA_PAIRS = [
    {"question": "Who created Python and when was it released?",
     "ground_truth": "Python was created by Guido van Rossum and first released in 1991."},
    {"question": "What is pip and what is PyPI?",
     "ground_truth": "pip is Python's standard package manager. PyPI (Python Package Index) is where packages are distributed."},
    {"question": "How do list comprehensions work in Python?",
     "ground_truth": "List comprehensions create lists concisely: [x**2 for x in range(10)]. They are faster and more Pythonic than for-loops."},
    {"question": "What are Python decorators?",
     "ground_truth": "Decorators are functions that modify other functions, applied with @decorator syntax. Examples: @staticmethod, @classmethod."},
    {"question": "What is asyncio used for?",
     "ground_truth": "asyncio is Python's standard library for async programming using async/await, running coroutines concurrently via an event loop."},
    {"question": "What are Python type hints?",
     "ground_truth": "Type hints (PEP 484) are optional annotations checked by mypy/Pylance. Example: def greet(name: str) -> str."},
]

CHUNKING_CONFIGS = [
    {"chunk_size": 128, "overlap": 20},
    {"chunk_size": 256, "overlap": 40},
    {"chunk_size": 512, "overlap": 80},
]

# Metric thresholds for recommendations
THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.70,
}

FIXES = {
    "faithfulness":      "tighten system prompt to 'answer ONLY from context'",
    "answer_relevancy":  "improve answer generation prompt to stay on-topic",
    "context_precision": "raise similarity threshold or reduce top_k",
    "context_recall":    "increase top_k or try smaller chunk size",
}


# ── Shared embedding + retrieval (word-frequency based — works without sentence-transformers) ──

def _tf_embed(text: str, vocab: dict[str, int]) -> list[float]:
    vec = [0.0] * len(vocab)
    for w in text.lower().split():
        if w in vocab:
            vec[vocab[w]] += 1.0
    return vec

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb  = math.sqrt(sum(x * x for x in b)) + 1e-9
    return dot / (na * nb)


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Complete chunk_documents()
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Split each document into word-based chunks of `chunk_size` words,
    with `overlap` words of overlap between consecutive chunks.

    Algorithm:
    1. For each doc, split into words.
    2. Step through with stride = chunk_size - overlap.
    3. Each chunk = words[i : i + chunk_size] joined back to a string.
    4. Collect all chunks across all docs.

    TODO: implement the sliding-window chunking.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Complete configure_ragas_llm()
# ─────────────────────────────────────────────────────────────────────────────

def configure_ragas_llm():
    """
    Return a RAGAS-compatible LLM wrapper that uses our llm.py / LiteLLM
    instead of OpenAI, so RAGAS works with any model (Gemini, Anthropic, etc).

    RAGAS uses langchain_core BaseLanguageModel internally.
    The easiest way: return a LangChain ChatLiteLLM wrapper.

    TODO:
    1. Try: from langchain_community.chat_models import ChatLiteLLM
            return ChatLiteLLM(model=MODEL)
    2. If ImportError → return None (RAGAS will use its default).

    The returned object is passed as `llm=` to ragas evaluate().
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Complete build_eval_data()
# ─────────────────────────────────────────────────────────────────────────────

async def build_eval_data(chunks: list[str], qa_pairs: list[dict], top_k: int = 3) -> dict:
    """
    Run the RAG pipeline for each QA pair and collect RAGAS-format data.

    Steps per QA pair:
    1. Build vocab from all chunks (word → index mapping).
    2. Embed all chunks with _tf_embed.
    3. Embed the question with _tf_embed.
    4. Retrieve top_k chunks by cosine similarity.
    5. Build RAG prompt: "Context:\n{chunks}\n\nQuestion: {question}"
    6. Call achat() to get the answer (max_tokens=200).
    7. Append to questions, answers, contexts, ground_truths lists.

    Return dict with keys: "question", "answer", "contexts", "ground_truth".
    "contexts" must be list[list[str]] (list of retrieved chunk lists per question).

    TODO: implement the loop using asyncio.gather for the LLM calls.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 4: Complete compare_chunking_strategies()
# ─────────────────────────────────────────────────────────────────────────────

async def compare_chunking_strategies() -> list[dict]:
    """
    For each config in CHUNKING_CONFIGS:
    1. Chunk RAW_DOCUMENTS with chunk_documents().
    2. Build eval data with build_eval_data().
    3. Run RAGAS (try real library; fall back to simulated scores).
    4. Collect result dict: {"chunk_size": N, "scores": {...}}.
    Return list of result dicts.

    For the real RAGAS call:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        ragas_llm = configure_ragas_llm()
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        if ragas_llm:
            for m in metrics: m.llm = ragas_llm
        ds = Dataset.from_dict(data)
        result = evaluate(ds, metrics=metrics)
        scores = {k: round(float(v), 4) for k, v in result.items()}
    except ImportError:
        # Simulate plausible scores that vary by chunk size
        scores = _simulate_scores(chunk_size)

    TODO: implement the loop.
    """
    raise NotImplementedError


def _simulate_scores(chunk_size: int) -> dict:
    """Plausible simulated scores that vary realistically with chunk size."""
    base = {
        "faithfulness":      0.78 + (chunk_size / 4096),
        "answer_relevancy":  0.91 - (chunk_size / 5000),
        "context_precision": 0.72 + (chunk_size / 6000),
        "context_recall":    0.55 + (chunk_size / 2000),
    }
    return {k: round(min(v, 0.99), 3) for k, v in base.items()}


# ─────────────────────────────────────────────────────────────────────────────
# TODO 5: Complete generate_recommendations()
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(best_result: dict) -> None:
    """
    Given the best_result dict {"chunk_size": N, "scores": {...}},
    print one recommendation line per metric:
      - ✅ if score >= threshold
      - ⚠ if score < threshold, with a specific fix from FIXES dict

    TODO: loop over THRESHOLDS, compare, print.
    """
    raise NotImplementedError


# ── Print comparison table ─────────────────────────────────────────────────────

def print_comparison_table(results: list[dict]):
    metrics = list(THRESHOLDS.keys())
    sizes   = [r["chunk_size"] for r in results]

    print(f"\n{'Metric':<24}", end="")
    for s in sizes:
        print(f"  chunk={s:<5}", end="")
    print()
    print("─" * (24 + len(sizes) * 14))

    for metric in metrics:
        print(f"  {metric:<22}", end="")
        for r in results:
            score = r["scores"].get(metric, 0)
            flag  = " ✅" if score >= THRESHOLDS[metric] else " ⚠ "
            print(f"  {score:.3f}{flag}", end="")
        print()


def pick_best(results: list[dict]) -> dict:
    """Pick config with highest average score across all RAGAS metrics."""
    def avg(r):
        return sum(r["scores"].get(m, 0) for m in THRESHOLDS) / len(THRESHOLDS)
    return max(results, key=avg)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("RAGAS ADVANCED — Chunking Strategy Comparison")
    print("=" * 60)
    sizes = [c["chunk_size"] for c in CHUNKING_CONFIGS]
    print(f"Comparing chunk sizes: {' / '.join(str(s) for s in sizes)}\n")

    results = await compare_chunking_strategies()

    print_comparison_table(results)

    best = pick_best(results)
    print(f"\nBest overall: chunk_size={best['chunk_size']} "
          f"(avg score: {sum(best['scores'].values())/len(best['scores']):.3f})")

    print("\nRecommendations for best config:")
    generate_recommendations(best)


if __name__ == "__main__":
    asyncio.run(main())
