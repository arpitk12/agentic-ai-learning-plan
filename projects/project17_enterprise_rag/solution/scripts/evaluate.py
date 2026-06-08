"""
Hallucination evaluation harness — runs golden questions and reports faithfulness.

Golden dataset: 5 questions from the sample_docs corpus.
Pass criterion:  overall faithfulness ≥ 0.85 across all non-abstained answers.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --threshold 0.90
"""
from __future__ import annotations

import json
import sys
import argparse
import logging

logging.basicConfig(level=logging.WARNING)

GOLDEN_QA = [
    {
        "question": "What are the API rate limits?",
        "expected_keywords": ["rate limit", "requests", "per minute"],
        "must_abstain": False,
    },
    {
        "question": "How do I authenticate with the API?",
        "expected_keywords": ["api key", "auth", "header", "bearer"],
        "must_abstain": False,
    },
    {
        "question": "What is the maximum payload size?",
        "expected_keywords": ["payload", "size", "MB", "limit"],
        "must_abstain": False,
    },
    {
        "question": "How do I handle webhook retries?",
        "expected_keywords": ["webhook", "retry", "backoff"],
        "must_abstain": False,
    },
    {
        "question": "Who invented quantum physics?",   # out-of-corpus — must abstain
        "expected_keywords": [],
        "must_abstain": True,
    },
]


def main(threshold: float = 0.85) -> None:
    from src.agents.rag_agent import RAGAgent
    from src.cache.redis_cache import RedisCache
    from src.cache.semantic_cache import SemanticCache
    from src.config import cfg
    from src.hallucination.faithfulness_checker import FaithfulnessChecker
    from src.ingestion.embedder import Embedder
    from src.models import QueryRequest
    from src.retrieval.retriever import HybridRetriever
    from src.store.qdrant_store import QdrantStore
    from sentence_transformers import CrossEncoder

    store = QdrantStore()
    embedder = Embedder()
    reranker = CrossEncoder(cfg.reranker_model)
    checker = FaithfulnessChecker()
    retriever = HybridRetriever(qdrant_store=store, embedder=embedder)
    agent = RAGAgent(
        retriever=retriever,
        embedder=embedder,
        faithfulness_checker=checker,
        reranker_model=reranker,
        redis_cache=RedisCache(),
        semantic_cache=SemanticCache(),
    )

    results = []
    for qa in GOLDEN_QA:
        resp = agent.answer(QueryRequest(question=qa["question"]))

        passed = True
        notes = []

        if qa["must_abstain"] and not resp.abstained:
            passed = False
            notes.append("EXPECTED abstain but got an answer")
        if not qa["must_abstain"] and resp.abstained:
            passed = False
            notes.append(f"Unexpectedly abstained: {resp.abstain_reason}")
        if not resp.abstained and resp.faithfulness_score < threshold:
            passed = False
            notes.append(f"Faithfulness {resp.faithfulness_score:.2f} < {threshold}")

        icon = "✓" if passed else "✗"
        print(
            f"  {icon} Q: {qa['question'][:60]}\n"
            f"     abstained={resp.abstained}  faithfulness={resp.faithfulness_score:.2f}"
            + (f"  ← {notes[0]}" if notes else "")
        )
        results.append({"passed": passed, "faithfulness": resp.faithfulness_score, "abstained": resp.abstained})

    non_abstained = [r for r in results if not r["abstained"]]
    avg_faith = sum(r["faithfulness"] for r in non_abstained) / len(non_abstained) if non_abstained else 0.0
    all_passed = all(r["passed"] for r in results)

    print(f"\n  Overall faithfulness: {avg_faith:.2f}  (threshold: {threshold})")
    print(f"  Abstain rate: {sum(1 for r in results if r['abstained'])}/{len(results)}")
    print(f"  Result: {'PASS ✓' if all_passed else 'FAIL ✗'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    main(threshold=args.threshold)
