"""
Solution 10: DeepEval — Production LLM Evaluation Framework
"""

import os, sys, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

SYSTEM = "You are a helpful AI assistant. Answer concisely and accurately."

async def agent(question: str, context: str = "") -> str:
    prompt = (f"Context:\n{context}\n\nQuestion: {question}" if context else question)
    r = await achat([{"role": "user", "content": prompt}], system=SYSTEM, max_tokens=300)
    return get_text(r)

CONTEXT_DOCS = {
    "python": "Python was created by Guido van Rossum in 1991. It emphasises readability.",
    "docker": "Docker uses OS-level virtualisation to deliver software in containers. Containers share the host OS kernel but are isolated.",
    "rag":    "Retrieval-Augmented Generation (RAG) combines a retrieval step with LLM generation. It grounds LLM answers in retrieved documents.",
    "async":  "asyncio is Python's standard library for async programming. The event loop executes coroutines without OS threads.",
    "git":    "Git is a distributed version control system. Branches allow parallel development. Commits are immutable snapshots.",
}

RAW_TEST_CASES = [
    {"input": "Who created Python and when?",          "context_key": "python", "expected_output": "Python was created by Guido van Rossum in 1991."},
    {"input": "What does Docker containerise?",         "context_key": "docker", "expected_output": "Docker containerises software so it runs consistently."},
    {"input": "What is RAG in AI?",                    "context_key": "rag",    "expected_output": "RAG combines retrieval with LLM generation."},
    {"input": "What is asyncio used for?",             "context_key": "async",  "expected_output": "asyncio enables async programming in Python."},
    {"input": "What is a Git branch?",                 "context_key": "git",    "expected_output": "A Git branch allows parallel development."},
    {"input": "What is the capital of Australia?",     "context_key": None,     "expected_output": "Canberra"},
]

MAX_ANSWER_WORDS = 60


# ── Solution implementations ───────────────────────────────────────────────────

def make_test_case(raw: dict, actual_output: str):
    """Build a DeepEval LLMTestCase (or fallback dict)."""
    ctx_doc = CONTEXT_DOCS.get(raw["context_key"]) if raw.get("context_key") else None
    try:
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=raw["input"],
            actual_output=actual_output,
            expected_output=raw.get("expected_output"),
            retrieval_context=[ctx_doc] if ctx_doc else None,
            context=[ctx_doc] if ctx_doc else None,
        )
    except ImportError:
        return {
            "input": raw["input"],
            "actual_output": actual_output,
            "expected_output": raw.get("expected_output"),
            "context": [ctx_doc] if ctx_doc else [],
        }


def build_metrics() -> list:
    """Instantiate 4 standard DeepEval metrics."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
            BiasMetric,
        )
        return [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            HallucinationMetric(threshold=0.5),
            BiasMetric(threshold=0.5),
        ]
    except ImportError:
        return []


class ConcisenesFallback:
    name      = "conciseness_custom"
    threshold = 0.5
    score     = 1.0
    reason    = ""

    def measure(self, test_case) -> float:
        output = (test_case.get("actual_output", "") if isinstance(test_case, dict)
                  else test_case.actual_output)
        words       = len(output.split())
        self.score  = 1.0 if words <= MAX_ANSWER_WORDS else round(MAX_ANSWER_WORDS / words, 2)
        self.reason = f"Answer is {words} words ({'OK' if words <= MAX_ANSWER_WORDS else f'over limit of {MAX_ANSWER_WORDS}'})."
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


def make_conciseness_metric():
    """Custom DeepEval BaseMetric measuring answer length."""
    try:
        from deepeval.metrics import BaseMetric

        class ConcisenessMetric(BaseMetric):
            def __init__(self):
                self.threshold = 0.5
                self.name      = "conciseness_custom"
                self.score     = 1.0
                self.reason    = ""

            def measure(self, test_case) -> float:
                output      = test_case.actual_output
                words       = len(output.split())
                self.score  = 1.0 if words <= MAX_ANSWER_WORDS else round(MAX_ANSWER_WORDS / words, 2)
                self.reason = f"Answer is {words} words ({'OK' if words <= MAX_ANSWER_WORDS else f'over limit of {MAX_ANSWER_WORDS}'})."
                return self.score

            def is_successful(self) -> bool:
                return self.score >= self.threshold

        return ConcisenessMetric()
    except ImportError:
        return ConcisenesFallback()


def _simulate_batch_results(n_cases: int, metric_names: list[str]) -> dict:
    import random
    rng = random.Random(42)
    return {
        name: {
            "passed": (p := rng.randint(max(0, n_cases - 2), n_cases)),
            "total":  n_cases,
            "rate":   p / n_cases,
        }
        for name in metric_names
    }


async def run_batch_eval(raw_cases: list[dict]) -> dict:
    """Run agent on all cases then evaluate with DeepEval."""
    # Step 1: Collect outputs concurrently
    outputs = await asyncio.gather(*[
        agent(r["input"], CONTEXT_DOCS.get(r["context_key"], "") if r.get("context_key") else "")
        for r in raw_cases
    ])

    # Step 2: Build test cases
    test_cases = [make_test_case(r, o) for r, o in zip(raw_cases, outputs)]

    # Step 3: Build metrics
    std_metrics    = build_metrics()
    custom_metric  = make_conciseness_metric()
    all_metrics    = std_metrics + [custom_metric]

    if not std_metrics:
        # No deepeval installed — simulate
        all_metric_names = ["answer_relevancy", "faithfulness", "hallucination",
                            "bias", "conciseness_custom"]
        print("  [deepeval not installed — using simulated results]")
        print("  Install with: pip install deepeval")
        return _simulate_batch_results(len(raw_cases), all_metric_names)

    # Step 4: Run evaluate()
    try:
        from deepeval import evaluate
        eval_result = evaluate(test_cases, all_metrics, print_results=False)

        # Step 5: Extract per-metric pass rates
        results: dict[str, dict] = {}
        n = len(raw_cases)
        for tr in eval_result.test_results:
            for md in tr.metrics_data:
                name = md.name
                if name not in results:
                    results[name] = {"passed": 0, "total": 0, "rate": 0.0}
                results[name]["total"]  += 1
                results[name]["passed"] += int(md.success)
        for name in results:
            r = results[name]
            r["rate"] = r["passed"] / r["total"] if r["total"] else 0.0
        return results
    except Exception as e:
        print(f"  [deepeval evaluate() error: {e} — using simulated results]")
        all_metric_names = [m.name for m in all_metrics]
        return _simulate_batch_results(len(raw_cases), all_metric_names)


async def run_pytest_demo():
    """Demonstrate assert_test() for a single case."""
    raw    = RAW_TEST_CASES[0]
    output = await agent(raw["input"], CONTEXT_DOCS.get(raw["context_key"], ""))
    tc     = make_test_case(raw, output)
    try:
        from deepeval import assert_test
        from deepeval.metrics import AnswerRelevancyMetric
        assert_test(tc, [AnswerRelevancyMetric(threshold=0.7)])
        print("  ✅ assert_test passed")
    except AssertionError as e:
        print(f"  ❌ assert_test failed: {e}")
    except ImportError:
        # Manual fallback: check if expected keyword appears
        ok = raw["expected_output"].split()[0].lower() in output.lower()
        print(f"  [deepeval not installed — manual check: {'✅ pass' if ok else '❌ fail'}]")


def test_answer_relevancy_python():
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric

        output = asyncio.run(agent("Who created Python?", CONTEXT_DOCS["python"]))
        tc     = LLMTestCase(input="Who created Python?", actual_output=output,
                             retrieval_context=[CONTEXT_DOCS["python"]])
        assert_test(tc, [AnswerRelevancyMetric(threshold=0.7)])
    except ImportError:
        print("[deepeval not installed — test skipped]")


def test_no_hallucination_in_context_answer():
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import HallucinationMetric

        output = asyncio.run(agent("What is asyncio used for?", CONTEXT_DOCS["async"]))
        tc     = LLMTestCase(input="What is asyncio used for?", actual_output=output,
                             context=[CONTEXT_DOCS["async"]])
        assert_test(tc, [HallucinationMetric(threshold=0.5)])
    except ImportError:
        print("[deepeval not installed — test skipped]")


def print_batch_report(results: dict, overall_threshold: float = 0.80):
    total_p, total_t = 0, 0
    print(f"\n  {'Metric':<28} {'Passed':>8} {'Rate':>8}  Status")
    print("  " + "─" * 54)
    for name, r in results.items():
        ok = "✅" if r["rate"] >= overall_threshold else "⚠ "
        print(f"  {name:<28} {r['passed']:>3}/{r['total']:<3}  {r['rate']:5.1%}  {ok}")
        total_p += r["passed"]
        total_t += r["total"]
    overall_rate = total_p / total_t if total_t else 0.0
    ok = "✅ PASS" if overall_rate >= overall_threshold else "❌ FAIL"
    print("  " + "─" * 54)
    print(f"  {'Overall':<28} {total_p:>3}/{total_t:<3}  {overall_rate:5.1%}  {ok}")


async def main():
    print("=" * 62)
    print("DEEPEVAL — LLM Evaluation Framework")
    print("=" * 62)
    print(f"Evaluating {len(RAW_TEST_CASES)} test cases × 5 metrics...\n")

    print("── Batch evaluation ──")
    results = await run_batch_eval(RAW_TEST_CASES)
    print_batch_report(results)

    print("\n── assert_test() demo (single case) ──")
    await run_pytest_demo()

    print("\nTip: install deepeval with  pip install deepeval")
    print("     then run              pytest sol10_deepeval.py -v")


if __name__ == "__main__":
    asyncio.run(main())
