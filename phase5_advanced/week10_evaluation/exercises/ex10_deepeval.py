"""
Exercise 10: DeepEval — Production LLM Evaluation Framework
Goal: Use DeepEval to systematically evaluate agent outputs with 14+ built-in metrics
      and write your own custom metric.

Install: pip install deepeval

What DeepEval adds over LLM-as-judge:
  - Standardised metric API (score 0–1, reason, threshold)
  - pytest integration via @pytest.mark.deepeval + assert_test()
  - 14+ built-in metrics: AnswerRelevancy, Faithfulness, Hallucination,
    Bias, Toxicity, Contextual Precision/Recall/Relevancy, GEval, ...
  - Batch evaluation via evaluate()
  - Confidence scores + detailed reasoning per failure

Core concepts:
  LLMTestCase(input, actual_output, expected_output, context)  ← data container
  AnswerRelevancyMetric(threshold=0.7)                         ← metric with threshold
  assert_test(test_case, [metric1, metric2])                   ← raises if any fails
  evaluate([test_case1, ...], [metric1, ...])                  ← batch, returns EvaluationResult

Tasks:
  1. Complete make_test_case()    — build a DeepEval LLMTestCase from raw data.
  2. Complete build_metrics()     — instantiate 4 metrics with appropriate thresholds.
  3. Complete ConcisenessMetric   — a custom BaseMetric that checks answer length.
  4. Complete run_batch_eval()    — call evaluate() and extract pass/fail per metric.
  5. Complete run_pytest_demo()   — demonstrate assert_test() with pytest-style output.

Run:
  python ex10_deepeval.py           ← standalone batch eval
  pytest ex10_deepeval.py -v        ← runs the pytest test functions too

Expected output:
  DeepEval Evaluation — 6 test cases × 5 metrics
  ──────────────────────────────────────────────
  answer_relevancy    : 5/6 passed  (83.3%)  ✅
  faithfulness        : 6/6 passed  (100%)   ✅
  hallucination       : 6/6 passed  (100%)   ✅
  bias                : 6/6 passed  (100%)   ✅
  conciseness_custom  : 4/6 passed  (66.7%)  ⚠
  Overall             : 26/30       (86.7%)  ✅ (target ≥ 80%)
"""

import os, sys, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

# ── Agent under test ───────────────────────────────────────────────────────────

SYSTEM = "You are a helpful AI assistant. Answer concisely and accurately."

async def agent(question: str, context: str = "") -> str:
    prompt = (f"Context:\n{context}\n\nQuestion: {question}" if context
              else question)
    r = await achat([{"role": "user", "content": prompt}], system=SYSTEM, max_tokens=300)
    return get_text(r)


# ── Test data ──────────────────────────────────────────────────────────────────

CONTEXT_DOCS = {
    "python": "Python was created by Guido van Rossum in 1991. It emphasises readability.",
    "docker": "Docker uses OS-level virtualisation to deliver software in containers. Containers share the host OS kernel but are isolated.",
    "rag":    "Retrieval-Augmented Generation (RAG) combines a retrieval step with LLM generation. It grounds LLM answers in retrieved documents.",
    "async":  "asyncio is Python's standard library for async programming. The event loop executes coroutines without OS threads.",
    "git":    "Git is a distributed version control system. Branches allow parallel development. Commits are immutable snapshots.",
}

RAW_TEST_CASES = [
    {
        "input":           "Who created Python and when?",
        "context_key":     "python",
        "expected_output": "Python was created by Guido van Rossum in 1991.",
    },
    {
        "input":           "What does Docker containerise?",
        "context_key":     "docker",
        "expected_output": "Docker containerises software so it runs consistently across environments.",
    },
    {
        "input":           "What is RAG in AI?",
        "context_key":     "rag",
        "expected_output": "RAG combines retrieval with LLM generation, grounding answers in documents.",
    },
    {
        "input":           "What is asyncio used for?",
        "context_key":     "async",
        "expected_output": "asyncio enables async programming in Python via an event loop.",
    },
    {
        "input":           "What is a Git branch?",
        "context_key":     "git",
        "expected_output": "A Git branch allows parallel development without affecting the main codebase.",
    },
    {
        "input":           "What is the capital of Australia?",  # no context — tests hallucination guard
        "context_key":     None,
        "expected_output": "Canberra",
    },
]

MAX_ANSWER_WORDS = 60   # used by the custom conciseness metric


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Complete make_test_case()
# ─────────────────────────────────────────────────────────────────────────────

def make_test_case(raw: dict, actual_output: str):
    """
    Build a DeepEval LLMTestCase from a raw test dict and the agent's answer.

    DeepEval signature:
    LLMTestCase(
        input:           str,             # the user question
        actual_output:   str,             # what the agent said
        expected_output: str | None,      # ground truth (for some metrics)
        retrieval_context: list[str] | None,  # retrieved chunks (for faithfulness)
    )

    For `retrieval_context`, use [CONTEXT_DOCS[raw["context_key"]]] if key exists,
    else None.

    TODO:
    1. try: from deepeval.test_case import LLMTestCase
    2. Build and return LLMTestCase with the 4 fields above.
    3. If ImportError: return a plain dict with the same keys (fallback).
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Complete build_metrics()
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics() -> list:
    """
    Return a list of 4 DeepEval metric instances:
      - AnswerRelevancyMetric(threshold=0.7)
      - FaithfulnessMetric(threshold=0.7)
      - HallucinationMetric(threshold=0.5)   ← lower = less hallucination accepted
      - BiasMetric(threshold=0.5)

    DeepEval metrics live in:
      from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric,
                                   HallucinationMetric, BiasMetric

    TODO:
    1. try import; instantiate and return the 4 metrics.
    2. On ImportError: return [] (caller handles this gracefully).
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Complete ConcisenessMetric (custom BaseMetric)
# ─────────────────────────────────────────────────────────────────────────────

class ConcisenesFallback:
    """Fallback used when deepeval is not installed."""
    name = "conciseness_custom"
    threshold = 0.5

    def measure(self, test_case) -> float:
        output = test_case.get("actual_output", "") if isinstance(test_case, dict) else test_case.actual_output
        words  = len(output.split())
        self.score  = 1.0 if words <= MAX_ANSWER_WORDS else round(MAX_ANSWER_WORDS / words, 2)
        self.reason = (f"Answer is {words} words — {'OK' if words <= MAX_ANSWER_WORDS else f'too long, limit is {MAX_ANSWER_WORDS}'}.")
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold


def make_conciseness_metric():
    """
    Create a custom DeepEval metric that scores answers by length.

    A passing answer has ≤ MAX_ANSWER_WORDS words (score = 1.0).
    A longer answer is penalised: score = MAX_ANSWER_WORDS / word_count.
    Threshold: 0.5 (answer must not be more than 2× the max length).

    TODO:
    1. try:
         from deepeval.metrics import BaseMetric
         class ConcisessMetric(BaseMetric):
             def __init__(self): ...
             def measure(self, test_case) -> float: ...  # sets self.score, self.reason
             def is_successful(self) -> bool: ...
         return ConcisessMetric()
    2. On ImportError: return ConcisenesFallback()
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 4: Complete run_batch_eval()
# ─────────────────────────────────────────────────────────────────────────────

async def run_batch_eval(raw_cases: list[dict]) -> dict:
    """
    Run agent on all test cases, then run DeepEval batch evaluate().

    Steps:
    1. Collect actual outputs: run agent() for all cases concurrently.
    2. Build test cases: [make_test_case(raw, output) for each].
    3. Build metrics: build_metrics() + [make_conciseness_metric()].
    4. Try:
         from deepeval import evaluate
         result = evaluate(test_cases, metrics)
         # result.test_results is a list; each has .metrics_data
         # Extract per-metric pass rates.
       Except ImportError: simulate scores.
    5. Return dict: {metric_name: {"passed": N, "total": M, "rate": float}}.

    TODO: implement steps 1–5.
    """
    raise NotImplementedError


def _simulate_batch_results(n_cases: int, metric_names: list[str]) -> dict:
    """Simulate plausible DeepEval results when library not installed."""
    import random
    rng = random.Random(42)
    return {
        name: {
            "passed": (p := rng.randint(n_cases - 2, n_cases)),
            "total":  n_cases,
            "rate":   p / n_cases,
        }
        for name in metric_names
    }


# ─────────────────────────────────────────────────────────────────────────────
# TODO 5: Complete run_pytest_demo()
# ─────────────────────────────────────────────────────────────────────────────

async def run_pytest_demo():
    """
    Demonstrate assert_test() (the per-case pytest-style assertion).

    For ONE test case (the first in RAW_TEST_CASES):
    1. Run the agent to get actual_output.
    2. Build a LLMTestCase.
    3. Try:
         from deepeval import assert_test
         assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])
         print("  ✅ assert_test passed")
       Except AssertionError as e:
         print(f"  ❌ assert_test failed: {e}")
       Except ImportError:
         print("  [deepeval not installed — skipping assert_test demo]")

    TODO: implement this function.
    """
    raise NotImplementedError


# ── Pytest test functions (run with `pytest ex10_deepeval.py -v`) ──────────────

def test_answer_relevancy_python():
    """pytest: agent answer about Python should be relevant."""
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric
        import asyncio

        output = asyncio.run(agent("Who created Python?", CONTEXT_DOCS["python"]))
        tc     = LLMTestCase(
            input="Who created Python?",
            actual_output=output,
            retrieval_context=[CONTEXT_DOCS["python"]],
        )
        assert_test(tc, [AnswerRelevancyMetric(threshold=0.7)])
    except ImportError:
        print("[deepeval not installed — test skipped]")


def test_no_hallucination_in_context_answer():
    """pytest: agent should not hallucinate when context is provided."""
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import HallucinationMetric
        import asyncio

        output = asyncio.run(agent("What is asyncio used for?", CONTEXT_DOCS["async"]))
        tc     = LLMTestCase(
            input="What is asyncio used for?",
            actual_output=output,
            context=[CONTEXT_DOCS["async"]],         # HallucinationMetric uses context=
        )
        assert_test(tc, [HallucinationMetric(threshold=0.5)])
    except ImportError:
        print("[deepeval not installed — test skipped]")


# ── Report helper ──────────────────────────────────────────────────────────────

def print_batch_report(results: dict, overall_threshold: float = 0.80):
    total_p, total_t = 0, 0
    print(f"\n{'Metric':<28} {'Passed':>8} {'Rate':>8}  Status")
    print("─" * 56)
    for name, r in results.items():
        ok   = "✅" if r["rate"] >= overall_threshold else "⚠ "
        print(f"  {name:<26} {r['passed']:>3}/{r['total']:<3}  {r['rate']:5.1%}  {ok}")
        total_p += r["passed"]
        total_t += r["total"]
    overall_rate = total_p / total_t if total_t else 0.0
    ok = "✅ PASS" if overall_rate >= overall_threshold else "❌ FAIL"
    print("─" * 56)
    print(f"  {'Overall':<26} {total_p:>3}/{total_t:<3}  {overall_rate:5.1%}  {ok}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 62)
    print("DEEPEVAL — LLM Evaluation Framework")
    print("=" * 62)
    print(f"Evaluating {len(RAW_TEST_CASES)} test cases...\n")

    print("── Batch evaluation ──")
    results = await run_batch_eval(RAW_TEST_CASES)
    print_batch_report(results)

    print("\n── assert_test() demo ──")
    await run_pytest_demo()

    print("\nTip: run  pytest ex10_deepeval.py -v  to see pytest integration.")


if __name__ == "__main__":
    asyncio.run(main())
