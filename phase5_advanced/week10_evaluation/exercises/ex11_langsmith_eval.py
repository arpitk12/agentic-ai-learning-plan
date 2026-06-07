"""
Exercise 11: LangSmith Evaluation — Datasets, Custom Evaluators & Version Comparison
Goal: Use LangSmith as a full evaluation platform: upload datasets, run evaluators,
      compare two agent versions, and track quality over time.

Install: pip install langsmith

Setup:
  1. Get a free API key at https://smith.langchain.com
  2. Add to .env:
       LANGSMITH_API_KEY=ls__your_key_here
       LANGSMITH_TRACING=true
       LANGSMITH_PROJECT=week10-eval

  NOTE: This exercise runs fully WITHOUT a key — it simulates the LangSmith
  workflow locally so you understand the API before connecting to the hosted service.

What LangSmith evaluation adds:
  - Persistent dataset storage (upload once, reuse across experiments)
  - Versioned experiments (compare agent v1 vs v2 on the same dataset)
  - Custom evaluator functions (any Python function → numeric score)
  - Automatic result aggregation + dashboard

Core API flow:
  client  = langsmith.Client()
  dataset = client.create_dataset("my-eval-set")
  client.create_examples(inputs=[...], outputs=[...], dataset_id=dataset.id)
  results = evaluate(target_fn, data="my-eval-set", evaluators=[...])

Tasks:
  1. Complete create_eval_dataset()   — upload QA pairs to LangSmith as a Dataset.
  2. Complete exact_match_evaluator() — custom evaluator: 1.0 if expected in answer.
  3. Complete keyword_evaluator()     — custom evaluator: fraction of keywords present.
  4. Complete llm_score_evaluator()   — LLM judge that scores 1-5 and returns 0-1.
  5. Complete run_langsmith_eval()    — call evaluate() with all 3 evaluators; print summary.
  6. Complete compare_versions()      — run eval on agent_v1 and agent_v2, print diff table.

Run:
  python ex11_langsmith_eval.py

Expected output (with or without API key):
  ══════════════════════════════════════════════
  LANGSMITH EVALUATION — QA Agent
  ══════════════════════════════════════════════
  Dataset: "week10-qa-eval" (10 examples)

  Evaluator                Score    Status
  ──────────────────────────────────────
  exact_match              0.60     ⚠  (target ≥ 0.80)
  keyword_presence         0.87     ✅ (target ≥ 0.80)
  llm_score                0.82     ✅ (target ≥ 0.80)

  ══ Version Comparison ══
  Metric           agent_v1   agent_v2   Delta
  exact_match       0.60       0.70      +0.10 ✅
  keyword_presence  0.87       0.91      +0.04 ✅
  llm_score         0.82       0.86      +0.04 ✅
"""

import os, sys, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

LANGSMITH_AVAILABLE = bool(os.getenv("LANGSMITH_API_KEY"))

# ── QA Dataset ─────────────────────────────────────────────────────────────────

QA_EXAMPLES = [
    {"question": "Who created Python?",               "expected": "Guido van Rossum",       "keywords": ["guido", "van rossum"]},
    {"question": "What does RAG stand for?",          "expected": "Retrieval-Augmented Generation", "keywords": ["retrieval", "augmented", "generation"]},
    {"question": "What is a Docker container?",       "expected": "isolated environment for running applications", "keywords": ["container", "isolated", "docker"]},
    {"question": "What HTTP code means Not Found?",   "expected": "404",                    "keywords": ["404"]},
    {"question": "What is asyncio?",                  "expected": "Python async library",   "keywords": ["async", "asyncio", "python"]},
    {"question": "What does SOLID stand for?",        "expected": "Single responsibility, Open/closed, Liskov, Interface, Dependency", "keywords": ["single", "open", "liskov"]},
    {"question": "What is the purpose of a load balancer?", "expected": "distribute traffic across servers", "keywords": ["traffic", "distribute", "server"]},
    {"question": "What is a database index?",         "expected": "data structure for faster query lookup", "keywords": ["index", "faster", "query"]},
    {"question": "What is latency?",                  "expected": "time delay between request and response", "keywords": ["delay", "time", "response"]},
    {"question": "What is a context window in LLMs?", "expected": "maximum tokens an LLM can process at once", "keywords": ["token", "process", "maximum"]},
]

SYSTEM_V1 = "You are a helpful assistant. Answer accurately."
SYSTEM_V2 = "You are a helpful assistant. Answer accurately and concisely in 1-2 sentences maximum."


# ── Agent variants ─────────────────────────────────────────────────────────────

async def agent_v1(question: str) -> str:
    """Agent version 1: standard system prompt."""
    r = await achat([{"role": "user", "content": question}], system=SYSTEM_V1, max_tokens=300)
    return get_text(r)

async def agent_v2(question: str) -> str:
    """Agent version 2: conciseness-focused system prompt."""
    r = await achat([{"role": "user", "content": question}], system=SYSTEM_V2, max_tokens=150)
    return get_text(r)


# ── Local simulation (used when LANGSMITH_API_KEY not set) ────────────────────

@dataclass
class SimulatedDataset:
    name: str
    id:   str = "local-sim-001"

@dataclass
class SimulatedExample:
    inputs:  dict
    outputs: dict

@dataclass
class EvalRun:
    """Local simulation of a LangSmith evaluation run result."""
    evaluator_scores: dict[str, list[float]] = field(default_factory=dict)

    def add(self, evaluator_name: str, score: float):
        self.evaluator_scores.setdefault(evaluator_name, []).append(score)

    def summary(self) -> dict[str, float]:
        return {k: sum(v)/len(v) for k, v in self.evaluator_scores.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Complete create_eval_dataset()
# ─────────────────────────────────────────────────────────────────────────────

def create_eval_dataset(name: str = "week10-qa-eval"):
    """
    Create (or retrieve) a LangSmith dataset and upload QA_EXAMPLES to it.

    With LangSmith API key:
      import langsmith
      client  = langsmith.Client()
      # Check if dataset already exists
      try:
          dataset = client.read_dataset(dataset_name=name)
          print(f"  Using existing dataset: {name}")
      except Exception:
          dataset = client.create_dataset(name, description="Week 10 QA eval set")
          client.create_examples(
              inputs  = [{"question": ex["question"]} for ex in QA_EXAMPLES],
              outputs = [{"answer":   ex["expected"]} for ex in QA_EXAMPLES],
              dataset_id = dataset.id,
          )
          print(f"  Created dataset: {name} ({len(QA_EXAMPLES)} examples)")
      return dataset

    Without API key (fallback):
      return SimulatedDataset(name=name)

    TODO: implement with the try/except pattern above.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Complete exact_match_evaluator()
# ─────────────────────────────────────────────────────────────────────────────

def exact_match_evaluator(run, example) -> dict:
    """
    LangSmith custom evaluator: returns 1.0 if the expected answer appears
    (case-insensitive) in the agent's output, else 0.0.

    LangSmith evaluator contract:
    - `run`     has attribute `outputs` (dict with key "answer" = agent response)
    - `example` has attribute `outputs` (dict with key "answer" = expected)
    - Return:   {"key": "exact_match", "score": float}

    For local simulation (when objects are dicts):
    - `run`     may be a dict: {"outputs": {"answer": "..."}}
    - `example` may be a dict: {"outputs": {"answer": "..."}}

    TODO:
    1. Extract agent_output  = run.outputs["answer"]     (or run["outputs"]["answer"])
    2. Extract expected       = example.outputs["answer"] (or example["outputs"]["answer"])
    3. score = 1.0 if expected.lower() in agent_output.lower() else 0.0
    4. Return {"key": "exact_match", "score": score}

    Hint: use getattr(obj, "outputs", None) or obj["outputs"] pattern.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Complete keyword_evaluator()
# ─────────────────────────────────────────────────────────────────────────────

def keyword_evaluator(run, example) -> dict:
    """
    LangSmith custom evaluator: score = fraction of required keywords present
    in the agent output.

    Keywords are stored in example.inputs["keywords"] (list of strings).
    If no keywords provided, return score=1.0.

    TODO:
    1. agent_output = run.outputs["answer"]
    2. keywords     = example.inputs.get("keywords", [])  ← from the input side
    3. score = fraction of keywords that appear in agent_output.lower()
    4. Return {"key": "keyword_presence", "score": score}
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 4: Complete llm_score_evaluator()
# ─────────────────────────────────────────────────────────────────────────────

def llm_score_evaluator(run, example) -> dict:
    """
    LangSmith custom evaluator: use an LLM to score the answer 1-5,
    normalised to 0-1.

    TODO:
    1. Extract agent_output and expected (same as exact_match_evaluator).
    2. Build a prompt: "Question: {q}\nExpected: {expected}\nAnswer: {agent_output}\n
       Score 1-5 for accuracy. Reply ONLY with a number."
    3. Call achat() synchronously using asyncio.run() (evaluators run in sync context).
    4. Parse the integer from the response; normalise: score = (int_val - 1) / 4.
    5. Return {"key": "llm_score", "score": score}.

    On any error, return {"key": "llm_score", "score": 0.5}.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 5: Complete run_langsmith_eval()
# ─────────────────────────────────────────────────────────────────────────────

async def run_langsmith_eval(agent_fn, dataset_name: str = "week10-qa-eval") -> dict:
    """
    Run all evaluators against the dataset. Return summary {evaluator: avg_score}.

    With LangSmith:
      from langsmith.evaluation import evaluate
      results = evaluate(
          lambda inputs: {"answer": asyncio.run(agent_fn(inputs["question"]))},
          data=dataset_name,
          evaluators=[exact_match_evaluator, keyword_evaluator, llm_score_evaluator],
          experiment_prefix="week10",
      )
      return {r["key"]: r["score"] for r in results.summary_results}

    Without LangSmith (local simulation):
      Run all QA_EXAMPLES through agent_fn concurrently.
      Apply each evaluator function manually.
      Return averaged scores.

    TODO: implement with try/except ImportError fallback.
    """
    raise NotImplementedError


async def _local_eval(agent_fn) -> dict:
    """Local simulation of evaluate() — no LangSmith key needed."""
    answers = await asyncio.gather(*[agent_fn(ex["question"]) for ex in QA_EXAMPLES])

    run_objs     = [{"outputs": {"answer": a}}                          for a in answers]
    example_objs = [{"inputs":  {"question": ex["question"], "keywords": ex["keywords"]},
                     "outputs": {"answer": ex["expected"]}}             for ex in QA_EXAMPLES]

    def _get(obj, *keys):
        for k in keys:
            obj = obj[k] if isinstance(obj, dict) else getattr(obj, k)
        return obj

    scores: dict[str, list[float]] = {
        "exact_match":      [],
        "keyword_presence": [],
        "llm_score":        [],
    }

    for run, ex in zip(run_objs, example_objs):
        scores["exact_match"].append(    exact_match_evaluator(run, ex)["score"])
        scores["keyword_presence"].append(keyword_evaluator(run, ex)["score"])
        scores["llm_score"].append(      llm_score_evaluator(run, ex)["score"])

    return {k: sum(v)/len(v) if v else 0.0 for k, v in scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
# TODO 6: Complete compare_versions()
# ─────────────────────────────────────────────────────────────────────────────

async def compare_versions() -> tuple[dict, dict]:
    """
    Evaluate agent_v1 and agent_v2 on the same dataset and return both result dicts.

    TODO:
    1. v1_results = await run_langsmith_eval(agent_v1, "week10-qa-eval-v1")
    2. v2_results = await run_langsmith_eval(agent_v2, "week10-qa-eval-v2")
    3. Return (v1_results, v2_results)
    """
    raise NotImplementedError


# ── Reporting ──────────────────────────────────────────────────────────────────

EVAL_TARGETS = {
    "exact_match":       0.80,
    "keyword_presence":  0.80,
    "llm_score":         0.80,
}

def print_eval_summary(results: dict, title: str = ""):
    if title:
        print(f"\n  {title}")
    print(f"  {'Evaluator':<24} {'Score':>8}   Status")
    print("  " + "─" * 44)
    for name, score in results.items():
        target = EVAL_TARGETS.get(name, 0.80)
        ok     = "✅" if score >= target else "⚠ "
        print(f"  {name:<24} {score:8.2f}   {ok}  (target ≥ {target:.2f})")

def print_version_comparison(v1: dict, v2: dict):
    print("\n  ══ Version Comparison ══")
    print(f"  {'Metric':<24} {'v1':>8} {'v2':>8} {'Delta':>8}")
    print("  " + "─" * 52)
    for metric in v1:
        s1    = v1.get(metric, 0.0)
        s2    = v2.get(metric, 0.0)
        delta = s2 - s1
        icon  = "✅" if delta >= 0 else "❌"
        print(f"  {metric:<24} {s1:8.2f} {s2:8.2f} {delta:+8.2f} {icon}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    mode    = "LangSmith hosted" if has_key else "LOCAL SIMULATION (no API key)"

    print("═" * 56)
    print(f"LANGSMITH EVALUATION — QA Agent [{mode}]")
    print("═" * 56)

    if not has_key:
        print("\n  💡 To use hosted LangSmith:")
        print("     1. Sign up at https://smith.langchain.com (free)")
        print("     2. Add LANGSMITH_API_KEY=ls__... to your .env")
        print("     3. Re-run this script\n")

    print(f"  Dataset: 'week10-qa-eval' ({len(QA_EXAMPLES)} examples)")

    print("\n── Single agent evaluation ──")
    results = await run_langsmith_eval(agent_v1)
    print_eval_summary(results, "Agent v1 results:")

    print("\n── Two-version comparison ──")
    v1_results, v2_results = await compare_versions()
    print_version_comparison(v1_results, v2_results)


if __name__ == "__main__":
    asyncio.run(main())
