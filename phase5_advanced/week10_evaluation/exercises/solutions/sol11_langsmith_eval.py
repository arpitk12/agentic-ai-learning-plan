"""
Solution 11: LangSmith Evaluation — Datasets, Custom Evaluators & Version Comparison
"""

import os, sys, json, asyncio, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

QA_EXAMPLES = [
    {"question": "Who created Python?",               "expected": "Guido van Rossum",       "keywords": ["guido", "van rossum"]},
    {"question": "What does RAG stand for?",          "expected": "Retrieval-Augmented Generation", "keywords": ["retrieval", "augmented", "generation"]},
    {"question": "What is a Docker container?",       "expected": "isolated environment for running applications", "keywords": ["container", "isolated"]},
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


async def agent_v1(question: str) -> str:
    r = await achat([{"role": "user", "content": question}], system=SYSTEM_V1, max_tokens=300)
    return get_text(r)

async def agent_v2(question: str) -> str:
    r = await achat([{"role": "user", "content": question}], system=SYSTEM_V2, max_tokens=150)
    return get_text(r)


@dataclass
class SimulatedDataset:
    name: str
    id:   str = "local-sim-001"


EVAL_TARGETS = {
    "exact_match":      0.80,
    "keyword_presence": 0.80,
    "llm_score":        0.80,
}


# ── Helpers for accessing run/example objects (dict OR LangSmith object) ──────

def _get_field(obj, *keys):
    """Navigate obj via attribute OR dict access."""
    for k in keys:
        obj = obj[k] if isinstance(obj, dict) else getattr(obj, k)
    return obj


# ── Solution implementations ───────────────────────────────────────────────────

def create_eval_dataset(name: str = "week10-qa-eval"):
    """Create or retrieve a LangSmith dataset, or return a simulation object."""
    if not os.getenv("LANGSMITH_API_KEY"):
        print(f"  [local simulation] dataset: {name} ({len(QA_EXAMPLES)} examples)")
        return SimulatedDataset(name=name)

    try:
        import langsmith
        client = langsmith.Client()
        try:
            dataset = client.read_dataset(dataset_name=name)
            print(f"  Using existing LangSmith dataset: {name}")
        except Exception:
            dataset = client.create_dataset(name, description="Week 10 QA eval set")
            client.create_examples(
                inputs    = [{"question": ex["question"], "keywords": ex["keywords"]} for ex in QA_EXAMPLES],
                outputs   = [{"answer":   ex["expected"]}                             for ex in QA_EXAMPLES],
                dataset_id= dataset.id,
            )
            print(f"  Created LangSmith dataset: {name} ({len(QA_EXAMPLES)} examples)")
        return dataset
    except ImportError:
        print("  [langsmith not installed — local simulation]")
        return SimulatedDataset(name=name)


def exact_match_evaluator(run, example) -> dict:
    """1.0 if expected appears in agent output (case-insensitive), else 0.0."""
    agent_output = _get_field(run,     "outputs", "answer")
    expected     = _get_field(example, "outputs", "answer")
    score = 1.0 if str(expected).lower() in str(agent_output).lower() else 0.0
    return {"key": "exact_match", "score": score}


def keyword_evaluator(run, example) -> dict:
    """Fraction of required keywords present in agent output."""
    agent_output = _get_field(run,     "outputs", "answer")
    try:
        keywords = _get_field(example, "inputs", "keywords")
    except (KeyError, AttributeError):
        keywords = []
    if not keywords:
        return {"key": "keyword_presence", "score": 1.0}
    lower  = str(agent_output).lower()
    score  = sum(1 for kw in keywords if kw.lower() in lower) / len(keywords)
    return {"key": "keyword_presence", "score": round(score, 3)}


def llm_score_evaluator(run, example) -> dict:
    """LLM judge scoring the answer 1-5, normalised to 0-1."""
    try:
        agent_output = _get_field(run,     "outputs", "answer")
        expected     = _get_field(example, "outputs", "answer")
        try:
            q = _get_field(example, "inputs", "question")
        except Exception:
            q = "(unknown)"

        prompt = (
            f"Question: {q}\nExpected: {expected}\nAnswer: {agent_output}\n\n"
            "Score the answer 1 (wrong) to 5 (perfect). Reply ONLY with the integer."
        )
        resp   = asyncio.run(achat([{"role": "user", "content": prompt}], max_tokens=5))
        text   = get_text(resp).strip()
        m      = re.search(r"[1-5]", text)
        val    = int(m.group()) if m else 3
        score  = round((val - 1) / 4, 3)
    except Exception:
        score = 0.5
    return {"key": "llm_score", "score": score}


async def run_langsmith_eval(agent_fn, dataset_name: str = "week10-qa-eval") -> dict:
    """Run all evaluators. Uses LangSmith if key present, otherwise local simulation."""
    if os.getenv("LANGSMITH_API_KEY"):
        try:
            from langsmith.evaluation import evaluate as ls_evaluate

            def target(inputs: dict) -> dict:
                return {"answer": asyncio.run(agent_fn(inputs["question"]))}

            results = ls_evaluate(
                target,
                data=dataset_name,
                evaluators=[exact_match_evaluator, keyword_evaluator, llm_score_evaluator],
                experiment_prefix="week10",
            )
            summary: dict[str, float] = {}
            for r in results.summary_results:
                summary[r["key"]] = round(r["score"], 3)
            return summary
        except ImportError:
            pass  # fall through to local

    return await _local_eval(agent_fn)


async def _local_eval(agent_fn) -> dict:
    """Run evaluation locally without LangSmith."""
    answers = await asyncio.gather(*[agent_fn(ex["question"]) for ex in QA_EXAMPLES])

    run_objs = [
        {"outputs": {"answer": a}}
        for a in answers
    ]
    example_objs = [
        {
            "inputs":  {"question": ex["question"], "keywords": ex["keywords"]},
            "outputs": {"answer": ex["expected"]},
        }
        for ex in QA_EXAMPLES
    ]

    scores: dict[str, list[float]] = {
        "exact_match":      [],
        "keyword_presence": [],
        "llm_score":        [],
    }
    for run, ex in zip(run_objs, example_objs):
        scores["exact_match"].append(      exact_match_evaluator(run, ex)["score"])
        scores["keyword_presence"].append(  keyword_evaluator(run, ex)["score"])
        scores["llm_score"].append(         llm_score_evaluator(run, ex)["score"])

    return {k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in scores.items()}


async def compare_versions() -> tuple[dict, dict]:
    """Evaluate both agent versions and return their result dicts."""
    print("  Evaluating agent_v1...")
    v1_results = await run_langsmith_eval(agent_v1, "week10-qa-eval-v1")
    print("  Evaluating agent_v2...")
    v2_results = await run_langsmith_eval(agent_v2, "week10-qa-eval-v2")
    return v1_results, v2_results


def print_eval_summary(results: dict, title: str = ""):
    if title:
        print(f"\n  {title}")
    print(f"  {'Evaluator':<24} {'Score':>8}   Status")
    print("  " + "─" * 44)
    for name, score in results.items():
        target = EVAL_TARGETS.get(name, 0.80)
        ok     = "✅" if score >= target else "⚠ "
        print(f"  {name:<24} {score:8.3f}   {ok}  (target ≥ {target:.2f})")


def print_version_comparison(v1: dict, v2: dict):
    print("\n  ══ Version Comparison ══")
    print(f"  {'Metric':<24} {'v1':>8} {'v2':>8} {'Delta':>8}")
    print("  " + "─" * 52)
    for metric in v1:
        s1    = v1.get(metric, 0.0)
        s2    = v2.get(metric, 0.0)
        delta = s2 - s1
        icon  = "✅" if delta >= 0 else "❌"
        print(f"  {metric:<24} {s1:8.3f} {s2:8.3f} {delta:+8.3f} {icon}")


async def main():
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    mode    = "LangSmith hosted" if has_key else "LOCAL SIMULATION"

    print("═" * 56)
    print(f"LANGSMITH EVALUATION — QA Agent [{mode}]")
    print("═" * 56)

    if not has_key:
        print("\n  💡 To use hosted LangSmith:")
        print("     1. Sign up at https://smith.langchain.com (free)")
        print("     2. Add LANGSMITH_API_KEY=ls__... to your .env\n")

    create_eval_dataset()

    print("\n── Single agent evaluation ──")
    results = await run_langsmith_eval(agent_v1)
    print_eval_summary(results, "Agent v1 results:")

    print("\n── Two-version comparison ──")
    v1_results, v2_results = await compare_versions()
    print_version_comparison(v1_results, v2_results)


if __name__ == "__main__":
    asyncio.run(main())
