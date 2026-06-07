"""
TODO — Implement the LLM-judge evaluation suite.

Evaluation components:
  1. Golden dataset — 5 hand-crafted QA pairs based on sample_docs/
  2. faithfulness_score(answer, context_chunks) — does answer stay grounded in context?
  3. relevancy_score(question, answer)          — does answer actually address the question?
  4. run_eval(retriever) — runs all golden cases through the full RAG pipeline

LLM judge prompt pattern:
  "Score 0.0–1.0 how [faithful/relevant] the Answer is. Reply with ONLY a decimal."

Aggregate:
  overall_per_case = (faithfulness + relevancy) / 2
  aggregate        = mean(overall_per_case for all cases)
  gate_passed      = aggregate >= cfg.EVAL_PASS_THRESHOLD  (default 0.75)
"""
from __future__ import annotations
import logging
from litellm import completion
from src.config import cfg
from src.models import EvalCase, EvalCaseResult, EvalReport, QueryRequest
from src.agents.orchestrator import handle

logger = logging.getLogger(__name__)

# ── golden dataset ─────────────────────────────────────────────────────────────
# These 5 QA pairs are based on data/sample_docs/product_overview.md
# and data/sample_docs/api_reference.md — don't change the questions.

GOLDEN_CASES: list[EvalCase] = [
    EvalCase(
        question="What are the three pricing tiers for TechFlow?",
        reference_answer="TechFlow offers Starter, Professional, and Enterprise tiers.",
    ),
    EvalCase(
        question="What is the rate limit for the TechFlow API?",
        reference_answer="The API allows 1000 requests per hour on the Professional plan.",
    ),
    EvalCase(
        question="How do I authenticate with the TechFlow API?",
        reference_answer="Authentication is done via Bearer tokens in the Authorization header.",
    ),
    EvalCase(
        question="What analytics features are included in the Enterprise plan?",
        reference_answer="Enterprise includes advanced analytics, custom dashboards, and data export.",
    ),
    EvalCase(
        question="Which endpoint is used to list all projects?",
        reference_answer="GET /v1/projects returns a paginated list of all projects.",
    ),
]


def _judge(prompt: str) -> float:
    """
    TODO 1: Call litellm.completion(model=cfg.MODEL, messages=[{"role":"user","content":prompt}],
                                    max_tokens=10, temperature=0)
    TODO 2: Parse the score from the response (handle "0.85", "85", "8.5/10" formats)
    TODO 3: Clamp to [0.0, 1.0]
    TODO 4: Return 0.0 on any exception
    """
    raise NotImplementedError


def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    """
    TODO 5: Build a faithfulness judge prompt:
              Context = "\n---\n".join(context_chunks)
              "Score 0.0–1.0 how well the Answer is grounded in Context.
               1.0=fully grounded, 0.0=fabricated. Reply ONLY a decimal."
    TODO 6: Return _judge(prompt)
    """
    raise NotImplementedError


def relevancy_score(question: str, answer: str) -> float:
    """
    TODO 7: Build a relevancy judge prompt:
              "Score 0.0–1.0 how well the Answer addresses the Question.
               1.0=fully relevant, 0.0=off-topic. Reply ONLY a decimal."
    TODO 8: Return _judge(prompt)
    """
    raise NotImplementedError


async def run_eval(retriever, request_id: str = "eval") -> EvalReport:
    """
    Run all GOLDEN_CASES through the RAG pipeline and compute scores.

    TODO 9:  For each case in GOLDEN_CASES:
               - Call await handle(QueryRequest(question=case.question), retriever, ...)
               - Extract answer + citation texts
               - Compute faithfulness_score + relevancy_score
               - Build EvalCaseResult(question, reference_answer, generated_answer,
                                      faithfulness, relevancy, overall=(f+r)/2)
    TODO 10: Compute aggregate_faithfulness, aggregate_relevancy, aggregate_overall
    TODO 11: gate_passed = aggregate_overall >= cfg.EVAL_PASS_THRESHOLD
    TODO 12: Return EvalReport(cases, aggregate_*, gate_passed, threshold)
    """
    raise NotImplementedError
