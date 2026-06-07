"""
LLM-judge evaluation suite.

Runs 5 golden QA pairs through the full RAG pipeline and scores each answer on:
  faithfulness — does the answer stay within the retrieved context?
  relevancy    — does the answer address the question?

Aggregate score = mean((faithfulness + relevancy) / 2) across all cases.
gate_passed    = aggregate >= cfg.EVAL_PASS_THRESHOLD
"""
from __future__ import annotations
import logging
from litellm import completion
from src.config import cfg
from src.models import EvalCase, EvalCaseResult, EvalReport, QueryRequest
from src.agents.orchestrator import handle

logger = logging.getLogger(__name__)

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
    TODO 1: Call the LLM with the given prompt and parse a 0.0–1.0 score from the response.
    TODO 2: Clamp the result to the valid range and convert percentages if needed.
    TODO 3: Return 0.0 on any failure.
    """
    raise NotImplementedError


def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    """
    TODO 4: Build a prompt asking the LLM to score how well the answer
            is grounded in the provided context chunks (not hallucinated).
    TODO 5: Return the judge score.
    """
    raise NotImplementedError


def relevancy_score(question: str, answer: str) -> float:
    """
    TODO 6: Build a prompt asking the LLM to score how well the answer
            addresses the original question.
    TODO 7: Return the judge score.
    """
    raise NotImplementedError


async def run_eval(retriever, request_id: str = "eval") -> EvalReport:
    """
    Run all golden cases and return an EvalReport.

    TODO 8: For each case, run the question through the full RAG pipeline
    TODO 9: Score the generated answer for faithfulness and relevancy
    TODO 10: Compute aggregate scores across all cases
    TODO 11: Determine whether the gate passes based on cfg.EVAL_PASS_THRESHOLD
    TODO 12: Return a fully populated EvalReport
    """
    raise NotImplementedError
