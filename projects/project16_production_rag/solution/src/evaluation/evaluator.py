"""
Evaluation module.

Golden dataset: 5 curated QA pairs about the sample docs.

Metrics (LLM-judge, 0–1 scale):
  faithfulness  — does the answer contain only facts from the retrieved chunks?
  relevancy     — does the answer actually address the question?

aggregate_score = mean(faithfulness + relevancy) / 2

`gate_passed = aggregate_score >= cfg.EVAL_PASS_THRESHOLD` (default 0.75)
"""
from __future__ import annotations
import json
import logging
from litellm import completion

from src.config import cfg
from src.models import (
    EvalCase, EvalCaseResult, EvalReport,
    QueryRequest,
)
from src.agents.orchestrator import handle

logger = logging.getLogger(__name__)


# ── golden dataset ────────────────────────────────────────────────────────────

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
        reference_answer=(
            "Authentication is done via Bearer tokens in the Authorization header."
        ),
    ),
    EvalCase(
        question="What analytics features are included in the Enterprise plan?",
        reference_answer=(
            "Enterprise includes advanced analytics, custom dashboards, and data export."
        ),
    ),
    EvalCase(
        question="Which endpoint is used to list all projects?",
        reference_answer="GET /v1/projects returns a paginated list of all projects.",
    ),
]


# ── LLM judges ────────────────────────────────────────────────────────────────

def _judge(prompt: str) -> float:
    """Ask the LLM for a score and parse a float in [0, 1]."""
    try:
        resp = completion(
            model=cfg.MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        # Accept "0.85", "85", "8.5/10"
        raw = raw.split("/")[0].strip()
        score = float(raw)
        if score > 1:          # percentage → fraction
            score = score / 100
        return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return 0.0


def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    """
    0 = answer contradicts or fabricates info not in context
    1 = answer contains only facts grounded in context
    """
    context = "\n---\n".join(context_chunks) if context_chunks else "(no context)"
    prompt = (
        "You are a faithfulness judge. Score 0.0–1.0 how well the Answer is "
        "grounded in the Context. 1.0 = fully grounded, 0.0 = entirely fabricated.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Reply with ONLY a decimal number between 0.0 and 1.0."
    )
    return _judge(prompt)


def relevancy_score(question: str, answer: str) -> float:
    """
    0 = answer is off-topic
    1 = answer fully addresses the question
    """
    prompt = (
        "You are a relevancy judge. Score 0.0–1.0 how well the Answer "
        "addresses the Question. 1.0 = fully relevant, 0.0 = completely off-topic.\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
        "Reply with ONLY a decimal number between 0.0 and 1.0."
    )
    return _judge(prompt)


# ── main evaluation runner ────────────────────────────────────────────────────

async def run_eval(retriever, request_id: str = "eval") -> EvalReport:
    """
    Run all golden cases against the RAG pipeline and return an EvalReport.
    """
    results: list[EvalCaseResult] = []

    for i, case in enumerate(GOLDEN_CASES):
        req = QueryRequest(question=case.question)
        logger.info("[%s] eval case %d/%d: %r", request_id, i + 1, len(GOLDEN_CASES), case.question)

        try:
            response = await handle(req, retriever, f"{request_id}-{i}")
            answer   = response.answer
            chunks   = [c.text for c in response.citations] if response.citations else []
        except Exception as exc:
            logger.warning("[%s] eval case %d failed: %s", request_id, i, exc)
            answer = ""
            chunks = []

        f = faithfulness_score(answer, chunks)
        r = relevancy_score(case.question, answer)

        results.append(
            EvalCaseResult(
                question=case.question,
                reference_answer=case.reference_answer,
                generated_answer=answer,
                faithfulness=f,
                relevancy=r,
                overall=(f + r) / 2,
            )
        )
        logger.info("[%s] case %d scores — faith=%.2f  rel=%.2f", request_id, i, f, r)

    aggregate = sum(r.overall for r in results) / len(results) if results else 0.0
    passed    = aggregate >= cfg.EVAL_PASS_THRESHOLD

    report = EvalReport(
        cases=results,
        aggregate_faithfulness=sum(r.faithfulness for r in results) / len(results),
        aggregate_relevancy=sum(r.relevancy for r in results) / len(results),
        aggregate_overall=aggregate,
        gate_passed=passed,
        threshold=cfg.EVAL_PASS_THRESHOLD,
    )

    logger.info(
        "[%s] eval complete — overall=%.3f  gate=%s",
        request_id, aggregate, "PASS" if passed else "FAIL",
    )
    return report
