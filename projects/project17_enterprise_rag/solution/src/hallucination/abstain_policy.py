"""
Abstain policy — decides when to refuse to answer rather than risk hallucination.

Three abstain triggers
──────────────────────
  1. no_relevant_documents
     The maximum cosine similarity of all retrieved chunks is below MIN_RETRIEVAL_SCORE.
     This means the corpus doesn't contain information relevant to the question.
     Returning an answer here would require the LLM to draw on parametric memory → hallucination.

  2. insufficient_grounding
     The NLI faithfulness checker found that fewer than OVERALL_FAITHFULNESS_THRESHOLD
     fraction of sentences in the answer are entailed by the retrieved context.
     Even after removing ungrounded sentences, the answer quality is too low.

  3. all_sentences_ungrounded
     Every sentence in the answer was removed by the faithfulness checker.
     The remaining answer is empty — nothing safe to return.

Tuning guidance
───────────────
  Raising thresholds → fewer hallucinations, more abstains (higher precision)
  Lowering thresholds → fewer abstains, more hallucinations (higher recall)

  For internal knowledge bases: MIN_RETRIEVAL_SCORE=0.60, FAITHFULNESS=0.70
  For customer-facing support:  MIN_RETRIEVAL_SCORE=0.70, FAITHFULNESS=0.85
  For medical/legal:            MIN_RETRIEVAL_SCORE=0.75, FAITHFULNESS=0.95
"""
from __future__ import annotations

import logging

from src.config import cfg
from src.models import FaithfulnessResult

logger = logging.getLogger(__name__)

# Abstain reason constants (returned in QueryResponse.abstain_reason)
REASON_NO_DOCS = "no_relevant_documents"
REASON_LOW_FAITH = "insufficient_grounding"
REASON_ALL_UNGROUNDED = "all_sentences_ungrounded"


class AbstainPolicy:
    def __init__(
        self,
        min_retrieval_score: float = cfg.min_retrieval_score,
        overall_faithfulness_threshold: float = cfg.overall_faithfulness_threshold,
    ) -> None:
        self._min_score = min_retrieval_score
        self._faith_threshold = overall_faithfulness_threshold

    def should_abstain(
        self,
        retrieval_max_score: float,
        faithfulness_result: FaithfulnessResult,
    ) -> tuple[bool, str]:
        """
        Returns (should_abstain: bool, reason: str).
        reason is empty string when should_abstain is False.
        """
        # Check 1: no relevant documents
        if retrieval_max_score < self._min_score:
            logger.info(
                "Abstaining: retrieval_max_score=%.3f < threshold=%.3f",
                retrieval_max_score, self._min_score,
            )
            return True, REASON_NO_DOCS

        # Check 2: overall faithfulness too low
        if not faithfulness_result.passed:
            logger.info(
                "Abstaining: faithfulness=%.3f < threshold=%.3f",
                faithfulness_result.faithfulness_score, self._faith_threshold,
            )
            return True, REASON_LOW_FAITH

        # Check 3: all sentences removed
        if not faithfulness_result.grounded_answer.strip():
            logger.info("Abstaining: no grounded sentences remain")
            return True, REASON_ALL_UNGROUNDED

        return False, ""
