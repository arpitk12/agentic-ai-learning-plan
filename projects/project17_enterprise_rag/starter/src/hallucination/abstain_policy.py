"""
Abstain policy — decides when to refuse to answer.
See GUIDE.md Phase 4.5 for tuning guidance.
"""
from __future__ import annotations

import logging

from src.config import cfg
from src.models import FaithfulnessResult

logger = logging.getLogger(__name__)

REASON_NO_DOCS = "no_relevant_documents"
REASON_LOW_FAITH = "insufficient_grounding"
REASON_ALL_UNGROUNDED = "all_sentences_ungrounded"


class AbstainPolicy:
    def __init__(
        self,
        min_retrieval_score: float = cfg.min_retrieval_score,
        overall_faithfulness_threshold: float = cfg.overall_faithfulness_threshold,
    ) -> None:
        """
        TODO 1: Store min_retrieval_score as self._min_score.
                Store overall_faithfulness_threshold as self._faith_threshold.
        """
        raise NotImplementedError

    def should_abstain(
        self,
        retrieval_max_score: float,
        faithfulness_result: FaithfulnessResult,
    ) -> tuple[bool, str]:
        """
        Returns (should_abstain: bool, reason: str).
        reason is empty string when should_abstain is False.

        TODO 2: If retrieval_max_score < self._min_score:
                  return True, REASON_NO_DOCS

        TODO 3: If not faithfulness_result.passed:
                  return True, REASON_LOW_FAITH

        TODO 4: If faithfulness_result.grounded_answer.strip() is empty:
                  return True, REASON_ALL_UNGROUNDED

        TODO 5: Return False, ""
        """
        raise NotImplementedError
