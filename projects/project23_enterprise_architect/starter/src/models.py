"""Starter stub — Project 23: PydanticAI typed contracts.

Define all Pydantic models that serve as contracts between agents.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    """Contract for the Risk Analysis Agent output.

    PydanticAI enforces this schema — the agent retries if it can't conform.
    """
    # TODO 1: Add field 'risk_level' as Literal["low", "medium", "high", "critical"]
    # TODO 2: Add field 'risk_factors' as list[str] with min_length=1
    # TODO 3: Add field 'regulatory_concerns' as list[str]
    # TODO 4: Add field 'requires_human_review' as bool
    # TODO 5: Add field 'confidence' as float with ge=0.0, le=1.0
    # TODO 6: Add field 'reasoning' as str
    pass


class PolicyViolation(BaseModel):
    """A single policy violation."""
    # TODO 7: Add fields: policy_id, policy_name, violation_description,
    #         severity (Literal["minor","major","blocker"]), remediation_steps: list[str]
    pass


class PolicyCheckResult(BaseModel):
    """Contract for the Policy Check Agent output."""
    # TODO 8: Add fields: is_compliant (bool), violations (list[PolicyViolation]),
    #         compliance_score (float ge=0 le=100), checked_policies (list[str]), summary (str)
    pass


class ComplianceReport(BaseModel):
    """Final output report."""
    # TODO 9: Add fields: document_id, document_type, status (Literal[...]),
    #         risk_level, compliance_score, violations, processing_time_seconds,
    #         llm_cost_usd, manual_cost_equivalent_usd, recommendations: list[str]
    pass
