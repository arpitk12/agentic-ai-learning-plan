"""PydanticAI typed contracts for the compliance review system."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    """Output contract for the Risk Analysis Agent."""
    risk_level: Literal["low", "medium", "high", "critical"]
    risk_factors: list[str] = Field(min_length=1, description="Specific risks identified")
    regulatory_concerns: list[str] = Field(description="Applicable regulations: GDPR, SOX, HIPAA, etc.")
    requires_human_review: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in assessment")
    reasoning: str = Field(description="Detailed justification of the risk level")


class PolicyViolation(BaseModel):
    """A single policy violation found during policy check."""
    policy_id: str
    policy_name: str
    violation_description: str
    severity: Literal["minor", "major", "blocker"]
    remediation_steps: list[str]


class PolicyCheckResult(BaseModel):
    """Output contract for the Policy Check Agent."""
    is_compliant: bool
    violations: list[PolicyViolation]
    compliance_score: float = Field(ge=0.0, le=100.0)
    checked_policies: list[str] = Field(description="Policy IDs that were evaluated")
    summary: str


class AuditEntry(BaseModel):
    """Immutable audit record for a workflow step."""
    document_id: str
    step: str
    action: str
    actor: str           # "system" | "risk-agent" | "policy-agent" | reviewer name
    timestamp: str
    details: dict
    prev_hash: str
    entry_hash: str


class ComplianceReport(BaseModel):
    """Final output report for a compliance review run."""
    document_id: str
    document_type: str
    status: Literal["approved", "rejected", "escalated", "pending"]
    risk_level: str
    compliance_score: float
    violations: list[PolicyViolation]
    human_reviewer: str | None
    human_notes: str | None
    processing_time_seconds: float
    llm_cost_usd: float
    manual_cost_equivalent_usd: float   # what this would have cost manually
    audit_trail_hash: str               # hash of final audit entry
    recommendations: list[str]
