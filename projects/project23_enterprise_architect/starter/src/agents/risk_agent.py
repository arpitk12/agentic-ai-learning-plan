"""Starter stub — Project 23: PydanticAI Risk Agent."""
from __future__ import annotations

import os
from pydantic_ai import Agent, ModelRetry, RunContext
from src.models import RiskAssessment

# TODO 1: Create risk_agent = Agent(
#   model="openai:gpt-4o-mini",
#   result_type=RiskAssessment,
#   system_prompt="You are a compliance risk analyst. Analyze for GDPR, SOX, HIPAA, PCI-DSS risks..."
# )

# TODO 2: Add a @risk_agent.result_validator that:
#   - Raises ModelRetry if risk_level=="critical" and confidence < 0.75
#   - Raises ModelRetry if reasoning is shorter than 50 chars

async def assess_document_risk(document_content: str, document_type: str) -> tuple[RiskAssessment, float]:
    """
    TODO 3: Call risk_agent.run(f"Document type: {document_type}\n\n{document_content}")
    TODO 4: Calculate cost from result.usage().total_tokens * 0.00000015
    TODO 5: Return (result.data, cost)
    """
    raise NotImplementedError
