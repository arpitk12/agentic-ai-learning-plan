"""PydanticAI Risk Analysis Agent."""
from __future__ import annotations

import os

from pydantic_ai import Agent, ModelRetry, RunContext

from src.models import RiskAssessment


risk_agent = Agent(
    model=f"openai:{os.getenv('MODEL', 'gpt-4o-mini')}",
    result_type=RiskAssessment,
    system_prompt="""You are a senior compliance risk analyst with expertise in:
- GDPR and EU data protection law
- SOX financial reporting requirements
- HIPAA healthcare data privacy
- PCI-DSS payment card security
- Contract law and liability exposure

Analyze the document thoroughly. Always provide:
1. An accurate risk_level (low/medium/high/critical)
2. Specific, actionable risk_factors
3. Named regulatory_concerns with article/section references
4. Clear reasoning for your assessment

For CRITICAL risk: require human review. For HIGH risk: recommend human review.""",
)


@risk_agent.result_validator
async def validate_risk_result(ctx: RunContext, result: RiskAssessment) -> RiskAssessment:
    """Enforce quality thresholds — retry if assessment is weak."""
    if result.risk_level == "critical" and result.confidence < 0.75:
        raise ModelRetry(
            "CRITICAL assessment requires confidence >= 0.75. "
            "Provide specific regulatory article citations and retry."
        )
    if not result.reasoning or len(result.reasoning) < 50:
        raise ModelRetry("Reasoning must be at least 50 characters. Elaborate on your assessment.")
    return result


async def assess_document_risk(document_content: str, document_type: str) -> tuple[RiskAssessment, float]:
    """Run risk assessment and return (result, cost_usd)."""
    result = await risk_agent.run(
        f"Document type: {document_type}\n\n{document_content}"
    )
    usage = result.usage()
    cost = (usage.total_tokens or 0) * 0.00000015  # gpt-4o-mini blended rate
    return result.data, cost
