"""PydanticAI Policy Check Agent with dependency injection."""
from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from src.models import PolicyCheckResult, PolicyViolation


@dataclass
class PolicyDeps:
    """Dependencies injected into the policy agent at runtime."""
    policies: dict[str, str]     # policy_id → full policy text
    document_type: str
    reviewer_id: str


policy_agent = Agent(
    model=f"openai:{os.getenv('MODEL', 'gpt-4o-mini')}",
    result_type=PolicyCheckResult,
    deps_type=PolicyDeps,
)


@policy_agent.system_prompt
async def build_policy_system_prompt(ctx: RunContext[PolicyDeps]) -> str:
    policy_names = ", ".join(ctx.deps.policies.keys())
    return (
        f"You are a compliance officer checking a {ctx.deps.document_type} document "
        f"against {len(ctx.deps.policies)} company policies: {policy_names}. "
        "For each violation, identify the policy ID, describe the violation clearly, "
        "rate its severity, and provide remediation steps. "
        "Score compliance from 0-100 where 100 is fully compliant."
    )


@policy_agent.tool
async def get_policy_text(ctx: RunContext[PolicyDeps], policy_id: str) -> str:
    """Retrieve full text of a specific policy by ID."""
    return ctx.deps.policies.get(policy_id, f"Policy {policy_id} not found in database.")


async def check_document_policies(
    document_content: str,
    policies: dict[str, str],
    document_type: str,
    reviewer_id: str = "system",
) -> tuple[PolicyCheckResult, float]:
    """Run policy check and return (result, cost_usd)."""
    deps = PolicyDeps(
        policies=policies,
        document_type=document_type,
        reviewer_id=reviewer_id,
    )
    result = await policy_agent.run(document_content, deps=deps)
    usage = result.usage()
    cost = (usage.total_tokens or 0) * 0.00000015
    return result.data, cost
