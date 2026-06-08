"""Starter stub — Project 23: PydanticAI Policy Agent with dependency injection."""
from __future__ import annotations

from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from src.models import PolicyCheckResult

@dataclass
class PolicyDeps:
    """Dependencies injected at runtime — type-checked by PydanticAI."""
    policies: dict[str, str]    # policy_id → policy text
    document_type: str
    reviewer_id: str

# TODO 1: Create policy_agent = Agent(
#   model="openai:gpt-4o-mini",
#   result_type=PolicyCheckResult,
#   deps_type=PolicyDeps,
# )

# TODO 2: Add @policy_agent.system_prompt (async, receives RunContext[PolicyDeps])
#   Build a dynamic system prompt using ctx.deps.policies and ctx.deps.document_type

# TODO 3: Add @policy_agent.tool named "get_policy_text"
#   It receives (ctx: RunContext[PolicyDeps], policy_id: str) and returns the policy text

async def check_document_policies(
    document_content: str,
    policies: dict[str, str],
    document_type: str,
    reviewer_id: str = "system",
) -> tuple[PolicyCheckResult, float]:
    """
    TODO 4: Create PolicyDeps(policies=..., document_type=..., reviewer_id=...)
    TODO 5: Run policy_agent.run(document_content, deps=deps)
    TODO 6: Return (result.data, cost)
    """
    raise NotImplementedError
