"""Starter stub — Project 22: GroupChat + custom speaker selection."""
from __future__ import annotations

import autogen

from src.agents.agents import (
    make_architect,
    make_developer,
    make_executor,
    make_product_manager,
    make_reviewer,
    make_tester,
)


def custom_speaker_selection(last_speaker: autogen.Agent, groupchat: autogen.GroupChat):
    """Deterministic speaker selection: PM → Architect → Developer → Tester → Executor → Reviewer → loop.

    Returns either an Agent object or the string "auto" to fall back to LLM selection.
    """
    # TODO 1: Build a dict {agent.name: agent} for all agents in groupchat.agents
    # TODO 2: Define the pipeline order list
    # TODO 3: If last_speaker.name is in order, return the NEXT agent in the order (mod len)
    # TODO 4: Return "auto" as fallback
    raise NotImplementedError


def run_team(task: str, max_rounds: int = 20, work_dir: str = "workspace") -> dict:
    """Run the full 5-agent AutoGen coding team.

    Returns dict with chat_history and final_status.
    """
    # TODO 5: Instantiate all 6 agents (pm, arch, dev, tester, executor, reviewer)
    # TODO 6: Create autogen.GroupChat(
    #   agents=[pm, arch, dev, tester, executor, reviewer],
    #   messages=[],
    #   max_round=max_rounds,
    #   speaker_selection_method=custom_speaker_selection,
    # )
    # TODO 7: Create autogen.GroupChatManager(groupchat=groupchat, llm_config=pm.llm_config)
    # TODO 8: executor.initiate_chat(manager, message=f"Team, we need to build: {task}")
    # TODO 9: Check final messages for TERMINATE / APPROVED / CHANGES_REQUESTED
    # TODO 10: Return {"chat_history": ..., "final_status": ..., "rounds_used": ...}
    raise NotImplementedError
