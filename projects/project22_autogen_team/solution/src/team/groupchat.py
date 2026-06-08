"""GroupChat orchestration for the AutoGen coding team."""
from __future__ import annotations

from typing import Callable

import autogen

from src.agents.agents import (
    make_architect,
    make_developer,
    make_executor,
    make_product_manager,
    make_reviewer,
    make_tester,
)


def custom_speaker_selection(last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent | str:
    """Deterministic speaker selection: PM → Architect → Developer → Tester → Executor → Reviewer → loop.

    Falls back to 'auto' (LLM-based) when the pattern doesn't match.
    """
    agents_by_name = {a.name: a for a in groupchat.agents}
    order = ["ProductManager", "Architect", "Developer", "Tester", "Executor", "Reviewer"]

    # If Reviewer says APPROVED, let executor wrap up
    last_msg = groupchat.messages[-1]["content"] if groupchat.messages else ""
    if "APPROVED" in last_msg and last_speaker.name == "Reviewer":
        return agents_by_name.get("Executor", "auto")

    # Follow the pipeline order
    if last_speaker.name in order:
        idx = order.index(last_speaker.name)
        next_name = order[(idx + 1) % len(order)]
        return agents_by_name.get(next_name, "auto")

    return "auto"


def run_team(task: str, max_rounds: int = 20, work_dir: str = "workspace") -> dict:
    """Run the full 5-agent coding team on a task.

    Returns:
        dict with 'chat_history' and 'final_status'.
    """
    pm = make_product_manager()
    arch = make_architect()
    dev = make_developer()
    tester = make_tester()
    executor = make_executor(work_dir=work_dir)
    reviewer = make_reviewer()

    groupchat = autogen.GroupChat(
        agents=[pm, arch, dev, tester, executor, reviewer],
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=custom_speaker_selection,
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=pm.llm_config,
    )

    # Initiate conversation from executor (human proxy initiates)
    executor.initiate_chat(
        manager,
        message=f"Team, we need to build the following:\n\n{task}\n\nProductManager, please start by defining the specifications.",
    )

    final_status = "COMPLETED"
    for msg in reversed(groupchat.messages):
        if "TERMINATE" in msg.get("content", ""):
            final_status = "TERMINATED"
            break
        if "CHANGES_REQUESTED" in msg.get("content", ""):
            final_status = "CHANGES_REQUESTED"
            break

    return {
        "chat_history": groupchat.messages,
        "final_status": final_status,
        "rounds_used": len(groupchat.messages),
    }
