"""Two-agent nested chat pattern — simplest AutoGen pattern."""
from __future__ import annotations

import os

import autogen


def run_two_agent_chat(task: str) -> str:
    """Classic two-agent pattern: AssistantAgent + UserProxyAgent.

    The UserProxy terminates when it sees TERMINATE in the assistant's reply.
    Returns the final response.
    """
    llm_config = {
        "config_list": [{
            "model": os.getenv("MODEL", "gpt-4o-mini"),
            "api_base": os.getenv("LITELLM_API_BASE", "http://localhost:4000"),
            "api_key": os.getenv("API_KEY", "dummy"),
            "api_type": "openai",
        }],
        "temperature": 0.1,
    }

    assistant = autogen.AssistantAgent(
        name="CodingAssistant",
        system_message=(
            "You are a helpful coding assistant. Write complete, runnable Python code. "
            "After writing code, add TERMINATE to signal you are done."
        ),
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config={
            "executor": autogen.coding.LocalCommandLineCodeExecutor(
                timeout=30,
                work_dir="workspace",
            )
        },
    )

    result = user_proxy.initiate_chat(
        recipient=assistant,
        message=task,
        summary_method="reflection_with_llm",
    )

    return result.summary


def run_nested_chat_with_carryover(tasks: list[str]) -> list[str]:
    """Demonstrate nested chats with carryover — previous results flow into next chat.

    Args:
        tasks: List of sequential tasks, each building on the previous.

    Returns:
        List of summaries for each task.
    """
    llm_config = {
        "config_list": [{
            "model": os.getenv("MODEL", "gpt-4o-mini"),
            "api_base": os.getenv("LITELLM_API_BASE", "http://localhost:4000"),
            "api_key": os.getenv("API_KEY", "dummy"),
        }],
    }

    assistant = autogen.AssistantAgent(
        name="CodingAssistant",
        system_message="You are a helpful coding assistant. Append TERMINATE when done.",
        llm_config=llm_config,
    )
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
    )

    summaries = []
    for task in tasks:
        result = user_proxy.initiate_chat(
            recipient=assistant,
            message=task,
            carryover=summaries[-1] if summaries else None,
            summary_method="last_msg",
            max_turns=5,
        )
        summaries.append(result.summary)

    return summaries
