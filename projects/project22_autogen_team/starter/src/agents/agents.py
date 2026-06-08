"""Starter stub — Project 22: AutoGen agents."""
from __future__ import annotations

import os


def _llm_config(model: str | None = None) -> dict:
    """Build an AutoGen LLM config dict.

    AutoGen expects: {"config_list": [{"model": ..., "api_base": ..., "api_key": ...}]}
    """
    # TODO 1: Return the config dict using os.getenv("MODEL"), os.getenv("LITELLM_API_BASE")
    raise NotImplementedError


def make_product_manager():
    """ProductManager AssistantAgent — translates requirements into specifications."""
    # TODO 2: import autogen
    # TODO 3: Return autogen.AssistantAgent(
    #   name="ProductManager",
    #   system_message="You are a PM...",
    #   llm_config=_llm_config(),
    # )
    raise NotImplementedError


def make_architect():
    """Architect AssistantAgent — designs solution structure."""
    # TODO 4: Similar to above with Architect role and system message about SOLID principles
    raise NotImplementedError


def make_developer():
    """Developer AssistantAgent — writes complete runnable code."""
    # TODO 5: Developer agent with system message emphasizing no placeholders
    raise NotImplementedError


def make_tester():
    """Tester AssistantAgent — writes comprehensive pytest tests."""
    # TODO 6: Tester agent with system message about fixtures, parametrize, mocks
    raise NotImplementedError


def make_reviewer():
    """Reviewer AssistantAgent — code review, returns APPROVED or CHANGES_REQUESTED."""
    # TODO 7: Reviewer agent that ends messages with APPROVED or CHANGES_REQUESTED
    raise NotImplementedError


def make_executor(work_dir: str = "workspace"):
    """UserProxyAgent with local code execution.

    human_input_mode="NEVER" means it runs fully automatically.
    is_termination_msg checks for TERMINATE in the last message.
    """
    # TODO 8: from autogen.coding import LocalCommandLineCodeExecutor
    # TODO 9: Return autogen.UserProxyAgent(
    #   name="Executor",
    #   human_input_mode="NEVER",
    #   max_consecutive_auto_reply=5,
    #   code_execution_config={"executor": LocalCommandLineCodeExecutor(timeout=60, work_dir=work_dir)},
    # )
    raise NotImplementedError
