"""AutoGen agents for the multi-agent coding team."""
from __future__ import annotations

import os
from typing import Any

import autogen


def _llm_config(model: str | None = None) -> dict[str, Any]:
    """Build an LLM config dict compatible with AutoGen."""
    return {
        "config_list": [
            {
                "model": model or os.getenv("MODEL", "gpt-4o-mini"),
                "api_base": os.getenv("LITELLM_API_BASE", "http://localhost:4000"),
                "api_key": os.getenv("API_KEY", "dummy"),
                "api_type": "openai",
            }
        ],
        "temperature": 0.1,
        "cache_seed": None,  # disable caching for fresh responses
    }


def make_product_manager() -> autogen.AssistantAgent:
    return autogen.AssistantAgent(
        name="ProductManager",
        system_message=(
            "You are a Product Manager. Translate user requirements into clear, "
            "specific technical specifications. Break complex features into "
            "well-defined tasks. Prioritize ruthlessly. Ask clarifying questions "
            "when requirements are ambiguous. End your messages with 'PM_DONE' "
            "when specifications are complete."
        ),
        llm_config=_llm_config(),
    )


def make_architect() -> autogen.AssistantAgent:
    return autogen.AssistantAgent(
        name="Architect",
        system_message=(
            "You are a Senior Software Architect. Design clean, scalable solutions. "
            "Propose the right data structures, design patterns, and architecture. "
            "Justify technical decisions. Review code for architectural compliance. "
            "Focus on: SOLID principles, separation of concerns, testability."
        ),
        llm_config=_llm_config(),
    )


def make_developer() -> autogen.AssistantAgent:
    return autogen.AssistantAgent(
        name="Developer",
        system_message=(
            "You are an expert Python developer. Write clean, idiomatic, well-documented "
            "code based on the architect's design. Include type hints, docstrings, and "
            "error handling. Follow the existing code style. Write code blocks that are "
            "ready to execute — no placeholders."
        ),
        llm_config=_llm_config(),
    )


def make_tester() -> autogen.AssistantAgent:
    return autogen.AssistantAgent(
        name="Tester",
        system_message=(
            "You are a QA Engineer. Write comprehensive pytest test suites for code "
            "written by the Developer. Cover: happy path, edge cases, error cases. "
            "Use fixtures, parametrize, and mocks appropriately. "
            "Aim for >80% coverage."
        ),
        llm_config=_llm_config(),
    )


def make_reviewer() -> autogen.AssistantAgent:
    return autogen.AssistantAgent(
        name="Reviewer",
        system_message=(
            "You are a Senior Code Reviewer. Review code for: bugs, security issues, "
            "performance problems, code smells, and missing test coverage. "
            "Be specific — reference exact lines. Suggest concrete fixes. "
            "End with APPROVED or CHANGES_REQUESTED."
        ),
        llm_config=_llm_config(),
    )


def make_executor(work_dir: str = "workspace") -> autogen.UserProxyAgent:
    """UserProxyAgent with local code execution (no Docker for simplicity).

    Switch to DockerCommandLineCodeExecutor for production isolation.
    """
    from autogen.coding import LocalCommandLineCodeExecutor
    executor = LocalCommandLineCodeExecutor(
        timeout=60,
        work_dir=work_dir,
    )
    return autogen.UserProxyAgent(
        name="Executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config={"executor": executor},
        system_message=(
            "Execute code blocks provided by the Developer and Tester. "
            "Report the exact stdout/stderr output. "
            "Reply TERMINATE when the task is fully complete and tests pass."
        ),
    )
