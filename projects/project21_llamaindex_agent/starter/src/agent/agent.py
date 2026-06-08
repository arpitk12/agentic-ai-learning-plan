"""Starter stub — Project 21: LlamaIndex ReActAgent."""
from __future__ import annotations


def build_react_agent(query_engine_tools: list, extra_tools: list | None = None):
    """Build a ReActAgent over document indexes.

    The ReActAgent iteratively:
    1. Reasons about which tool to use
    2. Acts (calls a tool)
    3. Observes the result
    4. Repeats until it has an answer
    """
    # TODO 1: from llama_index.core.agent import ReActAgent
    # TODO 2: Combine query_engine_tools + (extra_tools or [])
    # TODO 3: Return ReActAgent.from_tools(tools=all_tools, verbose=True, max_iterations=10, context="...")
    raise NotImplementedError


def make_calculator_tool():
    """Create a safe calculator as a LlamaIndex FunctionTool."""
    # TODO 4: from llama_index.core.tools import FunctionTool
    # TODO 5: Define a calculator(expression: str) -> str function using ast
    # TODO 6: Return FunctionTool.from_defaults(fn=calculator)
    raise NotImplementedError
