"""LlamaIndex ReActAgent with custom tools over document indexes."""
from __future__ import annotations

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool


def build_react_agent(
    query_engine_tools: list[QueryEngineTool],
    extra_tools: list | None = None,
) -> ReActAgent:
    """Build a ReActAgent that can query documents and use custom tools.

    Args:
        query_engine_tools: List of QueryEngineTool wrapping vector/summary engines.
        extra_tools: Optional additional FunctionTools.

    Returns:
        Configured ReActAgent ready for inference.
    """
    all_tools = list(query_engine_tools) + (extra_tools or [])

    return ReActAgent.from_tools(
        tools=all_tools,
        verbose=True,
        max_iterations=10,
        context=(
            "You are a research assistant with access to a knowledge base. "
            "Use the available tools to answer questions accurately. "
            "Always cite your sources from the retrieved context."
        ),
    )


# ── Custom function tools ────────────────────────────────────────────────────

def make_calculator_tool() -> FunctionTool:
    """Simple safe calculator as a FunctionTool."""
    import ast

    def calculator(expression: str) -> str:
        """Evaluate a simple math expression safely. Input: expression string."""
        try:
            result = ast.literal_eval(expression)
            return str(result)
        except Exception:
            try:
                # Allow basic arithmetic only
                allowed = set("0123456789.+-*/() ")
                if all(c in allowed for c in expression):
                    return str(eval(expression))  # noqa: S307
                return "Expression contains disallowed characters"
            except Exception as e:
                return f"Calculation error: {e}"

    return FunctionTool.from_defaults(fn=calculator)


def make_summary_tool(index) -> QueryEngineTool:
    """Wrap an index's summary engine as a named tool."""
    return QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(response_mode="tree_summarize"),
        name="document_summarizer",
        description="Summarizes documents or sections. Use for 'summarize' or 'overview' queries.",
    )
