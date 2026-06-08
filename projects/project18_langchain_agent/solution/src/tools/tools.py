"""Custom tools for the LangChain research agent."""
from __future__ import annotations

import ast
import operator
from typing import Optional

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Input should be a valid math expression like '15 * 23 + 7' or '(100 / 4) ** 2'.
    Supports: +, -, *, /, **, parentheses, integers and floats."""
    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def wikipedia_search(query: str) -> str:
    """Look up information on Wikipedia. Use for factual background knowledge,
    definitions, historical context, or topics where accuracy matters.
    Input: a clear search query."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        page = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return page
    except wikipedia.exceptions.DisambiguationError as e:
        # Try first disambiguation option
        try:
            page = wikipedia.summary(e.options[0], sentences=5)
            return f"(Showing results for '{e.options[0]}')\n{page}"
        except Exception:
            return f"Ambiguous query. Options: {e.options[:5]}"
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


def get_tavily_tool(api_key: str, max_results: int = 5):
    """Build a Tavily web search tool."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    return TavilySearchResults(
        max_results=max_results,
        api_key=api_key,
        description=(
            "Search the web for current information, news, and recent events. "
            "Use for topics that require up-to-date information."
        ),
    )


def get_rag_tool(retriever):
    """Wrap a LangChain retriever as a tool."""
    @tool
    def rag_search(query: str) -> str:
        """Search the internal knowledge base for relevant information.
        Use this first before web search for questions about the documented domain."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found in knowledge base."
        return "\n\n".join(f"[Source {i+1}]: {d.page_content}" for i, d in enumerate(docs))
    return rag_search
