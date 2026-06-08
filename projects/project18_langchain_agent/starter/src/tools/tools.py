"""Starter stub — Project 18: LangChain tools.

Implement each tool using the @tool decorator and LangChain tool helpers.
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Input: math expression string."""
    # TODO 1: Use ast.parse() to validate the expression contains only safe nodes
    #         (Num, BinOp, UnaryOp, Add, Sub, Mul, Div, Pow, Mod, USub)
    # TODO 2: If safe, evaluate with eval() and return str(result)
    # TODO 3: Return "Unsafe expression" for anything containing disallowed nodes
    raise NotImplementedError


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return the first 500 characters of the top result."""
    # TODO 4: pip install wikipedia-api
    # TODO 5: Create a wikipedia.Wikipedia("ResearchBot/1.0", "en") client
    # TODO 6: Call .page(query) and return page.summary[:500] or a "not found" message
    raise NotImplementedError


def get_tavily_tool():
    """Return a pre-built Tavily search tool.

    Requires TAVILY_API_KEY env var.
    Returns a BaseTool compatible with AgentExecutor.
    """
    # TODO 7: from langchain_community.tools.tavily_search import TavilySearchResults
    # TODO 8: Return TavilySearchResults(max_results=5)
    raise NotImplementedError


def get_rag_tool(retriever):
    """Wrap a retriever as a @tool that returns formatted document context."""
    # TODO 9: Define a nested function rag_search(query: str) -> str
    #         that calls retriever.invoke(query) and formats the results
    # TODO 10: Decorate it with @tool and return it
    raise NotImplementedError
