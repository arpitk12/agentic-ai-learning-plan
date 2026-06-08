"""Starter stub — Project 20: CrewAI agents."""
from __future__ import annotations

import os

from crewai import Agent, LLM

from src.tools.tools import TavilySearchTool, read_file_tool, write_file_tool


def _llm() -> LLM:
    # TODO 1: Return LLM(model=os.getenv("MODEL", ...), api_base=os.getenv("LITELLM_API_BASE"))
    raise NotImplementedError


def make_researcher() -> Agent:
    """Research analyst agent — searches the web and compiles facts."""
    # TODO 2: Return Agent(
    #   role="Senior Research Analyst",
    #   goal="...",
    #   backstory="...",
    #   tools=[TavilySearchTool(), read_file_tool],
    #   llm=_llm(),
    #   verbose=True,
    #   memory=True,
    # )
    raise NotImplementedError


def make_writer() -> Agent:
    """Content writer agent — transforms research into articles."""
    # TODO 3: Similar to make_researcher() but with writer role/goal/backstory
    raise NotImplementedError


def make_editor() -> Agent:
    """Senior editor agent — polishes grammar, flow, and clarity."""
    # TODO 4: Editor agent with no tools (reviews text only)
    raise NotImplementedError


def make_seo_analyst() -> Agent:
    """SEO strategist agent — keyword research and meta optimization."""
    # TODO 5: SEO agent with TavilySearchTool for keyword research
    raise NotImplementedError
