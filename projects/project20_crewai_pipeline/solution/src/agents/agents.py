"""Four specialized CrewAI agents for the content pipeline."""
from __future__ import annotations

import os

from crewai import Agent, LLM

from src.tools.tools import TavilySearchTool, read_file_tool, write_file_tool


def _llm() -> LLM:
    return LLM(
        model=os.getenv("MODEL", "openai/gpt-4o-mini"),
        api_base=os.getenv("LITELLM_API_BASE"),
    )


def make_researcher() -> Agent:
    return Agent(
        role="Senior Research Analyst",
        goal="Gather comprehensive, accurate information on the given topic from reliable sources.",
        backstory=(
            "You are a meticulous research analyst with 10 years of experience distilling "
            "complex topics into clear, factual reports. You always cite your sources."
        ),
        tools=[TavilySearchTool(), read_file_tool],
        llm=_llm(),
        verbose=True,
        memory=True,
        max_iter=5,
    )


def make_writer() -> Agent:
    return Agent(
        role="Expert Content Writer",
        goal="Transform research findings into engaging, well-structured long-form articles.",
        backstory=(
            "You are a skilled writer who has published hundreds of in-depth articles. "
            "You excel at making complex topics accessible and compelling."
        ),
        tools=[write_file_tool],
        llm=_llm(),
        verbose=True,
        memory=True,
    )


def make_editor() -> Agent:
    return Agent(
        role="Senior Editor",
        goal="Polish articles for clarity, flow, grammar, and factual accuracy.",
        backstory=(
            "You have edited for top publications and have an eagle eye for inconsistencies, "
            "passive voice overuse, and logical gaps."
        ),
        tools=[],
        llm=_llm(),
        verbose=True,
    )


def make_seo_analyst() -> Agent:
    return Agent(
        role="SEO Content Strategist",
        goal="Optimize articles for search engine discoverability without sacrificing readability.",
        backstory=(
            "You are an SEO expert who understands both Google's ranking signals and "
            "user intent. You suggest data-driven optimizations."
        ),
        tools=[TavilySearchTool(max_results=3)],
        llm=_llm(),
        verbose=True,
    )
