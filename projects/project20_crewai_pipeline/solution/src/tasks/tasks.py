"""Four CrewAI tasks — each maps to one agent, with Pydantic output models."""
from __future__ import annotations

from crewai import Task

from src.agents.agents import make_editor, make_researcher, make_seo_analyst, make_writer
from src.models import Article, EditedArticle, ResearchReport, SEOAnalysis


def make_research_task(topic: str, researcher) -> Task:
    return Task(
        description=f"""Research the topic: "{topic}".
        - Search for recent developments, statistics, and expert opinions.
        - Identify at least 5 reliable sources.
        - Extract key facts and compile a structured research report.""",
        expected_output="A detailed research report with key facts, sources, and a 300-word summary.",
        agent=researcher,
        output_pydantic=ResearchReport,
    )


def make_writing_task(writer, research_task: Task) -> Task:
    return Task(
        description="""Using the research report provided, write a comprehensive 1200-word article.
        Structure it with a compelling introduction, 3-4 body sections with subheadings,
        and a strong conclusion. Make it engaging and accessible to a general audience.""",
        expected_output="A complete 1200-word article with title, introduction, body sections, and conclusion.",
        agent=writer,
        context=[research_task],           # ← gets researcher output
        output_pydantic=Article,
    )


def make_editing_task(editor, writing_task: Task) -> Task:
    return Task(
        description="""Edit the draft article for:
        1. Grammar, spelling, and punctuation
        2. Sentence variety and flow
        3. Passive voice reduction
        4. Factual consistency with the research
        List all changes you make.""",
        expected_output="Polished article in markdown format with a list of changes made.",
        agent=editor,
        context=[writing_task],
        output_pydantic=EditedArticle,
    )


def make_seo_task(seo_analyst, editing_task: Task, research_task: Task) -> Task:
    return Task(
        description="""Perform an SEO analysis of the edited article. Provide:
        - Primary and secondary keyword recommendations
        - Optimized meta title (max 60 chars) and meta description (max 160 chars)
        - Suggested heading structure improvements
        - Overall SEO score (0-100) with justification
        - 3-5 actionable recommendations""",
        expected_output="Complete SEO analysis with keywords, meta tags, score, and recommendations.",
        agent=seo_analyst,
        context=[editing_task, research_task],
        output_pydantic=SEOAnalysis,
    )
