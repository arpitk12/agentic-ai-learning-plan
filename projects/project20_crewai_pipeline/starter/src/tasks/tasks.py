"""Starter stub — Project 20: CrewAI tasks with Pydantic output."""
from __future__ import annotations

from crewai import Task

from src.agents.agents import make_editor, make_researcher, make_seo_analyst, make_writer
from src.models import Article, EditedArticle, ResearchReport, SEOAnalysis


def make_research_task(topic: str, researcher) -> Task:
    """Task for the researcher agent.

    Key fields:
    - description: detailed instructions
    - expected_output: human-readable description of what's expected
    - agent: the agent that executes this task
    - output_pydantic: ResearchReport (enforces structured output)
    """
    # TODO 1: Return Task(description=..., expected_output=..., agent=researcher, output_pydantic=ResearchReport)
    raise NotImplementedError


def make_writing_task(writer, research_task: Task) -> Task:
    """Task for the writer agent.

    Important: use context=[research_task] so the writer gets the research output.
    """
    # TODO 2: Return Task(..., context=[research_task], output_pydantic=Article)
    raise NotImplementedError


def make_editing_task(editor, writing_task: Task) -> Task:
    """Task for the editor agent."""
    # TODO 3: Return Task(..., context=[writing_task], output_pydantic=EditedArticle)
    raise NotImplementedError


def make_seo_task(seo_analyst, editing_task: Task, research_task: Task) -> Task:
    """Task for the SEO analyst agent.

    Note: this task needs context from BOTH editing and research tasks.
    """
    # TODO 4: Return Task(..., context=[editing_task, research_task], output_pydantic=SEOAnalysis)
    raise NotImplementedError
