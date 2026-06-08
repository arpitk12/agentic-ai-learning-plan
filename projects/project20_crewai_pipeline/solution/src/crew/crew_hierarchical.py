"""Hierarchical CrewAI crew — manager LLM coordinates all agents."""
from __future__ import annotations

import os

from crewai import Agent, Crew, LLM, Process

from src.agents.agents import make_editor, make_researcher, make_seo_analyst, make_writer
from src.tasks.tasks import make_editing_task, make_research_task, make_seo_task, make_writing_task


def run_hierarchical_pipeline(topic: str) -> dict:
    """Hierarchical crew: a manager agent orchestrates worker agents.

    Key difference from sequential:
    - manager_llm decides task assignment order dynamically
    - Agents can be called multiple times or skipped
    - Better for complex, non-linear workflows
    """
    researcher = make_researcher()
    writer = make_writer()
    editor = make_editor()
    seo_analyst = make_seo_analyst()

    research_task = make_research_task(topic, researcher)
    writing_task = make_writing_task(writer, research_task)
    editing_task = make_editing_task(editor, writing_task)
    seo_task = make_seo_task(seo_analyst, editing_task, research_task)

    crew = Crew(
        agents=[researcher, writer, editor, seo_analyst],
        tasks=[research_task, writing_task, editing_task, seo_task],
        process=Process.hierarchical,
        manager_llm=LLM(
            model=os.getenv("MANAGER_MODEL", os.getenv("MODEL", "openai/gpt-4o")),
            api_base=os.getenv("LITELLM_API_BASE"),
        ),
        memory=True,
        verbose=True,
    )

    result = crew.kickoff(inputs={"topic": topic})
    return {"raw": result.raw, "usage": result.token_usage}
