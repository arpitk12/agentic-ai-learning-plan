"""Starter stub — Project 20: Assemble and run the CrewAI pipeline."""
from __future__ import annotations

from crewai import Crew, Process

from src.agents.agents import make_editor, make_researcher, make_seo_analyst, make_writer
from src.tasks.tasks import make_editing_task, make_research_task, make_seo_task, make_writing_task


def run_sequential_pipeline(topic: str) -> dict:
    """Build and run a sequential CrewAI pipeline.

    Steps:
    1. Instantiate all 4 agents
    2. Create all 4 tasks (with proper context chaining)
    3. Create Crew(process=Process.sequential, memory=True)
    4. Call crew.kickoff(inputs={"topic": topic})
    5. Return dict with each task's pydantic output
    """
    # TODO 1: Create all agents
    # TODO 2: Create all tasks (pass previous task as context)
    # TODO 3: Create Crew with sequential process and memory=True
    # TODO 4: crew.kickoff(inputs={"topic": topic})
    # TODO 5: Return {"research": ..., "article": ..., "edited": ..., "seo": ...}
    raise NotImplementedError


async def run_async_pipeline(topic: str) -> dict:
    """Async version using crew.kickoff_async()."""
    # TODO 6: Same setup as above but use await crew.kickoff_async(inputs={"topic": topic})
    raise NotImplementedError
