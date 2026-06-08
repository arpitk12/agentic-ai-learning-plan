"""Sequential CrewAI pipeline: Researcher → Writer → Editor → SEO Analyst."""
from __future__ import annotations

from crewai import Crew, Process

from src.agents.agents import make_editor, make_researcher, make_seo_analyst, make_writer
from src.tasks.tasks import make_editing_task, make_research_task, make_seo_task, make_writing_task


def run_sequential_pipeline(topic: str) -> dict:
    """Run the full content pipeline sequentially.

    Returns a dict with results from each agent.
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
        process=Process.sequential,
        memory=True,          # shared memory across agents
        verbose=True,
        output_log_file="output/crew_log.txt",
    )

    result = crew.kickoff(inputs={"topic": topic})
    return {
        "research": result.tasks_output[0].pydantic if result.tasks_output else None,
        "article": result.tasks_output[1].pydantic if len(result.tasks_output) > 1 else None,
        "edited": result.tasks_output[2].pydantic if len(result.tasks_output) > 2 else None,
        "seo": result.tasks_output[3].pydantic if len(result.tasks_output) > 3 else None,
        "raw": result.raw,
    }


async def run_async_pipeline(topic: str) -> dict:
    """Async variant — useful for FastAPI integration."""
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
        process=Process.sequential,
        memory=True,
        verbose=True,
    )

    result = await crew.kickoff_async(inputs={"topic": topic})
    return {"raw": result.raw, "usage": result.token_usage}
