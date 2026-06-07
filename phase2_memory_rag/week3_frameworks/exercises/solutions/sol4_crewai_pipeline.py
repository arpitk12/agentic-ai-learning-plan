"""
SOLUTION — Exercise 4: CrewAI Multi-Agent Research Pipeline

Key concepts demonstrated:
- Agent: role + goal + backstory → shapes LLM persona
- Task: description + expected_output + agent
- context=[task]: pass prior task output to next task automatically
- Process.sequential: tasks run in order
- Process.hierarchical: manager LLM dynamically delegates tasks

pip install crewai
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process


def build_research_crew(topic: str) -> Crew:
    model = os.getenv("MODEL", "gemini/gemini-2.0-flash")

    # ── Agents ────────────────────────────────────────────────────────────────
    # Backstories are crucial — the more specific, the better the persona.
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover accurate, comprehensive, and current information on the given topic",
        backstory=(
            "You are an expert researcher with 15 years of experience synthesizing "
            "complex information from multiple sources. You always note confidence "
            "levels (HIGH/MEDIUM/LOW) and structure findings for easy reference."
        ),
        llm=model,
        verbose=True,
        max_iter=3,
    )

    writer = Agent(
        role="Technical Content Writer",
        goal="Transform research findings into clear, engaging content for senior engineers",
        backstory=(
            "You have written documentation for Google, AWS, and OpenAI. You excel at "
            "making complex concepts accessible without sacrificing accuracy. "
            "You always include concrete code examples and analogies."
        ),
        llm=model,
        verbose=True,
    )

    reviewer = Agent(
        role="Quality Assurance Critic",
        goal="Ensure technical accuracy, completeness, and clarity of all content",
        backstory=(
            "You are a meticulous editor who catches logical errors, missing context, "
            "and factual inaccuracies. Your reviews are structured and actionable, "
            "always providing the corrected version alongside your critique."
        ),
        llm=model,
        verbose=True,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────
    # context=[prior_task] → the agent automatically receives that task's output.
    research_task = Task(
        description=(
            f"Research the following topic thoroughly: {topic}\n\n"
            "Your output MUST include:\n"
            "- 3-5 key concepts with clear technical definitions\n"
            "- Current state and notable developments (2024-2025)\n"
            "- Practical use cases with real-world examples\n"
            "- Key limitations and trade-offs\n"
            "- Confidence level for each section: HIGH / MEDIUM / LOW"
        ),
        expected_output=(
            "A structured research report (600-800 words) covering key concepts, "
            "recent developments, practical applications, trade-offs, and confidence levels."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Write a technical blog post about: {topic}\n\n"
            "Use the researcher's findings as your factual basis.\n"
            "Structure:\n"
            "1. Hook — why this topic matters to engineers right now\n"
            "2. Core concept with at least one concrete analogy\n"
            "3. Three practical use cases with code snippets or pseudocode\n"
            "4. When to use this vs alternatives\n"
            "5. Three clear takeaways\n\n"
            "Target: senior software engineers who learn by doing."
        ),
        expected_output=(
            "A polished blog post (500-700 words) with a title, clear sections, "
            "at least one code snippet, and actionable developer insights."
        ),
        agent=writer,
        context=[research_task],  # receives researcher output automatically
    )

    review_task = Task(
        description=(
            f"Review the blog post about: {topic}\n\n"
            "Cross-check all claims against the original research.\n"
            "Evaluate:\n"
            "1. Technical accuracy — is everything factually correct?\n"
            "2. Clarity — can any senior engineer understand every paragraph?\n"
            "3. Completeness — are there obvious gaps?\n"
            "4. Code — is the snippet correct and runnable?\n\n"
            "Output:\n"
            "1. Overall quality score: N/10\n"
            "2. Issues found (severity: HIGH / MEDIUM / LOW + specific location)\n"
            "3. Final corrected blog post with all issues fixed"
        ),
        expected_output=(
            "A quality review with score, issues list, and the final "
            "corrected blog post ready for publication."
        ),
        agent=reviewer,
        context=[research_task, writing_task],  # sees both prior outputs
    )

    return Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    topic = "LangGraph for building stateful AI agent workflows"

    print(f"\n{'='*60}")
    print(f"CrewAI Pipeline — Topic: {topic}")
    print(f"{'='*60}\n")

    crew = build_research_crew(topic)
    result = crew.kickoff(inputs={"topic": topic})

    print(f"\n{'='*60}")
    print("FINAL OUTPUT (Reviewed Blog Post):")
    print(f"{'='*60}")
    print(result.raw)

    if hasattr(result, "token_usage") and result.token_usage:
        print(f"\nToken usage: {result.token_usage}")
