"""
Exercise 4: CrewAI Multi-Agent Research Pipeline
Guide Section: §2.5 — CrewAI: Role-Based Multi-Agent Orchestration

Goal: Build a 3-agent team (Researcher → Writer → Reviewer) where each agent
has a specialist role, and tasks pass output to the next agent automatically.

Why CrewAI?
- Role-based: describe agents in human terms ("Senior Research Analyst")
- Task dependencies: writer automatically receives researcher's output
- Simpler than LangGraph for sequential pipelines with clear role separation
- Best for: content pipelines, research workflows, structured team tasks

vs LangGraph (week3/ex1): LangGraph gives you explicit state + conditional routing;
CrewAI gives you a higher-level role abstraction with easier sequential setup.

pip install crewai
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process


# ─── Step 1: Define Specialist Agents ─────────────────────────────────────────
# Each agent = role (who) + goal (what drives them) + backstory (LLM persona)
# The backstory shapes how the LLM responds — more specific = better results.

def build_research_crew(topic: str) -> Crew:
    model = os.getenv("MODEL", "gemini/gemini-2.0-flash")

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover accurate, comprehensive, and current information on the given topic",
        backstory="""You are an expert researcher with 15 years of experience synthesizing
        complex information from multiple sources. You prioritize accuracy, always note
        your confidence level, and structure findings so they are easy to reference.""",
        llm=model,
        verbose=True,
        max_iter=3,  # max reasoning loops before the agent stops
    )

    writer = Agent(
        role="Technical Content Writer",
        goal="Transform research findings into clear, engaging technical content for engineers",
        backstory="""You are a technical writer who has produced documentation and blog posts
        for Google, AWS, and OpenAI. You excel at making complex concepts accessible to
        software engineers without sacrificing accuracy. You always use concrete examples.""",
        llm=model,
        verbose=True,
    )

    reviewer = Agent(
        role="Quality Assurance Critic",
        goal="Ensure technical accuracy, completeness, and clarity of all written content",
        backstory="""You are a meticulous editor and technical expert. You catch logical
        errors, missing context, factual inaccuracies, and unclear explanations.
        Your reviews are structured, specific, and actionable.""",
        llm=model,
        verbose=True,
    )

    # ─── Step 2: Define Tasks with Dependencies ─────────────────────────────────
    # Each task specifies WHAT to do, WHO does it, and WHAT output to expect.
    # context=[previous_task] means this agent gets the prior task's output.

    research_task = Task(
        description=f"""Research the following topic thoroughly: {topic}

        Your output MUST include:
        - 3-5 key concepts with clear, technical definitions
        - Current state and notable developments (2024-2025)
        - Practical use cases and real-world examples
        - Key limitations, trade-offs, or common misconceptions
        - Confidence level for each section: HIGH / MEDIUM / LOW""",
        expected_output=(
            "A structured research report (600-800 words) covering key concepts, "
            "recent developments, practical applications, trade-offs, and confidence levels."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=f"""Write a technical blog post about: {topic}

        Use the research provided by the Research Analyst as your factual basis.
        Structure:
        1. Hook introduction — why this topic matters to engineers right now
        2. Core concept explanation with at least one concrete analogy
        3. Three practical use cases with code snippets or pseudocode
        4. Comparison with alternatives — when to use this, when not to
        5. Conclusion with 3 clear takeaways

        Target audience: Senior software engineers who learn by doing.""",
        expected_output=(
            "A polished blog post (500-700 words) with a title, clear sections, "
            "at least one code snippet, and actionable developer insights."
        ),
        agent=writer,
        context=[research_task],  # writer automatically receives researcher's full output
    )

    review_task = Task(
        description=f"""Review the blog post about: {topic}

        Cross-check the post against the original research findings.
        Evaluate:
        - Technical accuracy: Is everything factually correct?
        - Clarity: Can a senior engineer understand every paragraph?
        - Completeness: Are there obvious gaps or missing context?
        - Quality: Is the writing professional, engaging, and well-structured?

        Output format:
        1. Overall quality score: N/10
        2. Issues found (severity: HIGH / MEDIUM / LOW, with specific location)
        3. Final improved version incorporating all your corrections""",
        expected_output=(
            "A quality review with a score, structured issues list, "
            "and the final corrected blog post ready for publication."
        ),
        agent=reviewer,
        context=[research_task, writing_task],  # reviewer sees both prior outputs
    )

    # ─── Step 3: Assemble the Crew ───────────────────────────────────────────────
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,  # tasks execute in order; each gets prior output
        verbose=True,
    )

    return crew


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # TODO: Change the topic to something you want to learn about
    topic = "LangGraph for building stateful AI agent workflows"

    print(f"\n{'='*60}")
    print(f"Starting CrewAI Pipeline")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")

    crew = build_research_crew(topic)
    result = crew.kickoff(inputs={"topic": topic})

    print(f"\n{'='*60}")
    print("FINAL OUTPUT (Reviewed & Corrected Blog Post):")
    print(f"{'='*60}")
    print(result.raw)

    if hasattr(result, "token_usage") and result.token_usage:
        print(f"\nToken Usage: {result.token_usage}")

    # ─── CHALLENGES ───────────────────────────────────────────────────────────
    # CHALLENGE 1: Switch to Process.hierarchical
    #   A manager LLM dynamically delegates tasks to agents.
    #   crew = Crew(..., process=Process.hierarchical, manager_llm=model)
    #
    # CHALLENGE 2: Add a 4th agent — "Fact Checker"
    #   Role: "Senior Fact Checker"
    #   Goal: Verify all specific claims with web search
    #   (add crewai-tools: from crewai_tools import SerperDevTool)
    #
    # CHALLENGE 3: Add output_file to each task to save results to disk
    #   Task(..., output_file="research.md")
    #
    # CHALLENGE 4: Change the topic to "RAG vs Fine-tuning: when to use each"
    #   and compare the output quality across multiple runs.
