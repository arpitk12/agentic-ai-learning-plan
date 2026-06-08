# CrewAI Content Pipeline — Build Guide

## Prerequisites
```bash
pip install -r requirements.txt
# Tavily API key: https://tavily.com
```

---

## Phase 1 — Tools

### 1.1 BaseTool subclass
```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information on any topic."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        from tavily import TavilyClient
        client = TavilyClient(api_key=cfg.tavily_api_key)
        results = client.search(query, max_results=5)
        return "\n\n".join(r["content"] for r in results["results"])
```

### 1.2 `@tool` decorator (simpler)
```python
from crewai.tools import tool

@tool("file_writer")
def write_file(filename: str, content: str) -> str:
    """Write content to a file. Input: filename and content string."""
    Path(filename).write_text(content)
    return f"Written {len(content)} chars to {filename}"
```

**Checkpoint:** `python -c "from src.tools.search_tool import TavilySearchTool; print(TavilySearchTool()._run('AI news 2025')[:200])"`

---

## Phase 2 — Agents

### 2.1 Define agents
```python
from crewai import Agent
from langchain_litellm import ChatLiteLLM

llm = ChatLiteLLM(model=cfg.model, temperature=0.3)

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find accurate, current, and relevant information about {topic}",
    backstory="""You are a world-class research analyst with 20 years of experience.
    You dig deep into topics, verify facts from multiple sources, and provide
    comprehensive research reports. You never make up information.""",
    tools=[TavilySearchTool(), WebsiteReadTool()],
    llm=llm,
    verbose=True,
    max_iter=5,         # max tool calls before forced output
    memory=True,        # remember within this agent's context
)

writer = Agent(
    role="Content Writer",
    goal="Write compelling, accurate articles based on research findings about {topic}",
    backstory="""You are an expert content writer specializing in technology and science.
    You transform research data into engaging narratives. You always cite sources.""",
    tools=[],           # writer uses LLM reasoning only
    llm=llm,
    verbose=True,
)
```

---

## Phase 3 — Tasks

### 3.1 Task with Pydantic output
```python
from crewai import Task
from pydantic import BaseModel, Field

class ResearchReport(BaseModel):
    topic: str
    summary: str = Field(description="Executive summary (2-3 sentences)")
    key_findings: list[str] = Field(description="5-7 key facts")
    sources: list[str] = Field(description="URLs of sources used")
    confidence: float = Field(ge=0, le=1, description="Confidence in accuracy")

research_task = Task(
    description="""Research the topic: {topic}

    Find:
    1. Current state and recent developments
    2. Key statistics and data points
    3. Expert opinions and analysis
    4. Future outlook

    Use at least 3 different sources.""",
    expected_output="A comprehensive research report with facts, statistics, and sources.",
    agent=researcher,
    output_pydantic=ResearchReport,
)
```

### 3.2 Task with context (depends on previous task)
```python
writing_task = Task(
    description="""Using the research provided, write a {word_count}-word article about {topic}.

    Requirements:
    - Engaging headline and introduction
    - Clear section headers
    - Include specific data points from research
    - Conclusion with key takeaways
    - Natural, conversational tone""",
    expected_output="A polished article in Markdown format.",
    agent=writer,
    context=[research_task],   # research_task output is injected into context
)
```

**Key insight:** `context=[research_task]` makes the previous task's output available
to this task's agent. This is how CrewAI chains agents without explicit data passing.

---

## Phase 4 — Crew

### 4.1 Sequential process
```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer, editor, seo_analyst],
    tasks=[research_task, writing_task, editing_task, seo_task],
    process=Process.sequential,  # tasks run in order, each feeds next
    verbose=True,
    memory=True,                 # shared memory across agents
)

result = crew.kickoff(inputs={"topic": "Quantum computing in 2025", "word_count": 1500})
print(result.raw)          # final output
print(result.token_usage)  # total tokens used
```

### 4.2 Hierarchical process
```python
from crewai import Crew, Process, Agent

manager = Agent(
    role="Content Director",
    goal="Oversee the creation of high-quality content about {topic}",
    backstory="Experienced content director who delegates and reviews work.",
    llm=llm,
    allow_delegation=True,   # manager can delegate to other agents
)

crew = Crew(
    agents=[researcher, writer, editor, seo_analyst],
    tasks=[research_task, writing_task, editing_task, seo_task],
    process=Process.hierarchical,
    manager_agent=manager,
    verbose=True,
)
```

### 4.3 Async kickoff
```python
import asyncio
result = asyncio.run(crew.kickoff_async(inputs={"topic": "..."}))
```

---

## Phase 5 — Callbacks & Observability

```python
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess

class MetricsCallback:
    def on_task_start(self, task, **kwargs):
        print(f"▶  Task started: {task.description[:60]}")

    def on_task_end(self, task, output, **kwargs):
        print(f"✓  Task done: {len(output.raw)} chars output")

    def on_agent_action(self, agent, action, **kwargs):
        print(f"  [{agent.role}] using tool: {action}")

crew = Crew(..., step_callback=MetricsCallback().on_agent_action,
            task_callback=MetricsCallback().on_task_end)
```

---

## Framework Comparison

| | LangChain | CrewAI |
|---|---|---|
| Unit of abstraction | Chain / Runnable | Agent + Task + Crew |
| Multi-agent | Manual (AgentExecutor) | First-class |
| Task dependencies | Custom code | `context=[task]` |
| Output models | Pydantic via `with_structured_output` | `output_pydantic=Model` |
| Memory | Multiple memory types | `memory=True` (opinionated) |
| When to use | Fine-grained control | Fast multi-agent pipelines |
