# Project 20 — CrewAI: Content Creation Pipeline

A **production-grade CrewAI application** demonstrating multi-agent collaboration:
four specialized agents (Researcher, Writer, Editor, SEO Analyst) work sequentially
on a shared content brief, each agent using tools and producing structured outputs
that feed into the next agent's task.

---

## 🎯 What You Learn

| Concept | Where |
|---------|-------|
| **Agent** — role, goal, backstory, tools | `src/agents/agents.py` |
| **Task** — description, expected_output, agent | `src/tasks/tasks.py` |
| **Crew** — agents + tasks + process | `src/crew/crew.py` |
| **Tools** — `@tool` + `BaseTool` subclass | `src/tools/` |
| **Sequential process** — task chain | `src/crew/crew.py` |
| **Hierarchical process** — manager agent | `src/crew/crew_hierarchical.py` |
| **Structured output** — Pydantic output models | `src/tasks/tasks.py` |
| **Memory** — short-term + long-term + entity | `src/crew/crew.py` |
| **Callbacks** — step + task hooks | `src/observability/callbacks.py` |

---

## 🏗 Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║              SEQUENTIAL CONTENT PIPELINE                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Topic brief (input)                                            ║
║       │                                                          ║
║  ┌────▼──────────────────────────────────────────────────────┐  ║
║  │  RESEARCHER AGENT                                          │  ║
║  │  goal: find accurate, up-to-date facts on the topic       │  ║
║  │  tools: TavilySearchTool, WebsiteReadTool                 │  ║
║  │  output: ResearchReport (Pydantic model)                  │  ║
║  └────────────────────────────────────┬───────────────────────┘  ║
║                                        │ research_report          ║
║  ┌─────────────────────────────────────▼───────────────────────┐  ║
║  │  WRITER AGENT                                               │  ║
║  │  goal: write compelling article from research              │  ║
║  │  tools: FileReadTool (style guide)                         │  ║
║  │  output: Article (Pydantic: title, body, word_count)       │  ║
║  └─────────────────────────────────────┬───────────────────────┘  ║
║                                         │ article                  ║
║  ┌──────────────────────────────────────▼───────────────────────┐  ║
║  │  EDITOR AGENT                                                │  ║
║  │  goal: improve clarity, fix grammar, ensure accuracy        │  ║
║  │  tools: none (LLM reasoning only)                           │  ║
║  │  output: EditedArticle with tracked changes                 │  ║
║  └──────────────────────────────────────┬───────────────────────┘  ║
║                                          │ edited                   ║
║  ┌───────────────────────────────────────▼───────────────────────┐  ║
║  │  SEO ANALYST AGENT                                            │  ║
║  │  goal: optimize for search, add meta tags, keyword density   │  ║
║  │  tools: none                                                  │  ║
║  │  output: SEOReport + optimized content                       │  ║
║  └───────────────────────────────────────────────────────────────┘  ║
║                                                                     ║
║  Published article (markdown) + SEO report (JSON)                  ║
╚═════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║         HIERARCHICAL PROCESS (bonus crew)                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  MANAGER AGENT  ←─── orchestrates all agents                   ║
║    ├── delegates research_task to RESEARCHER                    ║
║    ├── delegates writing_task to WRITER                         ║
║    ├── delegates editing_task to EDITOR                         ║
║    └── delegates seo_task to SEO_ANALYST                        ║
║                                                                  ║
║  Manager decides order and can re-assign tasks on poor output   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Folder Structure

```
project20_crewai_pipeline/
├── README.md
├── GUIDE.md
├── starter/
│   ├── requirements.txt
│   ├── .env.example
│   └── src/
│       ├── config.py             ← given
│       ├── models.py             ← given (Pydantic output models)
│       ├── agents/
│       │   └── agents.py         ← TODO (8 tasks) — 4 specialized agents
│       ├── tasks/
│       │   └── tasks.py          ← TODO (8 tasks) — 4 tasks with context
│       ├── tools/
│       │   ├── search_tool.py    ← TODO (4 tasks) — Tavily wrapper
│       │   └── file_tool.py      ← TODO (3 tasks) — file read/write
│       ├── crew/
│       │   ├── crew.py           ← TODO (5 tasks) — sequential crew
│       │   └── crew_hierarchical.py ← TODO (4 tasks) — hierarchical crew
│       ├── observability/
│       │   └── callbacks.py      ← TODO (3 tasks) — step callbacks
│       └── main.py               ← TODO (3 tasks) — CLI entry point
└── solution/
    └── src/
```

---

## ⚡ Key CrewAI Patterns

| Pattern | Code | Why |
|---------|------|-----|
| Define agent | `Agent(role=..., goal=..., backstory=..., tools=[...], llm=...)` | Personality + capability |
| Define task | `Task(description=..., expected_output=..., agent=..., output_pydantic=Model)` | Structured output |
| Task context | `Task(..., context=[previous_task])` | Pass prior output as context |
| Sequential crew | `Crew(agents=..., tasks=..., process=Process.sequential)` | Each task feeds next |
| Hierarchical crew | `Crew(..., process=Process.hierarchical, manager_llm=...)` | Manager delegates |
| Enable memory | `Crew(..., memory=True)` | Short-term + entity memory |
| Kickoff | `crew.kickoff(inputs={"topic": "..."})`| Run the pipeline |
| Async | `crew.kickoff_async(inputs=...)` | Non-blocking |

---

## 🚀 Quick Start

```bash
cd projects/project20_crewai_pipeline/starter
pip install -r requirements.txt
cp .env.example .env  # add TAVILY_API_KEY + LLM key

# Run sequential pipeline
python -m src.main --topic "The future of quantum computing" --output article.md

# Run hierarchical crew
python -m src.main --topic "AI in healthcare" --mode hierarchical
```

---

## Milestones

1. **Tools** — implement and test each tool independently
2. **Agents** — define all 4 agents, verify instantiation
3. **Tasks** — define tasks with Pydantic output models
4. **Sequential Crew** — wire together, run end-to-end
5. **Hierarchical Crew** — add manager agent, compare quality vs sequential
6. **Memory + Callbacks** — enable memory, add step logging
