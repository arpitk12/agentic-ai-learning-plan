# About n8n & No-Code Workflow Tools

## Short Answer

**n8n is NOT covered in this curriculum.** This is intentional.

---

## Why Not n8n?

### 1. **Different Paradigm**
- **This curriculum**: Programmatic agents in Python (LangChain, LangGraph, CrewAI)
- **n8n**: No-code/low-code workflow orchestration UI

### 2. **Different Audience**
- **Target**: AI engineers, Python developers, backend teams building production agents
- **n8n audience**: Business analysts, workflow automators, teams without programming skills

### 3. **Different Use Cases**
| When to use Python agents (this course) | When to use n8n |
|---|---|
| Complex multi-step reasoning | Simple integrations (Slack → Google Sheets) |
| Tool calling with LLMs | Connecting SaaS tools without code |
| Real-time AI logic | One-off workflows, no reasoning needed |
| Custom ML/RAG pipelines | Business process automation |
| High-throughput production | Quick prototypes, rapid iteration |

---

## The Covered Workflow Frameworks

This curriculum covers **6 production-grade Python frameworks** for building stateful AI workflows:

1. **LangChain LCEL** — Linear chains, document pipelines
2. **LangGraph** — Stateful workflows with branching, HITL, checkpointing
3. **CrewAI** — Role-based multi-agent teams
4. **LlamaIndex** — Document RAG with indexing strategies
5. **AutoGen** — Coding-focused agent teams with code execution
6. **Raw libraries** (FastAPI + asyncio) — Maximum control, production scale

See **[`framework_selection_guide.md`](framework_selection_guide.md)** for details.

---

## If You Want No-Code Workflow Tools

Popular alternatives to n8n (all NOT covered here):

| Tool | Best for | Free? |
|---|---|---|
| **n8n** | Generic workflow automation, 350+ integrations | ✅ Self-hosted |
| **Zapier** | Business process automation, integrations | ⚠️ Paid |
| **Make (Integromat)** | Visual workflow builder, flexible logic | ⚠️ Paid |
| **Activepieces** | Open-source Zapier alternative | ✅ Yes |
| **Windmill** | Developer-friendly workflow engine | ✅ Self-hosted |

### If You Need n8n + Python Agents

You can **integrate** them:

```
┌─────────────────────┐
│   n8n (workflow)    │
│  - Trigger webhook  │
│  - Schedule tasks   │
│  - Route data       │
└──────────┬──────────┘
           │
           ▼ HTTP POST to FastAPI endpoint
┌─────────────────────┐
│ Python Agent API    │
│ (Projects 4, 16)    │
│ - LLM reasoning     │
│ - Tool calling      │
│ - RAG lookups       │
└──────────┬──────────┘
           │
           ▼ Return JSON result
┌─────────────────────┐
│   n8n (continues)   │
│  - Process response │
│  - Send to Slack    │
└─────────────────────┘
```

**Example**: Use n8n to trigger your agent API on a Slack command, then post results back. This is NOT covered in this course, but it's possible.

---

## What IS Covered for Workflows

### Stateful Workflows
- **LangGraph** (`project19_langgraph_workflow`, `project23_enterprise_architect`)
  - Branching logic (if/else routes)
  - Human-in-the-loop with `interrupt()`
  - State persistence across sessions
  - Conditional edges and loops

### Multi-Agent Workflows
- **CrewAI** (`project20_crewai_pipeline`)
  - Role-based agent teams
  - Sequential and hierarchical execution
  - Task dependencies

### Async Parallelism
- **Raw asyncio + FastAPI** (`project6_capstone`, `project9_batch_pipeline`)
  - Parallel tool execution
  - Semaphore-based concurrency
  - Map-reduce patterns for large datasets

### Resilience & Saga Pattern
- **Project 34** (`project34_resilience`)
  - Multi-step workflow compensation
  - Rollback on failure
  - Saga coordinator for distributed workflows

---

## Recommendation

### For Teams Building AI Agents in Production
→ Follow **this curriculum** (Python frameworks)

### For Business Users Automating Routine Tasks
→ Use **n8n or Zapier** (outside scope of this course)

### For Teams with Both Needs
→ **Combine both**:
- n8n handles: scheduling, data routing, non-AI tasks
- Python agents handle: reasoning, tool calling, complex logic
- Connect via REST API (Project 4 covers this)

---

## Resources

- **Framework selection**: [`framework_selection_guide.md`](framework_selection_guide.md)
- **LangGraph stateful workflows**: [`projects/project19_langgraph_workflow/`](projects/project19_langgraph_workflow/)
- **Multi-agent orchestration**: [`projects/project20_crewai_pipeline/`](projects/project20_crewai_pipeline/)
- **Production API + FastAPI**: [`projects/project4_agent_api/`](projects/project4_agent_api/)
- **LLMOps workflows**: [`guide/14_llmops.md`](guide/14_llmops.md)
