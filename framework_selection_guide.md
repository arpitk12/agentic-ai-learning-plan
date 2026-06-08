# Framework Selection Guide — Agentic AI

> **How to pick the right tool for the job.**  
> Covers: LangChain · LangGraph · CrewAI · LlamaIndex · AutoGen · Raw Libraries

---

## 1. Quick Decision Flowchart

```
Start: What are you building?
          │
          ├─► RAG / document Q&A?
          │         │
          │         ├─► Multi-document, complex queries?  ──► LlamaIndex (Project 21)
          │         └─► Simple pipeline, custom control?  ──► LangChain LCEL (Project 18)
          │
          ├─► A workflow with branching / loops / human approval?
          │         │
          │         ├─► Need visual graph + state machine?  ──► LangGraph (Project 19)
          │         └─► Need full async + human-in-the-loop? ── LangGraph (Project 19)
          │
          ├─► Autonomous agents collaborating as a team?
          │         │
          │         ├─► Role-based, content/research pipeline?  ──► CrewAI (Project 20)
          │         └─► Code generation + execution team?       ──► AutoGen (Project 22)
          │
          ├─► Production API / high-throughput / custom infra?
          │         └─► Raw Libraries (FastAPI + LiteLLM + Qdrant) (Projects 16-17)
          │
          └─► Building a simple chatbot or one-shot LLM call?
                    └─► Raw API (OpenAI/LiteLLM SDK directly)
```

---

## 2. Framework Comparison Table

| Dimension | LangChain | LangGraph | CrewAI | LlamaIndex | AutoGen |
|---|---|---|---|---|---|
| **Core abstraction** | Chain / LCEL pipe | StateGraph (DAG/cyclic) | Agent + Task + Crew | Index + QueryEngine | AssistantAgent + GroupChat |
| **Control flow** | Linear / sequential | Graph with conditions & loops | Sequential or hierarchical | Query-time routing | Conversation-driven |
| **Human-in-the-loop** | ✅ (manual) | ✅ **Native** (`interrupt()`) | ⚠️ Limited | ❌ | ✅ (`human_input_mode`) |
| **State persistence** | ❌ (manual) | ✅ **Native** (SqliteSaver / Postgres) | ❌ | ❌ | ❌ |
| **RAG / indexing** | ✅ Basic | ⚠️ (via LangChain) | ⚠️ (via tools) | ✅ **Best-in-class** | ⚠️ (via tools) |
| **Multi-agent** | ⚠️ (hacks) | ✅ (multi-graph) | ✅ **Native** | ❌ | ✅ **Native** |
| **Code execution** | ❌ | ❌ | ❌ | ❌ | ✅ **Native** (Docker/local) |
| **Streaming** | ✅ `astream_events` | ✅ `astream` | ⚠️ Limited | ✅ | ❌ |
| **Learning curve** | Medium | High | Low | Medium | Medium |
| **Production readiness** | ✅ | ✅ | ⚠️ | ✅ | ⚠️ |
| **Best for** | Chains, tools, RAG | Stateful workflows | Role-based teams | Document RAG | Coding agents |

---

## 3. When to Use Each Framework

### 3.1 LangChain — Use for chains, tools, and basic RAG

**Choose LangChain when:**
- You need composable pipelines (prompt → LLM → parser)
- You want a large ecosystem of pre-built tools (Tavily, Wikipedia, SQL, etc.)
- You need structured output with Pydantic schemas
- You're adding LLM capabilities to an existing Python app

**Avoid when:** You need stateful loops, complex branching, or multi-agent coordination.

**Example scenario:**
> *"Build an endpoint that takes a user question, searches a PDF knowledge base, and returns a structured JSON answer with citations."*

```python
# Classic LangChain LCEL pattern
chain = (
    RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
    | RAG_PROMPT
    | llm.with_structured_output(AnswerWithCitations)
)
answer = chain.invoke("What is the refund policy?")
```

---

### 3.2 LangGraph — Use for stateful workflows and human-in-the-loop

**Choose LangGraph when:**
- Your workflow has branching logic (if/else, retry loops)
- You need to pause and resume mid-workflow (human approval, external APIs)
- You need to persist state across sessions (multi-turn, async workflows)
- You're building a workflow where the next step depends on previous results

**Avoid when:** You just need a simple linear chain — LangGraph's overhead isn't worth it.

**Example scenario:**
> *"Build a code review bot that parses code, flags security issues, generates a review, pauses for a human to approve or request changes, then loops back to revise."*

```python
# LangGraph conditional edge pattern
builder.add_conditional_edges(
    "human_approval",
    route_approval,
    {"approved": "finalize", "needs_revision": "generate_review"},
)
graph = builder.compile(
    checkpointer=SqliteSaver(...),
    interrupt_before=["human_approval"],   # pause here
)

# Later, resume with human input:
result = graph.invoke(Command(resume="revise: add more detail on SQL injection"), config=config)
```

**LangGraph vs LangChain decision rule:**
> If you can draw your workflow as a **flowchart with decision diamonds**, use LangGraph.  
> If it's a **straight pipe**, use LangChain LCEL.

---

### 3.3 CrewAI — Use for role-based multi-agent teams

**Choose CrewAI when:**
- You want to assign distinct roles, goals, and backstories to agents
- Your workflow maps naturally to a team of specialists (researcher, writer, reviewer)
- You want agents to share memory and pass structured outputs between each other
- You need a manager/hierarchical mode where one agent coordinates others

**Avoid when:** You need precise control over execution order or need code to be run and tested — CrewAI's agent autonomy can make exact flow hard to control.

**Example scenario:**
> *"Build a content factory where a Researcher gathers facts, a Writer drafts an article, an Editor polishes it, and an SEO Analyst optimizes keywords — all automatically."*

```python
# CrewAI task chaining with structured output
research_task = Task(description="Research quantum computing...", agent=researcher,
                     output_pydantic=ResearchReport)
writing_task  = Task(description="Write a 1200-word article...", agent=writer,
                     context=[research_task],          # ← gets researcher output
                     output_pydantic=Article)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task],
            process=Process.sequential, memory=True)
result = crew.kickoff(inputs={"topic": "Quantum Computing"})
```

**CrewAI vs AutoGen decision rule:**
> CrewAI = **role-playing specialists** on a content/research pipeline.  
> AutoGen = **coding team** where actual code needs to be written and executed.

---

### 3.4 LlamaIndex — Use for document-heavy RAG

**Choose LlamaIndex when:**
- You have lots of documents and need efficient indexing and retrieval
- You need multiple index types (vector for facts, summary for overviews)
- You want smart routing between query strategies
- You need incremental ingestion with caching (large doc collections)
- You're building a knowledge base that needs rich metadata extraction

**Avoid when:** You only have a handful of documents — FAISS + LangChain is simpler.

**Example scenario:**
> *"Index 500 research papers. Some queries need specific facts ('what did paper X say about Y?'), others need summaries ('summarize all papers on topic Z'). Route automatically."*

```python
# LlamaIndex RouterQueryEngine
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),  # LLM picks the right engine
    query_engine_tools=[
        QueryEngineTool.from_defaults(vector_engine,
            description="Specific facts and definitions"),
        QueryEngineTool.from_defaults(summary_engine,
            description="High-level overviews and summaries"),
    ]
)
response = router_engine.query("Summarize the papers on transformer attention mechanisms")
# LLM selects → summary engine → tree_summarize across all relevant chunks
```

**LlamaIndex vs LangChain RAG decision rule:**
> LangChain RAG: < 50 docs, simple use-case, already using LangChain ecosystem.  
> LlamaIndex: 50+ docs, need metadata extraction, multiple index types, or sub-question decomposition.

---

### 3.5 AutoGen — Use for code generation and execution agents

**Choose AutoGen when:**
- You need agents that actually write and **run** code
- You want a full software team (PM → Architect → Dev → Tester → Reviewer)
- You need Docker-isolated code execution for safety
- You're building coding assistants, data analysis pipelines, or DevOps automation

**Avoid when:** You don't need code execution — the complexity isn't worth it for pure text tasks.

**Example scenario:**
> *"Build a system where a user describes a feature, and a team of agents designs the architecture, writes the code, runs tests, and reviews the result."*

```python
# AutoGen two-agent pattern (simplest)
assistant = autogen.AssistantAgent("CodingAssistant", llm_config=llm_config,
    system_message="Write runnable Python. Append TERMINATE when done.")
user_proxy = autogen.UserProxyAgent("UserProxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="workspace")})

user_proxy.initiate_chat(assistant, message="Write a Python function to parse CSV files with error handling")
# → Code is written AND executed → output returned → verified
```

---

### 3.6 Raw Libraries — Use for production systems

**Choose raw libraries (FastAPI + LiteLLM + Qdrant/FAISS) when:**
- You need maximum performance and control
- You're serving high-traffic production workloads
- You need custom retry logic, circuit breakers, or rate limiting
- Framework abstractions add latency or limit your architecture choices
- You need Celery/Kafka async pipelines that frameworks don't support cleanly

**Example scenario:**
> *"Serve a RAG API handling 1,000 RPS with hybrid search, BM25 + vector, reranking, and async Celery workers for ingestion."*  
> → Projects 16 & 17 — no framework overhead, full control.

---

## 4. Framework Combinations That Work Well

Some of the most powerful production systems combine frameworks:

| Combination | Why it works | Use case |
|---|---|---|
| **LangGraph + LangChain** | LangGraph for flow control, LangChain LCEL for node-level chains | Complex RAG with approval loops |
| **LangGraph + LlamaIndex** | LangGraph for orchestration, LlamaIndex for retrieval nodes | Document review workflow |
| **CrewAI + LangChain tools** | CrewAI agents use `@tool`-decorated LangChain tools | Research pipelines with web search |
| **AutoGen + LangChain RAG** | AutoGen coding team, RAG via LangChain tool | Code agents that consult docs |

```python
# Example: LangGraph node that calls a LlamaIndex query engine
def retrieve_node(state: WorkflowState) -> dict:
    # LlamaIndex handles the heavy retrieval
    engine = router_engine  # built with LlamaIndex
    response = engine.query(state["question"])
    return {"context": str(response)}

# This node lives inside a LangGraph StateGraph
builder.add_node("retrieve", retrieve_node)
```

---

## 5. Decision Checklist

Answer these questions to find your framework:

```
□ Do I need to run and test code?              → YES → AutoGen
□ Do I need branching / retry / pause-resume?  → YES → LangGraph
□ Do I have many documents to index?           → YES → LlamaIndex
□ Do I want role-based specialist agents?      → YES → CrewAI
□ Do I need a quick chain or RAG pipeline?     → YES → LangChain
□ Do I need maximum production performance?    → YES → Raw Libraries
```

If multiple boxes are checked, use a **combination** (see Section 4).

---

## 6. Complexity vs Control Trade-off

```
HIGH
│  Raw Libraries  ◄──── maximum control, maximum complexity
│
│  LangGraph      ◄──── stateful orchestration, medium complexity
│
│  LangChain      ◄──── good balance, wide ecosystem
CONTROL
│  LlamaIndex     ◄──── document-domain opinionated, easy RAG
│
│  AutoGen        ◄──── easy code-execution, less control over flow
│
│  CrewAI         ◄──── easiest to start, least control
LOW
└────────────────────────────────────────────────────────────────►
    HIGH complexity                                LOW complexity
```

---

## 7. Real-World Scenario Mapping

| Scenario | Recommended | Why |
|---|---|---|
| Customer support chatbot | **LangChain** | Simple chain + memory, no complex orchestration needed |
| Legal doc review with human sign-off | **LangGraph** | State machine + `interrupt()` for approvals |
| AI content marketing team | **CrewAI** | Role-based pipeline maps perfectly |
| Internal enterprise knowledge base | **LlamaIndex** | Multi-doc indexing, metadata, complex queries |
| AI pair programmer | **AutoGen** | Writes + runs + tests code autonomously |
| High-traffic RAG API (1k+ RPS) | **Raw Libraries** | Framework overhead unacceptable at scale |
| Research report with web search + RAG | **LangChain + Tavily** | Quick, composable, tool ecosystem |
| Approval workflow with retry on failure | **LangGraph** | Conditional edges + checkpointer handles retries |
| Multi-doc comparison across 1000 papers | **LlamaIndex SubQuestion** | Decomposes query per document automatically |
| Full-stack app built by AI agents | **AutoGen** | Docker executor runs and tests the full app |

---

## 8. Anti-Patterns to Avoid

| Anti-pattern | Problem | Fix |
|---|---|---|
| Using LangGraph for a simple chain | Massive overhead for no benefit | Use LangChain LCEL |
| Using CrewAI for code execution | No native executor, unreliable | Use AutoGen |
| Using LangChain for 500+ document RAG | Poor ingestion, no caching | Use LlamaIndex |
| Adding a framework to a 10-line script | Abstraction cost > benefit | Use raw `openai` SDK |
| Using AutoGen for pure text tasks | GroupChat overhead, hard to debug | Use CrewAI or LangChain |
| Multiple frameworks for one task | Dependency conflicts, confusion | Pick one, use combinations intentionally |

---

## 9. Framework Maturity & Community (as of June 2026)

| Framework | GitHub Stars | Production Use | Active Dev |
|---|---|---|---|
| LangChain | ★★★★★ | High | ✅ |
| LlamaIndex | ★★★★☆ | High | ✅ |
| LangGraph | ★★★★☆ | Growing fast | ✅ |
| AutoGen | ★★★★☆ | Medium | ✅ |
| CrewAI | ★★★☆☆ | Medium | ✅ |

---

*For deep dives on each framework, see the corresponding project guide:*
- *LangChain → `projects/project18_langchain_agent/GUIDE.md`*
- *LangGraph → `projects/project19_langgraph_workflow/GUIDE.md`*
- *CrewAI → `projects/project20_crewai_pipeline/GUIDE.md`*
- *LlamaIndex → `projects/project21_llamaindex_agent/GUIDE.md`*
- *AutoGen → `projects/project22_autogen_team/GUIDE.md`*
