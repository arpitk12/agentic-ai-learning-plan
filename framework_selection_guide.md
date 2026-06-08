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

---

## 10. Exercises

Work through these in order — each builds on the previous.

---

### Exercise 1 — Framework Mapping (No Code)

For each scenario, pick a framework and write a one-sentence justification.  
Check against Section 7 (Real-World Scenario Mapping).

| Scenario | Your pick | Justification |
|---|---|---|
| A Slack bot that answers HR policy questions from a 200-page PDF | | |
| Automated pipeline: scrape news → summarise → post to CMS | | |
| A coding assistant that writes, runs, and fixes failing tests | | |
| Legal review: draft → lawyer approval → redline → final | | |
| Document Q&A over 10,000 internal wiki pages | | |
| Data analyst agent: writes SQL, executes it, charts results | | |
| Real-time customer support chat with < 200ms latency requirement | | |

---

### Exercise 2 — LCEL Chain (LangChain)

**Goal**: Build a chain that takes a topic, retrieves 3 web results, and returns a structured JSON summary.

```python
# File: exercises/framework_selection/ex2_lcel_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

class ResearchSummary(BaseModel):
    topic: str
    key_findings: list[str] = Field(min_length=3, max_length=5)
    confidence: str = Field(description="high / medium / low")

# TODO 1: Import ChatLiteLLM and TavilySearchResults
# TODO 2: Create llm = ChatLiteLLM(model="openai/gpt-4o-mini")
# TODO 3: Create search = TavilySearchResults(max_results=3)
# TODO 4: Build format_results(results) that joins results into one string

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Synthesise search results into a structured research summary."),
    ("human", "Topic: {topic}\n\nSearch results:\n{search_results}"),
])

def build_research_chain(llm, search_tool):
    # TODO 5: RunnableParallel(topic=RunnablePassthrough(), search_results=search|format)
    #         | PROMPT | llm.with_structured_output(ResearchSummary)
    raise NotImplementedError

if __name__ == "__main__":
    chain = build_research_chain(...)
    result = chain.invoke("LangGraph human-in-the-loop patterns")
    print(result.model_dump_json(indent=2))
```

---

### Exercise 3 — StateGraph with Conditional Edge (LangGraph)

**Goal**: 3-node graph: `classify` → conditional → `answer_simple` or `answer_detailed`.

```python
# File: exercises/framework_selection/ex3_langgraph_router.py
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class RouterState(TypedDict):
    question: str
    complexity: str      # "simple" or "detailed"
    answer: str
    messages: Annotated[list, operator.add]

def classify_node(state: RouterState) -> dict:
    # TODO 1: len > 10 words → "detailed", else → "simple"
    raise NotImplementedError

def simple_answer_node(state: RouterState) -> dict:
    # TODO 2: Answer in one sentence
    raise NotImplementedError

def detailed_answer_node(state: RouterState) -> dict:
    # TODO 3: Answer with 3–5 bullet points
    raise NotImplementedError

def route(state: RouterState) -> str:
    # TODO 4: return state["complexity"]
    raise NotImplementedError

def build_router_graph():
    # TODO 5: START→classify→conditional→{simple_answer|detailed_answer}→END
    raise NotImplementedError

if __name__ == "__main__":
    graph = build_router_graph()
    for q in ["Hi", "Explain the difference between RAG and fine-tuning in production"]:
        result = graph.invoke({"question": q, "messages": []})
        print(f"[{result['complexity']}] {result['answer']}\n")
```

---

### Exercise 4 — Two-Agent CrewAI Pipeline

**Goal**: Researcher + Writer crew that outputs a structured blog outline.

```python
# File: exercises/framework_selection/ex4_crewai_pipeline.py
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field

class BlogOutline(BaseModel):
    title: str
    hook: str
    sections: list[dict]   # [{heading, key_points: list[str]}]
    cta: str

# TODO 1: Create researcher Agent (role, goal, backstory, llm)
# TODO 2: Create writer Agent
# TODO 3: Create research_task → output_pydantic=ResearchNotes
# TODO 4: Create outline_task → context=[research_task], output_pydantic=BlogOutline

def run_crew(topic: str) -> BlogOutline:
    # TODO 5: Crew(process=Process.sequential).kickoff(inputs={"topic": topic})
    raise NotImplementedError

if __name__ == "__main__":
    outline = run_crew("The practical benefits of LangGraph over raw LangChain")
    print(f"\n📝 {outline.title}")
    for s in outline.sections:
        print(f"\n## {s['heading']}")
        for pt in s.get("key_points", []):
            print(f"  - {pt}")
```

---

### Exercise 5 — Spot the Anti-Pattern

Read each snippet. Identify the anti-pattern and write the fix.

**Snippet A** — Building a simple 50-row FAQ bot with LangGraph: StateGraph, 5 nodes, SqliteSaver, interrupt/resume, conditional edges.  
*Anti-pattern:* ___ *Fix:* ___

**Snippet B** — Using CrewAI where the final agent needs to write a script, execute it, capture stdout, and fix errors.  
*Anti-pattern:* ___ *Fix:* ___

**Snippet C** — Using LangChain LCEL: `load_all_50000_tickets | prompt | llm` — all 50k tickets injected into every prompt.  
*Anti-pattern:* ___ *Fix:* ___

**Snippet D** — Using raw OpenAI SDK for a stateful workflow that may pause for hours waiting for user input.  
*Anti-pattern:* ___ *Fix:* ___

*(Answers in `exercises/framework_selection/ex5_answers.md`)*

---

### Exercise 6 — Framework Overhead Comparison

**Goal**: Implement the same RAG task with raw litellm, LangChain LCEL, and LangGraph. Compare token counts.

```python
# File: exercises/framework_selection/ex6_overhead_comparison.py
QUESTION = "What are the key differences between RAG and fine-tuning?"
CONTEXT = [
    "RAG retrieves relevant documents at inference time without changing model weights.",
    "Fine-tuning updates model parameters on domain-specific data.",
]

# TODO 1: raw_litellm(question, context) → (answer, input_tokens, output_tokens, calls=1)
# TODO 2: langchain_lcel(question, context) → (answer, input_tokens, output_tokens, calls)
# TODO 3: langgraph_graph(question, context) → (answer, input_tokens, output_tokens, calls)
# TODO 4: print a comparison table showing tokens and calls per approach

# Expected finding: same answer quality, but raw litellm uses fewest tokens.
# Framework overhead appears in prompt templates and chain bookkeeping.
```

---

## 11. References and Resources

### 📄 Papers

| Paper | Framework | Link |
|---|---|---|
| **ReAct: Synergizing Reasoning and Acting** (Yao et al., 2022) | LangChain / LangGraph | [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) |
| **ToolFormer** (Schick et al., 2023) | Tool use | [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761) |
| **AutoGen: Enabling Next-Gen LLM Applications** (Wu et al., 2023) | AutoGen | [arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155) |
| **RAG vs Fine-tuning** (Ovadia et al., 2023) | LlamaIndex / LangChain | [arxiv.org/abs/2312.05934](https://arxiv.org/abs/2312.05934) |
| **Self-RAG** (Asai et al., 2023) | RAG patterns | [arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511) |

### 📚 Official Documentation

| Resource | Link |
|---|---|
| LangChain LCEL | [python.langchain.com/docs/expression_language](https://python.langchain.com/docs/expression_language/) |
| LangGraph concepts | [langchain-ai.github.io/langgraph/concepts](https://langchain-ai.github.io/langgraph/concepts/) |
| LangGraph HITL how-to | [langchain-ai.github.io/langgraph/how-tos/human_in_the_loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/) |
| CrewAI docs | [docs.crewai.com](https://docs.crewai.com) |
| LlamaIndex core concepts | [docs.llamaindex.ai/en/stable/getting_started/concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) |
| AutoGen docs | [microsoft.github.io/autogen](https://microsoft.github.io/autogen/) |

### 🎓 Free Courses (DeepLearning.AI)

| Course | Link |
|---|---|
| Functions, Tools and Agents with LangChain | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) |
| AI Agents in LangGraph | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) |
| Multi AI Agent Systems with CrewAI | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) |
| Building Agentic RAG with LlamaIndex | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) |
| LangGraph Academy (interactive) | [academy.langchain.com](https://academy.langchain.com) |

### 📖 Further Reading

- **"Emerging Architectures for LLM Applications"** — [a16z.com](https://a16z.com/emerging-architectures-for-llm-applications/)
- **"The State of AI Agents"** — [lmsys.org/blog](https://lmsys.org/blog/)
- **"AI Engineering"** by Chip Huyen — [oreilly.com](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)

---

> 💰 **Also see**: [`token_optimization_guide.md`](token_optimization_guide.md) — how to reduce LLM costs by 70–90% regardless of which framework you choose.
