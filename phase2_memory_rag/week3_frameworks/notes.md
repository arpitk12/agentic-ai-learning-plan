# Week 3 — Agent Frameworks: LangGraph, LangChain, CrewAI, AutoGen

## What This Week Is About
Raw ReAct loops work, but production agents need structure: state management, routing logic, observability, and composability. This week surveys the major frameworks — what each one IS, why it exists, and when to choose it over the others.

---

## 1. Framework Decision Map

```
Need a single agent with complex state/routing?  → LangGraph
Need pipelines with many off-the-shelf tools?    → LangChain LCEL
Need multiple role-based agents collaborating?   → CrewAI
Need research-grade multi-agent experimentation? → AutoGen
Need maximum control with minimal abstraction?   → Raw llm.py (ReAct loop)
```

---

## 2. LangGraph — Stateful Agent Workflows

**What it is**: A library for building agents as **directed graphs** where nodes are functions and edges are transitions. State flows through the graph and is modified at each node.

**Purpose**: Makes complex agent logic (branches, loops, conditions, parallel paths) explicit and inspectable. Instead of spaghetti if/else code, you see a visual graph.

**Install**: `pip install langgraph`

### Core Concepts

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Define the state — all data the agent carries
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only
    query: str
    search_results: str
    final_answer: str
    step_count: int

# 2. Define nodes — functions that receive and return state
def search_node(state: AgentState) -> dict:
    results = web_search(state["query"])
    return {"search_results": results, "step_count": state["step_count"] + 1}

def answer_node(state: AgentState) -> dict:
    answer = llm_answer(state["query"], state["search_results"])
    return {"final_answer": answer}

def should_search(state: AgentState) -> str:
    """Router: decides which node to visit next."""
    if state["step_count"] < 3 and state["final_answer"] == "":
        return "search"
    return "answer"

# 3. Build the graph
graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("answer", answer_node)
graph.set_entry_point("search")
graph.add_conditional_edges("search", should_search, {"search": "search", "answer": "answer"})
graph.add_edge("answer", END)

agent = graph.compile()

# 4. Run
result = agent.invoke({"query": "What is the GDP of France?", "search_results": "", "final_answer": "", "step_count": 0})
print(result["final_answer"])
```

### Key LangGraph Concepts

| Concept | Description |
|---------|-------------|
| `StateGraph` | The graph container — holds nodes and edges |
| `TypedDict` state | Typed Python dict that flows through all nodes |
| `Annotated[list, operator.add]` | Appends to list instead of replacing it |
| `add_conditional_edges` | Dynamic routing based on state |
| `set_entry_point` | Which node runs first |
| `END` | Terminal node — graph stops here |
| `compile()` | Returns a runnable `CompiledGraph` |
| `invoke()` | Synchronous run; `ainvoke()` for async |

### Human-in-the-Loop with LangGraph
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer, interrupt_before=["dangerous_action"])

# Agent pauses before "dangerous_action" node, waits for approval
result = agent.invoke(state, config={"configurable": {"thread_id": "user-123"}})
# User reviews, then resume:
result = agent.invoke(None, config={"configurable": {"thread_id": "user-123"}})
```

---

## 3. LangChain — The LCEL Pipeline Framework

**What it is**: A framework for chaining LLM calls, prompts, parsers, and tools into declarative pipelines using the **L**ang**C**hain **E**xpression **L**anguage (LCEL).

**Purpose**: Rapid prototyping. Huge library of pre-built integrations (100+ vector stores, 50+ LLMs, dozens of document loaders). The `|` operator chains components together.

**Install**: `pip install langchain langchain-core langchain-openai`

### LCEL Syntax

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Chain: prompt | model | parser
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate to {language}."),
    ("user", "{text}")
])
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"language": "French", "text": "Hello world"})
```

### LangChain Strengths
- **Document loaders**: PDF, Word, web pages, databases — unified interface
- **Text splitters**: Recursive, semantic, token-based chunking built in
- **Vector store integrations**: Chroma, Pinecone, Weaviate, FAISS all work out-of-box
- **Agent toolkits**: Pre-built tools for SQL, Wikipedia, Bash, etc.

---

## 4. LangSmith — Observability for LangChain/LangGraph

**What it is**: A tracing and evaluation platform from the LangChain team.

**Purpose**: See exactly what your agent did, what prompts it sent, what responses it got, how long each step took, and what it cost — all in a web UI.

```python
import os
os.environ["LANGSMITH_API_KEY"] = "your-key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-agent-project"

# Now every LangChain/LangGraph call is automatically traced
result = agent.invoke({"query": "What is quantum computing?"})
# Visit smith.langchain.com to see the full trace
```

**When to use**: Any time you're debugging why an agent made a bad decision. Essential in production.

---

## 5. CrewAI — Role-Based Multi-Agent Orchestration

**What it is**: A framework where you define a **crew** of AI agents, each with a **role**, **goal**, and **backstory**, and assign them **tasks** to complete collaboratively.

**Purpose**: Decompose complex problems into sub-problems, assign each to a specialist agent. Like hiring a team: one agent researches, one writes, one edits.

**Install**: `pip install crewai crewai-tools`

```python
from crewai import Agent, Task, Crew, Process

# Define specialist agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find accurate, comprehensive information on any topic",
    backstory="Expert at finding reliable sources and synthesizing information",
    tools=[web_search_tool, arxiv_tool],
    llm="gemini/gemini-2.0-flash",
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Transform research into clear, engaging content",
    backstory="Experienced at explaining complex topics to general audiences",
    llm="gemini/gemini-2.0-flash",
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest developments in quantum computing for 2025",
    expected_output="A comprehensive summary with key breakthroughs, key players, and market outlook",
    agent=researcher
)

writing_task = Task(
    description="Write a 500-word article based on the research",
    expected_output="A polished article with title, intro, body, and conclusion",
    agent=writer,
    context=[research_task]  # receives research output automatically
)

# Assemble and run the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential  # or Process.hierarchical
)

result = crew.kickoff()
print(result.raw)
```

### CrewAI Process Types
| Process | Description | Use When |
|---------|-------------|----------|
| `sequential` | Tasks run in order, each feeds next | Linear pipeline (research → write → review) |
| `hierarchical` | Manager agent delegates to workers | Complex tasks requiring coordination |

---

## 6. AutoGen — Conversational Multi-Agent Framework

**What it is**: A Microsoft Research framework for multi-agent conversations. Agents talk to each other in a chat loop until a task is done.

**Purpose**: Research and experimentation. Excellent for agent-to-agent dialogue, debate patterns, and code generation with automatic execution and error correction.

**Install**: `pip install pyautogen`

```python
import autogen

config = {"model": "gpt-4o", "api_key": "your-key"}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": [config]},
    system_message="You are a Python expert. Write clean, tested code."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",      # no human intervention
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding", "use_docker": False},
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "")
)

# Start the conversation — agents debate until task complete
user_proxy.initiate_chat(assistant, message="Write a Python function to find prime numbers up to N using a sieve.")
```

---

## 7. Framework Comparison

| Framework | Best For | Abstraction Level | Learning Curve | Production-Ready |
|-----------|---------|------------------|---------------|-----------------|
| **Raw llm.py** | Full control, learning | Lowest | Lowest | Yes (manual work) |
| **LangChain LCEL** | Pipelines, RAG, prototyping | High | Low-Medium | Yes |
| **LangGraph** | Complex stateful agents | Medium | Medium | Yes (recommended) |
| **CrewAI** | Multi-agent teams | High | Low | Yes |
| **AutoGen** | Agent research, code gen | High | Medium | Experimental |

### The Architecture Principle
Frameworks are training wheels and accelerators. Knowing the raw patterns (Week 1–2) makes you effective with any framework. When a framework's abstraction hurts more than it helps, drop to the layer below.

---

## Tools & Libraries Used This Week — Deep Dive

### LangGraph — Why You Need It Beyond Raw ReAct

**The problem with raw loops**: A raw `while True:` ReAct loop has hidden problems:
- No state persistence: if the process crashes mid-agent-run, all progress is lost
- No branching: you can't have "if research insufficient → search more, else → write"
- No checkpointing: you can't pause and resume an agent run
- No visibility: you can't see what state the agent is in right now

LangGraph solves ALL of these by making your agent's logic explicit as a graph.

**How to think about LangGraph**:
- **Nodes** are functions: `def search_node(state): → new_state_fields`
- **Edges** are transitions: `search → analyze`
- **State** is a TypedDict: all data the agent carries
- **Conditional edges** are routers: `should_search → "search" or "answer"`

The graph is compiled once and then run many times. You can visualize it (`agent.get_graph().draw_ascii()`), test each node in isolation, and checkpoint state between runs.

```python
# When LangGraph shines: multi-step agent with conditional replanning
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver  # persistent checkpoints

# This shows what raw loops can't do easily:
graph = StateGraph(AgentState)
graph.add_node("plan", create_plan_node)
graph.add_node("execute", execute_step_node)
graph.add_node("replan", replan_if_failed_node)
graph.add_node("answer", final_answer_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "execute")
graph.add_conditional_edges("execute", route_after_execution, {
    "continue": "execute",   # more steps to execute
    "replan": "replan",      # step failed, need new plan
    "done": "answer"         # all done
})
graph.add_edge("replan", "execute")
graph.add_edge("answer", END)

# Persistent checkpoints — agent survives process restart
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
agent = graph.compile(checkpointer=checkpointer)

# Run with thread_id — each user conversation has its own state
result = agent.invoke(state, config={"configurable": {"thread_id": "user_42_session_1"}})
```

---

### LangChain — When to Use vs When to Skip

**Use LangChain for**:
1. **Document ingestion** — PDF, Word, HTML loaders: `PyPDFLoader`, `WebBaseLoader`
2. **Text splitting** — `RecursiveCharacterTextSplitter` is excellent and battle-tested
3. **Vector store integrations** — single unified API for Chroma, Qdrant, Pinecone, etc.
4. **LCEL pipelines** — `prompt | model | parser` is elegant for simple workflows

**Skip LangChain for**:
1. Complex stateful agents (use LangGraph instead)
2. When you want maximum control (use raw llm.py)
3. When debugging is critical (LangChain's abstractions can hide errors)

```python
# LangChain's REAL superpower: document loading
from langchain_community.document_loaders import (
    PyPDFLoader,        # PDF files
    TextLoader,         # .txt files  
    WebBaseLoader,      # web pages (uses BeautifulSoup)
    CSVLoader,          # CSV files
    NotionDBLoader,     # Notion databases (needs token)
    YoutubeLoader,      # YouTube transcripts
    GitLoader,          # Git repository files
)

# Load a PDF
loader = PyPDFLoader("quarterly_report.pdf")
documents = loader.load()  # returns list of Document objects with page_content + metadata
print(f"Loaded {len(documents)} pages")
print(f"Metadata: {documents[0].metadata}")
# → {'source': 'quarterly_report.pdf', 'page': 0}

# Load any webpage
loader = WebBaseLoader("https://docs.python.org/3/library/asyncio.html")
docs = loader.load()

# Then split → embed → store in one pipeline:
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Single line retrieval:
results = retriever.invoke("What are the revenue highlights?")
```

---

### LangSmith — Invisible Until You Need It, Essential When You Do

**What it captures for every LangChain/LangGraph run**:
- Complete prompt text sent to LLM (including system prompt)
- Full LLM response
- Token counts and costs
- Every node in the graph and its input/output
- Duration of each step
- Any errors with full stack traces

**The debug workflow**:
```python
# Set up tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "ls__your_key"
os.environ["LANGSMITH_PROJECT"] = "debugging-agent-v2"

# Run your agent (automatically traced)
result = agent.invoke({"query": "Why is my agent giving wrong answers?"})

# Visit https://smith.langchain.com
# → Click the trace for this run
# → See: which retrieval chunks were used
# → See: what the LLM actually received
# → See: which node took 45 seconds
# → See: the exact error message with context
```

**The non-LangChain way** — use the `@traceable` decorator:
```python
from langsmith import traceable

@traceable(name="my-rag-retrieval")
def retrieve(query: str) -> list[str]:
    # This function's input/output is now traced in LangSmith
    return collection.query(query_texts=[query], n_results=3)["documents"][0]
```

---

### CrewAI — Understanding the Internals

When you `crew.kickoff()`, CrewAI:
1. Takes the first task's `description` + agent's `role`, `goal`, `backstory`
2. Constructs a system prompt: "You are {role}. Your goal is {goal}. Background: {backstory}."
3. Adds the task description and any context from previous tasks
4. Calls the LLM with any tools the agent has
5. Processes tool calls in a ReAct loop
6. Returns the `expected_output`
7. Passes this output as `context` to the next task

Understanding this means you can debug CrewAI the same way you debug any agent — look at what prompts are being sent.

```python
# Debug mode — shows all prompts and responses
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,          # prints step-by-step execution
    full_output=True,      # returns all task outputs, not just final
)

result = crew.kickoff()
print(result.raw)           # final task output
print(result.tasks_output)  # list of all task outputs
print(result.token_usage)   # token counts and cost

# Access individual task results:
for task_output in result.tasks_output:
    print(f"Task: {task_output.description[:50]}")
    print(f"Output: {task_output.raw[:200]}")
```

**CrewAI's Memory System** — what agents remember:
```python
# Short-term memory: current conversation within one task (always on)
# Long-term memory: stored in SQLite, persists across crew runs (opt-in)
# Entity memory: extracted entities (people, places, concepts) — opt-in
# Contextual memory: combined recent interactions — opt-in

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,            # enable long-term + entity + contextual memory
    embedder={              # needed for vector memory storage
        "provider": "google",
        "config": {"model": "models/embedding-001"}
    }
)
```

---

### AutoGen — When Agents Talk to Each Other

**The mental model**: AutoGen agents are like Slack users. They take turns sending messages in a group chat. The conversation ends when a termination condition is met.

```python
# The "termination message" pattern
is_done = lambda msg: msg.get("content", "").endswith("TERMINATE")

# Two-agent code review:
coder = autogen.AssistantAgent("coder", llm_config=config,
    system_message="Write Python code. End with TERMINATE when done.")

reviewer = autogen.AssistantAgent("reviewer", llm_config=config,
    system_message="Review code for bugs, style, security. Say TERMINATE when approved.")

# GroupChat — more than 2 agents
groupchat = autogen.GroupChat(
    agents=[coder, reviewer, tester],
    messages=[],
    max_round=20,
    speaker_selection_method="round_robin"  # or "auto" for LLM-selected speaker
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)
```

---

## Framework Maturity & Production Considerations

| Framework | GitHub Stars | Age | Active Maintainers | Production Deployments |
|-----------|-------------|-----|-------------------|----------------------|
| LangChain | 90K+ | 2022 | 15+ | Thousands |
| LangGraph | 10K+ | 2024 | 5+ | Hundreds |
| CrewAI | 25K+ | 2024 | 3+ | Hundreds |
| AutoGen | 30K+ | 2023 | 5+ | Research-heavy |

**Honest assessment**:
- **LangChain**: Most mature, most integrations, some abstraction debt from rapid early growth
- **LangGraph**: Best production architecture for complex agents, growing fast
- **CrewAI**: Best DX for multi-agent prototyping, some rough edges in production
- **AutoGen**: Best for code-execution workflows, less suited for non-code tasks

---

## Common Pitfalls — Week 3

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Using LangChain LCEL for stateful agent | State management becomes messy | Switch to LangGraph for any agent with loops |
| Missing `context=[prev_task]` in CrewAI | Writer agent doesn't know what researcher found | Always connect tasks with `context=` |
| Not setting `verbose=True` during dev | Can't see what agents are doing | Use verbose during development, disable in prod |
| LangSmith not capturing traces | Missed bugs in production | Set all 3 env vars (`API_KEY`, `TRACING=true`, `PROJECT`) |
| LangGraph state annotation missing | Messages get overwritten not appended | Use `Annotated[list, operator.add]` for lists |
| AutoGen code execution without Docker | Security risk | Always `use_docker=True` in production |
| CrewAI with too many tasks | Context loss, very slow | Keep crews focused: 3-5 tasks max per crew |
- `ex2_langchain_rag_chain.py` — LCEL chain: retrieve → prompt → LLM → parse
- `ex3_crewai_research_team.py` — researcher + writer crew
- `ex4_framework_comparison.py` — solve same task with two frameworks, compare

## Checklist
- [ ] Built a LangGraph agent with StateGraph — at least 3 nodes, 1 conditional edge
- [ ] Traced a LangGraph run in LangSmith
- [ ] Built a CrewAI crew with at least 2 agents and 2 tasks
- [ ] Understood what each framework is doing under the hood (read source for 1 node)
- [ ] Chose a framework for your phase project and justified the choice
