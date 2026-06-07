[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §1 Agentic Stack](guide/01_agentic_stack.md) | [§3 RAG Architecture →](guide/03_rag_architecture.md)

---

## 2. Framework & Tool Selection Guide — The Complete Reference

This section explains every major tool in the agentic AI ecosystem: what it is, why it exists, when to use it, and how to use it. Read this before choosing your stack.

---

### 2.1 The Decision Tree

```
What does your agent need?
│
├─ Single LLM call, structured output?
│    └─ Raw llm.py + Pydantic ← SIMPLEST, always start here
│
├─ ReAct loop (reason + tool calls)?
│    └─ Raw llm.py react_agent() OR LangGraph
│
├─ Complex conditional routing (A→B if X, A→C if Y)?
│    └─ LangGraph StateGraph ← explicit state, testable
│
├─ Multiple specialist agents collaborating?
│    ├─ Simple pipeline (research → write → review)?
│    │    └─ CrewAI ← fastest setup, role-based
│    └─ Complex branching / HITL / checkpointing?
│         └─ LangGraph ← most flexible
│
├─ Agents that have conversations with each other?
│    └─ AutoGen ← conversational, code-exec loops
│
├─ Off-the-shelf document loaders, vector store integrations?
│    └─ LangChain LCEL ← 300+ integrations built in
│
└─ Tracing / debugging agent behavior in production?
     └─ LangSmith ← essential with LangChain/LangGraph
```

---

### 2.2 LiteLLM — The Universal LLM Proxy

**What it is**: A Python library that provides a single, unified API for 100+ LLM providers. It translates your code into provider-specific API formats behind the scenes.

**Why you need it**: Every LLM provider has a different API format, different response structures, and different parameter names. Without LiteLLM, switching from OpenAI to Google means rewriting all your code. With LiteLLM, you change one environment variable.

**Where it's used**: The base layer of everything. Every single LLM call in this project goes through LiteLLM.

**Install**: `pip install litellm`

```python
import litellm

# Works identically for ALL providers — just change the model string
response = litellm.completion(
    model="gemini/gemini-2.0-flash",   # ← swap to "openai/gpt-4o" or "anthropic/claude-3-5-sonnet" 
    messages=[{"role": "user", "content": "Explain RAG in 2 sentences"}],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)
print(f"Tokens: {response.usage.total_tokens}")

# Async version for concurrent calls
response = await litellm.acompletion(model="gemini/gemini-2.0-flash", messages=[...])

# Streaming
for chunk in litellm.completion(model="...", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

**Supported model strings** (prefix/model-name format):
```
"gemini/gemini-2.0-flash"           # Google AI — large context, cheap
"gemini/gemini-1.5-pro"             # Google AI — best quality
"openai/gpt-4o"                     # OpenAI — reliable, widely tested
"openai/gpt-4o-mini"                # OpenAI — cheap and fast
"anthropic/claude-3-5-sonnet"       # Anthropic — best reasoning
"anthropic/claude-3-haiku"          # Anthropic — fastest, cheapest
"groq/llama-3.3-70b-versatile"      # Groq — extremely fast inference
"ollama/llama3.2"                   # Local — free, private, no internet needed
"mistral/mistral-large-latest"      # Mistral — good European alternative
```

**When NOT to use LiteLLM**: Never. It's always the right choice. The only exception is if you need provider-specific features not yet supported (rare).

---

### 2.3 LangChain — Pipelines & Integrations

**What it is**: A framework for building "chains" of LLM calls using the LangChain Expression Language (LCEL). Famous for having 300+ pre-built integrations with document loaders, vector stores, and tools.

**Why it exists**: Rapid prototyping. Instead of writing code to load a PDF, split it, embed it, store it in Chroma, and query it — LangChain gives you pre-built components for each step.

**Where it's used**: RAG pipelines, document processing, any time you want pre-built connectors instead of writing everything from scratch.

**Install**: `pip install langchain langchain-core langchain-community`

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LCEL Pipeline syntax: prompt | model | parser
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at {topic}."),
    ("user", "{question}")
])
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

chain = prompt | model | parser

# Invoke synchronously
result = chain.invoke({"topic": "Python", "question": "What is a decorator?"})

# Stream response
for token in chain.stream({"topic": "Python", "question": "Explain generators"}):
    print(token, end="", flush=True)

# Batch multiple requests
results = chain.batch([
    {"topic": "Python", "question": "What is a closure?"},
    {"topic": "Go", "question": "What is a goroutine?"},
])

# RAG chain example
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | model
    | parser
)
```

**LangChain Strengths**:
- **Document Loaders**: PDF, Word, HTML, JSON, CSV, databases, YouTube transcripts, etc.
- **Text Splitters**: Recursive, semantic, by tokens, by sentences
- **Vector Store Integrations**: Chroma, Pinecone, Weaviate, FAISS, Qdrant — all with same API
- **Agent Toolkits**: SQL, Wikipedia, Bash, Python REPL, Tavily, Serper — ready to use

**When to use**: You need pre-built connectors and fast prototyping. Not ideal for complex stateful agents (use LangGraph instead).

---

### 2.4 LangGraph — Stateful Agent Graphs

**What it is**: A library from the LangChain team for building agents as **directed graphs** with explicit, persistent state. Nodes are Python functions, edges are transitions, and state flows between them.

**Why it exists**: LangChain chains are linear (A→B→C). Real agents have loops, conditional branches, and the need to checkpoint state (so they can resume if they crash). LangGraph adds cycles, conditional routing, and built-in persistence to agent workflows.

**Where it's used**: Complex single agents (multiple steps with branching), multi-agent systems where you want explicit control, HITL (human-in-the-loop) workflows, and any agent where state persistence is needed.

**Install**: `pip install langgraph`

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
import operator

# 1. Define the state — what data flows through all nodes
class ResearchAgentState(TypedDict):
    messages: Annotated[list, operator.add]   # append-only list
    query: str
    search_results: list[str]
    analysis: str
    final_answer: str
    step_count: int
    needs_more_research: bool

# 2. Define nodes — each is a function that returns a state update
def search_node(state: ResearchAgentState) -> dict:
    """Node 1: Perform web search"""
    results = web_search(state["query"])
    return {
        "search_results": results,
        "step_count": state["step_count"] + 1
    }

def analyze_node(state: ResearchAgentState) -> dict:
    """Node 2: Analyze search results"""
    context = "\n".join(state["search_results"])
    analysis = get_text(chat([{
        "role": "user",
        "content": f"Analyze these search results for: {state['query']}\n\n{context}"
    }]))
    needs_more = "insufficient" in analysis.lower() or "need more" in analysis.lower()
    return {"analysis": analysis, "needs_more_research": needs_more}

def answer_node(state: ResearchAgentState) -> dict:
    """Node 3: Generate final answer"""
    answer = get_text(chat([{
        "role": "user",
        "content": f"Based on this analysis, answer: {state['query']}\n\n{state['analysis']}"
    }]))
    return {"final_answer": answer}

# 3. Router function — returns the name of the next node
def route_after_analysis(state: ResearchAgentState) -> Literal["search", "answer"]:
    if state["needs_more_research"] and state["step_count"] < 3:
        return "search"   # Loop back for more research
    return "answer"       # Proceed to final answer

# 4. Build and compile the graph
graph = StateGraph(ResearchAgentState)
graph.add_node("search", search_node)
graph.add_node("analyze", analyze_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("search")
graph.add_edge("search", "analyze")
graph.add_conditional_edges("analyze", route_after_analysis, {
    "search": "search",   # loop back
    "answer": "answer"    # proceed
})
graph.add_edge("answer", END)

# With checkpointing (state persists to disk/DB)
checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "research-session-42"}}
result = agent.invoke({
    "query": "Latest breakthroughs in quantum computing 2025",
    "search_results": [],
    "analysis": "",
    "final_answer": "",
    "step_count": 0,
    "needs_more_research": False
}, config=config)

print(result["final_answer"])

# Resume paused agent (human-in-the-loop)
agent_hitl = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["answer"]  # pause before answering for human review
)
```

**LangGraph Key Concepts**:
- `StateGraph`: The container for your agent graph
- `TypedDict`: Typed state schema — all agents in the graph share this state
- `Annotated[list, operator.add]`: Makes the field append-only (for messages)
- `add_conditional_edges`: Routing based on state — the "brain" of complex agents
- `MemorySaver`: In-memory checkpoint (use `SqliteSaver` or `PostgresSaver` for production)
- `interrupt_before`: Pause graph before a specific node (HITL pattern)

**When to use**: Any production agent with more than 3 steps. The explicit state and graph structure make testing and debugging much easier than raw loops.

---

### 2.5 CrewAI — Role-Based Multi-Agent Orchestration

**What it is**: A framework for building teams of AI agents where each agent has a **role**, **goal**, and **backstory**. Agents are assigned **tasks** and collaborate sequentially or hierarchically.

**Why it exists**: Decomposing complex tasks into specialist agents is a powerful pattern (researcher + writer + reviewer). CrewAI makes this natural by letting you define agents in role-based language, similar to describing a human team.

**Where it's used**: Content creation pipelines, research workflows, code review systems, any workflow that can be described as "Agent A does X, then Agent B does Y with A's output."

**Install**: `pip install crewai crewai-tools`

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool

search_tool = SerperDevTool()    # web search
file_tool = FileReadTool()       # read files

# Define specialist agents with rich role descriptions
research_analyst = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments and synthesize comprehensive findings",
    backstory="""You are an expert researcher with 15 years of experience at leading 
    think tanks. You're known for finding obscure but highly relevant sources and 
    for presenting information with appropriate confidence levels.""",
    tools=[search_tool],
    llm="gemini/gemini-2.0-flash",
    verbose=True,
    max_iter=5,          # max reasoning iterations
    memory=True,         # remember past actions in this run
)

tech_writer = Agent(
    role="Technical Content Writer",
    goal="Transform complex research into clear, engaging technical content",
    backstory="""You are a technical writer who has written documentation for 
    Google, AWS, and OpenAI. You excel at explaining difficult concepts simply 
    without sacrificing accuracy.""",
    tools=[],
    llm="gemini/gemini-2.0-flash",
    verbose=True,
)

quality_reviewer = Agent(
    role="Quality Assurance Specialist",
    goal="Ensure content is accurate, complete, and free of errors",
    backstory="""You are a meticulous reviewer who catches logical errors, 
    factual mistakes, and unclear explanations. You are known for your 
    high standards.""",
    tools=[],
    llm="gemini/gemini-2.0-flash",
    verbose=True,
)

# Define tasks (with dependencies via context=[])
research_task = Task(
    description="Research the current state of AI agents in 2025, including key frameworks, use cases, and market adoption",
    expected_output="Comprehensive research summary (800+ words) with key findings, statistics, and citations",
    agent=research_analyst,
    output_file="research.md"   # save output to file
)

writing_task = Task(
    description="Write a 600-word technical blog post based on the research",
    expected_output="Polished blog post with title, introduction, 3 main sections, and conclusion",
    agent=tech_writer,
    context=[research_task],    # receives research_task output automatically
    output_file="draft.md"
)

review_task = Task(
    description="Review the blog post for accuracy, clarity, and completeness. Provide specific improvements.",
    expected_output="Reviewed post with inline corrections and a quality score (1-10)",
    agent=quality_reviewer,
    context=[research_task, writing_task],
    output_file="final.md"
)

# Assemble and run
crew = Crew(
    agents=[research_analyst, tech_writer, quality_reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,   # tasks run in order
    verbose=True,
    memory=True,                   # crew-level memory
    embedder={"provider": "google", "config": {"model": "models/embedding-001"}}
)

# Kick off with dynamic inputs
result = crew.kickoff(inputs={
    "topic": "AI agent frameworks comparison 2025",
    "target_audience": "Senior software engineers"
})

print(result.raw)          # final output text
print(result.token_usage)  # cost tracking
```

**Process Types**:
- `Process.sequential`: Tasks run one after another. Each task gets the output of the previous.
- `Process.hierarchical`: A manager LLM delegates tasks to worker agents dynamically.

**When to use**: Multi-step pipelines where each step has a clear role and the roles can be described in natural language. Excellent for content creation, research, and report generation.

**When NOT to use**: Complex conditional logic, real-time streaming, or when you need fine-grained control over agent communication.

---

### 2.6 AutoGen — Conversational Multi-Agent Framework

**What it is**: A Microsoft Research framework where agents communicate by **chatting with each other** in a conversation loop. One agent generates code, another executes it and reports the result, and they iterate until done.

**Why it exists**: Some tasks are best solved through dialogue — debate, code generation with automatic testing, and multi-perspective analysis. AutoGen models this as agent conversations.

**Where it's used**: Code generation with execution feedback, research debates, multi-perspective analysis, automated testing agents.

**Install**: `pip install pyautogen`

```python
import autogen

# Configuration
llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-..."}],
    "temperature": 0,
    "timeout": 120,
}

# The coding assistant agent — generates and fixes code
coder = autogen.AssistantAgent(
    name="senior_coder",
    llm_config=llm_config,
    system_message="""You are a senior Python engineer. Write complete, tested, 
    production-ready code. When code has bugs, analyze the error and fix it. 
    Say TERMINATE when the task is fully complete."""
)

# The user proxy — executes code and provides human-like feedback
executor = autogen.UserProxyAgent(
    name="code_executor",
    human_input_mode="NEVER",        # fully automated, no human input
    max_consecutive_auto_reply=10,   # max back-and-forth rounds
    code_execution_config={
        "work_dir": "/tmp/autogen",
        "use_docker": True,          # run in Docker for safety
        "timeout": 60,
    },
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)

# Start the agent conversation
executor.initiate_chat(
    coder,
    message="""Write a Python function that:
    1. Takes a list of URLs
    2. Fetches each URL concurrently using asyncio
    3. Returns a dict mapping URL → (status_code, response_time_ms)
    4. Handles timeouts and connection errors gracefully
    Include tests using pytest."""
)
# AutoGen handles the loop: coder writes → executor runs → reports errors → coder fixes → repeat
```

**When to use**: Code generation tasks where you want automatic execution and error correction. Research tasks where you want agents to debate. Generally more experimental/research-oriented than CrewAI.

---

### 2.7 LangSmith — Agent Observability & Evaluation

**What it is**: A hosted platform from the LangChain team for tracing every LLM call, measuring agent performance, managing test datasets, and running evaluations.

**Why you need it**: Without tracing, debugging an agent failure is like debugging without a debugger. LangSmith shows you exactly what prompt went to the LLM, what the LLM responded, which tool was called, and how long each step took — all in a visual UI.

**Where it's used**: Any agent built with LangChain or LangGraph automatically gets traced when you set the env vars. Also works with raw code via the `@traceable` decorator.

**Setup**:
```python
import os
# Add these to your .env file
os.environ["LANGSMITH_API_KEY"] = "ls__your_key_here"
os.environ["LANGSMITH_TRACING"] = "true"           # enable tracing
os.environ["LANGSMITH_PROJECT"] = "my-agent-v1"   # project name

# Now every LangChain/LangGraph call is automatically traced
agent.invoke({"query": "..."})
# Visit: https://smith.langchain.com to see the full trace

# For custom code, use the @traceable decorator
from langsmith import traceable

@traceable(name="my-react-agent")
def my_agent(query: str) -> str:
    # all LLM calls inside here are automatically traced
    return react_agent(query)
```

**LangSmith Evaluation**:
```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Create a dataset of test cases
client = langsmith.Client()
dataset = client.create_dataset("agent-eval-v1")
client.create_examples(
    inputs=[{"query": "What is 2+2?"}, {"query": "Capital of France?"}],
    outputs=[{"answer": "4"}, {"answer": "Paris"}],
    dataset_id=dataset.id
)

# Run evaluation
results = evaluate(
    my_agent,
    data="agent-eval-v1",
    evaluators=[LangChainStringEvaluator("cot_qa")],
    experiment_prefix="baseline-v1"
)
```

**When to use**: Always when using LangChain/LangGraph in production. It's the single best tool for understanding what your agent is actually doing.

---

### 2.8 Pydantic — Data Validation & Structured Output

**What it is**: A Python library for data validation and settings management using Python type annotations. Validates that data matches your schema at runtime and provides clear error messages.

**Why you need it for agents**: LLMs return unstructured text. Pydantic gives you a reliable way to parse and validate that text into typed Python objects. It's also how FastAPI validates API request/response bodies.

**Where it's used**: (1) Parsing LLM JSON output, (2) FastAPI request/response models, (3) Agent configuration, (4) Inter-agent message schemas.

**Install**: `pip install pydantic` (Pydantic v2 is default since 2023)

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
import json, re

# Define what you want the LLM to return
class CodeReviewResult(BaseModel):
    severity: Literal["critical", "high", "medium", "low", "none"]
    issues: list[str] = Field(default_factory=list, description="List of identified issues")
    suggestions: list[str] = Field(default_factory=list)
    overall_score: int = Field(ge=0, le=10, description="Code quality score 0-10")
    approved: bool
    explanation: str

    @field_validator("issues")
    @classmethod
    def issues_not_empty_on_critical(cls, v, info):
        if info.data.get("severity") == "critical" and not v:
            raise ValueError("Critical severity must include at least one issue")
        return v

def parse_llm_json(raw: str, model_class: type[BaseModel]) -> BaseModel:
    """Parse and validate LLM output into a Pydantic model with retries."""
    # Strip markdown code fences if present
    clean = re.sub(r"```json?\s*|\s*```", "", raw).strip()
    
    # Parse JSON
    data = json.loads(clean)
    
    # Validate with Pydantic (raises ValidationError with clear messages on failure)
    return model_class(**data)

# Usage in agent
def review_code(code: str) -> CodeReviewResult:
    raw = get_text(chat(
        messages=[{"role": "user", "content": f"Review this code:\n```python\n{code}\n```"}],
        system=f"""Review the code and respond ONLY with valid JSON matching this schema:
{CodeReviewResult.model_json_schema()}
No markdown, no explanation — just the JSON object."""
    ))
    
    for attempt in range(3):
        try:
            return parse_llm_json(raw, CodeReviewResult)
        except (json.JSONDecodeError, ValueError) as e:
            raw = get_text(chat([{"role": "user", "content": f"Previous JSON was invalid: {e}. Return corrected JSON only."}]))
    raise ValueError("Could not parse LLM output after 3 attempts")
```

**When to use**: Always when you need structured output from an LLM. Also use for all FastAPI request/response models.

---

### 2.9 FastAPI — Async Agent API Framework

**What it is**: A modern, high-performance Python web framework for building APIs. Built on Starlette (async) and Pydantic (validation). Generates OpenAPI (Swagger) documentation automatically.

**Why it's the right choice for agents**: Agents are async (LLM calls are I/O bound). FastAPI is natively async, meaning it can handle hundreds of concurrent agent requests without blocking. It also handles request validation, serialization, and documentation for free.

**Where it's used**: The HTTP layer that exposes your agent to the outside world. POST /agent/run, GET /agent/result/{id}, streaming endpoints.

**Install**: `pip install fastapi uvicorn[standard]`

```python
from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio, os, json

app = FastAPI(
    title="Production Agent API",
    description="AI Agent with RAG, tools, and streaming",
    version="1.0.0",
)

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=10000)
    session_id: str = "default"
    stream: bool = False
    max_steps: int = Field(default=10, ge=1, le=50)

class AgentResponse(BaseModel):
    result: str
    session_id: str
    steps_taken: int
    cost_usd: float
    run_id: str

# Dependency injection for auth
async def get_current_user(x_api_key: str = Header(...)):
    valid_key = os.getenv("API_KEY")
    if x_api_key != valid_key:
        raise HTTPException(401, "Invalid API key")
    return {"user": "authenticated"}

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(req: AgentRequest, user=Depends(get_current_user)):
    """Run agent synchronously — use for queries < 30 seconds."""
    result, steps, cost = await execute_agent_async(req.query, req.session_id, req.max_steps)
    return AgentResponse(result=result, session_id=req.session_id,
                         steps_taken=steps, cost_usd=cost, run_id=generate_id())

@app.post("/agent/stream")
async def stream_agent(req: AgentRequest) -> StreamingResponse:
    """Stream tokens in real-time via Server-Sent Events."""
    async def event_generator():
        async for token in stream_agent_tokens(req.query):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODEL"), "version": "1.0.0"}

# Run: uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

### 2.10 Celery + Redis — Background Task Queue

**Celery — What it is**: A distributed task queue. Submit a long-running task (your agent), get a task ID back immediately, run the task in a separate worker process, poll for the result.

**Why you need it**: HTTP requests timeout after 30-60 seconds. Agents can take 2-10 minutes. Celery decouples the "submit" from the "execute" — the API responds instantly with a task ID, the worker does the actual work.

**Redis — What it is**: An in-memory data store used by Celery as both the message **broker** (queue where tasks wait) and the **result backend** (where completed task results are stored). Redis is also used for caching, session storage, and rate limiting.

**Install**: `pip install celery redis`

```python
# celery_app.py — Celery configuration
from celery import Celery
import os

app = Celery(
    "agent_workers",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),   # task queue
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/1"),  # results store
    include=["worker.tasks"]
)
app.conf.update(
    task_serializer="json",
    result_expires=86400,          # results expire after 24 hours
    task_track_started=True,       # enables STARTED state
    worker_prefetch_multiplier=1,  # one task per worker at a time (good for long tasks)
    worker_max_tasks_per_child=50, # restart worker every 50 tasks (prevents memory leaks)
    task_acks_late=True,           # only ack task after completion (prevents lost tasks)
)

# worker/tasks.py — the actual tasks
from celery_app import app as celery_app
from llm import chat, get_text

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def run_agent_task(self, query: str, session_id: str, user_id: str) -> dict:
    """Long-running agent task."""
    try:
        self.update_state(state="STARTED", meta={"progress": 10, "status": "Planning..."})
        plan = create_plan(query)
        
        self.update_state(state="STARTED", meta={"progress": 30, "status": "Executing..."})
        result = execute_plan(plan, query)
        
        self.update_state(state="STARTED", meta={"progress": 90, "status": "Finalizing..."})
        return {"result": result, "session_id": session_id, "cost": calculate_cost()}
        
    except Exception as exc:
        raise self.retry(exc=exc)   # Auto-retry on failure

# api.py — submit and poll
from fastapi import FastAPI
from celery.result import AsyncResult
from worker.tasks import run_agent_task

@app.post("/agent/submit")
async def submit(req: AgentRequest):
    task = run_agent_task.delay(req.query, req.session_id, "user123")
    return {"task_id": task.id, "status": "queued"}

@app.get("/agent/status/{task_id}")
async def get_status(task_id: str):
    res = AsyncResult(task_id)
    if res.state == "SUCCESS":
        return {"status": "done", "result": res.result["result"]}
    elif res.state == "FAILURE":
        return {"status": "error", "error": str(res.result)}
    else:
        return {"status": res.state, "progress": (res.info or {}).get("progress", 0)}

# Start worker: celery -A celery_app worker --loglevel=info --concurrency=4
# Monitor: celery -A celery_app flower --port=5555
```

**Redis as Cache**:
```python
import redis, hashlib, json

r = redis.Redis(host="localhost", port=6379, db=2, decode_responses=True)

def cached_agent_call(query: str, ttl_seconds: int = 3600) -> str:
    key = f"cache:{hashlib.sha256(query.encode()).hexdigest()}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)["result"]
    result = react_agent(query)
    r.setex(key, ttl_seconds, json.dumps({"result": result}))
    return result
```

---

### 2.11 ChromaDB — Developer-Friendly Vector Database

**What it is**: An open-source vector database that runs locally (as a Python library) or as a server. Stores text documents + their embeddings + metadata. Supports filtering by metadata and semantic search.

**Why it's the development standard**: Zero configuration. `pip install chromadb`, create a client, add documents, query. Works completely locally with no external services. Persistent storage out of the box.

**Where it's used**: Development and small-to-medium production (up to ~1M vectors). The go-to choice for RAG prototyping.

**Install**: `pip install chromadb`

```python
import chromadb
from chromadb.utils import embedding_functions

# Setup (persistent to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Use a local embedding model (no API key needed)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or load collection
collection = client.get_or_create_collection(
    name="company_docs",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# Add documents (ChromaDB handles embedding)
collection.add(
    documents=["Python is a general-purpose language", "FastAPI is for building APIs"],
    metadatas=[{"source": "wiki", "type": "definition"}, {"source": "docs", "type": "tool"}],
    ids=["doc_001", "doc_002"]
)

# Query — semantic search
results = collection.query(
    query_texts=["what language should I use for web APIs?"],
    n_results=3,
    where={"type": "tool"},    # metadata filter
    include=["documents", "distances", "metadatas"]
)

# Update a document
collection.update(ids=["doc_001"], documents=["Updated content here"])

# Delete
collection.delete(ids=["doc_001"])
```

---

### 2.12 Qdrant — Production-Scale Vector Database

**What it is**: A Rust-based vector database built for production. Handles billions of vectors, supports complex filtering, payload indexing, hybrid search (dense + sparse), and horizontal scaling.

**Why it's the production choice**: ChromaDB is great for development but not designed for high throughput or large scale. Qdrant is built for production: it's fast, scalable, has a REST API, supports GRPC, and can run as a cluster.

**Where it's used**: Production RAG systems with large document corpora, high query throughput, or complex multi-tenant deployments.

**Install**: `pip install qdrant-client` + `docker run -p 6333:6333 qdrant/qdrant`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# Create collection (384 dimensions for MiniLM)
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Index documents with rich payloads (metadata)
points = [
    PointStruct(
        id=i,
        vector=encoder.encode(doc["text"]).tolist(),
        payload={
            "text": doc["text"],
            "source": doc["source"],
            "date": doc["date"],
            "category": doc["category"],
            "user_id": doc["user_id"],  # multi-tenant: filter by user
        }
    )
    for i, doc in enumerate(documents)
]
client.upsert(collection_name="knowledge_base", points=points)

# Search with complex filters
results = client.search(
    collection_name="knowledge_base",
    query_vector=encoder.encode("API authentication best practices").tolist(),
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="security")),
            FieldCondition(key="user_id", match=MatchValue(value="user_123")),
        ],
        should=[
            FieldCondition(key="date", range=Range(gte="2024-01-01"))
        ]
    ),
    limit=5,
    with_payload=True,
    score_threshold=0.7   # only return results above 70% similarity
)

for result in results:
    print(f"Score: {result.score:.3f} | {result.payload['text'][:100]}")
```

---

### 2.13 FAISS — In-Memory Vector Search Library

**What it is**: Facebook AI Similarity Search — a C++ library with Python bindings for fast vector search. Runs entirely in memory (or on disk), no server required. The fastest option for batch similarity search.

**Why it exists**: Sometimes you don't need a database — you just need fast similarity search over a fixed set of vectors. FAISS is the library used inside many other vector DBs.

**Where it's used**: Research, batch processing, when you need maximum search speed on a fixed dataset, offline document processing pipelines.

**Install**: `pip install faiss-cpu` (or `faiss-gpu` for GPU acceleration)

```python
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare corpus
texts = ["What is Python?", "How does RAG work?", "Explain transformers"]
vectors = model.encode(texts, convert_to_numpy=True).astype(np.float32)

# Normalize for cosine similarity
faiss.normalize_L2(vectors)

# Create index — options:
# IndexFlatIP: exact search, dot product (use with normalized = cosine)
# IndexIVFFlat: approximate, faster for large datasets
# IndexHNSWFlat: graph-based, best for real-time search
dim = vectors.shape[1]  # 384
index = faiss.IndexFlatIP(dim)  # exact inner product (= cosine after normalization)
index.add(vectors)

# Search
query_vec = model.encode(["tell me about Python"]).astype(np.float32)
faiss.normalize_L2(query_vec)
distances, indices = index.search(query_vec, k=2)

for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"Result {i+1}: '{texts[idx]}' (similarity: {dist:.3f})")

# Persist to disk
faiss.write_index(index, "index.faiss")
index2 = faiss.read_index("index.faiss")
```

---

### 2.14 sentence-transformers — Local Embedding Models

**What it is**: A Python library providing 100+ pre-trained sentence embedding models. Runs locally on CPU/GPU with no API calls or costs.

**Why it's the default choice for embeddings**: Zero cost, no API key, no data sent externally, works offline. The `all-MiniLM-L6-v2` model is 80MB and runs at ~1000 sentences/second on a laptop CPU.

**Where it's used**: The embedding step in every RAG pipeline. Converting text → vectors for storage in ChromaDB/Qdrant/FAISS.

**Install**: `pip install sentence-transformers`

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Bi-encoder: fast, for retrieval (embed query + documents separately)
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Encode single string or list
single = bi_encoder.encode("What is machine learning?")           # shape: (384,)
batch = bi_encoder.encode(["text1", "text2", "text3"],            # shape: (3, 384)
                          batch_size=32, show_progress_bar=True)

# Cross-encoder: slow but very accurate, for reranking top results
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score query-document pairs (much slower but more accurate)
pairs = [("What is Python?", "Python is a programming language"),
         ("What is Python?", "Python is a type of snake")]
scores = cross_encoder.predict(pairs)  # [0.98, 0.02]

# Full retrieval + reranking pipeline
def retrieve_and_rerank(query: str, docs: list[str], top_k: int = 3) -> list[str]:
    # Step 1: Fast retrieval with bi-encoder (get top 20)
    query_vec = bi_encoder.encode(query)
    doc_vecs = bi_encoder.encode(docs)
    
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity([query_vec], doc_vecs)[0]
    top_20_idx = scores.argsort()[-20:][::-1]
    candidates = [docs[i] for i in top_20_idx]
    
    # Step 2: Accurate reranking with cross-encoder
    pairs = [(query, doc) for doc in candidates]
    rerank_scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(rerank_scores, candidates), reverse=True)
    
    return [doc for _, doc in ranked[:top_k]]
```

---

### 2.15 Tavily — Web Search for Agents

**What it is**: A search API specifically designed for AI agents. Returns clean, structured results (title, URL, content extract) without HTML/JS noise. Much better than raw Google/Bing APIs for agent use.

**Why it beats raw search APIs**: Returns agent-friendly data: summaries, relevant content chunks, and context. Handles news, research, and general search. Designed to be injected directly into LLM context.

**Where it's used**: The `web_search` tool in ReAct agents. Any agent that needs current information beyond its training cutoff.

**Install**: `pip install tavily-python`

```python
from tavily import TavilyClient

client = TavilyClient(api_key="tvly-your-key")

# Simple search
results = client.search(
    query="LLM agent frameworks comparison 2025",
    max_results=5,
    search_depth="advanced",    # "basic" (faster) or "advanced" (more results)
    include_raw_content=False,  # True for full page content
    include_images=False,
)

# Each result: {"title": "...", "url": "...", "content": "...", "score": 0.95}
for r in results["results"]:
    print(f"[{r['score']:.2f}] {r['title']}: {r['content'][:200]}")

# As an agent tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web. Use for current events, news, prices, or facts."""
    response = client.search(query=query, max_results=max_results)
    return "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in response["results"]
    ])
```

---

### 2.16 rank-bm25 — Keyword Search for Hybrid RAG

**What it is**: A Python implementation of BM25 (Best Match 25) — the industry-standard keyword ranking algorithm used by search engines like Elasticsearch. Ranks documents by keyword overlap.

**Why it matters for RAG**: Vector search is great for semantic similarity but misses exact keyword matches (product codes, names, technical terms). BM25 catches these. Combining both (hybrid search) gives the best retrieval quality.

**Where it's used**: The keyword search component of hybrid RAG pipelines. Often combined with vector search using Reciprocal Rank Fusion (RRF).

**Install**: `pip install rank-bm25`

```python
from rank_bm25 import BM25Okapi
import numpy as np

corpus = [
    "FastAPI is a Python web framework for building APIs",
    "LangGraph is used for building stateful agent workflows",
    "BM25 is a classic information retrieval algorithm",
    "ChromaDB is a vector database for embeddings",
]

# Tokenize (simple whitespace split; use NLTK for production)
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# BM25 keyword search
query = "API framework Python"
tokenized_query = query.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)  # array of scores per document

# Get ranked results
ranked_indices = np.argsort(bm25_scores)[::-1]
for idx in ranked_indices[:3]:
    print(f"BM25 {bm25_scores[idx]:.3f}: {corpus[idx]}")

# Full hybrid search: BM25 + Vector + RRF
def hybrid_search(query: str, corpus: list[str], k: int = 3) -> list[str]:
    # BM25 ranking
    bm25_scores = BM25Okapi([d.lower().split() for d in corpus]).get_scores(query.lower().split())
    bm25_ranks = np.argsort(bm25_scores)[::-1]
    
    # Vector ranking
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(corpus + [query])
    from sklearn.metrics.pairwise import cosine_similarity
    vec_scores = cosine_similarity([vectors[-1]], vectors[:-1])[0]
    vec_ranks = np.argsort(vec_scores)[::-1]
    
    # Reciprocal Rank Fusion
    rrf_scores = {}
    for rank, doc_idx in enumerate(bm25_ranks):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (60 + rank + 1)
    for rank, doc_idx in enumerate(vec_ranks):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (60 + rank + 1)
    
    top_k = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [corpus[i] for i in top_k]
```

---

### 2.17 Prometheus + Grafana — Metrics & Dashboards

**Prometheus — What it is**: An open-source monitoring system that scrapes metrics from your services on a schedule (pull-based). Stores time-series data. Has a powerful query language (PromQL).

**Grafana — What it is**: A visualization tool that connects to Prometheus (and other data sources) to create dashboards with graphs, alerts, and panels.

**Why you need them**: Without metrics, you're blind. You won't know your agent's P95 latency, cost per hour, error rate, or when something is going wrong until users complain.

**Install**: `pip install prometheus-client` + Docker for Prometheus + Grafana

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics (module level, imported everywhere)
llm_calls_total = Counter("llm_calls_total", "LLM API calls", ["model", "status"])
llm_cost_usd = Counter("llm_cost_usd_total", "LLM spend in USD", ["model"])
llm_latency = Histogram("llm_duration_seconds", "LLM call duration", ["model"],
                         buckets=[0.5, 1, 2, 5, 10, 30, 60])
agent_steps = Histogram("agent_steps_per_run", "Steps per agent run",
                         buckets=[1, 2, 3, 5, 8, 13, 21])
active_agents = Gauge("agents_active", "Currently running agents")

# Instrument your agent
def instrumented_chat(messages: list, **kwargs) -> dict:
    model = kwargs.get("model", MODEL)
    start = time.time()
    try:
        response = chat(messages, **kwargs)
        llm_calls_total.labels(model=model, status="success").inc()
        cost = calc_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        llm_cost_usd.labels(model=model).inc(cost)
        return response
    except Exception as e:
        llm_calls_total.labels(model=model, status="error").inc()
        raise
    finally:
        llm_latency.labels(model=model).observe(time.time() - start)

# Expose /metrics endpoint (scraped by Prometheus every 15s)
start_http_server(9090)
```

---

### 2.18 OpenTelemetry — Distributed Tracing

**What it is**: An open standard for collecting telemetry (traces, metrics, logs) from distributed systems. A single trace shows the full journey of one request across all services.

**Why it matters**: When an agent request fails after 45 seconds, you need to know: was it the LLM call? The vector DB query? The tool execution? OTel traces answer this with precise timing.

**Install**: `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup (once at startup)
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Instrument agent
def traced_react_agent(query: str) -> str:
    with tracer.start_as_current_span("react_agent") as span:
        span.set_attribute("query", query[:200])
        
        for step in range(max_steps):
            with tracer.start_as_current_span(f"step_{step}"):
                with tracer.start_as_current_span("llm_call"):
                    response = chat(messages)
                
                if stop_reason(response) == "tool_calls":
                    for tc in get_tool_calls(response):
                        with tracer.start_as_current_span(f"tool_{tc['name']}") as tool_span:
                            tool_span.set_attribute("tool.name", tc["name"])
                            result = dispatch_tool(tc["name"], tc["arguments"])
                            tool_span.set_attribute("tool.result_length", len(result))
```

---

### 2.19 Docker — Container Packaging

**What it is**: A platform for packaging applications and all their dependencies into lightweight, portable containers. "It works on my machine" → "It works everywhere."

**Why every production agent needs Docker**: Guarantees your agent runs identically in development, CI, staging, and production. Solves dependency conflicts. Enables Kubernetes deployment.

```dockerfile
# Multi-stage build for small image
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH

# Security: run as non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

### 2.20 Framework Comparison Matrix

| Criteria | Raw llm.py | LangChain | LangGraph | CrewAI | AutoGen |
|---------|-----------|-----------|-----------|--------|---------|
| **Learning curve** | Lowest | Medium | Medium | Low | Medium |
| **Flexibility** | Maximum | High | High | Medium | Medium |
| **Multi-agent** | Manual | Limited | ✅ Full | ✅ Full | ✅ Full |
| **Observability** | Manual | LangSmith | LangSmith | Built-in | Built-in |
| **Production maturity** | ✅ | ✅ | ✅ | ✅ | ⚠️ Research |
| **Async support** | ✅ | ✅ | ✅ | Partial | Partial |
| **Checkpointing** | Manual | ❌ | ✅ | ❌ | ❌ |
| **HITL support** | Manual | ❌ | ✅ Native | ❌ | ❌ |
| **Streaming** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Best for** | Learning | RAG/Pipelines | Complex agents | Multi-agent teams | Code gen |

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §1 Agentic Stack](guide/01_agentic_stack.md) | [§3 RAG Architecture →](guide/03_rag_architecture.md)
