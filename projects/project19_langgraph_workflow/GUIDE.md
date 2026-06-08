# LangGraph Code Review Workflow — Build Guide

## Prerequisites
```bash
pip install -r requirements.txt
```

---

## Phase 1 — State & Graph Fundamentals

### 1.1 Why TypedDict + Annotated?
LangGraph passes state between nodes. Each node receives the current state and returns
a **partial update**. The `Annotated` reducers define how updates merge:

```python
import operator
from typing import Annotated, TypedDict, Optional

class ReviewState(TypedDict):
    # Annotated[list, operator.add] = append-only list (new items added, not replaced)
    messages: Annotated[list, operator.add]

    # Plain fields = last-write-wins (node's return value replaces previous)
    code: str
    language: str
    security_issues: list[str]
    quality_score: float
    review: str
    human_feedback: Optional[str]
    status: str   # pending | approved | needs_revision | finalized
```

### 1.2 Node function signature
```python
def my_node(state: ReviewState) -> dict:
    # Read from state
    code = state["code"]
    # Return only the fields you want to update
    return {"security_issues": ["SQL injection risk on line 12"]}
```

**Checkpoint:** `python -c "from src.graph.state import ReviewState; print('OK')`

---

## Phase 2 — Nodes

### 2.1 parse_code node
```python
import ast
from src.graph.state import ReviewState

def parse_code(state: ReviewState) -> dict:
    code = state["code"]
    try:
        tree = ast.parse(code)
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        return {
            "parsed_info": {"functions": functions, "classes": classes, "lines": len(code.splitlines())},
            "status": "parsed",
            "messages": [{"role": "system", "content": f"Parsed: {len(functions)} functions, {len(classes)} classes"}],
        }
    except SyntaxError as e:
        return {"status": "syntax_error", "messages": [{"role": "error", "content": str(e)}]}
```

### 2.2 generate_review node (LLM call)
```python
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate

llm = ChatLiteLLM(model=cfg.model, temperature=0.1)

review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior software engineer performing code review."),
    ("human", """Review this {language} code:
```{language}
{code}
```
Security issues found: {security_issues}
Quality score: {quality_score}/10

Write a thorough code review with: summary, issues list, specific suggestions.""")
])

def generate_review(state: ReviewState) -> dict:
    chain = review_prompt | llm
    response = chain.invoke(state)
    return {
        "review": response.content,
        "status": "review_ready",
        "messages": [{"role": "assistant", "content": response.content}],
    }
```

### 2.3 human_approval node (with interrupt)
```python
from langgraph.types import interrupt

def human_approval(state: ReviewState) -> dict:
    # This pauses execution and waits for human input
    feedback = interrupt({
        "review": state["review"],
        "question": "Approve this review? (approve / revise: <your notes>)",
    })
    # Execution resumes here after human provides input
    if feedback.lower().startswith("approve"):
        return {"status": "approved", "human_feedback": feedback}
    else:
        return {
            "status": "needs_revision",
            "human_feedback": feedback.replace("revise:", "").strip(),
        }
```

---

## Phase 3 — Graph Construction

### 3.1 Build the graph
```python
from langgraph.graph import StateGraph, END
from src.graph.state import ReviewState
from src.graph.nodes import parse_code, analyze_security, analyze_quality, generate_review, human_approval, revise_review, finalize_review

def build_graph():
    graph = StateGraph(ReviewState)

    # Add nodes
    graph.add_node("parse_code", parse_code)
    graph.add_node("analyze_security", analyze_security)
    graph.add_node("analyze_quality", analyze_quality)
    graph.add_node("generate_review", generate_review)
    graph.add_node("human_approval", human_approval)
    graph.add_node("revise_review", revise_review)
    graph.add_node("finalize_review", finalize_review)

    # Linear edges
    graph.set_entry_point("parse_code")
    graph.add_edge("parse_code", "analyze_security")
    graph.add_edge("analyze_security", "analyze_quality")
    graph.add_edge("analyze_quality", "generate_review")
    graph.add_edge("generate_review", "human_approval")

    # Conditional routing after approval
    graph.add_conditional_edges(
        "human_approval",
        route_approval,           # function: State → str
        {
            "approved": "finalize_review",
            "needs_revision": "revise_review",
        },
    )
    graph.add_edge("revise_review", "generate_review")  # loop back
    graph.add_edge("finalize_review", END)

    return graph
```

### 3.2 Route function
```python
def route_approval(state: ReviewState) -> str:
    return state["status"]  # "approved" or "needs_revision"
```

**Checkpoint (without checkpointer):**
```python
graph = build_graph().compile()
result = graph.invoke({"code": "def add(a,b): return a+b", "language": "python", "messages": []})
print(result["review"])
```

---

## Phase 4 — Checkpointer + Human-in-the-Loop

### 4.1 Add SQLite checkpointer
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("reviews.db")
graph = build_graph().compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
```

### 4.2 Run with thread_id (enables per-session state)
```python
config = {"configurable": {"thread_id": "review-001"}}

# Run until interrupt
events = list(graph.stream(
    {"code": "...", "language": "python", "messages": []},
    config=config,
))
print("Paused at human_approval")
print("Review:", events[-1]["generate_review"]["review"])

# Human provides input — resume
from langgraph.types import Command
result = graph.invoke(
    Command(resume="approve — well written, approve"),
    config=config,
)
print("Final status:", result["status"])
```

### 4.3 Replay from checkpoint
```python
# Any previous state can be replayed
history = list(graph.get_state_history(config=config))
for state in history:
    print(state.created_at, state.values.get("status"))
```

---

## Phase 5 — FastAPI Streaming

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/review")
async def review_code(request: ReviewRequest):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    async def event_stream():
        async for event in graph.astream(
            {"code": request.code, "language": request.language, "messages": []},
            config=config,
            stream_mode="updates",
        ):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```
