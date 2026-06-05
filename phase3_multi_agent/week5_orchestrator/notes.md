# Week 5 — Multi-Agent Systems: Orchestrator Patterns

## What This Week Is About
Single agents hit limits. Complex tasks benefit from specialization — multiple agents with different roles, tools, and expertise collaborating toward a goal. This week covers multi-agent architecture patterns, the orchestrator-worker model, and how to coordinate agents safely.

---

## 1. Why Multi-Agent?

A single LLM context has limits — you can't give one agent expert-level prompts for 5 different domains simultaneously. Multi-agent solves this by:

- **Specialization**: Each agent has a focused role and a tuned system prompt
- **Parallelism**: Multiple agents work simultaneously on different subtasks
- **Verification**: Agents check each other's work (reviewer pattern)
- **Scale**: Tasks too large for one context window can be split across agents

**When NOT to use multi-agent:**
- Simple tasks that one well-prompted agent handles fine
- When latency is critical (multi-agent adds overhead)
- When you haven't gotten a single-agent solution working first

---

## 2. The Orchestrator-Worker Pattern

The most common multi-agent architecture. An **orchestrator** (manager) receives the task, decomposes it, delegates subtasks to **specialist workers**, and synthesizes the results.

```
User Request
     │
     ▼
[Orchestrator Agent]
 - Understands the full task
 - Decides which specialists to involve
 - Delegates with clear instructions
     │
     ├──→ [Research Worker]   → "Find market data for..."
     ├──→ [Code Worker]       → "Write a Python script to..."
     └──→ [Writer Worker]     → "Draft the executive summary..."
     │
     ▼
[Orchestrator synthesizes results]
     │
     ▼
Final Response to User
```

### Basic Implementation

```python
from llm import chat, get_text

# Specialist worker agents
def research_agent(task: str) -> str:
    return get_text(chat(
        messages=[{"role": "user", "content": task}],
        system="""You are a research specialist. Your job is to find and synthesize
information from multiple sources. Always cite your sources. Be thorough and accurate."""
    ))

def code_agent(task: str) -> str:
    return get_text(chat(
        messages=[{"role": "user", "content": task}],
        system="""You are a senior Python engineer. Write clean, tested, documented code.
Always include error handling and type annotations."""
    ))

def writer_agent(task: str, context: str) -> str:
    return get_text(chat(
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nTask: {task}"}],
        system="You are a technical writer. Produce clear, engaging, well-structured content."
    ))

# Orchestrator
def orchestrator(user_request: str) -> str:
    # Step 1: Plan the work
    plan_response = get_text(chat(
        messages=[{"role": "user", "content": f"Break this task into specialist subtasks: {user_request}"}],
        system="""You are a project manager. Break complex tasks into specialist subtasks.
Output JSON: {"subtasks": [{"agent": "research|code|writer", "task": "..."}]}"""
    ))
    
    # Step 2: Execute each subtask
    results = {}
    for subtask in parse_plan(plan_response):
        if subtask["agent"] == "research":
            results["research"] = research_agent(subtask["task"])
        elif subtask["agent"] == "code":
            results["code"] = code_agent(subtask["task"])
    
    # Step 3: Synthesize
    context = "\n\n".join(f"[{k}]: {v}" for k, v in results.items())
    return writer_agent("Write the final response using the context.", context)
```

---

## 3. Agent Roles & System Prompts

The system prompt defines an agent's "personality," expertise, and behavioral constraints. Well-crafted system prompts are the key to effective specialist agents.

### Role Template Pattern

```python
AGENT_ROLES = {
    "researcher": {
        "system": """You are a Senior Research Analyst with 15 years of experience.
EXPERTISE: Finding, validating, and synthesizing information from multiple sources.
APPROACH: Always cross-reference claims. Acknowledge uncertainty. Cite sources.
OUTPUT FORMAT: Structured findings with confidence levels (High/Medium/Low).
NEVER: Fabricate sources. Speculate without flagging it as speculation.""",
        "tools": ["web_search", "arxiv_search", "wikipedia"]
    },
    
    "coder": {
        "system": """You are a Staff Software Engineer specializing in Python and distributed systems.
EXPERTISE: Clean architecture, performance optimization, security best practices.
APPROACH: Write production-quality code with tests. Prefer clarity over cleverness.
OUTPUT FORMAT: Complete, runnable code with docstrings and inline comments.
NEVER: Skip error handling. Use deprecated libraries. Write code with known security issues.""",
        "tools": ["run_code", "read_file", "write_file"]
    },
    
    "critic": {
        "system": """You are a Quality Assurance specialist and adversarial reviewer.
EXPERTISE: Finding flaws, edge cases, security issues, and logical errors.
APPROACH: Actively look for problems. Assume the worst case.
OUTPUT FORMAT: Numbered list of issues with severity (Critical/High/Medium/Low).
NEVER: Give a clean bill of health without thorough examination.""",
        "tools": []
    }
}
```

---

## 4. The Debate Pattern

Two agents with opposing viewpoints argue about a problem. A judge synthesizes the best answer. Excellent for decisions where you want multiple perspectives.

```python
def debate_agent(topic: str, rounds: int = 2) -> str:
    protagonist = []
    antagonist = []
    
    # Initial positions
    pro_response = get_text(chat(
        messages=[{"role": "user", "content": f"Argue FOR: {topic}"}],
        system="You are a strong advocate. Make the most compelling case FOR the position."
    ))
    protagonist.append(pro_response)
    
    ant_response = get_text(chat(
        messages=[{"role": "user", "content": f"Argue AGAINST: {topic}"}],
        system="You are a strong critic. Make the most compelling case AGAINST the position."
    ))
    antagonist.append(ant_response)
    
    # Debate rounds
    for _ in range(rounds):
        pro_response = get_text(chat(
            messages=[{"role": "user", "content": f"Counter this argument: {ant_response}\n\nYour original position: {protagonist[-1]}"}],
            system="You are defending a position. Counter the opponent's strongest points."
        ))
        protagonist.append(pro_response)
        
        ant_response = get_text(chat(
            messages=[{"role": "user", "content": f"Counter this argument: {pro_response}\n\nYour original position: {antagonist[-1]}"}],
            system="You are the critic. Counter the defender's strongest points."
        ))
        antagonist.append(ant_response)
    
    # Judge synthesizes
    debate_transcript = f"FOR:\n{protagonist[-1]}\n\nAGAINST:\n{antagonist[-1]}"
    return get_text(chat(
        messages=[{"role": "user", "content": f"Debate transcript:\n{debate_transcript}\n\nProvide a balanced, nuanced conclusion."}],
        system="You are an impartial judge. Synthesize the best answer from both sides."
    ))
```

---

## 5. Human-in-the-Loop (HITL)

Some agent actions are too consequential to automate fully. HITL lets the agent pause and ask a human before proceeding.

```python
def hitl_agent(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    
    while True:
        response = chat(messages=messages, tools=ALL_TOOLS)
        reason = stop_reason(response)
        messages.append(assistant_message(response))
        
        if reason == "tool_calls":
            for tc in get_tool_calls(response):
                # Check if this action needs human approval
                if requires_approval(tc["name"]):
                    print(f"\n⚠️  Agent wants to: {tc['name']}({tc['arguments']})")
                    approval = input("Approve? (y/n/modify): ")
                    
                    if approval == "n":
                        result = "User denied this action. Try a different approach."
                    elif approval == "modify":
                        new_args = input("Enter new arguments (JSON): ")
                        tc["arguments"] = json.loads(new_args)
                        result = dispatch_tool(tc["name"], tc["arguments"])
                    else:
                        result = dispatch_tool(tc["name"], tc["arguments"])
                else:
                    result = dispatch_tool(tc["name"], tc["arguments"])
                
                messages.append(tool_result_message(tc["id"], result))
        else:
            return get_text(response)

def requires_approval(tool_name: str) -> bool:
    HIGH_RISK_TOOLS = {"send_email", "delete_file", "write_database", "execute_code", "post_social_media"}
    return tool_name in HIGH_RISK_TOOLS
```

---

## 6. Agent Communication Protocols

When agents talk to each other, structure the communication to avoid confusion:

```python
# Agent message format for inter-agent communication
def agent_message(from_agent: str, to_agent: str, task: str, context: dict = None) -> dict:
    return {
        "from": from_agent,
        "to": to_agent,
        "task": task,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
        "requires_response": True
    }

# Task result format
def task_result(agent: str, task_id: str, output: str, confidence: float, issues: list = None) -> dict:
    return {
        "agent": agent,
        "task_id": task_id,
        "output": output,
        "confidence": confidence,  # 0.0 - 1.0
        "issues": issues or [],
        "completed_at": datetime.now().isoformat()
    }
```

---

## 7. CrewAI for Orchestration

CrewAI is the most practical framework for the orchestrator pattern (see Week 3 for full CrewAI guide):

```python
from crewai import Agent, Task, Crew, Process

# Orchestrator approach: hierarchical process
crew = Crew(
    agents=[researcher, analyst, writer, reviewer],
    tasks=[research_task, analysis_task, writing_task, review_task],
    process=Process.hierarchical,  # manager agent orchestrates automatically
    manager_llm="gemini/gemini-2.0-flash",
    verbose=True
)

result = crew.kickoff(inputs={"topic": "AI trends in healthcare 2025"})
```

---

## 8. Avoiding Multi-Agent Failure Modes

| Failure Mode | What Happens | Prevention |
|-------------|-------------|------------|
| **Context loss** | Agent doesn't know what others did | Pass full context between agents |
| **Contradicting outputs** | Agents give conflicting answers | Use a synthesizer/judge agent |
| **Infinite loops** | Agents delegate back and forth | Set max_steps, max_rounds |
| **Hallucination compounding** | Errors multiply across agents | Add a critic/verifier agent |
| **Cost explosion** | 5 agents × 10 turns = 50 LLM calls | Track and cap per-run costs |

---

## Tools & Libraries Used This Week — Deep Dive

### CrewAI vs. LangGraph for Orchestration — The Real Difference

Both can implement the orchestrator-worker pattern. The key difference is the level of control:

**CrewAI's approach**: "Describe what agents should do in natural language. Let the framework handle the mechanics."
```python
# CrewAI: roles, goals, backstories in English
researcher = Agent(role="Research Analyst", goal="Find accurate data", backstory="...")
crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff()
```

**LangGraph's approach**: "Define every state, every transition, every routing decision explicitly."
```python
# LangGraph: explicit graph with typed state
graph = StateGraph(AgentState)
graph.add_node("research", research_fn)
graph.add_conditional_edges("research", should_write_or_search_more)
```

**When to choose CrewAI**: You need to ship fast and your workflow maps naturally to "Agent A does X, then Agent B does Y." Content pipelines, research workflows, report generation.

**When to choose LangGraph**: You need precise control — custom retry logic, specific HITL points, complex branching that CrewAI's sequential/hierarchical modes don't support.

---

### Agent Roles — The Psychology Behind Good Prompts

Why does a "Senior Research Analyst" with 15 years of experience produce better output than just "researcher"? Because:

1. The model has seen thousands of examples of how senior analysts write vs junior ones
2. The role triggers associated knowledge about professional norms, uncertainty language, citation practices
3. The backstory provides behavioral anchors: "known for cross-referencing claims" → model adds cross-references

```python
# The anatomy of an effective agent role prompt:
ROLE_TEMPLATE = """
ROLE: {job_title} (e.g., "Senior Research Analyst at a top consulting firm")

EXPERTISE: {specific_skills} (e.g., "Market research, data synthesis, competitive intelligence")

BEHAVIORAL NORMS: {how_they_work} (e.g., "Cross-reference all claims. Flag uncertainty. Be precise.")

OUTPUT STANDARDS: {output_format} (e.g., "Always include confidence level: High/Medium/Low")

CONSTRAINTS: {what_not_to_do} (e.g., "Never fabricate statistics. Don't speculate without flagging it.")
"""

# The "NEVER" clause is crucial — tells the model what the role-holder would REFUSE to do
# This is how you prevent hallucination and out-of-scope behavior
```

**The "15 years experience" effect**: Adding seniority to roles measurably improves output quality in benchmarks. The model associates seniority with:
- More careful language ("this suggests" vs "this proves")
- Acknowledgment of limitations
- Reference to prior cases ("similar to what happened in 2019...")
- Better structured output

---

### Orchestrator Pattern — State Management

The orchestrator must track which workers have been called, what they returned, and what comes next. Three approaches:

```python
# Approach 1: Simple dict (works for sequential)
results = {}
for worker_name, task in subtasks:
    results[worker_name] = workers[worker_name](task)
final = synthesize(results)

# Approach 2: Dataclass for structured tracking
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    DONE = "done"
    FAILED = "failed"

@dataclass
class TaskRecord:
    task_id: str
    worker: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

# Approach 3: LangGraph state (best for complex orchestration)
class OrchestratorState(TypedDict):
    goal: str
    plan: list[dict]
    completed_tasks: dict[int, str]   # task_id → result
    failed_tasks: list[int]
    current_task_id: int
    final_answer: str
    total_cost: float
```

---

### The Debate Pattern — When It Actually Matters

The debate pattern isn't just academic. Here's when it produces measurably better results:

**Use debate for**:
1. **Code security review**: Proposition: "This code is secure." Opposition: "Find all vulnerabilities."
2. **Architecture decisions**: Proposition: "Use microservices." Opposition: "Monolith is better for this scale."
3. **Investment/business analysis**: Proposition: "Enter this market." Opposition: "Reasons this will fail."
4. **Medical/scientific claims**: Proposition: "This treatment works." Opposition: "Methodological flaws."

**Don't use debate for**:
- Simple factual questions (wastes tokens)
- Tasks with objective correct answers
- Time-sensitive queries

```python
# Measuring debate quality — track when debates change the conclusion
def run_controlled_experiment(topic: str, n_trials: int = 10):
    """Compare single-agent vs debate outcomes."""
    single_answers = [
        get_text(chat([{"role": "user", "content": f"What's your recommendation on: {topic}?"}]))
        for _ in range(3)
    ]
    
    debate_answers = [
        debate_agent(topic)["verdict"]
        for _ in range(3)
    ]
    
    # Measure consistency (single agent may hallucinate differently each time)
    # and depth of analysis (debate answers tend to be more nuanced)
    return {
        "single_agent_variance": measure_variance(single_answers),
        "debate_variance": measure_variance(debate_answers),
        "debate_answer_length": sum(len(a) for a in debate_answers) / len(debate_answers),
    }
```

---

### Human-in-the-Loop — Production Implementation

In production, HITL needs more than a `input()` prompt. You need:
1. A way to asynchronously pause the agent
2. A UI or notification where humans can see and approve/reject
3. A timeout (if no response in 1 hour, do X)
4. An audit log of all approvals

```python
# LangGraph HITL — the production way
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

# interrupt_before creates a pause at the specified node
graph.compile(
    checkpointer=PostgresSaver.from_conn_string(DATABASE_URL),
    interrupt_before=["dangerous_action_node"]
)

# When agent hits "dangerous_action_node":
# 1. State is checkpointed to PostgreSQL
# 2. Agent pauses and returns control to caller
# 3. API returns {"status": "awaiting_approval", "thread_id": "...", "pending_action": {...}}
# 4. Frontend shows action to user
# 5. User approves: POST /agent/resume {"thread_id": "...", "approved": true}
# 6. Agent resumes from checkpoint

@app.post("/agent/resume")
async def resume_agent(thread_id: str, approved: bool, modification: dict = None):
    config = {"configurable": {"thread_id": thread_id}}
    
    if not approved:
        # Inject rejection into state
        agent.update_state(config, {"human_feedback": "Action rejected. Choose an alternative."})
    elif modification:
        # Update action arguments
        agent.update_state(config, {"pending_tool_args": modification})
    
    # Resume from checkpoint
    result = agent.invoke(None, config=config)
    return {"result": result["final_answer"]}
```

---

## Common Pitfalls — Week 5

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Orchestrator makes too many subtasks | Cost explosion (10+ LLM calls) | Cap at 4-5 subtasks per orchestration |
| Worker agents don't get enough context | Hallucination, incomplete work | Pass full relevant context from prior steps |
| No timeout on worker agents | One slow worker blocks everything | `asyncio.wait_for(worker_task, timeout=60)` |
| Debate rounds > 3 | Diminishing returns, high cost | 1-2 rounds usually sufficient |
| HITL blocking synchronous HTTP | Request timeout after 30s | Use Celery task + webhook for async HITL |
| Synthesizer not given all worker outputs | Incomplete final answer | Always pass `all_results` dict to synthesizer |
| Critic agent too harsh | Rejects everything, loops forever | Add confidence threshold: reject only if score < 6/10 |
- `ex2_debate_pattern.py` — two-agent debate with judge synthesis
- `ex3_hitl_agent.py` — agent that pauses for human approval on risky actions
- `ex4_crewai_crew.py` — CrewAI crew with hierarchical process

## Checklist
- [ ] Built an orchestrator that decomposes a task and delegates to 2+ workers
- [ ] Implemented the debate pattern — two opposing agents + judge
- [ ] Added HITL: agent asks approval before high-risk tool calls
- [ ] Measured cost of multi-agent run vs equivalent single-agent run
- [ ] Identified which tasks genuinely benefit from multi-agent vs which are overkill
