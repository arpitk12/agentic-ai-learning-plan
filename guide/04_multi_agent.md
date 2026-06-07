[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §3 RAG Architecture](guide/03_rag_architecture.md) | [§5 Vector Search →](guide/05_vector_search.md)

---

## 4. Multi-Agent Design Patterns

Multi-agent systems are more than just "multiple LLMs." Each pattern addresses specific problems and comes with specific trade-offs. Here is every major pattern with full implementation code.

### 4.1 Pattern Decision Tree

```
Does the task have multiple distinct specializations?
├─ Yes: Can they run sequentially?
│    ├─ Yes → Pipeline Pattern (research → write → review)
│    └─ No → Fan-Out/Fan-In (parallel specialists)
└─ No: Is this a quality-critical decision?
     ├─ Yes → Debate Pattern (adversarial review)
     └─ No → ReAct (single agent is enough)

Does the task need self-correction?
└─ Yes → Reflexion (retry with verbal feedback)

Does the task need human oversight?
└─ Yes → HITL Pattern (pause for approval)

Is the task a large dataset?
└─ Yes → Map-Reduce Pattern (process items in parallel)
```

---

### 4.2 Pattern 1: Orchestrator-Worker (Most Common)

**When to use**: Tasks that can be broken into specialist subtasks. A manager decomposes, delegates, and synthesizes.

**Cost**: High — N+1 LLM calls (1 for planning + N for workers)

```python
from llm import chat, get_text
import json, re
from typing import Callable

# Worker definitions
WORKERS: dict[str, Callable] = {
    "researcher": lambda task: get_text(chat(
        messages=[{"role": "user", "content": task}],
        system="You are a meticulous research analyst. Find accurate, current information. "
               "Always note the confidence level of your findings (High/Medium/Low)."
    )),
    "analyst": lambda task: get_text(chat(
        messages=[{"role": "user", "content": task}],
        system="You are a data analyst. Identify patterns, trends, anomalies, and insights. "
               "Support every claim with specific numbers or evidence."
    )),
    "coder": lambda task: get_text(chat(
        messages=[{"role": "user", "content": task}],
        system="You are a senior Python engineer. Write clean, tested, documented code. "
               "Include error handling and type annotations. Code must be runnable."
    )),
    "writer": lambda task: get_text(chat(
        messages=[{"role": "user", "content": task}],
        system="You are a technical writer. Produce clear, engaging, well-structured content. "
               "Tailor tone to the audience. Use concrete examples."
    )),
    "critic": lambda task: get_text(chat(
        messages=[{"role": "user", "content": task}],
        system="You are an adversarial reviewer. Find flaws, gaps, logical errors, and missing cases. "
               "Rate severity: CRITICAL / HIGH / MEDIUM / LOW."
    )),
}

def orchestrate(user_request: str, max_workers: int = 4) -> str:
    """
    Orchestrator-Worker pattern.
    1. Planner decomposes task → JSON plan
    2. Workers execute subtasks
    3. Synthesizer combines results
    """
    # Step 1: Create execution plan
    plan_raw = get_text(chat(
        messages=[{"role": "user", "content": f"Decompose this task into specialist subtasks:\n\n{user_request}"}],
        system=f"""You are a project manager. Break the task into specialist subtasks.
Available workers: {list(WORKERS.keys())}

Output ONLY valid JSON:
{{
  "subtasks": [
    {{"id": 1, "worker": "researcher", "task": "Find information about...", "depends_on": []}},
    {{"id": 2, "worker": "writer", "task": "Write a report using...", "depends_on": [1]}}
  ]
}}"""
    ))
    
    clean = re.sub(r"```json?\s*|\s*```", "", plan_raw).strip()
    plan = json.loads(clean)
    subtasks = plan["subtasks"][:max_workers]
    
    print(f"Plan: {len(subtasks)} subtasks")
    
    # Step 2: Execute subtasks respecting dependencies
    results: dict[int, str] = {}
    
    for subtask in subtasks:
        # Wait for dependencies
        dep_context = "\n\n".join([
            f"[Result from step {dep}]: {results[dep]}"
            for dep in subtask.get("depends_on", [])
            if dep in results
        ])
        
        full_task = subtask["task"]
        if dep_context:
            full_task += f"\n\nContext from previous steps:\n{dep_context}"
        
        worker = subtask["worker"]
        if worker not in WORKERS:
            worker = "researcher"  # fallback
        
        print(f"  [{worker}] {subtask['task'][:60]}...")
        results[subtask["id"]] = WORKERS[worker](full_task)
    
    # Step 3: Synthesize final answer
    all_results = "\n\n".join([f"=== Step {k} Result ===\n{v}" for k, v in results.items()])
    
    return get_text(chat(
        messages=[{"role": "user", "content": f"Original request: {user_request}\n\nAll step results:\n{all_results}\n\nSynthesize into a final, comprehensive response."}],
        system="Synthesize all provided information into a coherent, complete response. "
               "Don't just concatenate — integrate and improve."
    ))
```

---

### 4.3 Pattern 2: Debate / Adversarial Review

**When to use**: High-stakes decisions (code security review, investment analysis, architecture choices). Forces both pro and con perspectives.

**Cost**: High — 2 × (rounds) + 1 (judge)

```python
from llm import chat, get_text

def debate_agent(
    topic: str,
    rounds: int = 2,
    proposition_role: str = "strong advocate",
    opposition_role: str = "skeptical critic"
) -> dict:
    """
    Structured debate between two agents, judged by a third.
    
    Returns: {"conclusion": str, "pro_arguments": list, "con_arguments": list, "verdict": str}
    """
    pro_msgs = [{"role": "user", "content": f"You will argue FOR this position: {topic}. Make the strongest case you can."}]
    con_msgs = [{"role": "user", "content": f"You will argue AGAINST this position: {topic}. Find all weaknesses and risks."}]
    
    pro_args = []
    con_args = []
    
    # Opening statements
    pro_statement = get_text(chat(pro_msgs, system=f"You are a {proposition_role}. State your opening argument."))
    pro_args.append(pro_statement)
    
    con_statement = get_text(chat(con_msgs, system=f"You are a {opposition_role}. State your opening argument."))
    con_args.append(con_statement)
    
    # Debate rounds
    for r in range(rounds):
        # Proposition counters opposition
        pro_msgs.append({"role": "assistant", "content": pro_statement})
        pro_msgs.append({"role": "user", "content": f"Opposition argues: {con_statement}\n\nCounter their strongest points."})
        pro_statement = get_text(chat(pro_msgs, system=f"You are a {proposition_role}. Counter decisively."))
        pro_args.append(pro_statement)
        
        # Opposition counters proposition
        con_msgs.append({"role": "assistant", "content": con_statement})
        con_msgs.append({"role": "user", "content": f"Proposition argues: {pro_statement}\n\nExpose the flaws in their reasoning."})
        con_statement = get_text(chat(con_msgs, system=f"You are a {opposition_role}. Find every weakness."))
        con_args.append(con_statement)
    
    # Judge synthesizes
    transcript = f"""
TOPIC: {topic}

PROPOSITION ({proposition_role}):
{chr(10).join(f'Round {i+1}: {arg}' for i, arg in enumerate(pro_args))}

OPPOSITION ({opposition_role}):
{chr(10).join(f'Round {i+1}: {arg}' for i, arg in enumerate(con_args))}
"""
    
    verdict = get_text(chat(
        messages=[{"role": "user", "content": transcript}],
        system="""You are an impartial expert judge. Based on the debate:
1. Identify the strongest arguments from each side
2. Point out logical fallacies or unsupported claims
3. Synthesize a nuanced, balanced conclusion
4. State a clear verdict: APPROVE / REJECT / CONDITIONAL APPROVAL with conditions
Output as structured analysis."""
    ))
    
    return {
        "topic": topic,
        "verdict": verdict,
        "pro_arguments": pro_args,
        "con_arguments": con_args,
        "rounds": rounds,
    }
```

---

### 4.4 Pattern 3: Fan-Out / Fan-In (Parallel Processing)

**When to use**: Same task applied to many independent items. Classify 100 support tickets, summarize 50 articles, analyze 200 code files.

**Cost**: Very efficient — all items processed simultaneously

```python
import asyncio
import litellm
from llm import MODEL

async def parallel_agent(
    items: list[str],
    task_template: str,
    max_concurrent: int = 5,
    timeout_per_item: float = 30.0,
) -> list[dict]:
    """
    Fan-out: process all items in parallel.
    Fan-in: collect results with error handling.
    
    task_template: string with {item} placeholder, e.g. "Classify this ticket: {item}"
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_one(item: str, idx: int) -> dict:
        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=MODEL,
                        messages=[{"role": "user", "content": task_template.format(item=item)}],
                        max_tokens=500,
                    ),
                    timeout=timeout_per_item
                )
                return {
                    "index": idx,
                    "item": item,
                    "result": response.choices[0].message.content,
                    "status": "success",
                    "tokens": response.usage.total_tokens,
                }
            except asyncio.TimeoutError:
                return {"index": idx, "item": item, "result": None, "status": "timeout", "tokens": 0}
            except Exception as e:
                return {"index": idx, "item": item, "result": None, "status": f"error: {e}", "tokens": 0}
    
    # Fan-out: create all tasks
    tasks = [process_one(item, i) for i, item in enumerate(items)]
    
    # Fan-in: gather all results
    results = await asyncio.gather(*tasks)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"Processed {len(successful)}/{len(items)} items successfully. "
          f"Total tokens: {sum(r['tokens'] for r in successful)}")
    
    return sorted(results, key=lambda x: x["index"])

# Usage
async def main():
    support_tickets = [
        "My payment failed three times",
        "Can I upgrade my subscription?",
        "The app crashes on startup",
        # ... 100 more
    ]
    
    results = await parallel_agent(
        items=support_tickets,
        task_template="Classify this support ticket into exactly one category "
                      "(billing, technical, account, general). Reply with just the category word.\n\nTicket: {item}",
        max_concurrent=5,
    )
    
    for r in results:
        print(f"[{r['result']}] {r['item'][:60]}")
```

---

### 4.5 Pattern 4: Map-Reduce

**When to use**: Large datasets where each item is processed, then all results are combined. Processing 500 documents to answer one question.

```python
async def map_reduce_agent(
    documents: list[str],
    question: str,
    map_batch_size: int = 5,
    max_concurrent: int = 5,
) -> str:
    """
    Map: extract relevant information from each document in parallel.
    Reduce: synthesize all extracts into a final comprehensive answer.
    
    Handles large document sets that don't fit in one context window.
    """
    # MAP PHASE
    async def extract_relevant(doc: str) -> str:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": f"""Question: {question}

Document:
{doc[:3000]}

Extract ONLY the information from this document relevant to answering the question.
If nothing is relevant, respond with exactly: NO_RELEVANT_INFO"""}],
            max_tokens=400,
        )
        return response.choices[0].message.content
    
    semaphore = asyncio.Semaphore(max_concurrent)
    async def safe_extract(doc: str) -> str:
        async with semaphore:
            return await extract_relevant(doc)
    
    print(f"MAP: Processing {len(documents)} documents...")
    extracts = await asyncio.gather(*[safe_extract(doc) for doc in documents])
    
    # Filter irrelevant
    relevant = [e for e in extracts if "NO_RELEVANT_INFO" not in e and len(e.strip()) > 20]
    print(f"MAP: Found relevant info in {len(relevant)}/{len(documents)} documents")
    
    if not relevant:
        return "No relevant information found across any document."
    
    # REDUCE PHASE — handle large result sets with hierarchical reduction
    async def reduce_batch(batch: list[str], step: str) -> str:
        combined = "\n\n---\n\n".join(batch)
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": f"Question: {question}\n\nInformation extracts:\n{combined}\n\nSynthesize all relevant information."}],
            max_tokens=800,
        )
        return response.choices[0].message.content
    
    # Two-level reduction if too many extracts
    current = relevant
    while len(current) > 5:
        batches = [current[i:i+5] for i in range(0, len(current), 5)]
        print(f"REDUCE: Combining {len(batches)} batches...")
        current = list(await asyncio.gather(*[reduce_batch(b, f"batch_{i}") for i, b in enumerate(batches)]))
    
    # Final synthesis
    final = await reduce_batch(current, "final")
    return final
```

---

### 4.6 Pattern 5: Reflexion (Self-Correcting Agent)

**When to use**: Tasks where quality matters more than speed. Research, code generation, complex analysis. Budget 2-4x cost for 40-80% quality improvement.

```python
from llm import chat, get_text, MODEL
import json, re

def reflexion_agent(
    task: str,
    max_attempts: int = 3,
    evaluator_system: str = None,
) -> dict:
    """
    Reflexion: agent generates → evaluates → reflects → improves.
    
    Each reflection is stored and provided to the next attempt.
    Returns best result with metadata.
    """
    reflections = []
    attempts = []
    
    eval_system = evaluator_system or """Evaluate if the result fully completes the task.
Be strict. Only mark success if the result is complete, accurate, and directly addresses all requirements.
Return JSON: {"success": true/false, "score": 0-10, "gaps": ["gap1", "gap2"], "improvements": ["improvement1"]}"""
    
    for attempt_num in range(max_attempts):
        print(f"\n{'='*50}")
        print(f"Attempt {attempt_num + 1}/{max_attempts}")
        
        # Build context from past reflections
        reflection_context = ""
        if reflections:
            reflection_context = "\n\nLearnings from previous attempts (USE THESE TO IMPROVE):\n"
            for i, r in enumerate(reflections):
                reflection_context += f"\nAttempt {i+1} failed because: {r['gap']}\nNext time I should: {r['improvement']}\n"
        
        # Attempt the task
        result = react_agent(task + reflection_context) if has_tools else get_text(chat(
            messages=[{"role": "user", "content": task + reflection_context}],
            system="Complete the task thoroughly. Use all learnings from previous attempts."
        ))
        
        attempts.append(result)
        print(f"Result preview: {result[:200]}...")
        
        # Evaluate result
        eval_raw = get_text(chat(
            messages=[{"role": "user", "content": f"Task: {task}\n\nResult:\n{result}\n\nEvaluate."}],
            system=eval_system
        ))
        
        clean = re.sub(r"```json?\s*|\s*```", "", eval_raw).strip()
        try:
            evaluation = json.loads(clean)
        except:
            evaluation = {"success": False, "score": 5, "gaps": ["Parse error"], "improvements": ["Be more precise"]}
        
        print(f"Score: {evaluation.get('score', '?')}/10, Success: {evaluation.get('success', False)}")
        
        if evaluation.get("success"):
            print(f"✅ Task completed successfully on attempt {attempt_num + 1}")
            return {"result": result, "attempts": attempt_num + 1, "final_score": evaluation.get("score", 10)}
        
        # Generate targeted reflection
        gaps = evaluation.get("gaps", ["Incomplete"])
        improvements = evaluation.get("improvements", ["Try harder"])
        
        reflections.append({
            "attempt": attempt_num + 1,
            "gap": ", ".join(gaps),
            "improvement": ", ".join(improvements),
        })
    
    # Return best attempt (highest score)
    return {
        "result": attempts[-1],
        "attempts": max_attempts,
        "note": f"Max attempts reached. Final attempt returned.",
        "reflections": reflections,
    }
```

---

### 4.7 Pattern 6: Human-in-the-Loop (HITL)

**When to use**: Actions that are irreversible (send email, delete data, make payments), high-risk (deploy to production, modify user data), or low-confidence decisions.

```python
from llm import chat, get_tool_calls, stop_reason, assistant_message, tool_result_message, get_text
import json

# Risk classification for tools
TOOL_RISK = {
    "web_search": "low",           # safe, read-only
    "read_file": "low",            # safe, read-only
    "calculate": "low",            # safe, no side effects
    "write_file": "medium",        # modifies files
    "run_code": "medium",          # executes code
    "send_email": "high",          # external communication
    "delete_file": "high",         # irreversible
    "post_to_api": "high",         # external side effect
    "modify_database": "critical", # data mutation
}

def hitl_react_agent(task: str, auto_approve_low_risk: bool = True) -> str:
    """
    ReAct agent with human-in-the-loop for risky actions.
    
    Low risk: auto-approved
    Medium risk: show summary, auto-approve after 5 seconds
    High/Critical: require explicit human approval
    """
    messages = [{"role": "user", "content": task}]
    approved_actions = []
    denied_actions = []
    
    for step in range(20):
        response = chat(messages=messages, tools=ALL_TOOLS)
        reason = stop_reason(response)
        messages.append(assistant_message(response))
        
        if reason == "tool_calls":
            for tc in get_tool_calls(response):
                tool_name = tc["name"]
                risk = TOOL_RISK.get(tool_name, "medium")
                
                print(f"\n🔧 Agent wants to call: {tool_name}")
                print(f"   Arguments: {json.dumps(tc['arguments'], indent=2)}")
                print(f"   Risk level: {risk.upper()}")
                
                if risk == "low" and auto_approve_low_risk:
                    result = dispatch_tool(tool_name, tc["arguments"])
                    print(f"   ✅ Auto-approved (low risk)")
                elif risk in {"medium", "high", "critical"}:
                    print(f"\n{'⚠️ ' * 3} APPROVAL REQUIRED {'⚠️ ' * 3}")
                    user_input = input(f"  Approve '{tool_name}'? [y=yes, n=no, m=modify args]: ").strip().lower()
                    
                    if user_input == "n":
                        result = f"Action denied by user. Choose a different approach."
                        denied_actions.append(tool_name)
                    elif user_input == "m":
                        new_args = input("  Enter new arguments (JSON): ")
                        tc["arguments"] = json.loads(new_args)
                        result = dispatch_tool(tool_name, tc["arguments"])
                        approved_actions.append(tool_name)
                    else:  # "y" or enter
                        result = dispatch_tool(tool_name, tc["arguments"])
                        approved_actions.append(tool_name)
                else:
                    result = dispatch_tool(tool_name, tc["arguments"])
                
                messages.append(tool_result_message(tc["id"], result))
        
        elif reason == "stop":
            final = get_text(response)
            print(f"\nApproved: {approved_actions}")
            print(f"Denied: {denied_actions}")
            return final
    
    return "Max steps reached"
```

---

### 4.8 Pattern Summary

| Pattern | Cost Multiplier | Latency | Best Use Case |
|---------|----------------|---------|---------------|
| **Orchestrator-Worker** | 3-10x | High | Complex decomposable tasks |
| **Debate/Adversarial** | 5-8x | High | High-stakes decisions |
| **Fan-Out/Fan-In** | 1x (parallel) | Low | Batch processing |
| **Map-Reduce** | 1-3x (parallel) | Low-Medium | Large dataset Q&A |
| **Reflexion** | 2-4x | High | Quality-critical tasks |
| **HITL** | 1-2x + human time | Very High | Risky irreversible actions |
| **Pipeline (CrewAI)** | N x tasks | Medium | Structured sequential work |

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §3 RAG Architecture](guide/03_rag_architecture.md) | [§5 Vector Search →](guide/05_vector_search.md)
