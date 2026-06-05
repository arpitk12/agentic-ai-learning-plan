# Week 6 — Parallelism & Async Agent Patterns

## What This Week Is About
Sequential agents are slow. When tasks are independent, running them concurrently can reduce latency by 5-10x. This week covers Python's `asyncio` for async LLM calls, fan-out/fan-in patterns, rate limiting, and the map-reduce pattern for processing large datasets with agents.

---

## 1. Why Async Matters for Agents

LLM API calls are I/O-bound — your code spends 80-95% of time waiting for the network. `asyncio` lets you handle thousands of concurrent LLM calls without threads, using a single event loop.

```
Sequential (slow):                Parallel with asyncio (fast):
  Call 1 → wait 2s                 Call 1 ──────────────────→ 2s
  Call 2 → wait 2s                 Call 2 ──────────────────→ 2s  
  Call 3 → wait 2s                 Call 3 ──────────────────→ 2s
  Total: 6s                        Total: 2s (3x faster)
```

---

## 2. Basic Async LLM Calls with LiteLLM

LiteLLM supports async natively via `acompletion`:

```python
import asyncio
import litellm
from llm import MODEL

async def async_chat(messages: list, system: str = None) -> str:
    """Async version of our llm.py chat() function."""
    if system:
        messages = [{"role": "system", "content": system}] + messages
    
    response = await litellm.acompletion(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content

# Running a single async call
async def main():
    result = await async_chat([{"role": "user", "content": "What is 2+2?"}])
    print(result)

asyncio.run(main())
```

---

## 3. The Fan-Out / Fan-In Pattern

**Fan-out**: Distribute a task across multiple workers simultaneously.
**Fan-in**: Collect all results and synthesize.

This is the core pattern for parallel agent work.

```python
import asyncio
from llm import MODEL

async def process_item(item: str, task_description: str) -> dict:
    """Process a single item asynchronously."""
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Task: {task_description}\n\nItem: {item}"
        }],
        max_tokens=500
    )
    return {
        "item": item,
        "result": response.choices[0].message.content,
        "tokens": response.usage.total_tokens
    }

async def fan_out_agent(items: list[str], task: str, max_concurrent: int = 5) -> list[dict]:
    """
    Fan-out: process all items in parallel.
    Fan-in: return all results.
    max_concurrent: prevents API rate limit errors.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item: str) -> dict:
        async with semaphore:
            return await process_item(item, task)
    
    # Fan-out: create all tasks
    tasks = [process_with_semaphore(item) for item in items]
    
    # Fan-in: wait for all and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    if failed:
        print(f"Warning: {len(failed)} items failed processing")
    
    return successful

# Usage
async def main():
    articles = ["Article 1 text...", "Article 2 text...", "Article 3 text..."]
    results = await fan_out_agent(
        items=articles,
        task="Summarize this article in 2 sentences.",
        max_concurrent=3
    )
    for r in results:
        print(f"Item: {r['item'][:30]}... → {r['result'][:100]}")

asyncio.run(main())
```

---

## 4. Semaphores — Rate Limit Protection

API providers enforce **rate limits**: max requests per minute (RPM) and max tokens per minute (TPM). Semaphores prevent your agent from hitting these limits and getting 429 errors.

```python
import asyncio
import time
from collections import deque

class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            # Remove calls outside the window
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()
            
            if len(self.calls) >= self.max_calls:
                # Wait until oldest call expires
                sleep_time = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
            
            self.calls.append(time.monotonic())

# Usage
rate_limiter = RateLimiter(max_calls=50, period=60.0)  # 50 RPM

async def rate_limited_chat(messages: list) -> str:
    await rate_limiter.acquire()
    return await async_chat(messages)
```

### Semaphore vs Rate Limiter

| Tool | Controls | Use For |
|------|---------|---------|
| `asyncio.Semaphore(N)` | Max concurrent connections | Prevent overwhelming the API |
| `RateLimiter(N, period)` | Calls per time window | Stay within RPM limits |
| Both together | Concurrency + time | Production agents |

---

## 5. The Map-Reduce Pattern

**Map**: Apply a function to every item in a large dataset (using fan-out).
**Reduce**: Combine all results into a final answer.

Excellent for analyzing large document collections, processing datasets, or building knowledge graphs.

```python
async def map_reduce_agent(documents: list[str], question: str) -> str:
    """
    Map: extract relevant info from each document.
    Reduce: synthesize into a final answer.
    """
    # MAP PHASE: extract relevant info from each doc in parallel
    async def extract_relevant(doc: str) -> str:
        return await async_chat([{
            "role": "user",
            "content": f"""Given this question: "{question}"
            
Extract ONLY the information relevant to answering this question from the document below.
If nothing is relevant, respond with "NO RELEVANT INFORMATION".

Document:
{doc[:2000]}"""  # truncate long docs
        }])
    
    print(f"Map phase: processing {len(documents)} documents...")
    semaphore = asyncio.Semaphore(5)
    
    async def safe_extract(doc: str) -> str:
        async with semaphore:
            return await extract_relevant(doc)
    
    extracted = await asyncio.gather(*[safe_extract(doc) for doc in documents])
    
    # Filter out irrelevant
    relevant = [e for e in extracted if "NO RELEVANT INFORMATION" not in e]
    print(f"Found relevant info in {len(relevant)}/{len(documents)} documents")
    
    if not relevant:
        return "No relevant information found in the documents."
    
    # REDUCE PHASE: synthesize all relevant extracts into final answer
    combined = "\n\n---\n\n".join(relevant)
    
    # If too many extracts, do a two-level reduce
    if len(combined) > 10000:
        # Batch reduce first
        batches = [relevant[i:i+5] for i in range(0, len(relevant), 5)]
        batch_summaries = await asyncio.gather(*[
            async_chat([{"role": "user", "content": f"Synthesize these extracts:\n\n" + "\n---\n".join(batch)}])
            for batch in batches
        ])
        combined = "\n\n".join(batch_summaries)
    
    return await async_chat([{
        "role": "user",
        "content": f"""Question: {question}

Information extracted from documents:
{combined}

Provide a comprehensive answer to the question using only the information above."""
    }])

# Usage
async def main():
    docs = ["doc1 content...", "doc2 content...", "doc3 content..."]  # 100+ docs
    answer = await map_reduce_agent(docs, "What are the key risks mentioned?")
    print(answer)

asyncio.run(main())
```

---

## 6. Async Tool Execution

When your ReAct loop encounters multiple tool calls, execute them in parallel:

```python
async def async_react_agent(user_query: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_query}]
    
    for step in range(max_steps):
        response = await litellm.acompletion(model=MODEL, messages=messages, tools=tools)
        reason = response.choices[0].finish_reason
        messages.append({"role": "assistant", "content": response.choices[0].message.content,
                         "tool_calls": response.choices[0].message.tool_calls})
        
        if reason == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls
            
            # Execute ALL tool calls in parallel
            async def execute_tool(tc) -> tuple:
                result = await asyncio.to_thread(  # run sync tools in thread pool
                    dispatch_tool, tc.function.name, json.loads(tc.function.arguments)
                )
                return tc.id, result
            
            results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
            
            for tool_id, result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result
                })
        else:
            return response.choices[0].message.content
    
    return "Max steps reached"
```

---

## 7. Batch Processing with Progress Tracking

For large-scale agent pipelines, track progress:

```python
import asyncio
from tqdm.asyncio import tqdm  # pip install tqdm

async def batch_process_with_progress(items: list, process_fn, batch_size: int = 10) -> list:
    """Process items in batches with progress bar."""
    all_results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await tqdm.gather(
            *[process_fn(item) for item in batch],
            desc=f"Batch {i//batch_size + 1}/{len(items)//batch_size + 1}"
        )
        all_results.extend(batch_results)
        
        # Pause between batches to avoid rate limits
        if i + batch_size < len(items):
            await asyncio.sleep(1)
    
    return all_results
```

---

## 8. Async with FastAPI (Preview — Week 7)

Async agents integrate naturally with async web frameworks:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    # This runs fully async — hundreds of concurrent requests possible
    result = await async_react_agent(request.query)
    return {"result": result}
```

---

## Tools & Libraries Used This Week — Deep Dive

### asyncio — Python's Concurrency Engine

**Why asyncio, not threads for LLM calls?**

Threads use OS-level concurrency. Each thread needs its own stack (~1MB). 1000 threads = 1GB RAM just for stacks.

asyncio uses **cooperative multitasking** — a single thread switches between tasks at `await` points. While one task waits for an LLM response (I/O), another task runs. Zero extra RAM per concurrent task (just Python function frame overhead ~500 bytes).

LLM API calls are 95% waiting. You're not doing computation — you're waiting for bytes to arrive over the network. asyncio is perfectly suited for this.

```python
# The mechanics of asyncio
import asyncio

# coroutine — a function that can pause and resume
async def call_llm(query: str) -> str:
    # "await" yields control back to the event loop
    # The event loop can run other coroutines while this one waits
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

# Task — a scheduled coroutine
async def main():
    # Create 10 tasks — they're all scheduled but not yet running
    tasks = [asyncio.create_task(call_llm(f"Query {i}")) for i in range(10)]
    
    # gather() — run all tasks concurrently, collect results when all done
    results = await asyncio.gather(*tasks)
    
    # With error handling:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successes = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]

asyncio.run(main())

# Event loop visualization:
# Time 0: Start task 1, task 2, task 3 (all pending API calls)
# Time 0.1: Task 1's API call is in flight... event loop checks task 2
# Time 0.5: Task 2 gets a response first → execute its callback
# Time 0.8: Task 1 gets response → execute
# Time 1.2: Task 3 gets response → execute
# Total: 1.2s (vs 0.8+0.5+1.2 = 2.5s sequential)
```

---

### asyncio.Semaphore — Rate Limit Protection

**What a semaphore IS**: A concurrency limiter. `Semaphore(5)` means "at most 5 coroutines can hold this at once." It's like a parking lot with 5 spaces.

**Why you MUST use it with LLM APIs**: Gemini allows 60 RPM (requests per minute). Without a semaphore, `asyncio.gather(*[call_llm(q) for q in 1000_queries])` fires 1000 requests instantly → you get 429 rate limit errors on ~940 of them.

```python
# Semaphore implementation — the complete pattern
import asyncio
import litellm

async def batch_llm_calls(
    queries: list[str],
    max_concurrent: int = 5,    # tune based on your API tier's RPM
    delay_between: float = 0.0  # extra delay if needed
) -> list[str | Exception]:
    """
    Process queries with bounded concurrency.
    Returns results in same order as input queries.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def call_with_backoff(query: str, attempt: int = 0) -> str:
        async with semaphore:  # "acquire" — blocks if 5 already acquired
            try:
                if delay_between > 0:
                    await asyncio.sleep(delay_between)
                
                response = await litellm.acompletion(
                    model=MODEL,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=500,
                )
                return response.choices[0].message.content
            
            except litellm.RateLimitError:
                if attempt >= 3:
                    raise
                wait = (2 ** attempt) * 5  # exponential backoff: 5s, 10s, 20s
                print(f"Rate limited. Waiting {wait}s...")
                await asyncio.sleep(wait)
                return await call_with_backoff(query, attempt + 1)
    
    results = await asyncio.gather(
        *[call_with_backoff(q) for q in queries],
        return_exceptions=True
    )
    return results

# Usage:
async def main():
    queries = [f"Summarize document {i}" for i in range(50)]
    results = await batch_llm_calls(queries, max_concurrent=5)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Query {i} failed: {result}")
        else:
            print(f"Query {i}: {result[:100]}")

asyncio.run(main())
```

---

### asyncio.gather — Fan-Out / Fan-In Explained

```python
# The gather() function is the fan-out/fan-in primitive

# Fan-out: create all tasks simultaneously
coros = [process(item) for item in items]

# Fan-in: wait for ALL to complete, collect results
results = await asyncio.gather(*coros)

# Variants:
# 1. Fail fast (default): any exception propagates immediately
results = await asyncio.gather(*coros)

# 2. Collect all, even on error: exceptions returned as values
results = await asyncio.gather(*coros, return_exceptions=True)

# 3. First completed wins: useful for "fastest LLM provider" pattern
done, pending = await asyncio.wait(coros, return_when=asyncio.FIRST_COMPLETED)

# 4. With timeout: cancel all if too slow
results = await asyncio.wait_for(asyncio.gather(*coros), timeout=30.0)
```

**The "as_completed" pattern** — process results as they arrive (better for UX):
```python
from asyncio import as_completed

async def process_with_streaming_results(items: list) -> None:
    """Process items and print results as they complete (not in order)."""
    coros = [process_item(item) for item in items]
    
    for future in as_completed(coros):
        result = await future
        print(f"Completed: {result}")  # prints as each one finishes
```

---

### tqdm — Progress Tracking for Agent Batches

```python
from tqdm.asyncio import tqdm_asyncio

# With progress bar
results = await tqdm_asyncio.gather(
    *[process(item) for item in items],
    desc="Processing items",
    total=len(items)
)

# Manual progress tracking with async
from tqdm import tqdm

async def process_with_progress(items: list) -> list:
    results = []
    semaphore = asyncio.Semaphore(5)
    
    with tqdm(total=len(items), desc="Processing") as pbar:
        async def process_one(item):
            async with semaphore:
                result = await call_llm(item)
                pbar.update(1)
                pbar.set_postfix({"last": result[:30]})
                return result
        
        results = await asyncio.gather(*[process_one(item) for item in items])
    
    return results
```

---

### The Map-Reduce Pattern — Mathematical Basis

Map-Reduce is a distributed computing paradigm from Google (MapReduce paper, 2004). For LLM agents:

**Map function**: `document → relevant_extract` (embarrassingly parallel)
**Reduce function**: `list[extract] → final_answer` (requires all map outputs)

The trick for large datasets is **hierarchical reduction**:
```
1000 docs
  → [Map] → 1000 extracts
  → [Reduce batch 1] 10 extracts → 1 summary (batch 1)
  → [Reduce batch 2] 10 extracts → 1 summary (batch 2)
  ...100 batches
  → [Final Reduce] 100 summaries → final answer
```

This keeps each LLM context window manageable regardless of total dataset size.

```python
# Token-budget-aware reduction
def split_for_context(texts: list[str], max_tokens_per_batch: int = 3000) -> list[list[str]]:
    """Split texts into batches that fit within token budget."""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = len(text.split()) * 1.3  # rough estimate
        
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches
```

---

## When Parallelism Actually Helps vs Hurts

**Use parallelism when**:
- ✅ Tasks are independent (no task depends on another's result)
- ✅ Tasks are I/O-bound (LLM calls, web requests, DB queries)
- ✅ You have many items to process (>10 items)
- ✅ Items can be processed with the same prompt template

**Don't use parallelism when**:
- ❌ Tasks must run in sequence (ReAct steps: step 2 depends on step 1's result)
- ❌ Your API tier has very low rate limits (5 RPM) — semaphore to 1
- ❌ You need deterministic ordering for debugging
- ❌ The overhead of task creation exceeds the benefit (< 3 items)

---

## Common Pitfalls — Week 6

| Mistake | Symptom | Fix |
|---------|---------|-----|
| No semaphore on gather | 429 rate limit errors | Always use `asyncio.Semaphore(N)` |
| Running sync functions in async without thread | Event loop blocks, no concurrency | Use `asyncio.to_thread(sync_fn, args)` for sync tools |
| Catching Exception in gather without return_exceptions=True | Some results silently None | Always `return_exceptions=True` when handling errors |
| Not setting timeout on gather | One slow request blocks all | `asyncio.wait_for(gather(...), timeout=60)` |
| Using `asyncio.run()` inside an async function | Event loop already running error | Use `await` directly inside async functions |
| Context variables not propagating to tasks | `contextvars` not shared across tasks | Use `asyncio.create_task()` — it copies context |
| Mixing sync and async without thought | Deadlock or blocking event loop | Profile with `asyncio-debug=True` to find blocking calls |
- `ex2_fan_out_agent.py` — process 20 items in parallel, max 5 concurrent
- `ex3_map_reduce.py` — summarize 50 documents using map-reduce
- `ex4_rate_limiter.py` — custom rate limiter with token bucket algorithm

## Checklist
- [ ] Rewrote a sequential agent loop as async — measured speedup
- [ ] Implemented semaphore-limited fan-out (max 5 concurrent)
- [ ] Built map-reduce pipeline for a corpus of documents
- [ ] Handled exceptions in asyncio.gather without crashing the whole batch
- [ ] Integrated async agent into a simple FastAPI endpoint
