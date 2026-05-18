# Week 6 — Parallelism & Fan-Out Patterns

## Topics
1. asyncio for concurrent LLM calls
2. Fan-out / fan-in: spawn N agents, collect results
3. Map-reduce over large document sets
4. Rate limiting and token budget management

## Key Concepts

### Why Parallelism Matters
Sequential agents are bottlenecked by LLM latency (~1-3s per call).
Running 5 subagents in parallel cuts wall time by 5x.

### asyncio Pattern
```python
import asyncio
import anthropic

client = anthropic.AsyncAnthropic()

async def call_agent(prompt: str) -> str:
    r = await client.messages.create(
        model="claude-opus-4-5", max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text

async def fan_out(prompts: list[str]) -> list[str]:
    tasks = [call_agent(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

### Rate Limiting
Use a semaphore to cap concurrent calls:
```python
sem = asyncio.Semaphore(5)  # max 5 concurrent

async def call_with_limit(prompt: str) -> str:
    async with sem:
        return await call_agent(prompt)
```

### Map-Reduce Pattern
```
[doc1, doc2, ..., docN]  →  MAP: summarize each (parallel)
                          →  REDUCE: synthesize summaries (single LLM call)
```

## Exercises
- `ex1_async_fan_out.py` — summarize 5 documents in parallel
- `ex2_map_reduce.py` — map-reduce over 20 chunks
- `ex3_rate_limiter.py` — respect API rate limits gracefully

## Checklist
- [ ] Fan-out 5 async calls, measure speedup vs sequential
- [ ] Implement map-reduce pipeline
- [ ] Handle asyncio exceptions with `asyncio.gather(return_exceptions=True)`
- [ ] Add exponential backoff for rate limit errors
