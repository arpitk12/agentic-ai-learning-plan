# Week 6 Resources — Parallelism & Fan-Out

## Official Docs
- Python asyncio: https://docs.python.org/3/library/asyncio.html
- Anthropic AsyncAnthropic: https://github.com/anthropics/anthropic-sdk-python#async-usage
- asyncio.gather: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
- asyncio.Semaphore: https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore

## Key Articles
- "Async Python: The Different Forms of Concurrency": https://www.encode.io/articles/python-async-frameworks
- Anthropic rate limits: https://docs.anthropic.com/en/api/rate-limits

## Patterns
- **Fan-out/Fan-in**: spawn N tasks, await all, merge results
- **Semaphore**: cap concurrency to stay under rate limits
- **gather(return_exceptions=True)**: don't let one failure kill all tasks
- **Map-Reduce**: parallel map over chunks, single reduce call

## Install
```
pip install anthropic python-dotenv
```
(asyncio is built-in to Python 3.11+)
