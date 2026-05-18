"""
Exercise 1: Async Fan-Out — Summarize 5 Documents in Parallel
Goal: Use asyncio to run 5 LLM calls concurrently.

Tasks:
  1. Implement async_summarize(text: str) → str using achat() from llm.py.
  2. Implement fan_out(texts: list[str]) → list[str] using asyncio.gather.
  3. Measure time for sequential vs parallel execution.
  4. Add a Semaphore(3) to cap concurrent requests.
  5. Print speedup ratio.
"""
import asyncio
import time
from llm import achat, get_text


DOCS = [
    "Quantum computing uses quantum bits (qubits) that can exist in superposition states. This enables quantum parallelism where many calculations happen simultaneously. Quantum entanglement allows qubits to be correlated across any distance.",
    "Machine learning models learn patterns from training data. Neural networks are inspired by the brain's structure. Backpropagation adjusts weights to minimize prediction errors. Regularization prevents overfitting to training data.",
    "The Internet of Things connects everyday devices to the internet. Smart sensors collect real-time environmental data. Edge computing processes data closer to where it is generated. 5G networks enable faster IoT communication.",
    "Blockchain is a distributed ledger technology. Transactions are grouped into blocks and chained together. Consensus mechanisms like proof-of-work validate new blocks. Smart contracts execute automatically when conditions are met.",
    "CRISPR-Cas9 is a gene editing tool derived from bacterial immune systems. It can cut DNA at specific locations with high precision. Gene therapy uses CRISPR to correct genetic diseases. Ethical debates surround germline editing in humans.",
]


async def async_summarize(text: str, sem: asyncio.Semaphore) -> str:
    """Summarize text with the async client, respecting the rate-limit semaphore."""
    raise NotImplementedError


async def fan_out(texts: list[str]) -> list[str]:
    """Summarize all texts concurrently with a Semaphore(3) cap."""
    sem = asyncio.Semaphore(3)
    raise NotImplementedError


async def sequential(texts: list[str]) -> list[str]:
    """Summarize texts one by one for comparison."""
    results = []
    sem = asyncio.Semaphore(1)  # forces sequential
    for text in texts:
        results.append(await async_summarize(text, sem))
    return results


async def main():
    print("Running sequential...")
    t0 = time.perf_counter()
    seq_results = await sequential(DOCS)
    seq_time = time.perf_counter() - t0
    print(f"Sequential: {seq_time:.2f}s")

    print("\nRunning parallel...")
    t0 = time.perf_counter()
    par_results = await fan_out(DOCS)
    par_time = time.perf_counter() - t0
    print(f"Parallel: {par_time:.2f}s")
    print(f"Speedup: {seq_time / par_time:.1f}x")

    for i, summary in enumerate(par_results, 1):
        print(f"\n[Doc {i}] {summary[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
