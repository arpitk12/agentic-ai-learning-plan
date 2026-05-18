"""
SOLUTION — Exercise 1: Async Fan-Out
"""
import asyncio
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

async_client = anthropic.AsyncAnthropic()

DOCS = [
    "Quantum computing uses quantum bits (qubits) that can exist in superposition states. This enables quantum parallelism where many calculations happen simultaneously. Quantum entanglement allows qubits to be correlated across any distance.",
    "Machine learning models learn patterns from training data. Neural networks are inspired by the brain's structure. Backpropagation adjusts weights to minimize prediction errors. Regularization prevents overfitting to training data.",
    "The Internet of Things connects everyday devices to the internet. Smart sensors collect real-time environmental data. Edge computing processes data closer to where it is generated. 5G networks enable faster IoT communication.",
    "Blockchain is a distributed ledger technology. Transactions are grouped into blocks and chained together. Consensus mechanisms like proof-of-work validate new blocks. Smart contracts execute automatically when conditions are met.",
    "CRISPR-Cas9 is a gene editing tool derived from bacterial immune systems. It can cut DNA at specific locations with high precision. Gene therapy uses CRISPR to correct genetic diseases. Ethical debates surround germline editing in humans.",
]


async def async_summarize(text: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        r = await async_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=128,
            messages=[{"role": "user", "content": f"Summarize in one sentence:\n\n{text}"}],
        )
    return r.content[0].text


async def fan_out(texts: list[str]) -> list[str]:
    sem = asyncio.Semaphore(3)
    tasks = [async_summarize(text, sem) for text in texts]
    return await asyncio.gather(*tasks)


async def sequential(texts: list[str]) -> list[str]:
    results = []
    sem = asyncio.Semaphore(1)
    for text in texts:
        results.append(await async_summarize(text, sem))
    return results


async def main():
    print("Running sequential...")
    t0 = time.perf_counter()
    await sequential(DOCS)
    seq_time = time.perf_counter() - t0
    print(f"Sequential: {seq_time:.2f}s")

    print("\nRunning parallel (semaphore=3)...")
    t0 = time.perf_counter()
    par_results = await fan_out(DOCS)
    par_time = time.perf_counter() - t0
    print(f"Parallel:   {par_time:.2f}s")
    print(f"Speedup:    {seq_time / par_time:.1f}x\n")

    for i, summary in enumerate(par_results, 1):
        print(f"[Doc {i}] {summary}")


if __name__ == "__main__":
    asyncio.run(main())
