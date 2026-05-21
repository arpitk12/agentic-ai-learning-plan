"""
SOLUTION — Exercise 3: Streaming with TTFT Measurement
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import time
from dotenv import load_dotenv
from llm import stream_chat, chat, get_text

load_dotenv()


def stream_response(prompt: str) -> str:
    """Stream tokens to stdout and return the full text."""
    full_text = ""
    start = time.perf_counter()
    first_token_time: float | None = None

    for chunk in stream_chat([{"role": "user", "content": prompt}]):
        if first_token_time is None:
            first_token_time = time.perf_counter() - start
        print(chunk, end="", flush=True)
        full_text += chunk

    elapsed = time.perf_counter() - start
    print(f"\n\n[TTFT: {first_token_time:.3f}s | Total: {elapsed:.2f}s | "
          f"{len(full_text.split())} words]")
    return full_text


def compare_streaming_vs_blocking(prompt: str):
    """Demonstrate why streaming matters for UX."""
    print("=== STREAMING ===")
    stream_response(prompt)

    print("\n=== BLOCKING (waits for full response) ===")
    start = time.perf_counter()
    response = chat([{"role": "user", "content": prompt}], max_tokens=1024)
    elapsed = time.perf_counter() - start
    print(get_text(response))
    print(f"\n[Total wait: {elapsed:.2f}s — user saw nothing until the end]")


if __name__ == "__main__":
    stream_response("Explain how neural networks learn, step by step.")
