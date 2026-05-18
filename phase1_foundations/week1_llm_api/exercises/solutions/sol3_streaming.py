"""
SOLUTION — Exercise 3: Streaming with TTFT Measurement
"""
import time
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()


def stream_response(prompt: str) -> str:
    """Stream tokens to stdout and return the full text."""
    full_text = ""
    start = time.perf_counter()
    first_token_time: float | None = None

    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            print(text, end="", flush=True)
            full_text += text

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
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - start
    print(response.content[0].text)
    print(f"\n[Total wait: {elapsed:.2f}s — user saw nothing until the end]")


if __name__ == "__main__":
    stream_response("Explain how neural networks learn, step by step.")
