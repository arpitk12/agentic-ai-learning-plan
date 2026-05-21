"""
Exercise 3: Streaming Responses
Goal: Stream tokens live to the terminal. Track time-to-first-token.

Uses llm.py — works with Ollama (local) or any cloud model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import time
from llm import stream_chat, MODEL


def stream_response(prompt: str) -> str:
    full_text = ""
    start = time.time()
    first_token_time = None

    # TODO: Stream tokens from the API to stdout, one at a time
    for chunk in stream_chat([{"role":"user","content":prompt}],system="You are a helpful teaching assistant"):
        print(chunk,end="",flush=True)
        if not first_token_time:
            first_token_time=time.time() - start
        full_text+=chunk
    # TODO: Record the time elapsed when the first token arrives (TTFT)

    # TODO: Accumulate tokens into full_text

    print(f"\n\n[TTFT: {first_token_time:.2f}s | Total: {time.time()-start:.2f}s]")
    return full_text


if __name__ == "__main__":
    stream_response("Explain how neural networks learn, step by step.")
