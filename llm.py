"""
llm.py — Unified LLM client using LiteLLM.
Swap between local (Ollama) and any cloud provider by changing
the MODEL env var — zero code changes needed.

Usage:
    from llm import chat, stream_chat, MODEL

    response = chat([{"role": "user", "content": "Hello"}])
    print(get_text(response))

Supported providers (set MODEL= in .env):
    Local (free, no key):
        ollama/llama3.2
        ollama/qwen2.5:7b

    Free cloud (API key, no credit card):
        groq/llama-3.3-70b-versatile    ← best all-round ⭐
        groq/qwen-qwq-32b               ← best reasoning
        gemini/gemini-2.0-flash         ← large context, great for RAG
        cerebras/llama3.1-70b           ← fastest
        openrouter/mistralai/mistral-7b-instruct:free

    Paid cloud:
        anthropic/claude-haiku-4-5-20251001
        gpt-4o-mini

See FREE_CLOUD_LLM.md for full setup instructions.

Install:
    pip install litellm python-dotenv

Local setup (Ollama):
    brew install ollama
    ollama pull llama3.2
    ollama pull qwen2.5:7b    # better at tool calling
    ollama serve              # runs on http://localhost:11434
"""
import os
from dotenv import load_dotenv
import litellm
from litellm import completion, acompletion

load_dotenv()

# ── Model selection ────────────────────────────────────────────────────────────
# Override with: MODEL=ollama/llama3.2 python ex1.py
# or set MODEL in .env

MODEL = os.getenv("MODEL", "ollama/llama3.2")

# ── Security: never log API keys ───────────────────────────────────────────────
# LiteLLM can print keys in debug mode — keep both off
litellm.suppress_debug_info = True
litellm.set_verbose = False

def _check_secrets_not_exposed():
    """Warn if any exercise accidentally tries to print env vars."""
    import sys
    _dangerous = ["API_KEY", "SECRET", "TOKEN", "PASSWORD"]
    for arg in sys.argv[1:]:
        if any(d in arg.upper() for d in _dangerous):
            raise RuntimeError(
                f"Refusing to use a secret as a CLI argument: {arg!r}\n"
                "Load secrets from .env only."
            )

_check_secrets_not_exposed()

# ── Cost table (local = $0, cloud = real rates per 1K tokens) ─────────────────
COST_PER_1K: dict[str, dict[str, float]] = {
    # ── Local (Ollama) — free ──────────────────────────────────────────────────
    "ollama/llama3.2":      {"input": 0.0, "output": 0.0},
    "ollama/qwen2.5:7b":    {"input": 0.0, "output": 0.0},
    "ollama/mistral":       {"input": 0.0, "output": 0.0},
    # ── Free cloud — Groq ─────────────────────────────────────────────────────
    "groq/llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
    "groq/qwen-qwq-32b":            {"input": 0.0, "output": 0.0},
    "groq/llama3-8b-8192":          {"input": 0.0, "output": 0.0},
    # ── Free cloud — Gemini ───────────────────────────────────────────────────
    "gemini/gemini-2.0-flash":      {"input": 0.0, "output": 0.0},
    "gemini/gemini-1.5-pro":        {"input": 0.0, "output": 0.0},
    # ── Free cloud — Cerebras ─────────────────────────────────────────────────
    "cerebras/llama3.1-70b":        {"input": 0.0, "output": 0.0},
    # ── Paid cloud — Anthropic ────────────────────────────────────────────────
    "anthropic/claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "anthropic/claude-sonnet-4-6":         {"input": 0.003,   "output": 0.015},
    "anthropic/claude-opus-4-5":           {"input": 0.015,   "output": 0.075},
    # ── Paid cloud — OpenAI ───────────────────────────────────────────────────
    "gpt-4o-mini":                         {"input": 0.00015, "output": 0.0006},
    "gpt-4o":                              {"input": 0.005,   "output": 0.015},
}


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def chat(
    messages: list[dict],
    model: str = MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    tools: list[dict] | None = None,
    **kwargs,
):
    """
    Unified sync chat call. Works with Ollama, Anthropic, and OpenAI.
    Returns a litellm ModelResponse (same shape regardless of provider).

    response.choices[0].message.content  → text
    response.usage.prompt_tokens         → input tokens
    response.usage.completion_tokens     → output tokens
    """
    if system:
        messages = [{"role": "system", "content": system}] + messages

    params = dict(model=model, messages=messages, max_tokens=max_tokens, **kwargs)
    if tools:
        params["tools"] = tools

    return completion(**params)


def get_text(response) -> str:
    """Extract text content from a LiteLLM response."""
    return response.choices[0].message.content or ""


def get_tool_calls(response) -> list:
    """Extract tool calls from a LiteLLM response (empty list if none)."""
    return response.choices[0].message.tool_calls or []


def stop_reason(response) -> str:
    """
    Normalise stop reason across providers.
    Returns 'tool_use' or 'end_turn'.
    """
    reason = response.choices[0].finish_reason or ""
    if reason in {"tool_calls", "tool_use"}:
        return "tool_use"
    return "end_turn"


# ── Async wrapper ──────────────────────────────────────────────────────────────

async def achat(
    messages: list[dict],
    model: str = MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    tools: list[dict] | None = None,
    **kwargs,
):
    """Async version of chat(). Use with asyncio / fan-out exercises."""
    if system:
        messages = [{"role": "system", "content": system}] + messages
    params = dict(model=model, messages=messages, max_tokens=max_tokens, **kwargs)
    if tools:
        params["tools"] = tools
    return await acompletion(**params)


# ── Stream wrapper ─────────────────────────────────────────────────────────────

def stream_chat(
    messages: list[dict],
    model: str = MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    **kwargs,
):
    """
    Streaming generator. Yields text chunks one at a time.

    Example:
        for chunk in stream_chat([{"role": "user", "content": "Hi"}]):
            print(chunk, end="", flush=True)
    """
    if system:
        messages = [{"role": "system", "content": system}] + messages
    response = completion(
        model=model, messages=messages, max_tokens=max_tokens,
        stream=True, **kwargs,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Testing model: {MODEL}\n")

    # Sync
    r = chat([{"role": "user", "content": "Say 'hello' and nothing else."}])
    print(f"Sync:    {get_text(r)!r}")
    print(f"Tokens:  {r.usage.prompt_tokens} in / {r.usage.completion_tokens} out")
    print(f"Cost:    ${calc_cost(MODEL, r.usage.prompt_tokens, r.usage.completion_tokens):.6f}")

    # Stream
    print("\nStream:  ", end="")
    for chunk in stream_chat([{"role": "user", "content": "Count to 5, one word per line."}]):
        print(chunk, end="", flush=True)
    print()
