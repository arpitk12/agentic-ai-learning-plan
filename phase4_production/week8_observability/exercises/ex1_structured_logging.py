"""
Exercise 1: Structured Logging — Instrument Every Agent Step
Goal: Every LLM call and tool use emits a structured JSON log line.

Install: pip install structlog

Tasks:
  1. Configure structlog to output JSON (not human-readable) — complete setup_logging().
  2. Complete log_llm_call() — emit an "llm_call" log with model, tokens, cost, step.
  3. Complete log_tool_call() — emit a "tool_call" log with name, inputs, result, duration_ms.
  4. Instrument run_agent() to log every step.
  5. Pipe output to a file: python ex1_structured_logging.py 2>agent.log
     Then: cat agent.log | python3 -c "import sys,json;[print(json.dumps(json.loads(l),indent=2)) for l in sys.stdin]"

Expected log lines (JSON):
  {"event": "agent_start", "query": "...", "model": "gemini/..."}
  {"event": "llm_call", "step": 1, "stop_reason": "tool_use", "input_tokens": 120, "cost_usd": 0.0}
  {"event": "tool_call", "step": 1, "tool": "calculator", "duration_ms": 0.3, "result": "1024"}
  {"event": "agent_complete", "steps": 2, "total_tokens": 340, "total_cost_usd": 0.0, "duration_ms": 1240}
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import math
import time
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

# ── Logging Setup ──────────────────────────────────────────────────────────────

def setup_logging():
    """
    Configure structlog to emit JSON to stderr.
    TODO:
      import structlog
      structlog.configure(
          processors=[
              structlog.processors.TimeStamper(fmt="iso"),
              structlog.processors.add_log_level,
              structlog.processors.JSONRenderer(),
          ],
          wrapper_class=structlog.BoundLogger,
          context_class=dict,
          logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
      )
      return structlog.get_logger()
    """
    # Fallback: plain print logger (no structlog installed)
    class SimpleLogger:
        def info(self, event, **kw):
            import json
            print(json.dumps({"event": event, **kw}), file=sys.stderr)
        def warning(self, event, **kw):
            self.info(event, **kw)
    return SimpleLogger()


log = setup_logging()


# ── Instrumented Helpers ───────────────────────────────────────────────────────

def log_llm_call(response, step: int):
    """
    TODO: log.info("llm_call",
        step=step,
        model=MODEL,
        stop_reason=stop_reason(response),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost_usd=calc_cost(MODEL, response.usage.prompt_tokens, response.usage.completion_tokens),
    )
    """
    raise NotImplementedError


def log_tool_call(name: str, inputs: dict, result: str, step: int, duration_ms: float):
    """
    TODO: log.info("tool_call",
        step=step,
        tool=name,
        inputs=inputs,
        result=result[:100],
        duration_ms=round(duration_ms, 2),
    )
    """
    raise NotImplementedError


# ── Tools ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "word_count",
        "description": "Count words in a text.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]

_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        return str(eval(args["expression"], {"__builtins__": {}}, _MATH))  # noqa: S307
    if name == "word_count":
        return str(len(args["text"].split()))
    return f"Unknown: {name}"


# ── Instrumented Agent ─────────────────────────────────────────────────────────

def run_agent(query: str, max_steps: int = 8) -> str:
    t_start = time.perf_counter()
    # TODO: log.info("agent_start", query=query, model=MODEL, max_steps=max_steps)

    messages = [{"role": "user", "content": query}]
    total_input = 0
    total_output = 0
    step = 0

    while step < max_steps:
        step += 1
        response = chat(messages, max_tokens=512, tools=TOOLS)

        # TODO: log_llm_call(response, step)
        # TODO: total_input += response.usage.prompt_tokens
        # TODO: total_output += response.usage.completion_tokens

        messages.append(assistant_message(response))

        if stop_reason(response) == "end_turn":
            answer = get_text(response)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            # TODO: log.info("agent_complete", steps=step, total_input_tokens=total_input,
            #     total_output_tokens=total_output,
            #     total_cost_usd=calc_cost(MODEL, total_input, total_output),
            #     duration_ms=round(elapsed_ms, 1))
            return answer

        for tc in get_tool_calls(response):
            t0 = time.perf_counter()
            result = run_tool(tc["name"], tc["arguments"])
            duration_ms = (time.perf_counter() - t0) * 1000
            # TODO: log_tool_call(tc["name"], tc["arguments"], result, step, duration_ms)
            messages.append(tool_result_message(tc["id"], result))

    return "[max_steps reached]"


if __name__ == "__main__":
    print("Running agent... (logs go to stderr)", flush=True)
    answer = run_agent("What is 2^10 + sqrt(256)? Also count the words in 'the quick brown fox jumps'")
    print(f"\nAnswer: {answer}")
    print("\nTip: Run with '2>agent.log' to capture logs separately.")
