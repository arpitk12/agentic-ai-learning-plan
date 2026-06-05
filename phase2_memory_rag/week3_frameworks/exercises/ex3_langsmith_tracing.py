"""
Exercise 3: Agent Tracing — Manual Trace Log + Optional LangSmith
Goal: Instrument an agent so every step is recorded and inspectable.

Without LangSmith (default): writes a structured JSON trace to disk.
With LangSmith: set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true in .env.

Tasks:
  1. Complete the Tracer class — record start, llm_call, tool_call, end events.
  2. Instrument run_agent() to emit trace events at each step.
  3. Save the trace to 'trace_{timestamp}.json'.
  4. (Bonus) Add the @traceable decorator if LangSmith key is set.
  5. Compute and print: total steps, total tokens, wall time.

Expected output:
  Step 1 | tool: calculator({"expression": "2**10"}) → 1024
  Step 2 | end_turn
  Trace saved → trace_20250601_120000.json
  Summary: 2 steps | 340 tokens | 1.2s
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
import time
import math
import datetime
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()

# ── Tracer ─────────────────────────────────────────────────────────────────────

@dataclass
class TraceEvent:
    event: str          # "start" | "llm_call" | "tool_call" | "end"
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


@dataclass
class Tracer:
    run_id: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    events: list[TraceEvent] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record_start(self, query: str):
        # TODO: append a TraceEvent(event="start", data={"query": query})
        raise NotImplementedError

    def record_llm_call(self, response, step: int):
        # TODO: append event="llm_call" with step, stop_reason, token counts
        # Hint: response.usage.prompt_tokens / completion_tokens
        raise NotImplementedError

    def record_tool_call(self, name: str, arguments: dict, result: str, step: int):
        # TODO: append event="tool_call" with step, name, arguments, result[:200]
        raise NotImplementedError

    def record_end(self, answer: str):
        # TODO: append event="end" with answer and elapsed seconds
        raise NotImplementedError

    def save(self) -> str:
        """Save trace to JSON file, return filename."""
        filename = f"trace_{self.run_id}.json"
        with open(filename, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "events": [asdict(e) for e in self.events],
            }, f, indent=2)
        return filename

    def summary(self) -> str:
        steps = sum(1 for e in self.events if e.event in {"llm_call", "tool_call"})
        elapsed = time.perf_counter() - self.start_time
        total_tokens = self.total_input_tokens + self.total_output_tokens
        return f"{steps} steps | {total_tokens} tokens | {elapsed:.1f}s"


# ── Tools ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "word_count",
        "description": "Count words in a string.",
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
    return f"Unknown tool: {name}"


# ── Instrumented Agent ─────────────────────────────────────────────────────────

def run_agent(query: str, max_steps: int = 8) -> str:
    tracer = Tracer()
    # TODO: call tracer.record_start(query)

    messages = [{"role": "user", "content": query}]
    step = 0

    while step < max_steps:
        step += 1
        response = chat(messages, max_tokens=512, tools=TOOLS)

        # TODO: call tracer.record_llm_call(response, step)

        reason = stop_reason(response)
        print(f"  Step {step} | {reason}", end="")

        messages.append(assistant_message(response))

        if reason == "end_turn":
            answer = get_text(response)
            print()
            # TODO: call tracer.record_end(answer)
            filename = tracer.save()
            print(f"  Trace saved → {filename}")
            print(f"  Summary: {tracer.summary()}")
            return answer

        for tc in get_tool_calls(response):
            result = run_tool(tc["name"], tc["arguments"])
            print(f" | tool: {tc['name']}({tc['arguments']}) → {result[:60]}")
            # TODO: call tracer.record_tool_call(tc["name"], tc["arguments"], result, step)
            messages.append(tool_result_message(tc["id"], result))

    return "[max_steps reached]"


if __name__ == "__main__":
    answer = run_agent("What is sqrt(144) + 2^8? Also count the words in: 'the quick brown fox'")
    print(f"\nAnswer: {answer}")
