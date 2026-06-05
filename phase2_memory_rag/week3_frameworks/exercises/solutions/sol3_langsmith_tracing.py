"""
SOLUTION — Exercise 3: Agent Tracing — Manual Trace Log + Optional LangSmith
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
import time
import math
import datetime
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()


@dataclass
class TraceEvent:
    event: str
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
        self.events.append(TraceEvent(event="start", data={"query": query}))

    def record_llm_call(self, response, step: int):
        in_tok = getattr(getattr(response, "usage", None), "prompt_tokens", 0)
        out_tok = getattr(getattr(response, "usage", None), "completion_tokens", 0)
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.events.append(TraceEvent(
            event="llm_call",
            data={
                "step": step,
                "stop_reason": stop_reason(response),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            },
        ))

    def record_tool_call(self, name: str, arguments: dict, result: str, step: int):
        self.events.append(TraceEvent(
            event="tool_call",
            data={"step": step, "name": name, "arguments": arguments, "result": result[:200]},
        ))

    def record_end(self, answer: str):
        elapsed = time.perf_counter() - self.start_time
        self.events.append(TraceEvent(
            event="end",
            data={"answer": answer[:200], "elapsed_seconds": round(elapsed, 3)},
        ))

    def save(self) -> str:
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


def run_agent(query: str, max_steps: int = 8) -> str:
    tracer = Tracer()
    tracer.record_start(query)

    messages = [{"role": "user", "content": query}]
    step = 0

    while step < max_steps:
        step += 1
        response = chat(messages, max_tokens=512, tools=TOOLS)
        tracer.record_llm_call(response, step)
        messages.append(assistant_message(response))

        if stop_reason(response) == "end_turn":
            answer = get_text(response)
            tracer.record_end(answer)
            filename = tracer.save()
            print(f"Trace saved → {filename}")
            print(f"Summary: {tracer.summary()}")
            return answer

        for tc in get_tool_calls(response):
            result = run_tool(tc["name"], tc["arguments"])
            print(f"Step {step} | tool: {tc['name']}({tc['arguments']}) → {result}")
            tracer.record_tool_call(tc["name"], tc["arguments"], result, step)
            messages.append(tool_result_message(tc["id"], result))

    tracer.record_end("[max_steps reached]")
    tracer.save()
    return "[max_steps reached]"


if __name__ == "__main__":
    answer = run_agent("What is 2**10 and how many words are in 'the quick brown fox jumps over the lazy dog'?")
    print(f"\nAnswer: {answer}")
