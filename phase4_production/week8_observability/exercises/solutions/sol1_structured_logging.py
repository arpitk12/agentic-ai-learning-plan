"""
SOLUTION — Exercise 1: Structured Logging — Instrument Every Agent Step
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import math
import time
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()


def setup_logging():
    try:
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
    except ImportError:
        import json

        class SimpleLogger:
            def info(self, event, **kw):
                print(json.dumps({"event": event, **kw}), file=sys.stderr)
            def warning(self, event, **kw):
                self.info(event, **kw)

        return SimpleLogger()


log = setup_logging()


def log_llm_call(response, step: int):
    in_tok = getattr(getattr(response, "usage", None), "prompt_tokens", 0)
    out_tok = getattr(getattr(response, "usage", None), "completion_tokens", 0)
    log.info(
        "llm_call",
        step=step,
        model=MODEL,
        stop_reason=stop_reason(response),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=round(calc_cost(MODEL, in_tok, out_tok), 8),
    )


def log_tool_call(name: str, inputs: dict, result: str, step: int, duration_ms: float):
    log.info(
        "tool_call",
        step=step,
        tool=name,
        inputs=inputs,
        result=result[:100],
        duration_ms=round(duration_ms, 2),
    )


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


def run_agent(query: str, max_steps: int = 8) -> str:
    t_start = time.perf_counter()
    log.info("agent_start", query=query, model=MODEL, max_steps=max_steps)

    messages = [{"role": "user", "content": query}]
    total_input = 0
    total_output = 0
    step = 0

    while step < max_steps:
        step += 1
        response = chat(messages, max_tokens=512, tools=TOOLS)

        log_llm_call(response, step)
        in_tok = getattr(getattr(response, "usage", None), "prompt_tokens", 0)
        out_tok = getattr(getattr(response, "usage", None), "completion_tokens", 0)
        total_input += in_tok
        total_output += out_tok

        messages.append(assistant_message(response))

        if stop_reason(response) == "end_turn":
            answer = get_text(response)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            log.info(
                "agent_complete",
                steps=step,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_cost_usd=round(calc_cost(MODEL, total_input, total_output), 8),
                duration_ms=round(elapsed_ms, 1),
            )
            return answer

        for tc in get_tool_calls(response):
            t0 = time.perf_counter()
            result = run_tool(tc["name"], tc["arguments"])
            duration_ms = (time.perf_counter() - t0) * 1000
            log_tool_call(tc["name"], tc["arguments"], result, step, duration_ms)
            messages.append(tool_result_message(tc["id"], result))

    return "[max_steps reached]"


if __name__ == "__main__":
    print("Running agent... (logs go to stderr)", flush=True)
    answer = run_agent("What is 2^10 + sqrt(256)? Also count the words in 'the quick brown fox jumps'")
    print(f"\nAnswer: {answer}")
    print("\nTip: Run with '2>agent.log' to capture logs separately.")
