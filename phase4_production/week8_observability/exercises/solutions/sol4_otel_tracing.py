"""
SOLUTION — Exercise 4: OpenTelemetry Tracing — Instrument Agent with Spans
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import math
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()


def setup_tracer():
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        return trace.get_tracer("agent.tracer")
    except ImportError:
        print("[OTel not installed] Using no-op tracer. Run: pip install opentelemetry-sdk")
        return _NoOpTracer()


class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_attribute(self, k, v): print(f"  [SPAN attr] {k}={v}")
    def add_event(self, name, attributes=None): print(f"  [SPAN event] {name}: {attributes}")
    def record_exception(self, e): print(f"  [SPAN exception] {e}")
    def set_status(self, *a): pass


class _NoOpTracer:
    def start_as_current_span(self, name, **kw):
        print(f"\n[SPAN] {name}")
        return _NoOpSpan()


tracer = setup_tracer()

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
    with tracer.start_as_current_span("agent.run") as root_span:
        root_span.set_attribute("query", query)
        root_span.set_attribute("model", MODEL)
        root_span.set_attribute("max_steps", max_steps)

        messages = [{"role": "user", "content": query}]
        total_input = 0
        total_output = 0
        step = 0

        while step < max_steps:
            step += 1

            with tracer.start_as_current_span("llm.chat") as llm_span:
                llm_span.set_attribute("step", step)
                llm_span.set_attribute("model", MODEL)
                try:
                    response = chat(messages, max_tokens=512, tools=TOOLS)
                    in_tok = getattr(getattr(response, "usage", None), "prompt_tokens", 0)
                    out_tok = getattr(getattr(response, "usage", None), "completion_tokens", 0)
                    llm_span.set_attribute("input_tokens", in_tok)
                    llm_span.set_attribute("output_tokens", out_tok)
                    llm_span.set_attribute("stop_reason", stop_reason(response))
                    llm_span.set_attribute("cost_usd", round(calc_cost(MODEL, in_tok, out_tok), 8))
                    total_input += in_tok
                    total_output += out_tok
                except Exception as e:
                    llm_span.record_exception(e)
                    raise

            messages.append(assistant_message(response))

            if stop_reason(response) == "end_turn":
                answer = get_text(response)
                root_span.set_attribute("total_steps", step)
                root_span.set_attribute("total_tokens", total_input + total_output)
                root_span.set_attribute("total_cost_usd", round(calc_cost(MODEL, total_input, total_output), 8))
                return answer

            for tc in get_tool_calls(response):
                with tracer.start_as_current_span(f"tool.{tc['name']}") as tool_span:
                    tool_span.set_attribute("step", step)
                    tool_span.set_attribute("tool.name", tc["name"])
                    tool_span.set_attribute("tool.arguments", str(tc["arguments"]))
                    result = run_tool(tc["name"], tc["arguments"])
                    tool_span.set_attribute("tool.result_preview", result[:100])
                    print(f"  [tool] {tc['name']} → {result}")
                messages.append(tool_result_message(tc["id"], result))

    return "[max_steps reached]"


if __name__ == "__main__":
    answer = run_agent("What is 15^2 - sqrt(225)? Also count: 'hello world foo bar'")
    print(f"\nAnswer: {answer}")
