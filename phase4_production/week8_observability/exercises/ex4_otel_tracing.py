"""
Exercise 4: OpenTelemetry Tracing — Instrument Agent with Spans
Goal: Each agent step becomes an OTel span, visible in any APM tool.

Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

Without a collector, spans are printed to console via ConsoleSpanExporter.
With a collector: set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

Tasks:
  1. Complete setup_tracer() — configure TracerProvider with ConsoleSpanExporter fallback.
  2. Complete run_agent() — wrap each LLM call in a span "llm.chat",
     each tool call in "tool.{name}", and the whole run in "agent.run".
  3. Set span attributes: model, tokens, cost, tool_name, tool_result_preview.
  4. Record exceptions on spans with span.record_exception(e).
  5. (Bonus) Export to Jaeger: pip install opentelemetry-exporter-jaeger
     and run: docker run -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one

Expected console output (abbreviated):
  {
    "name": "agent.run",
    "attributes": {"query": "...", "total_steps": 2, "total_tokens": 340},
    "events": [...],
    ...
  }
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import math
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

# ── OTel Setup ─────────────────────────────────────────────────────────────────

def setup_tracer():
    """
    Configure OTel tracer with ConsoleSpanExporter.
    TODO:
      from opentelemetry import trace
      from opentelemetry.sdk.trace import TracerProvider
      from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

      provider = TracerProvider()
      provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
      trace.set_tracer_provider(provider)
      return trace.get_tracer("agent.tracer")

    Fallback if opentelemetry not installed: return a no-op tracer mock.
    """
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


# ── Traced Agent ───────────────────────────────────────────────────────────────

def run_agent(query: str, max_steps: int = 8) -> str:
    """
    Wrap the full agent run in a parent span "agent.run".
    Each LLM call → child span "llm.chat".
    Each tool call → child span "tool.{name}".
    """
    with tracer.start_as_current_span("agent.run") as root_span:
        # TODO: root_span.set_attribute("query", query)
        # TODO: root_span.set_attribute("model", MODEL)
        # TODO: root_span.set_attribute("max_steps", max_steps)

        messages = [{"role": "user", "content": query}]
        total_input = 0
        total_output = 0
        step = 0

        while step < max_steps:
            step += 1

            with tracer.start_as_current_span("llm.chat") as llm_span:
                # TODO: llm_span.set_attribute("step", step)
                # TODO: llm_span.set_attribute("model", MODEL)
                try:
                    response = chat(messages, max_tokens=512, tools=TOOLS)
                    # TODO: llm_span.set_attribute("input_tokens", response.usage.prompt_tokens)
                    # TODO: llm_span.set_attribute("output_tokens", response.usage.completion_tokens)
                    # TODO: llm_span.set_attribute("stop_reason", stop_reason(response))
                    # TODO: llm_span.set_attribute("cost_usd", calc_cost(MODEL, ...))
                    total_input += response.usage.prompt_tokens
                    total_output += response.usage.completion_tokens
                except Exception as e:
                    # TODO: llm_span.record_exception(e)
                    raise

            messages.append(assistant_message(response))

            if stop_reason(response) == "end_turn":
                answer = get_text(response)
                # TODO: root_span.set_attribute("total_steps", step)
                # TODO: root_span.set_attribute("total_tokens", total_input + total_output)
                # TODO: root_span.set_attribute("total_cost_usd", calc_cost(MODEL, total_input, total_output))
                return answer

            for tc in get_tool_calls(response):
                with tracer.start_as_current_span(f"tool.{tc['name']}") as tool_span:
                    # TODO: tool_span.set_attribute("step", step)
                    # TODO: tool_span.set_attribute("tool.name", tc["name"])
                    # TODO: tool_span.set_attribute("tool.arguments", str(tc["arguments"]))
                    result = run_tool(tc["name"], tc["arguments"])
                    # TODO: tool_span.set_attribute("tool.result_preview", result[:100])
                    tool_span.set_attribute("tool.name", tc["name"])
                    print(f"  [tool] {tc['name']} → {result}")
                messages.append(tool_result_message(tc["id"], result))

    return "[max_steps reached]"


if __name__ == "__main__":
    answer = run_agent("What is 15^2 - sqrt(225)? Also count: 'hello world foo bar'")
    print(f"\nAnswer: {answer}")
