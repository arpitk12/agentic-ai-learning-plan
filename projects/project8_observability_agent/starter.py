"""
Project 8 Starter — Fully Observed Agent

Instrument a ReAct agent with three observability pillars:
  1. structlog   — structured JSON logs with run_id context
  2. Prometheus  — Counter / Histogram / Gauge metrics + /metrics endpoint
  3. OpenTelemetry — distributed traces (spans) for every LLM call and tool

Usage:
    pip install structlog prometheus-client opentelemetry-api opentelemetry-sdk
    python starter.py "What are the main patterns in multi-agent AI systems?"

What you need to implement (TODOs 1-5):
  1. setup_structlog()           — configure JSON logging with timestamp
  2. setup_prometheus_metrics()  — define all metrics + start metrics HTTP server
  3. instrumented_chat()         — wrap chat() with latency + cost + OTel span
  4. instrumented_tool_call()    — wrap tool dispatch with OTel span + metrics
  5. run_observed_agent()        — full ReAct loop with run_id, root span, histogram
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import logging
import uuid
import json
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

# ── Optional imports (gracefully handle missing packages) ─────────────────────

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    print("⚠  structlog not installed. Run: pip install structlog")

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    print("⚠  prometheus-client not installed. Run: pip install prometheus-client")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    print("⚠  opentelemetry not installed. Run: pip install opentelemetry-api opentelemetry-sdk")


# ── Module-level logger and tracer (filled in by setup functions) ─────────────

log = None       # set by setup_structlog()
tracer = None    # set by setup_otel_tracing()

# Prometheus metric objects (filled in by setup_prometheus_metrics())
LLM_CALLS: "Counter"    = None
LLM_LATENCY: "Histogram" = None
LLM_COST: "Counter"     = None
TOOL_CALLS: "Counter"   = None
AGENT_STEPS: "Histogram" = None
ACTIVE_AGENTS: "Gauge"  = None


# ── Mock Tools (already complete) ─────────────────────────────────────────────

def web_search(query: str) -> str:
    """Simulate a web search returning relevant snippets."""
    return (
        f"[Search results for: {query}]\n"
        "1. Orchestrator-Worker: A planner LLM decomposes the task, workers execute subtasks.\n"
        "2. Debate/Adversarial: Two agents argue pro/con, a judge synthesises the verdict.\n"
        "3. Reflexion: An agent generates output, evaluates it, reflects, and retries.\n"
        "4. Fan-Out/Fan-In: Same task applied to many items in parallel via asyncio.gather().\n"
        "5. Map-Reduce: Extract relevant info from each document, then hierarchically synthesise."
    )


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: Only arithmetic expressions are allowed."
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


TOOL_DISPATCH = {
    "web_search": lambda a: web_search(a.get("query", "")),
    "calculate":  lambda a: calculate(a.get("expression", "")),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query"}
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression e.g. '(3 + 4) * 2'"}
        }, "required": ["expression"]},
    }},
]

SYSTEM = "You are a helpful research assistant. Use web_search for current information and calculate for math."


# ── Setup Functions ────────────────────────────────────────────────────────────

def setup_structlog():
    """
    Configure structlog for structured JSON logging with ISO timestamp.

    TODO 1:
      a. Import logging and structlog.
      b. Call structlog.configure() with these processors in order:
           - structlog.contextvars.merge_contextvars     ← merges bound context (run_id)
           - structlog.processors.add_log_level          ← adds "level" key
           - structlog.processors.TimeStamper(fmt="iso") ← adds "timestamp" key
           - structlog.processors.JSONRenderer()         ← outputs valid JSON lines
      c. Set wrapper_class = structlog.make_filtering_bound_logger(logging.INFO)
      d. Set logger_factory = structlog.PrintLoggerFactory(sys.stdout)
      e. Return structlog.get_logger()

    After this call, every log.info()/log.error() produces a JSON line like:
        {"timestamp": "2026-06-07T10:00:00", "level": "info", "event": "llm_call",
         "run_id": "abc123", "latency_ms": 1240}
    """
    if not HAS_STRUCTLOG:
        # Fallback: basic Python logger
        logging.basicConfig(level=logging.INFO,
                            format='{"time": "%(asctime)s", "level": "%(levelname)s", "event": "%(message)s"}')
        return logging.getLogger("agent")

    # TODO 1: configure structlog and return structlog.get_logger()
    raise NotImplementedError("setup_structlog() not implemented yet")


def setup_otel_tracing():
    """Set up an OpenTelemetry TracerProvider with in-memory export (console-friendly)."""
    if not HAS_OTEL:
        class NoOpTracer:
            def start_as_current_span(self, name, **kwargs):
                from contextlib import nullcontext
                return nullcontext()
        return NoOpTracer()

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("agent")


def setup_prometheus_metrics(metrics_port: int = 9090):
    """
    Define all Prometheus metrics and start the HTTP metrics server.

    TODO 2:
      a. Define these metrics (module-level globals):
           LLM_CALLS   = Counter("llm_calls_total",        "LLM API calls",         ["status"])
           LLM_LATENCY = Histogram("llm_duration_seconds", "LLM call latency",
                                   buckets=[0.5, 1, 2, 5, 10, 30, 60])
           LLM_COST    = Counter("llm_cost_usd_total",     "LLM spend in USD")
           TOOL_CALLS  = Counter("tool_calls_total",       "Tool calls",            ["tool_name", "status"])
           AGENT_STEPS = Histogram("agent_steps_per_run",  "Steps per agent run",
                                   buckets=[1, 2, 3, 5, 8, 13, 21])
           ACTIVE_AGENTS = Gauge("agents_active", "Currently running agents")
      b. Call start_http_server(metrics_port) to expose /metrics endpoint.
      c. Return the tuple (LLM_CALLS, LLM_LATENCY, LLM_COST, TOOL_CALLS, AGENT_STEPS, ACTIVE_AGENTS)

    After this: curl http://localhost:9090/metrics returns Prometheus text format.
    """
    if not HAS_PROMETHEUS:
        # Return no-op stubs
        class NoOp:
            def labels(self, **kwargs): return self
            def inc(self, n=1): pass
            def observe(self, v): pass
            def set(self, v): pass
        noop = NoOp()
        return noop, noop, noop, noop, noop, noop

    # TODO 2: define metrics and start server
    raise NotImplementedError("setup_prometheus_metrics() not implemented yet")


# ── Instrumented Wrappers ──────────────────────────────────────────────────────

def instrumented_chat(messages: list, run_id: str = "", **kwargs) -> object:
    """
    Wrap chat() with full observability: Prometheus metrics + OTel trace span + structured log.

    TODO 3:
      a. Record start time: start = time.time()
      b. Use tracer.start_as_current_span("llm_call") as span:
           - span.set_attribute("run_id", run_id)
           - span.set_attribute("message_count", len(messages))
           try:
             response = chat(messages, **kwargs)
             latency = time.time() - start
             LLM_CALLS.labels(status="success").inc()
             LLM_LATENCY.observe(latency)
             # Extract token counts from response.usage if available:
             if hasattr(response, "usage") and response.usage:
                 cost = calc_cost(MODEL, response.usage.prompt_tokens,
                                  response.usage.completion_tokens)
                 LLM_COST.inc(cost)
             log.info("llm_call", run_id=run_id,
                      latency_ms=round(latency * 1000),
                      status="success")
             span.set_attribute("latency_ms", round(latency * 1000))
             return response
           except Exception as e:
             LLM_CALLS.labels(status="error").inc()
             log.error("llm_call_failed", run_id=run_id, error=str(e))
             raise
    """
    # TODO 3: implement instrumented LLM wrapper
    raise NotImplementedError("instrumented_chat() not implemented yet")


def instrumented_tool_call(tool_name: str, args: dict, run_id: str = "") -> str:
    """
    Wrap tool dispatch with OTel span + Prometheus counter + structured log.

    TODO 4:
      Use tracer.start_as_current_span(f"tool_{tool_name}") as span:
           span.set_attribute("tool.name", tool_name)
           span.set_attribute("run_id", run_id)
           try:
             result = TOOL_DISPATCH[tool_name](args)
             TOOL_CALLS.labels(tool_name=tool_name, status="success").inc()
             log.info("tool_call", run_id=run_id, tool=tool_name, result_len=len(str(result)))
             return result
           except Exception as e:
             TOOL_CALLS.labels(tool_name=tool_name, status="error").inc()
             log.error("tool_call_failed", run_id=run_id, tool=tool_name, error=str(e))
             return f"Error: {e}"
    """
    # TODO 4: implement instrumented tool wrapper
    raise NotImplementedError("instrumented_tool_call() not implemented yet")


# ── Main Agent Loop ────────────────────────────────────────────────────────────

def run_observed_agent(query: str) -> str:
    """
    Full ReAct loop with all observability layers wired in.

    TODO 5:
      a. Generate run_id = str(uuid.uuid4())[:8]
      b. If HAS_STRUCTLOG: structlog.contextvars.bind_contextvars(run_id=run_id)
         This makes run_id appear in EVERY log line automatically.
      c. log.info("agent_start", query=query[:100])
      d. Set ACTIVE_AGENTS.set(1) at start, ACTIVE_AGENTS.set(0) at end/finally.
      e. Record start = time.time(); steps = 0
      f. Open a root OTel span: tracer.start_as_current_span("react_agent") as span
           span.set_attribute("run_id", run_id)
           span.set_attribute("query", query[:200])
           ReAct loop (max 15 steps):
             response = instrumented_chat(messages, run_id=run_id, tools=TOOLS, system=SYSTEM)
             reason   = stop_reason(response)
             messages.append(assistant_message(response))
             steps += 1
             if reason == "tool_calls":
                 for tc in get_tool_calls(response):
                     result = instrumented_tool_call(tc["name"], tc["arguments"], run_id)
                     messages.append(tool_result_message(tc["id"], result))
             elif reason == "stop":
                 final = get_text(response)
                 duration = time.time() - start
                 AGENT_STEPS.observe(steps)
                 log.info("agent_done", steps=steps, duration_ms=round(duration * 1000))
                 span.set_attribute("steps", steps)
                 return final
      g. Log warning if max steps reached.
    """
    global log, tracer, LLM_CALLS, LLM_LATENCY, LLM_COST, TOOL_CALLS, AGENT_STEPS, ACTIVE_AGENTS

    # TODO 5: implement fully observed agent loop
    # For now: naive fallback so the file at least runs
    messages = [{"role": "user", "content": query}]
    for _ in range(15):
        response = chat(messages=messages, tools=TOOLS, system=SYSTEM)
        reason = stop_reason(response)
        messages.append(assistant_message(response))
        if reason == "tool_calls":
            for tc in get_tool_calls(response):
                result = TOOL_DISPATCH.get(tc["name"], lambda a: "Unknown tool")(tc["arguments"])
                messages.append(tool_result_message(tc["id"], result))
        elif reason == "stop":
            return get_text(response)
    return "Max steps reached."


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    global log, tracer, LLM_CALLS, LLM_LATENCY, LLM_COST, TOOL_CALLS, AGENT_STEPS, ACTIVE_AGENTS

    # Initialise observability stack
    log = setup_structlog()
    tracer = setup_otel_tracing()
    (LLM_CALLS, LLM_LATENCY, LLM_COST,
     TOOL_CALLS, AGENT_STEPS, ACTIVE_AGENTS) = setup_prometheus_metrics(metrics_port=9090)

    query = " ".join(sys.argv[1:]) if sys.argv[1:] else "What are the main patterns in multi-agent AI?"
    print(f"\n📊 Observed Agent  |  Metrics → http://localhost:9090/metrics\n")

    result = run_observed_agent(query)
    print(f"\n✅ Result:\n{result}")


if __name__ == "__main__":
    main()
