"""
SOLUTION — Project 8: Fully Observed Agent

Three observability pillars wired into a ReAct research agent:
  - structlog  → structured JSON logs with run_id bound to every line
  - Prometheus → counters + histograms exposed at http://localhost:9090/metrics
  - OTel       → traces (root span per run, child spans per LLM/tool call)

Run:
    pip install structlog prometheus-client opentelemetry-api opentelemetry-sdk
    python solution.py "What are the main patterns in multi-agent AI?"
    curl http://localhost:9090/metrics
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import time
import logging
import uuid
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

# ── Optional imports with graceful fallback ────────────────────────────────────

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

# ── Mock Tools ────────────────────────────────────────────────────────────────

def web_search(query: str) -> str:
    return (
        f"[Search: {query}]\n"
        "Key patterns: Orchestrator-Worker, Debate/Adversarial, Reflexion, "
        "Fan-Out/Fan-In, Map-Reduce, HITL, LangGraph StateGraph."
    )

def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: Only arithmetic allowed."
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
        "description": "Search the web for current information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}
        }, "required": ["expression"]},
    }},
]

SYSTEM = "You are a helpful research assistant. Use web_search for information and calculate for math."


# ── Setup: structlog ──────────────────────────────────────────────────────────

def setup_structlog():
    if not HAS_STRUCTLOG:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger("agent")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,           # includes run_id automatically
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )
    return structlog.get_logger()


# ── Setup: OpenTelemetry tracing ──────────────────────────────────────────────

def setup_otel_tracing():
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


# ── Setup: Prometheus metrics ─────────────────────────────────────────────────

def setup_prometheus_metrics(metrics_port: int = 9090):
    if not HAS_PROMETHEUS:
        class NoOp:
            def labels(self, **kwargs): return self
            def inc(self, n=1): pass
            def observe(self, v): pass
            def set(self, v): pass
        noop = NoOp()
        return noop, noop, noop, noop, noop, noop

    llm_calls   = Counter("llm_calls_total",        "LLM API calls",           ["status"])
    llm_latency = Histogram("llm_duration_seconds",  "LLM call duration (s)",
                             buckets=[0.5, 1, 2, 5, 10, 30, 60])
    llm_cost    = Counter("llm_cost_usd_total",      "LLM cost in USD")
    tool_calls  = Counter("tool_calls_total",         "Tool calls",             ["tool_name", "status"])
    agent_steps = Histogram("agent_steps_per_run",   "Steps per agent run",
                             buckets=[1, 2, 3, 5, 8, 13, 21])
    active      = Gauge("agents_active",              "Currently running agents")

    try:
        start_http_server(metrics_port)
        print(f"📈 Prometheus metrics → http://localhost:{metrics_port}/metrics")
    except OSError:
        pass  # port already in use

    return llm_calls, llm_latency, llm_cost, tool_calls, agent_steps, active


# ── Instrumented Wrappers ─────────────────────────────────────────────────────

def make_instrumented_chat(log, tracer, llm_calls, llm_latency, llm_cost):
    def instrumented_chat(messages: list, run_id: str = "", **kwargs) -> object:
        start = time.time()
        with tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("run_id", run_id)
            span.set_attribute("message_count", len(messages))
            try:
                response = chat(messages, **kwargs)
                latency = time.time() - start
                llm_calls.labels(status="success").inc()
                llm_latency.observe(latency)
                if hasattr(response, "usage") and response.usage:
                    cost = calc_cost(
                        MODEL,
                        getattr(response.usage, "prompt_tokens", 0),
                        getattr(response.usage, "completion_tokens", 0),
                    )
                    llm_cost.inc(cost)
                log.info("llm_call",
                         run_id=run_id,
                         latency_ms=round(latency * 1000),
                         status="success")
                span.set_attribute("latency_ms", round(latency * 1000))
                return response
            except Exception as e:
                llm_calls.labels(status="error").inc()
                log.error("llm_call_failed", run_id=run_id, error=str(e))
                span.set_attribute("error", str(e))
                raise
    return instrumented_chat


def make_instrumented_tool(log, tracer, tool_calls):
    def instrumented_tool_call(tool_name: str, args: dict, run_id: str = "") -> str:
        with tracer.start_as_current_span(f"tool_{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("run_id", run_id)
            try:
                fn = TOOL_DISPATCH.get(tool_name, lambda a: f"Unknown tool: {tool_name}")
                result = fn(args)
                tool_calls.labels(tool_name=tool_name, status="success").inc()
                log.info("tool_call",
                         run_id=run_id, tool=tool_name, result_len=len(str(result)))
                span.set_attribute("result_len", len(str(result)))
                return result
            except Exception as e:
                tool_calls.labels(tool_name=tool_name, status="error").inc()
                log.error("tool_call_failed", run_id=run_id, tool=tool_name, error=str(e))
                return f"Error: {e}"
    return instrumented_tool_call


# ── Main Agent ────────────────────────────────────────────────────────────────

def make_run_agent(log, tracer, instrumented_chat, instrumented_tool_call,
                   agent_steps, active_agents):
    def run_observed_agent(query: str) -> str:
        run_id = str(uuid.uuid4())[:8]
        if HAS_STRUCTLOG:
            structlog.contextvars.bind_contextvars(run_id=run_id)

        log.info("agent_start", query=query[:100])
        active_agents.set(1)
        start = time.time()
        steps = 0

        try:
            with tracer.start_as_current_span("react_agent") as span:
                span.set_attribute("run_id", run_id)
                span.set_attribute("query", query[:200])

                messages = [{"role": "user", "content": query}]

                for _ in range(15):
                    response = instrumented_chat(messages, run_id=run_id,
                                                 tools=TOOLS, system=SYSTEM)
                    reason = stop_reason(response)
                    messages.append(assistant_message(response))
                    steps += 1

                    if reason == "tool_calls":
                        for tc in get_tool_calls(response):
                            result = instrumented_tool_call(tc["name"], tc["arguments"], run_id)
                            messages.append(tool_result_message(tc["id"], result))
                    elif reason == "stop":
                        final = get_text(response)
                        duration = time.time() - start
                        agent_steps.observe(steps)
                        log.info("agent_done",
                                 steps=steps,
                                 duration_ms=round(duration * 1000))
                        span.set_attribute("steps", steps)
                        return final

                log.warning("agent_max_steps", steps=steps)
                return "Maximum steps reached."
        finally:
            active_agents.set(0)
            if HAS_STRUCTLOG:
                structlog.contextvars.unbind_contextvars("run_id")

    return run_observed_agent


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    log     = setup_structlog()
    tracer  = setup_otel_tracing()
    (llm_calls, llm_latency, llm_cost,
     tool_calls, agent_steps, active_agents) = setup_prometheus_metrics(9090)

    ichat = make_instrumented_chat(log, tracer, llm_calls, llm_latency, llm_cost)
    itool = make_instrumented_tool(log, tracer, tool_calls)
    run_agent = make_run_agent(log, tracer, ichat, itool, agent_steps, active_agents)

    query = " ".join(sys.argv[1:]) if sys.argv[1:] else "What are the main patterns in multi-agent AI?"
    print(f"\n📊 Observed Agent  |  Metrics → http://localhost:9090/metrics\n")

    result = run_agent(query)
    print(f"\n✅ Result:\n{result}")


if __name__ == "__main__":
    main()
