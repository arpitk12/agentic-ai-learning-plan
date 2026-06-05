"""
SOLUTION — Exercise 1: MCP Server — Expose Agent Tools via Model Context Protocol
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Run: pip install mcp")


# ── Tool Implementations ──────────────────────────────────────────────────────

def calculator(operation: str, a: float, b: float) -> float | str:
    match operation:
        case "add":      return a + b
        case "subtract": return a - b
        case "multiply": return a * b
        case "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        case "power":    return a ** b
        case "modulo":
            if b == 0:
                raise ValueError("Modulo by zero")
            return a % b
        case _:
            raise ValueError(f"Unknown operation: {operation}")


MOCK_SEARCH_DB: dict[str, list[str]] = {
    "python": [
        "Python is a high-level, general-purpose programming language.",
        "Python 3.13 was released in October 2024 with major performance improvements.",
        "pip install <package> is how you install Python packages.",
    ],
    "ai": [
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "Large Language Models (LLMs) are AI systems trained on vast text corpora.",
        "LiteLLM provides a unified interface to 100+ LLM providers.",
    ],
    "default": [
        "No specific results found. Try a more specific search term.",
    ],
}


def web_search(query: str, num_results: int = 3) -> list[dict]:
    query_lower = query.lower()
    results_raw = MOCK_SEARCH_DB["default"]
    for key, snippets in MOCK_SEARCH_DB.items():
        if key in query_lower:
            results_raw = snippets
            break
    results = [
        {"title": f"Result {i+1} for '{query}'", "snippet": s, "url": f"https://example.com/{i+1}"}
        for i, s in enumerate(results_raw[:num_results])
    ]
    return results


def summarize_url(url: str) -> str:
    try:
        import requests
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.texts = []
            def handle_data(self, data):
                self.texts.append(data)

        resp = requests.get(url, timeout=10)
        extractor = TextExtractor()
        extractor.feed(resp.text)
        text = " ".join(extractor.texts)[:3000]

        from llm import chat, get_text
        r = chat(
            [{"role": "user", "content": f"Summarize this webpage in 3 bullet points:\n{text}"}],
            max_tokens=200,
        )
        return get_text(r)
    except Exception as e:
        return f"[Mock summary] Page at {url}: {e}"


def get_weather(city: str) -> str:
    try:
        import requests
        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5)
        data = resp.json()
        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        desc = current["weatherDesc"][0]["value"]
        return f"{city}: {temp_c}°C, {desc}"
    except Exception:
        return f"[Mock] Weather in {city}: 22°C, partly cloudy."


TOOLS = [
    {
        "name": "calculator",
        "description": "Perform arithmetic: add, subtract, multiply, divide, power, modulo",
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide", "power", "modulo"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "num_results": {"type": "integer", "default": 3}},
            "required": ["query"],
        },
    },
    {
        "name": "summarize_url",
        "description": "Fetch and summarize a web page",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]


def dispatch_tool(name: str, args: dict) -> str:
    try:
        if name == "calculator":
            result = calculator(args["operation"], args["a"], args["b"])
            return str(result)
        elif name == "web_search":
            results = web_search(args["query"], args.get("num_results", 3))
            return json.dumps(results, indent=2)
        elif name == "summarize_url":
            return summarize_url(args["url"])
        elif name == "get_weather":
            return get_weather(args["city"])
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


async def run_mcp_server():
    if not MCP_AVAILABLE:
        return

    server = Server("agentic-ai-tools")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        result = dispatch_tool(name, arguments)
        return [types.TextContent(type="text", text=result)]

    print("Starting MCP server (stdio transport)...")
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())


def demo_tools():
    print("=== Available MCP Tools ===\n")
    for tool in TOOLS:
        print(f"  🔧 {tool['name']}: {tool['description']}")

    print("\n=== Tool Tests ===\n")
    tests = [
        ("calculator", {"operation": "add", "a": 15, "b": 27}),
        ("calculator", {"operation": "divide", "a": 100, "b": 4}),
        ("web_search", {"query": "python programming", "num_results": 2}),
        ("get_weather", {"city": "London"}),
    ]
    for name, args in tests:
        result = dispatch_tool(name, args)
        print(f"  {name}({args}) → {result[:80]}")


if __name__ == "__main__":
    if MCP_AVAILABLE:
        asyncio.run(run_mcp_server())
    else:
        demo_tools()
