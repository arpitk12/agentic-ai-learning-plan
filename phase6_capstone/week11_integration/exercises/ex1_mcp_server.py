"""
Exercise 1: MCP Server — Expose Agent Tools via Model Context Protocol
Goal: Build an MCP server that exposes calculator and web-search tools.

Install: pip install mcp

What is MCP?
  Model Context Protocol is an open standard (by Anthropic) that lets LLMs
  connect to external tools and data sources through a unified interface.
  Any MCP-compatible client (Claude Desktop, VS Code, custom agents) can
  connect to your server and use its tools.

Run server:
  python ex1_mcp_server.py

Test manually:
  # In a second terminal, test with the MCP inspector:
  npx @modelcontextprotocol/inspector python ex1_mcp_server.py

Tasks:
  1. Complete the `calculator` tool handler — parse operation and operands from args.
  2. Complete the `web_search` tool handler — return mock results (or real with requests).
  3. Complete the `summarize_url` tool handler — fetch URL and summarize with LLM.
  4. Add a `get_weather` tool for a city (mock or real via wttr.in).
  5. (Bonus) Add a resource that exposes the contents of a local file.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# ── MCP Server Setup ──────────────────────────────────────────────────────────

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Run: pip install mcp")
    print("Showing tool definitions only...\n")


# ── Tool Implementations ──────────────────────────────────────────────────────

def calculator(operation: str, a: float, b: float) -> float | str:
    """
    Perform a basic arithmetic operation.
    Operations: add, subtract, multiply, divide, power, modulo

    TODO:
    1. Use a match statement (or if/elif) for each operation.
    2. Return the numeric result (float).
    3. Raise ValueError for unknown operations or division by zero.
    """
    raise NotImplementedError


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
    """
    Return mock search results for the given query.

    TODO:
    1. Lowercase the query and check if any key in MOCK_SEARCH_DB appears in it.
    2. Return the matching snippets (or "default") as a list of dicts:
       [{"title": "Result 1", "snippet": "...", "url": "https://example.com/1"}, ...]
    3. Limit to num_results.
    """
    raise NotImplementedError


def summarize_url(url: str) -> str:
    """
    Fetch a URL and return a summary.

    TODO (real implementation):
    1. Use requests.get(url, timeout=10) to fetch the page.
    2. Extract text with BeautifulSoup (strip tags).
    3. Send to LLM: "Summarize this webpage in 3 bullet points: {text[:3000]}"
    4. Return the summary.

    For the mock implementation below, just return a placeholder.
    """
    # Mock implementation (replace with real when requests + bs4 are installed)
    return f"[Mock summary] The page at {url} contains information relevant to your query. Install requests and beautifulsoup4 for real summaries."


def get_weather(city: str) -> str:
    """
    TODO: Fetch weather for a city using wttr.in API (no key needed).
      url = f"https://wttr.in/{city}?format=j1"
      Parse JSON response for current conditions.
    """
    return f"[Mock] Weather in {city}: 22°C, partly cloudy. (Implement with requests for real data)"


# ── MCP Tool Definitions ──────────────────────────────────────────────────────

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
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize_url",
        "description": "Fetch and summarize a web page",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    },
]


def dispatch_tool(name: str, args: dict) -> str:
    """Route tool calls to implementations."""
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
    except NotImplementedError:
        return f"[TODO] Tool '{name}' is not implemented yet."
    except Exception as e:
        return f"Error: {e}"


# ── MCP Server (async) ────────────────────────────────────────────────────────

async def run_mcp_server():
    if not MCP_AVAILABLE:
        return

    server = Server("agentic-ai-tools")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        result = dispatch_tool(name, arguments)
        return [types.TextContent(type="text", text=result)]

    print("Starting MCP server (stdio transport)...")
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())


# ── Demo (without MCP) ────────────────────────────────────────────────────────

def demo_tools():
    """Show tool definitions and run quick tests."""
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
        print(f"  {name}({args}) → {result[:80]}...")


if __name__ == "__main__":
    if MCP_AVAILABLE:
        # Run as MCP server (connects via stdio to MCP clients)
        asyncio.run(run_mcp_server())
    else:
        # Fallback: demo tool definitions
        demo_tools()
