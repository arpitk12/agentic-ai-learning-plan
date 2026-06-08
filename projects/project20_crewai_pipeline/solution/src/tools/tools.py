"""Custom CrewAI tools — search and file I/O."""
from __future__ import annotations

from crewai.tools import BaseTool, tool
from pydantic import Field


class TavilySearchTool(BaseTool):
    """Web search via Tavily API."""
    name: str = "web_search"
    description: str = "Search the web for current information. Input: a search query string."
    max_results: int = Field(default=5)

    def _run(self, query: str) -> str:  # type: ignore[override]
        try:
            from tavily import TavilyClient
            import os
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            results = client.search(query, max_results=self.max_results)
            lines = []
            for r in results.get("results", []):
                lines.append(f"[{r['title']}]({r['url']})\n{r.get('content', '')[:500]}")
            return "\n\n".join(lines) or "No results found."
        except Exception as e:
            return f"Search error: {e}"


@tool("read_file")
def read_file_tool(path: str) -> str:
    """Read the contents of a local file. Input: absolute or relative file path."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"


@tool("write_file")
def write_file_tool(content: str, path: str = "output/article.md") -> str:
    """Write text content to a file. Input: content string and optional path."""
    import pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} chars to {path}"
