"""
Exercise 3: Tool Error Handling — Graceful Failure in ReAct Loops
Goal: Build a robust agent that handles tool failures without crashing.

Scenario: You have 3 unreliable tools (they randomly raise exceptions).
The agent must:
  1. Catch tool errors and return an error string to the LLM (not crash)
  2. Let the LLM decide whether to retry, use a different tool, or apologize
  3. Implement a max_steps guard to prevent infinite loops
  4. Log each step with its outcome

Tools:
  - flaky_search(query) → fails 50% of the time with a TimeoutError
  - flaky_database(id) → fails if id > 100 with a ValueError
  - always_works(message) → always returns "OK: {message}"
"""
import random
import json
from llm import chat, get_text, get_tool_calls, stop_reason, MODEL

TOOLS = [
    {
        "name": "flaky_search",
        "description": "Search for information. Unreliable — may timeout.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "flaky_database",
        "description": "Look up a record by ID (valid range: 1-100).",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "always_works",
        "description": "A reliable fallback tool that always succeeds.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
]


def flaky_search(query: str) -> str:
    if random.random() < 0.5:
        raise TimeoutError(f"Search timed out for query: '{query}'")
    return f"Search results for '{query}': [result1, result2, result3]"


def flaky_database(id: int) -> str:
    if id > 100:
        raise ValueError(f"ID {id} out of range (max 100)")
    return f"Record {id}: {{name: 'Item {id}', value: {id * 10}}}"


def always_works(message: str) -> str:
    return f"OK: {message}"


TOOL_MAP = {"flaky_search": flaky_search, "flaky_database": flaky_database, "always_works": always_works}


def safe_execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool safely — never raise, always return a string result."""
    raise NotImplementedError


def run_robust_agent(user_message: str, max_steps: int = 10) -> str:
    """Run a ReAct loop that gracefully handles tool errors."""
    messages = [{"role": "user", "content": user_message}]
    step = 0

    while step < max_steps:
        step += 1
        # TODO: Implement the loop:
        #   1. Call LLM with TOOLS
        #   2. If stop_reason == "end_turn" → return final text
        #   3. If stop_reason == "tool_use" → call safe_execute_tool for each
        #      tool_use block, collect results into tool_result messages
        #   4. Append assistant message + tool results to messages
        #   5. Continue loop
        raise NotImplementedError

    return "[Agent reached max_steps without a final answer]"


if __name__ == "__main__":
    print(run_robust_agent("Search for 'AI trends' and also look up database record 150."))
