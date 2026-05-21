"""
SOLUTION — Exercise 3: Tool Error Handling
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import random
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()

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
        raise ValueError(f"ID {id} out of valid range (1-100)")
    return f"Record {id}: {{name: 'Item {id}', value: {id * 10}}}"


def always_works(message: str) -> str:
    return f"OK: {message}"


TOOL_MAP = {
    "flaky_search": flaky_search,
    "flaky_database": flaky_database,
    "always_works": always_works,
}


def safe_execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool safely — never raise, always return a string."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"ERROR: Tool '{name}' not found"
    try:
        return fn(**inputs)
    except Exception as e:
        error_msg = f"ERROR [{type(e).__name__}]: {e}"
        print(f"    ⚠ Tool '{name}' failed: {error_msg}")
        return error_msg


def run_robust_agent(user_message: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n=== Step {step} ===")

        response = chat(messages, max_tokens=1024, tools=TOOLS)

        messages.append(assistant_message(response))
        print(f"  stop_reason: {stop_reason(response)}")

        if stop_reason(response) == "end_turn":
            return get_text(response)

        if stop_reason(response) == "tool_use":
            for tc in get_tool_calls(response):
                print(f"  [CALL] {tc['name']}({tc['arguments']})")
                result = safe_execute_tool(tc["name"], tc["arguments"])
                print(f"  [RESULT] → {result[:80]}")
                messages.append(tool_result_message(tc["id"], result))

    return "[Agent reached max_steps without a final answer]"


if __name__ == "__main__":
    print(run_robust_agent(
        "Search for 'AI trends'. Also look up database record 150. Then confirm with always_works."
    ))
