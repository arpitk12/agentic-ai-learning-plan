"""
Exercise 2: ReAct Loop from Scratch
Goal: Implement Reason → Act → Observe without any framework.
"""
import json
import math
import datetime
from llm import chat, get_text, get_tool_calls, stop_reason, MODEL

# --- Tool definitions ---
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Returns a float.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '2 ** 10 + sqrt(16)'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_datetime",
        "description": "Returns the current date and time as a string.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "word_count",
        "description": "Count words in a text string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        }
    }
]


def run_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return result as string."""
    if name == "calculator":
        # Safe eval with math functions available
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(inputs["expression"], {"__builtins__": {}}, allowed)
        return str(result)
    elif name == "get_datetime":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif name == "word_count":
        return str(len(inputs["text"].split()))
    else:
        raise ValueError(f"Unknown tool: {name}")


def react_agent(user_query: str, max_steps: int = 5) -> str:
    """
    TODO: Implement the full ReAct loop.
    The agent must reason, call tools, observe results, and return a final answer.
    Guard against infinite loops with max_steps.
    """
    messages = [{"role": "user", "content": user_query}]
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        # TODO: Call the API
        # TODO: Handle tool_use — execute each tool block, inject results
        # TODO: Handle end_turn — return the final text

        pass
    
    return "Max steps reached"


if __name__ == "__main__":
    answer = react_agent(
        "What is 2^16, and how many words are in that sentence? Also what time is it?"
    )
    print(f"\nFinal answer: {answer}")
