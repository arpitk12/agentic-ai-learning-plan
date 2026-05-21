"""
Exercise 2: ReAct Loop from Scratch
Goal: Implement Reason → Act → Observe without any framework.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import json
import math
import datetime
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, MODEL

# --- Tool definitions ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Returns a float.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '2 ** 10 + sqrt(16)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Returns the current date and time as a string.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count words in a text string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    }
]


def run_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return result as string."""
    if name == "calculator":
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
    messages = [{"role": "user", "content": user_query}]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        response = chat(messages=messages, tools=TOOLS)
        reason = stop_reason(response)
        print(f"Stop reason: {reason}")

        if reason == "end_turn":
            return get_text(response)

        if reason == "tool_use":
            tool_calls = get_tool_calls(response)
            messages.append(assistant_message(response))
            for tc in tool_calls:
                result = run_tool(tc["name"], tc["arguments"])
                print(f"  [Tool: {tc['name']}({tc['arguments']})] → {result}")
                messages.append(tool_result_message(tc["id"], result))

    return "Max steps reached"


if __name__ == "__main__":
    answer = react_agent(
        "What is 2^16, and how many words are in that sentence? Also what time is it?"
    )
    print(f"\nFinal answer: {answer}")
