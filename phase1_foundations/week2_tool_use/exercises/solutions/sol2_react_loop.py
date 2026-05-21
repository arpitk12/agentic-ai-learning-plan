"""
SOLUTION — Exercise 2: ReAct Loop from Scratch
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import math
import datetime
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Returns a float.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '2 ** 10 + sqrt(16)'"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_datetime",
        "description": "Returns the current date and time as a string.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "word_count",
        "description": "Count words in a text string.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def run_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(inputs["expression"], {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    elif name == "get_datetime":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif name == "word_count":
        return str(len(inputs["text"].split()))
    else:
        return f"Unknown tool: {name}"


def react_agent(user_query: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_query}]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        response = chat(messages, max_tokens=1024, tools=TOOLS)

        print(f"  stop_reason: {stop_reason(response)}")

        messages.append(assistant_message(response))

        if stop_reason(response) == "end_turn":
            return get_text(response)

        if stop_reason(response) == "tool_use":
            for tc in get_tool_calls(response):
                print(f"  [ACT]     {tc['name']}({tc['arguments']})")
                result = run_tool(tc["name"], tc["arguments"])
                print(f"  [OBSERVE] → {result}")
                messages.append(tool_result_message(tc["id"], result))

    return "[max_steps reached]"


if __name__ == "__main__":
    answer = react_agent(
        "What is 2^16, and how many words are in that sentence? Also what time is it?"
    )
    print(f"\nFinal answer: {answer}")
