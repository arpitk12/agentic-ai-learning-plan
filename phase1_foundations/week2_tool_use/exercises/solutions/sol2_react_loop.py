"""
SOLUTION — Exercise 2: ReAct Loop from Scratch
"""
import json
import math
import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

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

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        print(f"  stop_reason: {response.stop_reason}")

        # Always append the assistant turn first
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [ACT]     {block.name}({json.dumps(block.input)})")
                    result = run_tool(block.name, block.input)
                    print(f"  [OBSERVE] → {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

    return "[max_steps reached]"


if __name__ == "__main__":
    answer = react_agent(
        "What is 2^16, and how many words are in that sentence? Also what time is it?"
    )
    print(f"\nFinal answer: {answer}")
