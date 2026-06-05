"""
SOLUTION — Exercise 2: Human-in-the-Loop Approval Gate
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
import argparse
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a text file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (destructive — overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email (destructive — cannot be undone).",
        "input_schema": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory (safe, read-only).",
        "input_schema": {
            "type": "object",
            "properties": {"directory": {"type": "string"}},
            "required": ["directory"],
        },
    },
]

DESTRUCTIVE_TOOLS = {"write_file", "send_email"}


def read_file_tool(path: str) -> str:
    try:
        return open(path).read()[:1000]
    except FileNotFoundError:
        return f"[File not found: {path}]"


def write_file_tool(path: str, content: str) -> str:
    with open(path, "w") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


def send_email_tool(to: str, subject: str, body: str) -> str:
    print(f"  [SIMULATED] Email sent to {to}: '{subject}'")
    return f"Email sent to {to}"


def list_files_tool(directory: str) -> str:
    try:
        return "\n".join(os.listdir(directory))
    except Exception as e:
        return str(e)


TOOL_MAP = {
    "read_file": read_file_tool,
    "write_file": write_file_tool,
    "send_email": send_email_tool,
    "list_files": list_files_tool,
}


def classify_action(tool_name: str) -> str:
    return "destructive" if tool_name in DESTRUCTIVE_TOOLS else "safe"


def human_approval(tool_name: str, arguments: dict) -> bool:
    print(f"\n  ⚠ APPROVAL REQUIRED: {tool_name}")
    print(f"  Arguments:\n{json.dumps(arguments, indent=4)}")
    answer = input("  Approve? [y/n]: ").strip().lower()
    return answer == "y"


def execute_tool(name: str, arguments: dict, auto_approve: bool) -> str:
    action_type = classify_action(name)

    if action_type == "safe" or auto_approve:
        tag = "[AUTO-APPROVED]" if action_type == "destructive" else "[SAFE]"
        print(f"  {tag} {name}({json.dumps(arguments)})")
        fn = TOOL_MAP.get(name)
        return fn(**arguments) if fn else f"Unknown tool: {name}"
    else:
        approved = human_approval(name, arguments)
        if approved:
            fn = TOOL_MAP.get(name)
            return fn(**arguments) if fn else f"Unknown tool: {name}"
        else:
            return f"ACTION REJECTED by user: {name} was not approved. Please suggest an alternative."


SYSTEM = """You are a helpful file and email assistant.
You can read files, list directories, write files, and send emails.
Always read before writing. Ask clarifying questions if unsure.
If an action is rejected, acknowledge it and suggest a safe alternative."""


def run_agent_with_approval(task: str, auto_approve: bool = False) -> str:
    messages = [{"role": "user", "content": task}]
    print(f"\nTask: {task}")
    print(f"Mode: {'AUTO-APPROVE' if auto_approve else 'HUMAN-IN-THE-LOOP'}\n")

    for _ in range(10):
        response = chat(messages, system=SYSTEM, tools=TOOLS, max_tokens=512)
        messages.append(assistant_message(response))

        if stop_reason(response) == "end_turn":
            answer = get_text(response)
            print(f"\nAgent: {answer}")
            return answer

        for tc in get_tool_calls(response):
            result = execute_tool(tc["name"], tc["arguments"], auto_approve)
            messages.append(tool_result_message(tc["id"], result))

    return "[max iterations reached]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Auto-approve all actions")
    args = parser.parse_args()

    run_agent_with_approval(
        "List files in the current directory, then write a file called 'test_output.txt' with the text 'Hello from agent!'",
        auto_approve=args.auto,
    )
