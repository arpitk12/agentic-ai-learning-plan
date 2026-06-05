"""
Exercise 2: Human-in-the-Loop Approval Gate
Goal: Add a human approval checkpoint before the agent takes destructive actions.

Scenario: An agent can read files, write files, and send emails.
Write and send_email are "destructive" — require human approval before executing.

Tasks:
  1. Complete classify_action() — return "safe" or "destructive".
  2. Complete human_approval() — print the proposed action and ask for y/n.
  3. Complete run_agent_with_approval() — approve safe tools automatically,
     gate destructive tools behind human_approval().
  4. If the user rejects an action, return an error string to the LLM so it
     can adapt (e.g. suggest an alternative or stop).
  5. Add a --auto flag to auto-approve everything (for testing).

Expected output:
  [AUTO-APPROVED] read_file({"path": "notes.txt"})
  ⚠ APPROVAL REQUIRED: send_email({"to": "boss@corp.com", "body": "..."})
  Approve? [y/n]: n
  → Rejected. LLM adapts.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
import argparse
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()

# ── Tools ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a text file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (destructive — overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email (destructive — cannot be undone).",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
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


# ── Tool Implementations ───────────────────────────────────────────────────────

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


# ── Approval Logic ─────────────────────────────────────────────────────────────

def classify_action(tool_name: str) -> str:
    """Return 'safe' or 'destructive'."""
    # TODO: return "destructive" if tool_name in DESTRUCTIVE_TOOLS else "safe"
    raise NotImplementedError


def human_approval(tool_name: str, arguments: dict) -> bool:
    """Print the proposed action, prompt user, return True if approved."""
    # TODO: print warning, show tool_name + json.dumps(arguments, indent=2)
    # TODO: ask input("Approve? [y/n]: ")
    # TODO: return True if answer.strip().lower() == "y"
    raise NotImplementedError


def execute_tool(name: str, arguments: dict, auto_approve: bool) -> str:
    """Execute with approval gate. Return result string."""
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


# ── Agent ──────────────────────────────────────────────────────────────────────

SYSTEM = """You are a helpful file and email assistant.
You can read files, list directories, write files, and send emails.
Always read before writing. Ask clarifying questions if unsure of intent.
If an action is rejected, acknowledge it and suggest a safe alternative."""


def run_agent_with_approval(user_request: str, auto_approve: bool = False, max_steps: int = 8) -> str:
    messages = [{"role": "user", "content": user_request}]
    print(f"\n[Agent] Processing: {user_request}\n")

    for step in range(max_steps):
        response = chat(messages, system=SYSTEM, max_tokens=512, tools=TOOLS)
        messages.append(assistant_message(response))

        if stop_reason(response) == "end_turn":
            return get_text(response)

        for tc in get_tool_calls(response):
            result = execute_tool(tc["name"], tc["arguments"], auto_approve)
            print(f"    → {result[:80]}")
            messages.append(tool_result_message(tc["id"], result))

    return "[max_steps reached]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Auto-approve all actions")
    args = parser.parse_args()

    # Create a test file first
    with open("test_notes.txt", "w") as f:
        f.write("Meeting notes: discuss Q3 roadmap with team.")

    result = run_agent_with_approval(
        "Read test_notes.txt and send a summary email to team@company.com with subject 'Notes Summary'.",
        auto_approve=args.auto,
    )
    print(f"\nFinal answer: {result}")
