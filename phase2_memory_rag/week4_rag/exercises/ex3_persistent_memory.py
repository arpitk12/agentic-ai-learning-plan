"""
Exercise 3: Persistent Memory Agent
Goal: Agent that remembers user preferences/facts across sessions using SQLite.

Run multiple times — the agent should recall facts from previous sessions:
    python ex3_persistent_memory.py --session alice
    python ex3_persistent_memory.py --session alice   # second run remembers

Tasks:
  1. Complete init_db() — create messages + memories tables.
  2. Complete save_message() and load_history() for conversation persistence.
  3. Complete extract_and_store_facts() — ask LLM to pull key facts from
     the conversation and store them in the memories table.
  4. Complete load_memories() — retrieve relevant stored facts.
  5. Wire it all together in chat_loop() — inject memories into system prompt.

Expected behaviour:
  Session 1: "My name is Alice and I like Python"
  Session 2: "What's my name?" → Agent recalls "Alice"
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import sqlite3
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

DB_PATH = "memory_agent.db"

# ── Database ───────────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """Create tables if they don't exist. Return open connection."""
    conn = sqlite3.connect(DB_PATH)
    # TODO: CREATE TABLE IF NOT EXISTS messages (
    #   id INTEGER PRIMARY KEY AUTOINCREMENT,
    #   session_id TEXT NOT NULL,
    #   role TEXT NOT NULL,       -- 'user' or 'assistant'
    #   content TEXT NOT NULL,
    #   ts DATETIME DEFAULT CURRENT_TIMESTAMP
    # )
    # TODO: CREATE TABLE IF NOT EXISTS memories (
    #   id INTEGER PRIMARY KEY AUTOINCREMENT,
    #   session_id TEXT NOT NULL,
    #   fact TEXT NOT NULL,        -- extracted fact string
    #   ts DATETIME DEFAULT CURRENT_TIMESTAMP
    # )
    raise NotImplementedError


def save_message(conn: sqlite3.Connection, session_id: str, role: str, content: str):
    """Persist one message turn."""
    # TODO: INSERT INTO messages ...
    raise NotImplementedError


def load_history(conn: sqlite3.Connection, session_id: str, last_n: int = 10) -> list[dict]:
    """Return the last N messages for this session as list of {role, content}."""
    # TODO: SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?
    # Return reversed so oldest first
    raise NotImplementedError


def save_memory(conn: sqlite3.Connection, session_id: str, fact: str):
    """Store one extracted fact."""
    # TODO: INSERT INTO memories ...
    raise NotImplementedError


def load_memories(conn: sqlite3.Connection, session_id: str) -> list[str]:
    """Return all stored facts for this session."""
    # TODO: SELECT fact FROM memories WHERE session_id=? ORDER BY ts
    raise NotImplementedError


# ── Memory Extraction ──────────────────────────────────────────────────────────

EXTRACTOR_SYSTEM = """Extract key facts about the user from this conversation.
Return a JSON array of short fact strings. If no facts, return [].
Examples of facts: "User's name is Alice", "User prefers Python", "User works at Acme Corp"
JSON only, no markdown."""


def extract_and_store_facts(conn: sqlite3.Connection, session_id: str, history: list[dict]):
    """Ask LLM to extract facts from recent history and store them."""
    if len(history) < 2:
        return
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
    response = chat(
        [{"role": "user", "content": f"Conversation:\n{convo}"}],
        system=EXTRACTOR_SYSTEM,
        max_tokens=256,
    )
    raw = get_text(response).strip()
    try:
        facts = json.loads(raw)
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                save_memory(conn, session_id, fact.strip())
                print(f"  [Memory stored] {fact}")
    except (json.JSONDecodeError, TypeError):
        pass


# ── Chat Loop ──────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a helpful assistant with persistent memory.

{memories_section}
Use these memories naturally in conversation. Update your understanding as the user shares new info."""


def chat_loop(session_id: str):
    conn = init_db()
    history = load_history(conn, session_id)
    memories = load_memories(conn, session_id)

    memories_section = ""
    if memories:
        memories_section = "Known facts about this user:\n" + "\n".join(f"- {m}" for m in memories)
        print(f"[Loaded {len(memories)} memories for session '{session_id}']\n")
    else:
        print(f"[New session '{session_id}' — no prior memories]\n")

    system = SYSTEM_TEMPLATE.format(memories_section=memories_section)

    print("Chat with the memory agent. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            # Extract and store facts before leaving
            extract_and_store_facts(conn, session_id, history)
            break

        save_message(conn, session_id, "user", user_input)
        history.append({"role": "user", "content": user_input})

        trimmed = history[-20:]
        response = chat(trimmed, system=system, max_tokens=512)
        reply = get_text(response)

        save_message(conn, session_id, "assistant", reply)
        history.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="default", help="Session ID (e.g. 'alice')")
    args = parser.parse_args()
    chat_loop(args.session)
