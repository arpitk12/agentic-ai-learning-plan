"""
SOLUTION — Exercise 3: Persistent Memory Agent with SQLite
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import sqlite3
import json
import argparse
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

DB_PATH = "memory_agent.db"


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_message(conn: sqlite3.Connection, session_id: str, role: str, content: str):
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.commit()


def load_history(conn: sqlite3.Connection, session_id: str, last_n: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, last_n),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_memory(conn: sqlite3.Connection, session_id: str, fact: str):
    conn.execute(
        "INSERT INTO memories (session_id, fact) VALUES (?, ?)",
        (session_id, fact),
    )
    conn.commit()


def load_memories(conn: sqlite3.Connection, session_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT fact FROM memories WHERE session_id=? ORDER BY ts",
        (session_id,),
    ).fetchall()
    return [r[0] for r in rows]


EXTRACTOR_SYSTEM = """Extract key facts about the user from this conversation.
Return a JSON array of short fact strings. If no facts, return [].
Examples: "User's name is Alice", "User prefers Python", "User works at Acme Corp"
JSON only, no markdown."""


def extract_and_store_facts(conn: sqlite3.Connection, session_id: str, history: list[dict]):
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
        start = raw.find("[")
        end = raw.rfind("]") + 1
        facts = json.loads(raw[start:end])
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                save_memory(conn, session_id, fact.strip())
                print(f"  [Memory stored] {fact}")
    except (json.JSONDecodeError, TypeError):
        pass


SYSTEM_TEMPLATE = """You are a helpful assistant with persistent memory.

{memories_section}
Use these memories naturally in conversation."""


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
            extract_and_store_facts(conn, session_id, history)
            break

        save_message(conn, session_id, "user", user_input)
        history.append({"role": "user", "content": user_input})

        response = chat(history[-20:], system=system, max_tokens=512)
        reply = get_text(response)

        save_message(conn, session_id, "assistant", reply)
        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="default")
    args = parser.parse_args()
    chat_loop(args.session)
