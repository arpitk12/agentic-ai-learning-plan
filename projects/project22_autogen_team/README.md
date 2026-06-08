# Project 22 — AutoGen: Multi-Agent Software Development Team

A **production-grade AutoGen application** demonstrating multi-agent conversation:
a simulated software team (Product Manager, Architect, Developer, Tester, Reviewer)
that takes a feature request and produces working, tested code through structured
agent dialogue with code execution in a Docker sandbox.

---

## 🎯 What You Learn

| Concept | Where |
|---------|-------|
| **AssistantAgent** — LLM-powered agent | `src/agents/agents.py` |
| **UserProxyAgent** — human / code executor | `src/agents/agents.py` |
| **GroupChat** — multi-agent conversation | `src/team/groupchat.py` |
| **GroupChatManager** — orchestrate turns | `src/team/groupchat.py` |
| **Custom speaker selection** — who speaks next | `src/team/speaker_selection.py` |
| **Code execution** — Docker sandbox | `src/execution/executor.py` |
| **Tool use** — function calling | `src/tools/tools.py` |
| **Nested chats** — agent calls agent | `src/team/nested_chat.py` |
| **Termination conditions** | `src/team/groupchat.py` |
| **Structured output** — Pydantic models | `src/agents/agents.py` |

---

## 🏗 Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║              MULTI-AGENT TEAM                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Feature Request (input)                                         ║
║       │                                                          ║
║  ┌────▼────────────────────────────────────────────────────┐    ║
║  │  GroupChat  (max_round=20, select_speaker_auto_llm=True)│    ║
║  │                                                          │    ║
║  │  PRODUCT MANAGER (AssistantAgent)                        │    ║
║  │    → clarifies requirements, writes user stories         │    ║
║  │                                                          │    ║
║  │  ARCHITECT (AssistantAgent)                              │    ║
║  │    → proposes system design, selects patterns            │    ║
║  │                                                          │    ║
║  │  DEVELOPER (AssistantAgent)                              │    ║
║  │    → writes Python code blocks (``` python ... ```)      │    ║
║  │                                                          │    ║
║  │  EXECUTOR (UserProxyAgent — human_input_mode=NEVER)      │    ║
║  │    → detects code blocks, runs in Docker sandbox         │    ║
║  │    → returns stdout/stderr to group                      │    ║
║  │                                                          │    ║
║  │  TESTER (AssistantAgent)                                 │    ║
║  │    → writes pytest test cases, asks EXECUTOR to run them │    ║
║  │                                                          │    ║
║  │  REVIEWER (AssistantAgent)                               │    ║
║  │    → reviews code quality, suggests improvements         │    ║
║  │    → signals TERMINATE when done                         │    ║
║  └──────────────────────────────────────────────────────────┘    ║
║                                                                   ║
║  Docker executor:  python:3.11-slim, 30s timeout, no network     ║
║  Termination:      "TERMINATE" in last message OR max_round=20   ║
╚═══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║              TWO-AGENT NESTED CHAT (simpler pattern)            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  USER_PROXY ◄──────────────────────────────► ASSISTANT          ║
║    │  sends feature request                    │  thinks + codes ║
║    │  receives code, executes it               │  reads result   ║
║    │  sends error back if any                  │  fixes code     ║
║    │  sends "TERMINATE" when tests pass        │                 ║
║    └──────────────── loop ────────────────────┘                 ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Folder Structure

```
project22_autogen_team/
├── README.md
├── GUIDE.md
├── starter/
│   ├── requirements.txt
│   ├── .env.example
│   └── src/
│       ├── config.py                  ← given
│       ├── agents/
│       │   └── agents.py              ← TODO (8 tasks) — 5 specialized agents
│       ├── team/
│       │   ├── groupchat.py           ← TODO (6 tasks) — GroupChat + Manager
│       │   ├── speaker_selection.py   ← TODO (4 tasks) — custom LLM speaker select
│       │   └── nested_chat.py         ← TODO (4 tasks) — two-agent pattern
│       ├── execution/
│       │   └── executor.py            ← TODO (4 tasks) — Docker code executor
│       ├── tools/
│       │   └── tools.py               ← TODO (4 tasks) — function tools
│       └── main.py                    ← TODO (3 tasks)
└── solution/
    └── src/
```

---

## ⚡ Key AutoGen Patterns

| Pattern | Code | Why |
|---------|------|-----|
| AssistantAgent | `AssistantAgent(name=..., system_message=..., llm_config=...)` | LLM-powered |
| UserProxyAgent | `UserProxyAgent(name=..., human_input_mode="NEVER", code_execution_config=...)` | Runs code |
| Trigger execution | `UserProxyAgent` detects ``` code blocks and executes them | Autonomous coding |
| GroupChat | `GroupChat(agents=[...], messages=[], max_round=20)` | Multi-agent conversation |
| Manager | `GroupChatManager(groupchat=gc, llm_config=...)` | Selects speaker |
| Custom speaker | `GroupChat(..., speaker_selection_method=my_fn)` | Deterministic routing |
| Initiate chat | `user_proxy.initiate_chat(manager, message="Build me a ...")` | Start the loop |
| Docker executor | `LocalCommandLineCodeExecutor(work_dir=..., timeout=30)` | Safe execution |
| Register tool | `@agent.register_for_llm(name=..., description=...)` decorator | LLM tool-use |

---

## 🚀 Quick Start

```bash
cd projects/project22_autogen_team/starter
pip install -r requirements.txt
cp .env.example .env

# Ensure Docker is running (for code execution sandbox)
docker info

# Two-agent task (simpler — start here)
python -m src.team.nested_chat "Write a function to detect palindromes with tests"

# Full team (GroupChat)
python -m src.main "Build a REST API endpoint for user registration with validation"
```

---

## Milestones

1. **Two-agent nested chat** — implement + test: coding assistant + executor
2. **Docker executor** — verify code runs safely in container
3. **Agents** — define all 5 team agents with distinct personas
4. **GroupChat** — wire into GroupChat + GroupChatManager
5. **Custom speaker selection** — implement structured turn order
6. **Tool use** — add function tools, verify LLM calls them
