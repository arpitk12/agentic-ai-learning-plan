# Project 18 — LangChain: Multi-Tool Research Agent

A **production-ready LangChain agent** that demonstrates the full modern LangChain stack:
LCEL (LangChain Expression Language) chains, custom tools, conversation memory, streaming,
and structured output. Built without magic — every chain is explicit and inspectable.

---

## 🎯 What You Learn

| Concept | Where |
|---------|-------|
| **LCEL** — pipe `|` chains | `src/chains/rag_chain.py` |
| **Custom tools** (`@tool` decorator) | `src/tools/` |
| **ReAct agent** — reason + act loop | `src/agent/agent.py` |
| **Memory** — sliding window buffer | `src/memory/memory.py` |
| **Streaming** — `astream_events` | `src/agent/streaming.py` |
| **Structured output** — Pydantic + LLM | `src/chains/structured_chain.py` |
| **Retrieval chain** — RAG with LCEL | `src/chains/rag_chain.py` |
| **Callbacks** — custom logging | `src/observability/callbacks.py` |

---

## 🏗 Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                    LCEL CHAIN ANATOMY                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Input dict                                                      ║
║       │                                                          ║
║  RunnableParallel    ← fetch context + pass-through question    ║
║    ├── context  ← retriever.invoke(question)                    ║
║    └── question ← RunnablePassthrough()                         ║
║       │                                                          ║
║  ChatPromptTemplate  ← format context + question into messages  ║
║       │                                                          ║
║  ChatLiteLLM         ← call LLM (any model via LiteLLM)        ║
║       │                                                          ║
║  StrOutputParser     ← extract string from AIMessage            ║
║       │                                                          ║
║  Answer string                                                   ║
║                                                                  ║
║  LCEL:  chain = retriever | prompt | llm | parser               ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                    REACT AGENT LOOP                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  User question                                                   ║
║       │                                                          ║
║  AgentExecutor                                                   ║
║       │                                                          ║
║  ┌────▼──────────────────────────────────────────────────────┐  ║
║  │  THINK: LLM decides which tool to call                    │  ║
║  │    └─► tools:                                             │  ║
║  │         web_search(query)    ← Tavily search API          │  ║
║  │         wikipedia(query)     ← Wikipedia summary          │  ║
║  │         calculator(expr)     ← safe eval math             │  ║
║  │         rag_search(query)    ← FAISS local docs           │  ║
║  │  ACT:   call chosen tool                                   │  ║
║  │  OBSERVE: tool returns result                              │  ║
║  │  repeat until Final Answer                                 │  ║
║  └───────────────────────────────────────────────────────────┘  ║
║       │                                                          ║
║  Memory: ConversationBufferWindowMemory (k=5 turns)             ║
║  Streaming: token-by-token via astream_events                   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Folder Structure

```
project18_langchain_agent/
├── README.md
├── GUIDE.md
├── starter/
│   ├── requirements.txt
│   ├── .env.example
│   └── src/
│       ├── config.py               ← given
│       ├── chains/
│       │   ├── rag_chain.py        ← TODO (8 tasks) — LCEL retrieval chain
│       │   └── structured_chain.py ← TODO (5 tasks) — Pydantic structured output
│       ├── tools/
│       │   ├── web_search.py       ← TODO (4 tasks) — Tavily search tool
│       │   ├── wikipedia_tool.py   ← TODO (3 tasks) — Wikipedia tool
│       │   └── calculator.py       ← TODO (3 tasks) — safe math tool
│       ├── memory/
│       │   └── memory.py           ← TODO (4 tasks) — conversation memory
│       ├── agent/
│       │   ├── agent.py            ← TODO (6 tasks) — ReAct agent
│       │   └── streaming.py        ← TODO (4 tasks) — astream_events
│       └── observability/
│           └── callbacks.py        ← TODO (3 tasks) — custom callbacks
└── solution/
    └── src/   ← full implementation
```

---

## ⚡ Key LangChain Patterns

| Pattern | Code | Why |
|---------|------|-----|
| LCEL pipe | `chain = prompt \| llm \| parser` | Composable, lazy evaluation |
| Runnable parallel | `RunnableParallel(ctx=retriever, q=passthrough)` | Fetch context + question simultaneously |
| Structured output | `llm.with_structured_output(MyModel)` | Type-safe LLM responses |
| Bind tools | `llm.bind_tools(tools)` | Pass tool schemas to LLM |
| Streaming | `async for event in chain.astream_events(input, version="v2")` | Token-by-token output |
| RunnableLambda | `RunnableLambda(my_func)` | Wrap any function in LCEL |

---

## 🚀 Quick Start

```bash
cd projects/project18_langchain_agent/starter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add TAVILY_API_KEY + LLM key

# Run the agent interactively
python -m src.agent.agent

# Stream a single query
python -m src.agent.streaming "What is the population of Tokyo and how does it compare to London?"
```

---

## Milestones

1. **LCEL Chains** — implement `rag_chain.py`, run `python -m src.chains.rag_chain`
2. **Tools** — implement all 3 tools, verify each independently
3. **ReAct Agent** — wire tools + LLM into AgentExecutor, test multi-step reasoning
4. **Memory** — add ConversationBufferWindowMemory, verify context persists across turns
5. **Streaming** — implement `astream_events`, verify tokens arrive in real-time
