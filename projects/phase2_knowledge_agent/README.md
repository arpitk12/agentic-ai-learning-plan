# Project 2 — Personal Knowledge Agent

## Brief
An agent that ingests a folder of your documents (PDFs, markdown, text),
builds a RAG pipeline, and answers questions with source citations.
Uses LangGraph for routing and persists memory to SQLite.

## Requirements
- [ ] Ingests a folder of mixed files (PDF, .md, .txt)
- [ ] Chunks and embeds all documents into Chroma
- [ ] LangGraph router: RAG vs tool call vs direct answer
- [ ] Every answer includes source file + chunk reference
- [ ] Conversation memory persists to SQLite across sessions
- [ ] CLI interface with session history

## Setup
```bash
pip install langchain langgraph chromadb sentence-transformers pypdf \
            anthropic python-dotenv pydantic sqlalchemy
```

## Usage
```bash
python starter.py --docs ./my_docs/
# Then chat:
# You: What did the Q3 report say about revenue?
# Agent: [RAG] According to q3_report.pdf (page 4): revenue grew 23%...
```

## Architecture
```
User query
    ↓
[Router Node] → Is this answerable from docs? Tool needed? Or direct?
    ↓                    ↓                         ↓
[RAG Node]        [Tool Node]              [LLM Node]
    ↓                    ↓                         ↓
         → [Synthesizer Node] → Answer + citations
                    ↓
              [Memory Node] → save to SQLite
```

## Hints
- Use `LangGraph` checkpoint for memory persistence
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (free, fast)
- Router prompt: "Classify this query as: RAG / TOOL / DIRECT"
- Citation format: `[source: filename.pdf, chunk 3]`
