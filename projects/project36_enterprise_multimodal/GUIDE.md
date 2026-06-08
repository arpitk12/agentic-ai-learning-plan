# Implementation Guide — Enterprise Multimodal Compliance Agent

This guide walks you through each module in implementation order.
Read **before** writing code. Compare with `solution/` after.

---

## Module 1 — Config & Models (`src/config.py`, `src/models.py`)

**What:** Centralise all environment variables and define Pydantic schemas for
every request and response the API accepts.

**Key decisions:**
- Use `pydantic_settings.BaseSettings` so `.env` values are auto-loaded
- Separate `LLMConfig` (model names, fallback order) from `ServiceConfig` (URLs, ports)
- Define `AnalyzeRequest`, `IngestResult`, `SearchResult`, `MemoryEntry` as typed models

**Checklist:**
- [ ] `Config` class with `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `REDIS_URL`
- [ ] `LLM_PRIMARY`, `LLM_FALLBACK`, `LLM_FINE_TUNED` with sane defaults
- [ ] `AnalyzeRequest(user_id, question, include_graph, top_k)`
- [ ] `AnalyzeResponse(answer, sources, graph_facts, memories_used, cost_usd, latency_ms)`
- [ ] `IngestResult(doc_id, chunks, entities, images, audio_segments)`

---

## Module 2 — Multimodal Ingestion (`src/ingestion/`)

### 2a — PDF Extractor (`pdf_extractor.py`)

**What:** Open a PDF with PyMuPDF, extract text per page, extract embedded images.

```python
import fitz  # pip install pymupdf

def extract_text_chunks(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    doc = fitz.open(pdf_path)
    # For each page: page.get_text() → split into overlapping chunks
    # Return: [{"text": str, "page": int, "source": str}]

def extract_images(pdf_path: str) -> list[dict]:
    # fitz.open() → page.get_images(full=True) → doc.extract_image(xref)
    # Return: [{"bytes": bytes, "ext": str, "page": int, "xref": int}]
```

**Pitfalls:**
- `page.get_text()` returns empty string for scanned PDFs → use `extract_images` + OCR
- Image `xref` can duplicate across pages — deduplicate by xref
- Chunk on sentence boundaries if possible (split on `. `)

### 2b — Vision Analyzer (`vision_analyzer.py`)

**What:** Send an image to GPT-4o vision API as a base64 data URL.

```python
async def analyze_image(image_bytes: bytes, context: str = "") -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    # litellm.acompletion with content=[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]
    # Ask model: type (table/chart/diagram/photo), description, key data extracted
    # Return: {"description": str, "type": str, "data": str, "page": int}
```

**Pitfalls:**
- Max image size: 20MB for GPT-4o. Resize large images with Pillow before encoding
- Request JSON output with `response_format={"type": "json_object"}`
- Wrap in `asyncio.gather` to analyze multiple images concurrently

### 2c — Audio Transcriber (`audio_transcriber.py`)

**What:** Send an audio file to the Whisper API.

```python
def transcribe(audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        # openai.audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json")
        # verbose_json includes word-level timestamps
    # Chunk transcript into 500-char segments with metadata
    # Return: {"text": str, "segments": [{"text": str, "start": float, "end": float}]}
```

---

## Module 3 — Graph RAG (`src/graph/`)

### 3a — Entity Extractor (`entity_extractor.py`)

**What:** Use spaCy to extract named entities; optionally enhance with LLM for
relations.

```python
ENTITY_LABELS = {"ORG", "PERSON", "LAW", "GPE", "DATE", "MONEY", "PRODUCT"}

def extract_entities(text: str) -> list[dict]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # doc.ents → filter by ENTITY_LABELS
    # Return: [{"text": str, "label": str, "start": int, "end": int}]

async def extract_relations(text: str, entities: list[dict]) -> list[dict]:
    # LLM prompt: "Find relations between these entities: ..."
    # Return: [{"subject": str, "predicate": str, "object": str}]
```

**Tip:** Cache the spaCy model — `spacy.load` is slow. Load once at module level.

### 3b — Neo4j Store (`neo4j_store.py`)

**What:** Load entities + relations into Neo4j; convert natural language to Cypher.

```python
def connect(uri: str, user: str, password: str) -> GraphDatabase.driver:
    # neo4j.GraphDatabase.driver(uri, auth=(user, password))

def load_document(driver, doc_id: str, entities: list, relations: list):
    # MERGE (d:Document {id: $doc_id})
    # For each entity: MERGE (e:Entity {name: $name, type: $type})
    # MERGE (d)-[:MENTIONS]->(e)
    # For each relation: MERGE (a)-[r:RELATION {type: $pred}]->(b)

async def nl_to_cypher(question: str, schema: str) -> str:
    # Prompt: "Graph schema: {schema}\nConvert to Cypher: {question}"
    # Strip ```cypher ... ``` code fences from response

def run_query(driver, cypher: str) -> list[dict]:
    # driver.session().run(cypher) → [dict(record) for record in result]
```

**Pitfalls:**
- Always use `MERGE` not `CREATE` — prevents duplicates on re-ingestion
- Escape special characters in entity names (apostrophes break Cypher)
- Add `LIMIT 25` to generated Cypher — unbounded queries can OOM Neo4j

---

## Module 4 — Retrieval (`src/retrieval/`)

### 4a — Vector Store (`vector_store.py`)

**What:** Three ChromaDB collections — one per modality — with upsert logic.

```python
COLLECTIONS = ["text_chunks", "image_contexts", "audio_segments"]

def setup_store(persist_dir: str = "./chroma_db") -> dict[str, chromadb.Collection]:
    client = chromadb.PersistentClient(path=persist_dir)
    # Create or get each collection
    # Return: {"text": col, "images": col, "audio": col}

def upsert_chunks(collection, chunks: list[dict], ids: list[str]):
    # collection.upsert(ids=ids, documents=[c["text"] for c in chunks],
    #                   metadatas=[{k: v for k,v in c.items() if k != "text"} for c in chunks])

def query(collection, query_text: str, n: int = 5) -> list[dict]:
    # collection.query(query_texts=[query_text], n_results=n)
    # Return: [{"text": str, "score": float, "metadata": dict}]
```

### 4b — Hybrid Retriever (`hybrid_retriever.py`)

**What:** Combine vector results from ChromaDB with graph results from Neo4j;
rerank with a cross-encoder.

```python
async def hybrid_search(
    question: str, collections: dict, driver,
    top_k: int = 5, include_graph: bool = True,
) -> dict[str, list]:
    # 1. Query all 3 collections in asyncio.gather
    # 2. If include_graph: generate Cypher from question, run on Neo4j
    # 3. Rerank: sort combined results by score (distance → 1-distance for ChromaDB)
    # Return: {"text": [...], "images": [...], "audio": [...], "graph": [...]}
```

**Tip:** ChromaDB returns L2 distances. Convert to similarity: `score = 1 / (1 + distance)`.

---

## Module 5 — Guardrails (`src/guardrails/`)

### Layer 1 — Injection Checker (`injection_checker.py`)

```python
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    # ... 6 more patterns
]

def check(text: str) -> tuple[bool, str]:
    # Returns (safe: bool, reason: str)
    # Compile patterns once at module level for O(1) per call
```

### Layer 2 — PII Scanner (`pii_scanner.py`)

```python
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn":   r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}

def scan_and_anonymize(text: str) -> tuple[str, list[str]]:
    # Replace each match with [TYPE_REDACTED]
    # Return (sanitized_text, list_of_found_types)
```

### Layer 3 — Safety Checker (`safety_checker.py`)

```python
async def check_safety(text: str) -> tuple[bool, str]:
    # Try: load LlamaGuard model (transformers) and call it
    # Fallback: GPT-4o-mini with structured safety prompt
    # Return: (is_safe: bool, reason: str)
```

### Layer 4 — Pipeline (`pipeline.py`)

```python
async def run_pipeline(text: str) -> GuardrailResult:
    # L1: injection_checker.check(text)        → sync
    # L2: pii_scanner.scan_and_anonymize(text) → sync (use sanitized text going forward)
    # L3 + L4: asyncio.gather(safety_checker, topic_relevance)  ← run in PARALLEL
    # Return GuardrailResult(safe, sanitized_text, issues, pii_found)
```

**Key insight:** L1 and L2 are sync and sequential (L2 sanitizes text for L3). L3 and L4 are both async and independent — run them in `asyncio.gather` to halve latency.

---

## Module 6 — Memory (`src/memory/mem0_store.py`)

```python
from mem0 import Memory

MEMORY_TYPES = ["episodic", "semantic", "procedural", "profile"]

def create_client() -> Memory:
    # Memory() for local; for production: Memory(config={"vector_store": {"provider": "qdrant"}})

def add_memory(client, user_id: str, messages: list[dict], memory_type: str) -> str:
    # client.add(messages=messages, user_id=user_id, metadata={"type": memory_type})
    # Handle mem0 v1 (returns list) and v2 (returns dict) API differences

def search(client, query: str, user_id: str, limit: int = 5) -> list[dict]:
    # client.search(query=query, user_id=user_id, limit=limit)
    # Normalize response format across v1/v2

def inject_into_prompt(memories: list[dict]) -> str:
    # Format as "\n[Memory] {text}" lines to append to system prompt
```

---

## Module 7 — Resilience (`src/resilience/`)

### Circuit Breaker (`circuit_breaker.py`)

```python
class CircuitState(Enum):
    CLOSED = "closed"       # normal — all requests pass
    OPEN = "open"           # tripped — fail fast
    HALF_OPEN = "half_open" # recovery probe

class CircuitBreaker:
    # CLOSED → OPEN when failures >= threshold
    # OPEN → HALF_OPEN when recovery_timeout expires
    # HALF_OPEN → CLOSED on first success; → OPEN on failure
```

### Fallback Chain (`fallback_chain.py`)

```python
class FallbackChain:
    def __init__(self, models: list[str]):
        # One CircuitBreaker per model

    async def call(self, messages: list[dict], **kwargs) -> tuple[str, str]:
        # For each model in order:
        #   if breaker.can_attempt(): try call, record success/failure
        # Raise RuntimeError if all fail
```

---

## Module 8 — Agent (`src/agents/multimodal_agent.py`)

This is the main orchestrator. Wire everything together:

```python
async def analyze(user_id: str, question: str, include_graph: bool = True) -> AnalyzeResponse:
    # 1. run_pipeline(question)           → guardrails
    # 2. hybrid_search(sanitized_q, ...) → context (text + images + audio + graph)
    # 3. search_memory(user_id, q)        → past memories
    # 4. build messages: system + memories + context + question
    # 5. fallback_chain.call(messages)    → answer, model_used
    # 6. pii_scanner.scan_and_anonymize(answer)  → clean output
    # 7. add_memory(user_id, [user_msg, assistant_msg], "episodic")
    # 8. cost_tracker.record(...)
    # Return AnalyzeResponse with all metadata
```

---

## Module 9 — API (`src/api/`)

### `app.py`

```python
def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise Multimodal Compliance Agent")
    app.add_middleware(RateLimitMiddleware, rpm=60)
    app.include_router(router)
    return app
```

### `routes.py`

Key routes to implement:
- `POST /analyze` — main agent call
- `POST /ingest/pdf` — `UploadFile` → `extract_text_chunks + extract_images → upsert`
- `POST /ingest/audio` — `UploadFile` → `transcribe → upsert`
- `GET /search` — hybrid search without LLM
- `GET /graph/query` — NL → Cypher
- `GET /memories/{user_id}`
- `GET /health` — circuit breaker states + Neo4j + ChromaDB ping

### `middleware.py`

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    # In-memory sliding window: {ip: [timestamps]}
    # 429 if len(recent) >= rpm
```

---

## Module 10 — Evaluation (`scripts/evaluate.py`)

Run three evaluation suites:

### Retrieval Eval
- 10 (question, expected_doc_ids) pairs
- Measure precision@5, recall@5, MRR using `hybrid_search`

### Guardrail Eval
- 5 known injection attempts → expect blocked
- 5 safe messages → expect allowed (false positive rate)
- 5 PII-containing texts → expect anonymized

### Agent Eval
- 20 (question, expected_answer_keywords) pairs
- Score: does answer contain expected keywords?
- Measure: accuracy, p50 latency, p95 latency, avg cost/query

---

## Implementation Order

Work through the modules in this order:

```
1. config.py + models.py          → foundation for everything else
2. guardrails/ (all 4 layers)     → test independently before agent
3. ingestion/ (pdf → vision → audio)
4. graph/ (entity → neo4j → cypher)
5. retrieval/ (vector → hybrid)
6. memory/mem0_store.py
7. resilience/ (circuit_breaker → fallback_chain)
8. agents/multimodal_agent.py     → wire 1-7 together
9. api/ (routes → middleware → app)
10. observability/ (logger → cost_tracker)
11. scripts/ingest.py
12. scripts/evaluate.py
13. tests/
```

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| spaCy model loaded inside a loop | Load once at module level (`_nlp = spacy.load(...)`) |
| Neo4j entity names with quotes | Escape: `name.replace("'", "\\'")` before MERGE |
| ChromaDB distance vs similarity | Convert: `score = 1 / (1 + distance)` |
| Mem0 v1 vs v2 API | Check `isinstance(result, list)` vs `dict` |
| Images analyzed sequentially | Use `asyncio.gather(*[analyze_image(b) for b in images])` |
| Circuit breaker not thread-safe | Use `asyncio.Lock` for state transitions |
| LlamaGuard needs GPU | Fall back to GPT-4o-mini safety prompt when GPU unavailable |
| PII in system prompt | Always sanitize user input before injecting into prompts |
| Unbounded graph queries | Append `LIMIT 25` to all generated Cypher |
| Cost not tracked for retries | Track cost inside `FallbackChain.call` after each attempt |
