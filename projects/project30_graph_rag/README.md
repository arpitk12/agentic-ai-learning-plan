# Project 30 — Graph RAG (Entity Extraction + Neo4j + LLM Cypher)

> **Stack**: spaCy · Neo4j · Microsoft GraphRAG · LiteLLM · ChromaDB  
> **Phase 7 — Advanced Production** | Priority: P1 🟠

---

## What You'll Build

A knowledge graph-powered RAG system that answers multi-hop questions requiring relationship traversal — queries that vector RAG fundamentally cannot answer.

**Multi-hop question vector RAG cannot answer**:
> "Which vendors in our system are subsidiaries of GlobalTech, have active contracts, and failed a security audit in the past year?"

This requires traversing: `vendor → SUBSIDIARY_OF → GlobalTech`, `vendor → HAS_CONTRACT → active`, `vendor → FAILED_AUDIT → 2025` — three hops across the graph.

---

## Milestones

### Milestone 1 — Entity Extraction Pipeline
Use spaCy to extract named entities (ORG, PERSON, LAW, GPE, DATE, MONEY) from documents. Add rule-based relationship extraction (co-occurrence in same sentence with relation keywords).

### Milestone 2 — Neo4j Graph Schema + Load
Define schema: `Document`, `Entity`, `Regulation` nodes; `MENTIONS`, `GOVERNED_BY`, `SUBSIDIARY_OF`, `SIGNED_BY` edges. Create uniqueness constraints. Load 10 sample documents.

### Milestone 3 — LLM-to-Cypher Translator
Build a function that takes a natural language question + graph schema description → Cypher query using GPT-4o-mini. Run the Cypher and return records. Handle invalid Cypher gracefully.

### Milestone 4 — Hybrid Retrieval
Combine Neo4j graph results (structured, relational) + ChromaDB vector results (semantic) in a single retrieval call. Merge and de-duplicate. Synthesize final answer with both as context.

### Milestone 5 — Microsoft GraphRAG Integration
Run `graphrag index` on your document corpus. Compare: manual graph (Milestone 3) vs GraphRAG community summaries for global questions ("What are the main compliance themes?").

### Milestone 6 — Multi-Hop Q&A Benchmark
Create 15 questions (5 single-hop, 5 two-hop, 5 three-hop). Run through: vector-only RAG, graph-only, hybrid. Compare accuracy. Expected: graph RAG should win on 2+ hop questions.

### Milestone 7 — Cypher Quality Validation
Add a Cypher validator: before executing, check with `EXPLAIN` (dry run). If invalid, re-ask the LLM with the error message. Max 3 retries.

---

## Setup

```bash
pip install spacy neo4j graphrag chromadb sentence-transformers litellm python-dotenv networkx
python -m spacy download en_core_web_sm

# Neo4j local:
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  neo4j:5
```

---

## Expected Output

```
=== Graph RAG Results ===

Documents loaded: 20 | Entities: 143 | Relationships: 89

Multi-hop questions:
  Q: "Which subsidiaries of GlobalTech failed audits?"
  Cypher: MATCH (v:Entity)-[:SUBSIDIARY_OF]->(g:Entity {name:"GlobalTech"})
          -[:MENTIONED_IN]->(d:Document {type:"audit"})...
  Graph records: 2 entities found
  Answer: "DataVendor Ltd and CloudOps Inc (both GlobalTech subsidiaries)
            failed ISO 27001 audits in 2025..."
  ✅ Correct (vector RAG returned: "No relevant documents")

Accuracy comparison (15 questions):
  Vector-only:  5/5 single-hop ✅ | 1/5 two-hop ❌ | 0/5 three-hop ❌  = 6/15 (40%)
  Graph+Hybrid: 5/5 single-hop ✅ | 4/5 two-hop ✅ | 3/5 three-hop ✅  = 12/15 (80%)
```
