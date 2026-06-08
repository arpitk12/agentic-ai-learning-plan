"""
Exercise 1: Graph RAG with Entity Extraction + Neo4j + LLM-Generated Cypher
Phase 7 / Week 16 — Graph RAG · Resilience · A2A · Multi-Tenancy · Reasoning

Goal: Build a Graph RAG pipeline that extracts entities from documents, builds
      a Neo4j knowledge graph, and uses LLM-generated Cypher queries to answer
      multi-hop questions that vector RAG cannot answer.

Stack: spacy · neo4j · litellm · pydantic

pip install spacy neo4j litellm pydantic python-dotenv
python -m spacy download en_core_web_sm

# Neo4j local (Docker):
# docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

TODOs:
  1. Extract named entities and relationships from document text using spaCy
  2. Define graph schema (nodes: Document, Entity, Regulation; edges: MENTIONS, GOVERNS)
  3. Load documents + entities into Neo4j
  4. Build an LLM-to-Cypher translator
  5. Build a hybrid search: combine Neo4j graph + ChromaDB vector results
  6. Answer multi-hop questions that require graph traversal
  7. BONUS: Add community detection to group related documents
"""
from __future__ import annotations
import os, json, asyncio, re
from dataclasses import dataclass
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    name: str
    entity_type: str   # ORG, PERSON, LAW, GPE, DATE, MONEY, PRODUCT
    normalized: str    # lowercase, stripped

@dataclass
class Relationship:
    source: str        # entity name
    relation: str      # MENTIONS, GOVERNED_BY, OWNED_BY, SIGNED_BY, VALUED_AT
    target: str

@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    content: str
    entities: list[Entity]
    relationships: list[Relationship]

# ── Sample Documents ──────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "doc_id": "DOC-001",
        "title": "Acme Corp Vendor Agreement 2026",
        "doc_type": "contract",
        "content": "This agreement between Acme Corp and DataVendor Ltd (a subsidiary of GlobalTech Inc) "
                   "establishes data processing terms. Payment: $750,000 annually. "
                   "Governed by GDPR Article 28, SOX Section 404. "
                   "Signed by John Smith (Acme) and Maria Chen (DataVendor). "
                   "Data residency: European Union only.",
    },
    {
        "doc_id": "DOC-002",
        "title": "GlobalTech Inc Privacy Policy 2025",
        "doc_type": "policy",
        "content": "GlobalTech Inc processes personal data under GDPR and CCPA. "
                   "Data retention: 7 years per SOX requirements. "
                   "DPO: Sarah Johnson. Approved by EU Data Protection Authority.",
    },
    {
        "doc_id": "DOC-003",
        "title": "DataVendor Ltd Security Audit Q4 2025",
        "doc_type": "report",
        "content": "DataVendor Ltd failed ISO 27001 audit on November 15 2025. "
                   "Critical finding: unencrypted PII storage. "
                   "Remediation deadline: March 2026. "
                   "Auditor: PwC. Regulatory notification sent to UK ICO.",
    },
]

# ── TODO 1: Entity Extraction ─────────────────────────────────────────────────

def extract_entities_and_relations(
    text: str,
    doc_id: str,
    use_llm_enhancement: bool = False,
) -> tuple[list[Entity], list[Relationship]]:
    """
    TODO 1: Extract entities and relationships from document text.

    Step A — spaCy NER:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [
        Entity(name=ent.text, entity_type=ent.label_,
               normalized=ent.text.lower().strip())
        for ent in doc.ents
        if ent.label_ in {"ORG", "PERSON", "GPE", "LAW", "DATE", "MONEY", "PRODUCT"}
    ]

    Step B — Rule-based relationships from entity co-occurrence:
    - If "GDPR" or "SOX" appears + an ORG in same sentence → (ORG, GOVERNED_BY, regulation)
    - If a PERSON + ORG in same sentence with "signed" → (PERSON, SIGNED_FOR, ORG)
    - If ORG + "subsidiary of" + ORG → (child, SUBSIDIARY_OF, parent)
    - If ORG + MONEY in same sentence → (ORG, VALUED_AT, MONEY)

    Step C (optional, use_llm_enhancement=True):
    Use litellm to extract additional relationships in JSON format.
    Prompt: "Extract all relationships between entities in this text as JSON:
             [{"source": "...", "relation": "...", "target": "..."}]"

    Return (entities, relationships).
    """
    # TODO 1: implement here
    raise NotImplementedError

# ── TODO 2: Define Graph Schema + Connect to Neo4j ────────────────────────────

def connect_neo4j(uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "password")):
    """
    TODO 2a: Connect to Neo4j and create schema constraints.

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=auth)

    Create uniqueness constraints:
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE")

    Return the driver.
    """
    # TODO 2a: implement here
    raise NotImplementedError

def get_graph_schema() -> str:
    """
    TODO 2b: Return the graph schema as a string for LLM context.

    Return a clear text description of:
    - Node types: Document(id, title, type, content), Entity(name, type, normalized)
    - Edge types: (Document)-[:MENTIONS]->(Entity)
                  (Entity)-[:GOVERNED_BY]->(Entity)
                  (Entity)-[:SUBSIDIARY_OF]->(Entity)
                  (Entity)-[:SIGNED_FOR]->(Entity)
    - Example Cypher queries for each relationship type

    This schema string will be injected into the LLM prompt for Cypher generation.
    """
    # TODO 2b: implement here
    raise NotImplementedError

# ── TODO 3: Load Documents into Neo4j ─────────────────────────────────────────

def load_document_to_graph(driver, doc: Document) -> None:
    """
    TODO 3: Load a document and its entities into Neo4j.

    With driver.session() as session:
    a) Create Document node:
       MERGE (d:Document {id: $id})
       SET d.title=$title, d.type=$type, d.content=$content[:500]

    b) For each entity: MERGE (e:Entity {name: $name, type: $type})
                        SET e.normalized = $normalized

    c) For each entity: create MENTIONS relationship:
       MATCH (d:Document {id: $doc_id}), (e:Entity {name: $name, type: $type})
       MERGE (d)-[:MENTIONS]->(e)

    d) For each relationship in doc.relationships:
       MATCH (s:Entity {name: $source}), (t:Entity {name: $target})
       MERGE (s)-[r:RELATED {type: $relation}]->(t)
       (or use specific rel types: MERGE (s)-[:GOVERNED_BY]->(t) etc.)
    """
    # TODO 3: implement here
    raise NotImplementedError

def load_all_documents(driver, documents: list[Document]) -> int:
    """Load all documents and return count loaded."""
    for doc in documents:
        load_document_to_graph(driver, doc)
    return len(documents)

# ── TODO 4: LLM-to-Cypher Translator ─────────────────────────────────────────

async def nl_to_cypher(question: str, schema: str) -> str:
    """
    TODO 4: Use an LLM to convert a natural language question to a Cypher query.

    Prompt:
    f\"\"\"You are a Neo4j expert. Given the graph schema and a question,
    write a Cypher query to answer it.

    Schema:
    {schema}

    Question: {question}

    Rules:
    - Return ONLY the Cypher query, no explanation
    - Use MATCH and RETURN, not CREATE or DELETE
    - Limit results to 10: add LIMIT 10 at the end
    - If you can't answer with the schema, return: MATCH (n) RETURN "No data" LIMIT 1
    \"\"\"

    Call litellm.acompletion with gpt-4o-mini.
    Clean the result: strip ```cypher ... ``` code fences.
    Return the raw Cypher string.
    """
    # TODO 4: implement here
    raise NotImplementedError

async def run_graph_query(driver, question: str, schema: str) -> list[dict]:
    """
    TODO 4 (continued): Translate question to Cypher and run it.

    a) cypher = await nl_to_cypher(question, schema)
    b) with driver.session() as session:
           results = session.run(cypher)
           records = [dict(r) for r in results]
    c) Print the Cypher query and first 3 results for debugging.
    d) Return records (list of dicts).

    Handle exceptions: if Cypher is invalid, return [{"error": str(e)}].
    """
    # TODO 4: implement here
    raise NotImplementedError

# ── TODO 5: Hybrid Search (Graph + Vector) ────────────────────────────────────

async def hybrid_graph_rag(
    driver,
    question: str,
    schema: str,
    text_collection=None,  # ChromaDB collection (optional)
    n_vector_results: int = 3,
) -> dict:
    """
    TODO 5: Combine graph traversal + vector similarity results.

    Step A — Graph search:
    graph_records = await run_graph_query(driver, question, schema)

    Step B — Vector search (if text_collection provided):
    vector_hits = text_collection.query(query_texts=[question], n_results=n_vector_results)
    vector_docs = vector_hits["documents"][0] if vector_hits else []

    Step C — Synthesize with LLM:
    context = f\"\"\"
    Graph database results:
    {json.dumps(graph_records[:5], indent=2)}

    Vector search results:
    {chr(10).join(vector_docs)}
    \"\"\"
    Use litellm to answer the question given the combined context.

    Return:
    {
        "question": question,
        "answer": str,
        "graph_records": graph_records,
        "vector_hits": vector_docs,
        "cypher_used": str,
    }
    """
    # TODO 5: implement here
    raise NotImplementedError

# ── TODO 6: Multi-Hop Questions ───────────────────────────────────────────────

MULTI_HOP_QUESTIONS = [
    # These require traversing relationships — vector RAG cannot answer them
    "Which organizations that are subsidiaries of GlobalTech Inc have compliance issues?",
    "What regulations govern the data processing agreement involving John Smith?",
    "Which vendors failed security audits and also have active contracts with Acme Corp?",
    "What is the total contract value for vendors governed by GDPR Article 28?",
    "Which DPO is responsible for the organization that failed the ISO 27001 audit?",
]

async def answer_multi_hop_questions(driver, schema: str) -> None:
    """
    TODO 6: Run all MULTI_HOP_QUESTIONS through hybrid_graph_rag and print results.

    For each question:
    a) Run hybrid_graph_rag(driver, question, schema)
    b) Print:
       Q: {question}
       Graph records: {len(result['graph_records'])} results
       A: {result['answer'][:200]}
       ---

    At the end, show which questions returned useful graph results vs empty.
    """
    # TODO 6: implement here
    raise NotImplementedError

# ── TODO 7 (BONUS): Community Detection ───────────────────────────────────────

def detect_communities(driver) -> list[list[str]]:
    """
    TODO 7: Use Neo4j Graph Data Science (GDS) to find communities of related documents.

    Requires: neo4j-graph-data-science plugin (available in Neo4j Desktop / AuraDB)

    a) Project a graph: all Document nodes connected via shared Entity nodes
       session.run(\"\"\"
           CALL gds.graph.project('doc-entity-graph',
               ['Document', 'Entity'],
               {MENTIONS: {orientation: 'UNDIRECTED'}})
       \"\"\")

    b) Run Louvain community detection:
       result = session.run(\"\"\"
           CALL gds.louvain.stream('doc-entity-graph', {nodeLabels: ['Document']})
           YIELD nodeId, communityId
           RETURN gds.util.asNode(nodeId).id AS doc_id, communityId
           ORDER BY communityId
       \"\"\")

    c) Group doc_ids by communityId and return list of lists.

    d) Print: "Community 1: [DOC-001, DOC-003] (2 docs share entities)"

    If GDS is not available, implement a manual approach using NetworkX:
    import networkx as nx
    Build a graph from entity co-occurrence and use nx.community.louvain_communities()
    """
    # TODO 7: implement here
    raise NotImplementedError

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Graph RAG Exercise ===\n")

    # Step 1: Extract entities from sample documents
    print("1. Extracting entities and relationships...")
    documents = []
    for raw in SAMPLE_DOCUMENTS:
        entities, relations = extract_entities_and_relations(
            raw["content"], raw["doc_id"]
        )
        documents.append(Document(
            doc_id=raw["doc_id"],
            title=raw["title"],
            doc_type=raw["doc_type"],
            content=raw["content"],
            entities=entities,
            relationships=relations,
        ))
        print(f"   {raw['doc_id']}: {len(entities)} entities, {len(relations)} relationships")

    # Step 2-3: Load into Neo4j
    print("\n2. Loading into Neo4j graph...")
    driver = connect_neo4j()
    n = load_all_documents(driver, documents)
    print(f"   Loaded {n} documents into graph")

    schema = get_graph_schema()

    # Steps 4-6: Multi-hop Q&A
    print("\n3. Answering multi-hop questions (require graph traversal):")
    await answer_multi_hop_questions(driver, schema)

    # Step 7: Communities
    print("4. Community detection:")
    communities = detect_communities(driver)
    for i, community in enumerate(communities):
        print(f"   Community {i+1}: {community}")

    driver.close()
    print("\n✅ Graph RAG exercise complete!")

if __name__ == "__main__":
    asyncio.run(main())
