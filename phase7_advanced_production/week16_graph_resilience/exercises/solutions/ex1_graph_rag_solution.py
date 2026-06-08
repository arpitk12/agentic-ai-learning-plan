"""
SOLUTION — Exercise 1: Graph RAG with Entity Extraction + Neo4j + LLM-Generated Cypher
Phase 7 / Week 16

How this solution works:
  TODO 1: spaCy's en_core_web_sm NER extracts ORG/PERSON/LAW/GPE/DATE/MONEY entities.
           Optional LLM enhancement adds relationship extraction (GOVERNS, OWNS, etc.)
  TODO 2: connect_neo4j() returns the driver; get_graph_schema() queries labels+rels.
  TODO 3: load_document_to_graph() creates Document node, merges Entity nodes,
           creates MENTIONS edges, and adds GOVERNED_BY/OWNED_BY/SIGNED_BY edges.
  TODO 4: nl_to_cypher() sends schema + question to GPT-4o-mini; returns Cypher string.
  TODO 5: run_graph_query() executes Cypher and returns records as list of dicts.
  TODO 6: hybrid_graph_rag() combines graph traversal + ChromaDB vector results.
  TODO 7: answer_multi_hop_questions() chains multiple Cypher queries.
  BONUS:  detect_communities() runs GDS Louvain algorithm for cluster detection.
"""
from __future__ import annotations
import os, json, asyncio, re
from dataclasses import dataclass, field
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Entity:
    name: str
    entity_type: str
    normalized: str

@dataclass
class Relationship:
    source: str
    relation: str
    target: str

@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    content: str
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


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


# ── TODO 1 SOLUTION: Entity Extraction ───────────────────────────────────────

def extract_entities_and_relations(
    text: str,
    doc_id: str,
    use_llm_enhancement: bool = False,
) -> tuple[list[Entity], list[Relationship]]:
    import spacy  # type: ignore
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    WANTED_TYPES = {"ORG", "PERSON", "GPE", "LAW", "DATE", "MONEY", "PRODUCT", "NORP"}

    entities: list[Entity] = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ in WANTED_TYPES and ent.text.strip() not in seen:
            entities.append(Entity(
                name=ent.text.strip(),
                entity_type=ent.label_,
                normalized=ent.text.strip().lower(),
            ))
            seen.add(ent.text.strip())

    # Basic relationship extraction: subsidiary-of, governed-by
    relationships: list[Relationship] = []
    subsidiary_pattern = re.compile(
        r"(\w[\w\s]+?)\s+\(a subsidiary of\s+([\w\s]+?)\)", re.IGNORECASE
    )
    for m in subsidiary_pattern.finditer(text):
        relationships.append(Relationship(
            source=m.group(1).strip(),
            relation="SUBSIDIARY_OF",
            target=m.group(2).strip(),
        ))

    governed_pattern = re.compile(r"governed by\s+([\w\s]+\d*)", re.IGNORECASE)
    for m in governed_pattern.finditer(text):
        relationships.append(Relationship(
            source=doc_id,
            relation="GOVERNED_BY",
            target=m.group(1).strip(),
        ))

    signed_pattern = re.compile(r"Signed by\s+([\w\s]+?)\s*\(", re.IGNORECASE)
    for m in signed_pattern.finditer(text):
        relationships.append(Relationship(
            source=doc_id,
            relation="SIGNED_BY",
            target=m.group(1).strip(),
        ))

    # LLM enhancement: extract more structured relationships
    if use_llm_enhancement:
        import asyncio as _asyncio

        async def llm_extract():
            resp = await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Extract relationships from this text as JSON array.
Each item: {{"source": "entity", "relation": "OWNS|GOVERNS|SIGNED|PROCESSES|AUDITED_BY", "target": "entity"}}
Text: {text}
Return only the JSON array, no explanation.""",
                }],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return data.get("relationships", data) if isinstance(data, dict) else data

        try:
            loop = _asyncio.get_event_loop()
            llm_rels = loop.run_until_complete(llm_extract())
            for r in llm_rels:
                if all(k in r for k in ("source", "relation", "target")):
                    relationships.append(Relationship(**r))
        except Exception:
            pass

    return entities, relationships


# ── TODO 2 SOLUTION: Neo4j connection + schema ────────────────────────────────

def connect_neo4j(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
):
    from neo4j import GraphDatabase  # type: ignore
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"  Connected to Neo4j at {uri}")
    return driver


def get_graph_schema(driver) -> str:
    with driver.session() as session:
        node_labels = [r["label"] for r in session.run("CALL db.labels()")]
        rel_types = [r["relationshipType"] for r in session.run("CALL db.relationshipTypes()")]
    schema = (
        f"Node labels: {', '.join(node_labels)}\n"
        f"Relationship types: {', '.join(rel_types)}\n"
        f"Example node properties: Document(doc_id, title, doc_type), "
        f"Entity(name, type), Regulation(name)"
    )
    return schema


# ── TODO 3 SOLUTION: Load documents into Neo4j ───────────────────────────────

def load_document_to_graph(driver, doc: Document) -> None:
    with driver.session() as session:
        # Create or merge the document node
        session.run(
            "MERGE (d:Document {doc_id: $id}) "
            "SET d.title=$title, d.doc_type=$doc_type",
            id=doc.doc_id, title=doc.title, doc_type=doc.doc_type,
        )

        # Create entity nodes and MENTIONS edges
        for ent in doc.entities:
            session.run(
                "MERGE (e:Entity {name: $name}) SET e.type=$type "
                "WITH e MATCH (d:Document {doc_id: $doc_id}) "
                "MERGE (d)-[:MENTIONS]->(e)",
                name=ent.name, type=ent.entity_type, doc_id=doc.doc_id,
            )

        # Create relationship edges
        for rel in doc.relationships:
            if rel.source == doc.doc_id:
                session.run(
                    f"MATCH (d:Document {{doc_id: $src}}) "
                    f"MERGE (t:Entity {{name: $tgt}}) "
                    f"MERGE (d)-[:{rel.relation}]->(t)",
                    src=rel.source, tgt=rel.target,
                )
            else:
                session.run(
                    f"MERGE (s:Entity {{name: $src}}) "
                    f"MERGE (t:Entity {{name: $tgt}}) "
                    f"MERGE (s)-[:{rel.relation}]->(t)",
                    src=rel.source, tgt=rel.target,
                )

    print(f"  Loaded {doc.doc_id}: {len(doc.entities)} entities, {len(doc.relationships)} relationships")


# ── TODO 4 SOLUTION: NL to Cypher ────────────────────────────────────────────

async def nl_to_cypher(question: str, schema: str) -> str:
    prompt = f"""You are a Neo4j Cypher expert.

Graph schema:
{schema}

Question: {question}

Write a single Cypher query to answer this question.
- Use MATCH, WHERE, RETURN
- Return meaningful labels for each column
- For multi-hop: chain MATCH clauses
- Return ONLY the Cypher query, no explanation, no markdown."""

    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    cypher = resp.choices[0].message.content.strip()
    # Strip markdown code blocks if present
    cypher = re.sub(r"```(?:cypher)?", "", cypher).strip()
    return cypher


# ── TODO 5 SOLUTION: Run graph query ─────────────────────────────────────────

def run_graph_query(driver, cypher: str) -> list[dict]:
    try:
        with driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]
    except Exception as e:
        print(f"  Cypher error: {e}")
        return []


# ── TODO 6 SOLUTION: Hybrid graph+vector search ───────────────────────────────

async def hybrid_graph_rag(
    driver,
    question: str,
    schema: str,
    vector_col=None,   # Optional ChromaDB collection
    n_vector: int = 3,
) -> str:
    # Step 1: Generate and run Cypher query
    cypher = await nl_to_cypher(question, schema)
    graph_records = run_graph_query(driver, cypher)

    # Step 2: Vector search (if collection provided)
    vector_context = ""
    if vector_col is not None:
        results = vector_col.query(query_texts=[question], n_results=n_vector)
        vector_docs = results.get("documents", [[]])[0]
        vector_context = "\n".join(vector_docs)

    # Step 3: Synthesise answer from both sources
    context_parts = []
    if graph_records:
        context_parts.append(f"Graph query results:\n{json.dumps(graph_records, indent=2)}")
    if vector_context:
        context_parts.append(f"Vector search results:\n{vector_context}")

    if not context_parts:
        return "No relevant information found in the knowledge graph."

    synthesis_prompt = f"""Question: {question}

{chr(10).join(context_parts)}

Answer the question based on the above information. Be specific and cite which documents or entities you're referencing."""

    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a compliance analyst with access to a knowledge graph."},
            {"role": "user", "content": synthesis_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── TODO 7 SOLUTION: Multi-hop questions ─────────────────────────────────────

async def answer_multi_hop_questions(driver, schema: str) -> None:
    MULTI_HOP_QUESTIONS = [
        "What regulations govern the vendor that is a subsidiary of GlobalTech Inc?",
        "Which documents mention entities that failed a security audit?",
        "What is the parent company of the vendor in the Acme Corp agreement?",
    ]

    print("\n  Multi-hop questions:")
    for q in MULTI_HOP_QUESTIONS:
        print(f"\n  Q: {q}")
        answer = await hybrid_graph_rag(driver, q, schema)
        print(f"  A: {answer[:200]}...")


# ── BONUS: Community Detection ────────────────────────────────────────────────

def detect_communities(driver) -> list[dict]:
    """Run Louvain community detection using Neo4j Graph Data Science (GDS)."""
    try:
        with driver.session() as session:
            # Project the graph for GDS
            session.run("""
                CALL gds.graph.project(
                  'compliance_graph',
                  ['Document', 'Entity'],
                  {MENTIONS: {orientation: 'UNDIRECTED'}}
                )
            """)
            # Run Louvain
            result = session.run("""
                CALL gds.louvain.stream('compliance_graph')
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).doc_id AS doc,
                       gds.util.asNode(nodeId).name AS entity,
                       communityId
                ORDER BY communityId
            """)
            communities = [dict(r) for r in result]
            # Clean up projected graph
            session.run("CALL gds.graph.drop('compliance_graph')")
            return communities
    except Exception as e:
        print(f"  GDS not available for community detection: {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Graph RAG — SOLUTION ===\n")
    print("PREREQUISITE: Neo4j running locally")
    print("  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5\n")

    try:
        driver = connect_neo4j()
    except Exception as e:
        print(f"Neo4j not available: {e}")
        print("Showing entity extraction only (Neo4j connection skipped)\n")
        # Demo entity extraction without Neo4j
        for raw in SAMPLE_DOCUMENTS:
            print(f"Document: {raw['doc_id']}")
            entities, rels = extract_entities_and_relations(raw["content"], raw["doc_id"])
            print(f"  Entities ({len(entities)}): {[e.name for e in entities[:5]]}")
            print(f"  Relationships ({len(rels)}): {[(r.source[:15], r.relation, r.target[:15]) for r in rels[:3]]}")
        return

    print("1. Extracting entities and loading to Neo4j...")
    for raw in SAMPLE_DOCUMENTS:
        entities, rels = extract_entities_and_relations(raw["content"], raw["doc_id"])
        doc = Document(
            doc_id=raw["doc_id"], title=raw["title"],
            doc_type=raw["doc_type"], content=raw["content"],
            entities=entities, relationships=rels,
        )
        load_document_to_graph(driver, doc)

    schema = get_graph_schema(driver)
    print(f"\n2. Graph schema:\n  {schema}\n")

    print("3. Answering multi-hop questions...")
    await answer_multi_hop_questions(driver, schema)

    print("\n4. Hybrid Graph RAG query...")
    q = "Which vendors have compliance gaps and what regulations govern them?"
    answer = await hybrid_graph_rag(driver, q, schema)
    print(f"  Q: {q}")
    print(f"  A: {answer}\n")

    driver.close()
    print("Neo4j connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
