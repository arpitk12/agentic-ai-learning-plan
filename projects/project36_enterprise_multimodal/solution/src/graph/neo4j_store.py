"""
solution/src/graph/neo4j_store.py — Full implementation.
"""
from __future__ import annotations
import asyncio
import json
import litellm  # type: ignore


def connect(uri: str, user: str, password: str):
    from neo4j import GraphDatabase  # type: ignore
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def load_document(driver, doc_id: str, source: str,
                  entities: list[dict], relations: list[dict]) -> dict:
    def _tx(tx, doc_id, source, entities, relations):
        # Merge document node
        tx.run("MERGE (d:Document {id: $id}) SET d.source = $source",
               id=doc_id, source=source)
        # Merge entities + MENTIONS edges
        for ent in entities:
            name = ent["text"].replace("'", "\\'")
            tx.run(
                "MERGE (e:Entity {name: $name, type: $type})\n"
                "WITH e\n"
                "MATCH (d:Document {id: $doc_id})\n"
                "MERGE (d)-[:MENTIONS]->(e)",
                name=ent["text"], type=ent["label"], doc_id=doc_id,
            )
        # Merge relations
        for rel in relations:
            tx.run(
                "MATCH (a:Entity {name: $subj}), (b:Entity {name: $obj})\n"
                "MERGE (a)-[:RELATION {type: $pred}]->(b)",
                subj=rel["subject"], obj=rel["object"], pred=rel["predicate"],
            )

    with driver.session() as session:
        session.execute_write(_tx, doc_id, source, entities, relations)
    return {"entities_loaded": len(entities), "relations_loaded": len(relations)}


def get_schema(driver) -> str:
    try:
        with driver.session() as session:
            labels = [r["label"] for r in session.run("CALL db.labels() YIELD label")]
            rel_types = [r["relationshipType"]
                         for r in session.run("CALL db.relationshipTypes() YIELD relationshipType")]
        return (f"Node labels: {', '.join(labels)}\n"
                f"Relationship types: {', '.join(rel_types)} (RELATION has 'type' property)")
    except Exception:
        return "Node labels: Document, Entity\nRelationship types: MENTIONS, RELATION"


async def nl_to_cypher(
    question: str,
    schema: str,
    model: str = "openai/gpt-4o-mini",
) -> str:
    resp = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content":
            f"Graph schema:\n{schema}\n\n"
            f"Convert to Cypher (Neo4j): {question}\n\n"
            "Rules: always add LIMIT 25, use MATCH not MERGE, return only Cypher."}],
        temperature=0.0,
    )
    cypher = resp.choices[0].message.content.strip()
    # Strip code fences
    for fence in ("```cypher", "```"):
        if cypher.startswith(fence):
            cypher = cypher[len(fence):]
    if cypher.endswith("```"):
        cypher = cypher[:-3]
    return cypher.strip()


def run_query(driver, cypher: str) -> list[dict]:
    try:
        with driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]
    except Exception as e:
        print(f"[neo4j] Cypher error: {e}")
        return []
