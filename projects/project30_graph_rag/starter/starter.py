"""Project 30 — Graph RAG: Starter File
pip install spacy neo4j litellm chromadb sentence-transformers python-dotenv
python -m spacy download en_core_web_sm
docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
"""
from __future__ import annotations
import os, json, asyncio
from dataclasses import dataclass
import litellm
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Entity:
    name: str; entity_type: str; normalized: str

@dataclass
class Relationship:
    source: str; relation: str; target: str

# TODO 1: Extract entities + relationships from document text using spaCy
def extract_entities(text: str) -> tuple[list[Entity], list[Relationship]]:
    """TODO 1: spaCy NER + rule-based relationship extraction. Return (entities, relations)."""
    # import spacy; nlp = spacy.load("en_core_web_sm")
    raise NotImplementedError

# TODO 2: Connect to Neo4j and create schema constraints
def connect_neo4j(uri: str = "bolt://localhost:7687", auth=("neo4j", "password")):
    """TODO 2: Connect + create uniqueness constraints for Document and Entity nodes."""
    # from neo4j import GraphDatabase
    raise NotImplementedError

def get_graph_schema() -> str:
    """TODO 2 (cont): Return graph schema description for LLM context."""
    raise NotImplementedError

# TODO 3: Load document entities into Neo4j
def load_document(driver, doc_id: str, title: str, entities: list[Entity], relations: list[Relationship]):
    """TODO 3: MERGE Document, Entity nodes; create MENTIONS and relationship edges."""
    raise NotImplementedError

# TODO 4: LLM → Cypher translator
async def nl_to_cypher(question: str, schema: str) -> str:
    """TODO 4: Prompt GPT-4o-mini to generate Cypher for the question. Return Cypher string."""
    raise NotImplementedError

async def run_graph_query(driver, question: str, schema: str) -> list[dict]:
    """TODO 4 (cont): Translate to Cypher, run on Neo4j, handle exceptions. Return records."""
    raise NotImplementedError

# TODO 5: Hybrid graph + vector search
async def hybrid_search(driver, question: str, schema: str, vector_col=None) -> dict:
    """TODO 5: Query graph + vector (if available), synthesize answer. Return {"answer": str, ...}"""
    raise NotImplementedError

# TODO 6: Multi-hop Q&A benchmark
MULTI_HOP_QUESTIONS = [
    "Which subsidiaries of GlobalTech failed security audits?",
    "What regulations govern the contract signed by John Smith?",
    "Which vendors have both active contracts and failed audits?",
]

async def answer_multi_hop(driver, schema: str):
    """TODO 6: Run all MULTI_HOP_QUESTIONS, print results + compare vs vector-only."""
    raise NotImplementedError

# TODO 7: Cypher validation with retry
async def validated_cypher_query(driver, question: str, schema: str, max_retries: int = 3) -> list[dict]:
    """TODO 7: Generate Cypher, validate with EXPLAIN, retry with error message if invalid."""
    raise NotImplementedError

SAMPLE_DOCS = [
    {"id": "DOC-001", "title": "Acme Vendor Agreement",
     "text": "Agreement between Acme Corp and DataVendor Ltd (subsidiary of GlobalTech). "
             "Governed by GDPR Article 28, SOX. Signed by John Smith."},
    {"id": "DOC-002", "title": "GlobalTech Privacy Policy",
     "text": "GlobalTech Inc processes data under GDPR. DPO: Sarah Johnson."},
    {"id": "DOC-003", "title": "DataVendor Audit Report",
     "text": "DataVendor Ltd failed ISO 27001 audit. Auditor: PwC."},
]

async def main():
    print("=== Project 30: Graph RAG ===\n")
    driver = connect_neo4j()
    schema = get_graph_schema()
    for doc in SAMPLE_DOCS:
        entities, relations = extract_entities(doc["text"])
        load_document(driver, doc["id"], doc["title"], entities, relations)
        print(f"  {doc['id']}: {len(entities)} entities, {len(relations)} relations")
    await answer_multi_hop(driver, schema)
    driver.close()

if __name__ == "__main__":
    asyncio.run(main())
