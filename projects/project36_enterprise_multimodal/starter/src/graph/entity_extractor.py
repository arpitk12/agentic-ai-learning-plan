"""
src/graph/entity_extractor.py
Extract named entities and relations from text using spaCy + optional LLM enhancement.

TODOs:
  1. implement extract_entities() — spaCy NER, filter to compliance-relevant labels
  2. implement extract_relations() — LLM prompt to find subject-predicate-object triples
  3. implement extract_all() — combine entities + relations from a list of text chunks
"""
from __future__ import annotations
import asyncio

# Labels relevant for compliance documents
ENTITY_LABELS = {"ORG", "PERSON", "LAW", "GPE", "DATE", "MONEY", "PRODUCT", "NORP"}

# Load spaCy model once at module level (slow to load — don't reload per call)
# TODO: uncomment after installing: python -m spacy download en_core_web_sm
# import spacy
# _NLP = spacy.load("en_core_web_sm")


# ── TODO 1: Extract entities with spaCy ──────────────────────────────────────
def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from text using spaCy.

    Steps:
      1a. doc = _NLP(text)
      1b. Filter doc.ents to those with .label_ in ENTITY_LABELS
      1c. Return [{"text": ent.text, "label": ent.label_, "start": ent.start_char,
                   "end": ent.end_char}]
      1d. Deduplicate by (text.lower(), label) — same entity can appear many times

    Returns:
        list[dict] — unique entities found in text
    """
    # doc = _NLP(text)
    # seen = set()
    # entities = []
    # for ent in doc.ents:
    #     if ent.label_ in ENTITY_LABELS:
    #         key = (ent.text.lower(), ent.label_)
    #         if key not in seen:
    #             seen.add(key)
    #             entities.append({"text": ent.text, "label": ent.label_, ...})
    raise NotImplementedError


# ── TODO 2: Extract relations with LLM ────────────────────────────────────────
async def extract_relations(
    text: str,
    entities: list[dict],
    model: str = "openai/gpt-4o-mini",
) -> list[dict]:
    """
    Use an LLM to find subject-predicate-object triples between the entities.

    Steps:
      2a. Build entity list string from entities
      2b. Prompt: "Given these entities: {entity_names}\n
                   Find relations in this text: {text}\n
                   Return JSON array: [{"subject": str, "predicate": str, "object": str}]\n
                   Predicates should be: GOVERNS, SIGNED_BY, SUBSIDIARY_OF, PARTY_TO,
                   APPLIES_TO, REGULATES, OWNED_BY, LOCATED_IN"
      2c. litellm.acompletion with response_format={"type":"json_object"}
      2d. Parse and validate — ensure subject + object appear in entity texts
      2e. Return [] if LLM call fails (don't block ingestion on relation extraction)

    Returns:
        list[dict] — [{"subject": str, "predicate": str, "object": str}]
    """
    # import litellm, json
    # entity_names = [e["text"] for e in entities]
    # ...
    raise NotImplementedError


# ── TODO 3: Extract from multiple chunks ─────────────────────────────────────
async def extract_all(
    chunks: list[dict],
    use_llm_relations: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Run entity + relation extraction across all text chunks.

    Steps:
      3a. entity extraction is sync → run per chunk, deduplicate across chunks
      3b. If use_llm_relations: run extract_relations on each chunk concurrently
          with asyncio.gather (each chunk independently)
      3c. Deduplicate relations by (subject.lower(), predicate, object.lower())
      3d. Return (all_entities, all_relations)

    Returns:
        tuple[list[dict], list[dict]] — (entities, relations)
    """
    # all_entities: dict[tuple, dict] = {}  # key=(text.lower(), label)
    # all_relations: dict[tuple, dict] = {}
    # ...
    raise NotImplementedError
