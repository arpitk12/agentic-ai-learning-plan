"""
solution/src/graph/entity_extractor.py — Full implementation.
"""
from __future__ import annotations
import asyncio
import json
import litellm  # type: ignore

ENTITY_LABELS = {"ORG", "PERSON", "LAW", "GPE", "DATE", "MONEY", "PRODUCT", "NORP"}

try:
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


def extract_entities(text: str) -> list[dict]:
    if _NLP is None:
        return []
    doc = _NLP(text)
    seen: set[tuple] = set()
    entities: list[dict] = []
    for ent in doc.ents:
        if ent.label_ not in ENTITY_LABELS:
            continue
        key = (ent.text.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


async def extract_relations(
    text: str,
    entities: list[dict],
    model: str = "openai/gpt-4o-mini",
) -> list[dict]:
    if not entities:
        return []
    entity_names = [e["text"] for e in entities]
    prompt = (
        f"Entities: {entity_names}\n\n"
        f"Text: {text[:1500]}\n\n"
        "Find relations between the entities. Use predicates: "
        "GOVERNS, SIGNED_BY, SUBSIDIARY_OF, PARTY_TO, APPLIES_TO, REGULATES, OWNED_BY, LOCATED_IN.\n"
        'Return JSON: {"relations": [{"subject": str, "predicate": str, "object": str}]}'
    )
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        relations = data.get("relations", [])
        # Validate: subject and object must be in known entities
        entity_set = {e["text"].lower() for e in entities}
        return [r for r in relations
                if r.get("subject", "").lower() in entity_set
                and r.get("object", "").lower() in entity_set]
    except Exception:
        return []


async def extract_all(
    chunks: list[dict],
    use_llm_relations: bool = True,
) -> tuple[list[dict], list[dict]]:
    all_entities: dict[tuple, dict] = {}
    all_relations: dict[tuple, dict] = {}

    # Entity extraction (sync, per chunk)
    chunk_entity_map: list[list[dict]] = []
    for chunk in chunks:
        ents = extract_entities(chunk["text"])
        for e in ents:
            key = (e["text"].lower(), e["label"])
            all_entities[key] = e
        chunk_entity_map.append(ents)

    if use_llm_relations:
        # Relation extraction (async, all chunks in parallel)
        tasks = [
            extract_relations(chunks[i]["text"], chunk_entity_map[i])
            for i in range(len(chunks))
        ]
        all_rels_lists = await asyncio.gather(*tasks)
        for rels in all_rels_lists:
            for r in rels:
                key = (r["subject"].lower(), r["predicate"], r["object"].lower())
                all_relations[key] = r

    return list(all_entities.values()), list(all_relations.values())
