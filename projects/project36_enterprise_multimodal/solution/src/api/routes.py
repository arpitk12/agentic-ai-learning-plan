"""
solution/src/api/routes.py — Full implementation.
"""
from __future__ import annotations
import tempfile, os, uuid, asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Request  # type: ignore
from pydantic import BaseModel  # type: ignore

router = APIRouter()


class AnalyzeRequest(BaseModel):
    user_id: str
    question: str
    include_graph: bool = True
    top_k: int = 5


@router.post("/analyze")
async def analyze_endpoint(req: AnalyzeRequest, request: Request):
    from src.agents.multimodal_agent import analyze  # type: ignore
    deps = request.app.state.deps
    result = await analyze(req.user_id, req.question, deps, req.include_graph, req.top_k)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), request: Request = None):
    from src.ingestion.pdf_extractor import extract_text_chunks, extract_images  # type: ignore
    from src.ingestion.vision_analyzer import analyze_images_batch  # type: ignore
    from src.graph.entity_extractor import extract_all  # type: ignore
    from src.retrieval.vector_store import upsert_text_chunks, upsert_image_contexts  # type: ignore
    from src.graph.neo4j_store import load_document  # type: ignore

    deps = request.app.state.deps
    doc_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        chunks = extract_text_chunks(tmp_path)
        raw_images = extract_images(tmp_path)
        annotated_images = await analyze_images_batch(raw_images) if raw_images else []
        entities, relations = await extract_all(chunks)
        n_chunks = upsert_text_chunks(deps.collections["text"], chunks, doc_id)
        n_images = upsert_image_contexts(deps.collections["images"], annotated_images, doc_id)
        if deps.neo4j_driver:
            load_document(deps.neo4j_driver, doc_id, file.filename or tmp_path, entities, relations)
    finally:
        os.unlink(tmp_path)

    return {
        "doc_id": doc_id,
        "source": file.filename,
        "chunks": n_chunks,
        "images": n_images,
        "entities": len(entities),
        "audio_segments": 0,
    }


@router.post("/ingest/audio")
async def ingest_audio(file: UploadFile = File(...), request: Request = None):
    from src.ingestion.audio_transcriber import transcribe, chunk_transcript  # type: ignore
    from src.retrieval.vector_store import upsert_audio_segments  # type: ignore

    deps = request.app.state.deps
    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "audio.mp3")[1]

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        transcript = transcribe(tmp_path)
        segments = chunk_transcript(transcript)
        n = upsert_audio_segments(deps.collections["audio"], segments, doc_id)
    finally:
        os.unlink(tmp_path)

    return {"doc_id": doc_id, "source": file.filename, "audio_segments": n,
            "chunks": 0, "images": 0, "entities": 0}


@router.get("/search")
async def search_endpoint(q: str = Query(...), top_k: int = 5,
                           include_graph: bool = True, request: Request = None):
    from src.retrieval.hybrid_retriever import hybrid_search  # type: ignore
    deps = request.app.state.deps
    return await hybrid_search(q, deps.collections, deps.neo4j_driver,
                                deps.neo4j_schema, top_k, include_graph)


@router.get("/graph/query")
async def graph_query(q: str = Query(...), request: Request = None):
    from src.graph.neo4j_store import nl_to_cypher, run_query  # type: ignore
    deps = request.app.state.deps
    if not deps.neo4j_driver:
        raise HTTPException(503, "Neo4j not available")
    cypher = await nl_to_cypher(q, deps.neo4j_schema)
    rows = run_query(deps.neo4j_driver, cypher)
    return {"cypher": cypher, "results": rows}


@router.get("/memories/{user_id}")
async def get_memories(user_id: str, limit: int = 10, request: Request = None):
    from src.memory.mem0_store import search_memories  # type: ignore
    deps = request.app.state.deps
    if not deps.mem0_client:
        raise HTTPException(503, "Memory not available")
    return search_memories(deps.mem0_client, "", user_id, limit=limit)


@router.get("/health")
async def health(request: Request):
    deps = request.app.state.deps
    neo4j_ok = deps.neo4j_driver is not None
    chroma_ok = deps.collections is not None
    circuits = deps.fallback_chain.circuits() if hasattr(deps.fallback_chain, "circuits") else []
    all_healthy = neo4j_ok and chroma_ok and all(c["state"] == "closed" for c in circuits)
    return {
        "status": "healthy" if all_healthy else "degraded",
        "neo4j": neo4j_ok,
        "chroma": chroma_ok,
        "mem0": deps.mem0_client is not None,
        "circuits": circuits,
    }


@router.get("/metrics")
async def metrics(request: Request):
    deps = request.app.state.deps
    summary = deps.cost_tracker.get_summary() if hasattr(deps.cost_tracker, "get_summary") else {}
    return {"cost": summary, "circuits": deps.fallback_chain.circuits()}
