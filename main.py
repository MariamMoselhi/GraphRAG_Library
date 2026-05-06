#!/usr/bin/env python3
"""
graphrag_api.py
===============
FastAPI service — bridges mobile application ↔ GraphRAG pipeline.

Flow
----
  Mobile App
    │
    ├─ POST /ingest   (lecture file)
    │     → wipe Neo4j completely  (MATCH (n) DETACH DELETE n)
    │     → extract text → chunk → embed → FAISS + BM25 indices
    │     → extract nodes + relationships → Neo4j
    │     → build RetrievalPipeline + AnswerGenerator
    │     → return  lecture_session_id
    │
    └─ POST /query    (student query — text or audio path)
          → retrieve  (BM25 + Vector + Graph → RRF → Reranker → Grader)
          → generate answer  (AnswerGenerator)
          → save answer as plain .txt  (NO timestamps, NO headers)
          → append entry to per-lecture JSON index file
          → return  answer + answer_file path

Endpoints
---------
  POST  /ingest           Ingest a new lecture file
  POST  /query            Answer a student query
  GET   /health           Liveness check

Output file layout
------------------
  ANSWERS_DIR/
      <lecture_session_id>__<student_session_id>__<YYYYMMDD_HHMMSS>.txt
          → plain answer text ONLY (fed directly to TTS)

  INDEX_DIR/
      <lecture_session_id>.json
          → one file per lecture, one entry appended per query
          {
            "lecture_session_id": "lecture_abc123",
            "created_at"        : "2026-05-05T14:00:00",
            "queries": [
              {
                "lecture_session_id" : "lecture_abc123",
                "student_session_id" : "student_xyz",
                "timestamp"          : "2026-05-05T14:32:01",
                "query"              : "What is XSS?",
                "answer_file"        : "/abs/path/to/answer.txt"
              }
            ]
          }

Environment variables  (.env loaded automatically)
--------------------------------------------------
  GROQ_API_KEY_GENERATE   answer generation
  GROQ_API_KEY_WHISPER    voice transcription
  GROQ_API_KEY_QUERY      intent / entity / Cypher extraction
  GROQ_API_KEY_GRADER     relevance grading
  NEO4J_URI               default: neo4j://127.0.0.1:7687
  NEO4J_USER              default: neo4j
  NEO4J_PASSWORD          default: neo4j1234
  WORK_DIR                index / memory / cache root  (default: ./rag_work)
  ANSWERS_DIR             answer .txt files            (default: ./answers)
  INDEX_DIR               JSON index files             (default: ./indexes)
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# ── directory defaults ────────────────────────────────────────────────────────
WORK_DIR    = Path(os.getenv("WORK_DIR",    "./rag_work")).resolve()
ANSWERS_DIR = Path(os.getenv("ANSWERS_DIR", "./answers")).resolve()
INDEX_DIR   = Path(os.getenv("INDEX_DIR",   "./indexes")).resolve()

for _d in (WORK_DIR, ANSWERS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Groq keys ─────────────────────────────────────────────────────────────────
_KEY_GENERATE = os.getenv("GROQ_API_KEY_GENERATE", "")
_KEY_WHISPER  = os.getenv("GROQ_API_KEY_WHISPER",  "")
_KEY_QUERY    = os.getenv("GROQ_API_KEY_QUERY",    "")
_KEY_GRADER   = os.getenv("GROQ_API_KEY_GRADER",   "")

# ── Neo4j connection ──────────────────────────────────────────────────────────
_NEO4J_URI  = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
_NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "neo4j1234")

# ── Shared in-process state ───────────────────────────────────────────────────
# Caches heavy objects (embed model, graph store, pipeline) across requests.
_state: dict = {
    "lecture_session_id" : None,
    "embed_model"        : None,
    "graph_store"        : None,
    "pipeline"           : None,
    "answer_generator"   : None,
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "GraphRAG Lecture API",
    description = "Lecture ingestion + per-student Q&A powered by GraphRAG",
    version     = "1.0.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    lecture_session_id  — returned by POST /ingest, must match active session.
    student_session_id  — unique per student, assigned by the mobile backend.
    query               — plain text question OR absolute path to audio file.
    """
    lecture_session_id : str
    student_session_id : str
    query              : str


class QueryResponse(BaseModel):
    lecture_session_id : str
    student_session_id : str
    query              : str
    answer             : str
    answer_file        : str   # absolute path to saved .txt
    timestamp          : str   # ISO-8601 UTC


class IngestResponse(BaseModel):
    lecture_session_id     : str
    chunks_indexed         : int
    nodes_written          : int
    relationships_written  : int
    message                : str


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status"             : "ok",
        "lecture_session_id" : _state["lecture_session_id"],
        "pipeline_ready"     : _state["pipeline"] is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_lecture(
    file: UploadFile = File(..., description="Lecture file: PDF / DOCX / PPTX / TXT / MD"),
):
    """
    Ingest a new lecture.

    Every call:
      - generates a fresh lecture_session_id
      - wipes Neo4j entirely before building the new graph
      - rebuilds FAISS, BM25, and Neo4j from the uploaded file
      - replaces the in-process RetrievalPipeline + AnswerGenerator
    """
    # ── 1. New session id ─────────────────────────────────────────────────────
    lecture_session_id = f"lecture_{uuid.uuid4().hex[:12]}"
    lecture_work_dir   = WORK_DIR / lecture_session_id
    lecture_work_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Save uploaded file ─────────────────────────────────────────────────
    suffix   = Path(file.filename).suffix.lower() if file.filename else ".bin"
    tmp_path = lecture_work_dir / f"lecture{suffix}"
    tmp_path.write_bytes(await file.read())

    # ── 3. Extract text ───────────────────────────────────────────────────────
    try:
        extracted_text = _extract_text(str(tmp_path), suffix)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    if len(extracted_text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Extracted text is too short.")

    # ── 4. Chunk ──────────────────────────────────────────────────────────────
    try:
        chunks = _chunk_text(extracted_text, source=str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {exc}")

    # ── 5. Embed + build FAISS & BM25 ─────────────────────────────────────────
    embed_model = _get_embed_model()
    try:
        _build_indices(embed_model, chunks, str(lecture_work_dir))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index building failed: {exc}")

    # ── 6. Neo4j — wipe everything, then build new graph ──────────────────────
    try:
        from graph.graph_store import GraphStore
        graph_store = GraphStore(
            uri      = _NEO4J_URI,
            user     = _NEO4J_USER,
            password = _NEO4J_PASS,
        )
        graph_store.init_schema()
        _wipe_neo4j(graph_store)   # full wipe before every new lecture
        nodes_written, rels_written = _build_graph(
            embed_model = embed_model,
            chunks      = chunks,
            graph_store = graph_store,
            api_key     = _KEY_QUERY or _KEY_GRADER,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph build failed: {exc}")

    # ── 7. RetrievalPipeline ──────────────────────────────────────────────────
    try:
        from retrieval.retrieval_pipeline import RetrievalPipeline
        pipeline = RetrievalPipeline(
            embedding_model   = embed_model,
            graph_store       = graph_store,
            faiss_dir         = str(lecture_work_dir),
            bm25_dir          = str(lecture_work_dir),
            session_id        = lecture_session_id,
            memory_dir        = str(lecture_work_dir),
            whisper_api_key   = _KEY_WHISPER,
            query_llm_api_key = _KEY_QUERY,
            grader_api_key    = _KEY_GRADER,
            enable_retry      = True,
            show_graph_viz    = False,
            verbose           = False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline init failed: {exc}")

    # ── 8. AnswerGenerator ────────────────────────────────────────────────────
    try:
        from retrieval.answer_generator import AnswerGenerator
        answer_gen = AnswerGenerator(
            api_key     = _KEY_GENERATE,
            answers_dir = str(ANSWERS_DIR),
            verbose     = False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AnswerGenerator init failed: {exc}")

    # ── 9. Persist everything in shared state ──────────────────────────────────
    _state["lecture_session_id"] = lecture_session_id
    _state["embed_model"]        = embed_model
    _state["graph_store"]        = graph_store
    _state["pipeline"]           = pipeline
    _state["answer_generator"]   = answer_gen

    return IngestResponse(
        lecture_session_id    = lecture_session_id,
        chunks_indexed        = len(chunks),
        nodes_written         = nodes_written,
        relationships_written = rels_written,
        message               = "Lecture ingested successfully. Pipeline is ready.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_lecture(req: QueryRequest):
    """
    Answer a student query against the currently loaded lecture.

    - Validates lecture_session_id matches the active session.
    - Accepts text queries or absolute audio file paths.
    - Saves the answer as a plain .txt file (no timestamps, no headers).
    - Appends one record to the per-lecture JSON index file.
    """
    # ── Guard: pipeline must be ready ─────────────────────────────────────────
    if _state["pipeline"] is None:
        raise HTTPException(
            status_code = 503,
            detail      = "No lecture has been ingested yet. Call POST /ingest first.",
        )

    # ── Guard: session must match ─────────────────────────────────────────────
    if req.lecture_session_id != _state["lecture_session_id"]:
        raise HTTPException(
            status_code = 409,
            detail      = (
                f"lecture_session_id mismatch. "
                f"Active session is '{_state['lecture_session_id']}'. "
                "Call POST /ingest to load a new lecture."
            ),
        )

    pipeline   = _state["pipeline"]
    answer_gen = _state["answer_generator"]
    timestamp  = datetime.utcnow()

    # ── Retrieval pipeline ────────────────────────────────────────────────────
    try:
        graded_result = pipeline.run(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval pipeline error: {exc}")

    # ── Answer generation ─────────────────────────────────────────────────────
    try:
        answer = answer_gen.generate(
            query         = req.query,
            graded_result = graded_result,
            session_id    = req.lecture_session_id,
        )
    except Exception:
        answer = (
            "I wasn't able to generate an answer at this time. "
            "Please try rephrasing your question."
        )

    # ── Record turn in memory (best-effort) ───────────────────────────────────
    try:
        pipeline.record_turn(
            user_query    = req.query,
            ai_response   = answer,
            graded_result = graded_result,
        )
    except Exception:
        pass

    # ── Save answer to plain .txt ─────────────────────────────────────────────
    answer_file = _save_answer_txt(
        lecture_session_id = req.lecture_session_id,
        student_session_id = req.student_session_id,
        answer             = answer,
        timestamp          = timestamp,
    )

    # ── Append to JSON index ──────────────────────────────────────────────────
    _append_index(
        lecture_session_id = req.lecture_session_id,
        student_session_id = req.student_session_id,
        query              = req.query,
        answer_file        = answer_file,
        timestamp          = timestamp,
    )

    return QueryResponse(
        lecture_session_id = req.lecture_session_id,
        student_session_id = req.student_session_id,
        query              = req.query,
        answer             = answer,
        answer_file        = str(answer_file),
        timestamp          = timestamp.isoformat(timespec="seconds"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wipe_neo4j(graph_store) -> None:
    """Delete ALL nodes and relationships from Neo4j (full database wipe)."""
    with graph_store._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def _get_embed_model():
    """
    Lazy-load the HuggingFace embedding model and cache it in _state.
    Loading happens only once per process lifetime.
    """
    if _state["embed_model"] is None:
        from embeddings.huggingFace import HuggingFaceEmbedding
        _state["embed_model"] = HuggingFaceEmbedding(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            normalize  = True,
        )
    return _state["embed_model"]


def _extract_text(file_path: str, suffix: str) -> str:
    """Extract raw text from a lecture file using the project extractors."""
    if suffix == ".pdf":
        from extractors.pdf_extractor import PDFExtractor
        result = PDFExtractor().extract(file_path)
    elif suffix in (".docx", ".doc"):
        from extractors.docx_extractor import DOCXExtractor
        result = DOCXExtractor().extract(file_path)
    elif suffix in (".pptx", ".ppt"):
        from extractors.pptx_extractor import PPTXExtractor
        result = PPTXExtractor().extract(file_path)
    else:  # .txt / .md / fallback
        result = {
            "success"        : True,
            "extracted_text" : Path(file_path).read_text(encoding="utf-8"),
        }

    if not result.get("success"):
        raise RuntimeError(result.get("error", "unknown extraction error"))

    return result.get("extracted_text", "")


def _chunk_text(text: str, source: str) -> list:
    """
    Chunk text using the semantic chunker.
    Falls back to paragraph chunker if semantic is unavailable.
    Tags every chunk with its source file path.
    """
    from chunking.chunking import get_chunker

    try:
        chunker = get_chunker("semantic", threshold=0.5, max_sentences_per_chunk=20)
        chunks  = chunker.chunk(text)
    except Exception:
        chunker = get_chunker("paragraph")
        chunks  = chunker.chunk(text)

    for c in chunks:
        if not hasattr(c, "metadata") or c.metadata is None:
            c.metadata = {}
        c.metadata["source"] = source

    return chunks


def _build_indices(embed_model, chunks: list, save_dir: str) -> None:
    """
    Batch-embed all chunks, then build and save FAISS + BM25 indices.

    Embeddings are stored in chunk.metadata['embedding'] before add_chunks()
    is called — this matches the FaissVectorStore contract used throughout
    the project (see interactive_retrieval_test.py).
    """
    from vectordb.faiss_store import FaissVectorStore
    from retrieval.bm25_retriever import BM25Retriever

    texts   = [c.text for c in chunks]
    vectors = embed_model.encode(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk.metadata["embedding"] = vec

    faiss_store = FaissVectorStore(dim=embed_model.dimension, verbose=False)
    faiss_store.add_chunks(chunks)
    faiss_store.save(save_dir)

    bm25 = BM25Retriever(verbose=False)
    bm25.build(chunks)
    bm25.save(save_dir)


def _build_graph(
    embed_model,
    chunks     : list,
    graph_store,
    api_key    : str,
) -> tuple[int, int]:
    """
    Extract entities + relationships from chunks and write to Neo4j.
    Uses Pipeline.from_components — same pattern as interactive_retrieval_test.py.
    Returns (nodes_written, relationships_written).
    """
    from graph.node_relation_extractor import CombinedExtractor
    from graph.llm_backend import LLMBackend
    from graph.Pipeline import Pipeline

    llm = LLMBackend(
        api_key    = api_key,
        model      = "llama-3.3-70b-versatile",
        max_tokens = 3000,
    )
    extractor = CombinedExtractor(
        llm          = llm,
        embedding_fn = embed_model.encode,
        batch_chunks = 2,
        mode         = "constrained",
    )
    pipeline = Pipeline.from_components(
        node_extractor         = extractor,
        relationship_extractor = extractor,
        graph_store            = graph_store,
        extract_cross_doc      = False,
        show_progress          = True,
    )
    stats = pipeline.run(chunks)
    return stats.nodes_written, stats.relationships_written


def _save_answer_txt(
    lecture_session_id : str,
    student_session_id : str,
    answer             : str,
    timestamp          : datetime,
) -> Path:
    """
    Save the plain answer text to a .txt file.

    Filename: <lecture_session_id>__<student_session_id>__<YYYYMMDD_HHMMSS>.txt

    The file contains ONLY the answer — no timestamps, no headers, no
    metadata — so it can be fed directly into TTS without post-processing.

    Returns the absolute resolved Path of the saved file.
    """
    ts_str   = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{lecture_session_id}__{student_session_id}__{ts_str}.txt"
    filepath = ANSWERS_DIR / filename
    filepath.write_text(answer, encoding="utf-8")
    return filepath.resolve()


def _append_index(
    lecture_session_id : str,
    student_session_id : str,
    query              : str,
    answer_file        : Path,
    timestamp          : datetime,
) -> None:
    """
    Append one query record to  INDEX_DIR/<lecture_session_id>.json.

    The file is created on the first query for a lecture and grows by one
    entry for every subsequent query in the same lecture session.

    Write is atomic: content is written to a .tmp file then renamed.

    JSON structure
    --------------
    {
      "lecture_session_id" : "lecture_abc123",
      "created_at"         : "2026-05-05T14:00:00",   ← set once, never changed
      "queries": [
        {
          "lecture_session_id" : "lecture_abc123",
          "student_session_id" : "student_xyz",
          "timestamp"          : "2026-05-05T14:32:01",
          "query"              : "What is XSS?",
          "answer_file"        : "/abs/path/to/answer.txt"
        },
        ...
      ]
    }
    """
    index_path = INDEX_DIR / f"{lecture_session_id}.json"

    # Load existing data or initialise a fresh structure
    data: Optional[dict] = None
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None  # corrupted — rebuild cleanly

    if data is None:
        data = {
            "lecture_session_id" : lecture_session_id,
            "created_at"         : datetime.utcnow().isoformat(timespec="seconds"),
            "queries"            : [],
        }

    # Append the new query record
    data["queries"].append({
        "lecture_session_id" : lecture_session_id,
        "student_session_id" : student_session_id,
        "timestamp"          : timestamp.isoformat(timespec="seconds"),
        "query"              : query,
        "answer_file"        : str(answer_file),
    })

    # Atomic write via .tmp → rename
    tmp_path = index_path.with_suffix(".tmp")
    try:
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(index_path)
    except OSError as exc:
        # Non-fatal: log but never crash the HTTP response
        print(f"[graphrag_api] WARNING: index write failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
#   uvicorn graphrag_api:app --host 0.0.0.0 --port 8000 --reload
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("graphrag_api:app", host="0.0.0.0", port=8000, reload=False)