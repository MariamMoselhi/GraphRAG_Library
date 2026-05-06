#!/usr/bin/env python3
"""
graphrag_api.py
===============
FastAPI service — bridges mobile application ↔ GraphRAG pipeline.

Architecture: Per-student pipeline isolation
--------------------------------------------
Each student gets their OWN RetrievalPipeline instance with their own:
  - MemoryStore  → conversation history is private to each student
  - QueryCache   → cached results are not shared between students
  - AnswerGenerator → separate answer generation context

All students in the same lecture share:
  - The FAISS index       (read-only, built once at /ingest)
  - The BM25 index        (read-only, built once at /ingest)
  - The Neo4j graph store (read-only queries after /ingest)
  - The embedding model   (stateless, thread-safe)

This means student A asking "what is XSS?" will NOT return student B's
cached answer, and "elaborate on that" will correctly resolve to each
student's own conversation history.

Flow
----
  Mobile App
    │
    ├─ POST /ingest   (lecture file — called once per lecture)
    │     → wipe Neo4j completely  (MATCH (n) DETACH DELETE n)
    │     → extract text → chunk → embed → FAISS + BM25 indices
    │     → extract nodes + relationships → Neo4j
    │     → store shared lecture components in _lecture_state
    │     → return  lecture_session_id
    │
    └─ POST /query    (one call per student question)
          → look up or create this student's pipeline (per student_session_id)
          → retrieve  (BM25 + Vector + Graph → RRF → Reranker → Grader)
          → generate answer
          → save answer as plain .txt  (answer text ONLY — no timestamps, no headers)
          → append entry to per-lecture JSON index (grows with every query)
          → return  answer + answer_file path

Endpoints
---------
  POST  /ingest           Ingest a new lecture file
  POST  /query            Answer a student query
  GET   /health           Liveness check
  GET   /students         List all student session IDs active in current lecture

Output file layout
------------------
  ANSWERS_DIR/
      <lecture_session_id>__<student_session_id>__<YYYYMMDD_HHMMSS>.txt
          → plain answer text ONLY (fed directly to TTS, no post-processing needed)

  INDEX_DIR/
      <lecture_session_id>.json
          → one file per lecture, one entry appended per query, never overwritten
          {
            "lecture_session_id" : "lecture_abc123",
            "created_at"         : "2026-05-05T14:00:00",
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

  WORK_DIR/
      <lecture_session_id>/
          faiss.index          shared FAISS index
          bm25.pkl             shared BM25 index
          lecture.<ext>        original uploaded file
          students/
              <student_session_id>/
                  memory_<student_session_id>.json   per-student memory
                  cache_<student_session_id>.pkl     per-student query cache

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
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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

# ─────────────────────────────────────────────────────────────────────────────
# Shared lecture state  (rebuilt on every POST /ingest)
# ─────────────────────────────────────────────────────────────────────────────
# Heavy read-only objects shared across all students for the current lecture.
_lecture_state: dict = {
    "lecture_session_id" : None,   # str
    "lecture_work_dir"   : None,   # Path
    "embed_model"        : None,   # HuggingFaceEmbedding  (stateless, thread-safe)
    "graph_store"        : None,   # GraphStore            (read-only after ingest)
    "answer_generator"   : None,   # AnswerGenerator       (stateless per call)
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-student pipeline registry
# ─────────────────────────────────────────────────────────────────────────────
# Maps student_session_id → RetrievalPipeline instance.
# Each pipeline has its own MemoryStore + QueryCache so students never
# interfere with each other's conversation state or cached results.
#
# Protected by _students_lock for thread-safe creation on first query.
_student_pipelines: Dict[str, object] = {}   # student_session_id → RetrievalPipeline
_students_lock = threading.Lock()

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
    answer_file        : str   # absolute path to the saved .txt
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
        "lecture_session_id" : _lecture_state["lecture_session_id"],
        "pipeline_ready"     : _lecture_state["graph_store"] is not None,
        "active_students"    : len(_student_pipelines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /students
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/students")
def list_students():
    """Return all student session IDs that have queried the current lecture."""
    return {
        "lecture_session_id" : _lecture_state["lecture_session_id"],
        "students"           : list(_student_pipelines.keys()),
        "count"              : len(_student_pipelines),
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
      1. Generates a fresh lecture_session_id.
      2. Wipes Neo4j entirely (MATCH (n) DETACH DELETE n).
      3. Extracts text → chunks → embeds → builds FAISS + BM25.
      4. Extracts nodes + relationships → writes to Neo4j.
      5. Stores shared components in _lecture_state.
      6. Clears all per-student pipelines from the previous lecture.
      7. Returns lecture_session_id for clients to use in /query calls.
    """
    global _student_pipelines

    # ── 1. New session id ─────────────────────────────────────────────────────
    lecture_session_id = f"lecture_{uuid.uuid4().hex[:12]}"
    lecture_work_dir   = WORK_DIR / lecture_session_id
    lecture_work_dir.mkdir(parents=True, exist_ok=True)
    (lecture_work_dir / "students").mkdir(exist_ok=True)

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
        _wipe_neo4j(graph_store)  # full wipe before every new lecture
        nodes_written, rels_written = _build_graph(
            embed_model = embed_model,
            chunks      = chunks,
            graph_store = graph_store,
            api_key     = _KEY_QUERY or _KEY_GRADER,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph build failed: {exc}")

    # ── 7. Build shared AnswerGenerator ───────────────────────────────────────
    try:
        from retrieval.answer_generator import AnswerGenerator
        answer_gen = AnswerGenerator(
            llm_api_key = _KEY_GENERATE,
            output_dir  = None,   # disabled — graphrag_api._save_answer_txt handles saving
            verbose     = False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AnswerGenerator init failed: {exc}")

    # ── 8. Update shared lecture state + clear all student pipelines ──────────
    _lecture_state["lecture_session_id"] = lecture_session_id
    _lecture_state["lecture_work_dir"]   = lecture_work_dir
    _lecture_state["embed_model"]        = embed_model
    _lecture_state["graph_store"]        = graph_store
    _lecture_state["answer_generator"]   = answer_gen

    with _students_lock:
        _student_pipelines = {}   # clear pipelines from previous lecture

    return IngestResponse(
        lecture_session_id    = lecture_session_id,
        chunks_indexed        = len(chunks),
        nodes_written         = nodes_written,
        relationships_written = rels_written,
        message               = (
            "Lecture ingested successfully. "
            "Students can now send queries with this lecture_session_id."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_lecture(req: QueryRequest):
    """
    Answer a student query against the currently loaded lecture.

    On first query from a student:
      - Creates a new RetrievalPipeline for this student with their own
        MemoryStore and QueryCache stored under WORK_DIR/.../students/<id>/

    On subsequent queries from the same student:
      - Reuses the existing pipeline (memory carries over between turns).

    Saves a plain .txt answer file and appends to the per-lecture JSON index.
    """
    # ── Guard: lecture must be ingested first ─────────────────────────────────
    if _lecture_state["graph_store"] is None:
        raise HTTPException(
            status_code = 503,
            detail      = "No lecture has been ingested yet. Call POST /ingest first.",
        )

    # ── Guard: lecture session must match ─────────────────────────────────────
    if req.lecture_session_id != _lecture_state["lecture_session_id"]:
        raise HTTPException(
            status_code = 409,
            detail      = (
                f"lecture_session_id mismatch. "
                f"Active session is '{_lecture_state['lecture_session_id']}'. "
                "Call POST /ingest to load a new lecture."
            ),
        )

    # ── Get or create this student's pipeline ─────────────────────────────────
    pipeline   = _get_or_create_student_pipeline(req.student_session_id)
    answer_gen = _lecture_state["answer_generator"]
    timestamp  = datetime.utcnow()

    # ── Retrieval pipeline ────────────────────────────────────────────────────
    try:
        graded_result = pipeline.run(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval pipeline error: {exc}")

    # ── Answer generation ─────────────────────────────────────────────────────
    # generate() returns an AnswerResult object; .text is the answer string.
    # When pipeline is passed, generate() calls pipeline.record_turn() internally.
    answer_result = None
    try:
        answer_result = answer_gen.generate(
            graded_result  = graded_result,
            pipeline       = pipeline,
            original_query = req.query,
        )
        answer = answer_result.text
    except Exception:
        answer = (
            "I wasn't able to generate an answer at this time. "
            "Please try rephrasing your question."
        )

    # ── Record turn in this student's memory (best-effort) ────────────────────
    # generate() already calls record_turn() internally when it succeeds.
    # Only call it here for the fallback case where generate() raised.
    if answer_result is None:
        try:
            pipeline.record_turn(
                user_query    = req.query,
                ai_response   = answer,
                graded_result = graded_result,
            )
        except Exception:
            pass  # memory is best-effort; never crash the response

    # ── Save plain answer .txt ─────────────────────────────────────────────────
    answer_file = _save_answer_txt(
        lecture_session_id = req.lecture_session_id,
        student_session_id = req.student_session_id,
        answer             = answer,
        timestamp          = timestamp,
    )

    # ── Append to per-lecture JSON index ──────────────────────────────────────
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
# Per-student pipeline factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_student_pipeline(student_session_id: str):
    """
    Return the existing RetrievalPipeline for this student, or create one.

    Each student gets:
      - Their own MemoryStore (JSON file under students/<id>/)
      - Their own QueryCache  (pickle file under students/<id>/)
      - Shared FAISS, BM25, and GraphStore (read-only, from _lecture_state)

    Thread-safe: the lock is held only for the dict read/write, NOT during
    the slow RetrievalPipeline construction. This prevents one slow student
    init from blocking all other concurrent queries.
    """
    # Fast path — already exists, no lock needed
    if student_session_id in _student_pipelines:
        return _student_pipelines[student_session_id]

    # Prepare paths before taking the lock (I/O outside critical section)
    lecture_work_dir = _lecture_state["lecture_work_dir"]
    student_dir      = lecture_work_dir / "students" / student_session_id
    student_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline outside the lock — construction is slow (loads models)
    try:
        from retrieval.retrieval_pipeline import RetrievalPipeline
        new_pipeline = RetrievalPipeline(
            # Shared read-only resources
            embedding_model   = _lecture_state["embed_model"],
            graph_store       = _lecture_state["graph_store"],
            faiss_dir         = str(lecture_work_dir),
            bm25_dir          = str(lecture_work_dir),
            # Per-student isolation
            session_id        = student_session_id,
            memory_dir        = str(student_dir),
            # API keys
            whisper_api_key   = _KEY_WHISPER,
            query_llm_api_key = _KEY_QUERY,
            grader_api_key    = _KEY_GRADER,
            # Behaviour
            enable_retry      = True,
            show_graph_viz    = False,
            verbose           = False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code = 500,
            detail      = f"Failed to create pipeline for student '{student_session_id}': {exc}",
        )

    # Lock only for the dict write — if two threads raced, keep the first winner
    with _students_lock:
        if student_session_id not in _student_pipelines:
            _student_pipelines[student_session_id] = new_pipeline
        return _student_pipelines[student_session_id]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wipe_neo4j(graph_store) -> None:
    """Delete ALL nodes and relationships from Neo4j (full database wipe)."""
    with graph_store._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def _get_embed_model():
    """
    Lazy-load the HuggingFace embedding model and cache it permanently.
    The model is stateless and thread-safe — safe to share across students.
    """
    if _lecture_state["embed_model"] is None:
        from embeddings.huggingFace import HuggingFaceEmbedding
        _lecture_state["embed_model"] = HuggingFaceEmbedding(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            normalize  = True,
        )
    return _lecture_state["embed_model"]


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
    Embeddings go into chunk.metadata['embedding'] before add_chunks().
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

    Filename:
        <lecture_session_id>__<student_session_id>__<YYYYMMDD_HHMMSS>.txt

    The file contains ONLY the answer text — no timestamps, no headers,
    no metadata — so it can be fed directly into TTS without any
    post-processing on the mobile side.

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

    The file is created on the first query for a lecture.
    It grows by exactly one entry per /query call and is never truncated.
    Write is atomic via .tmp → rename.

    JSON schema
    -----------
    {
      "lecture_session_id" : "lecture_abc123",
      "created_at"         : "2026-05-05T14:00:00",   ← set once on first write
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

    # Load existing or initialise fresh
    data: Optional[dict] = None
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None  # corrupted file — rebuild cleanly

    if data is None:
        data = {
            "lecture_session_id" : lecture_session_id,
            "created_at"         : datetime.utcnow().isoformat(timespec="seconds"),
            "queries"            : [],
        }

    data["queries"].append({
        "lecture_session_id" : lecture_session_id,
        "student_session_id" : student_session_id,
        "timestamp"          : timestamp.isoformat(timespec="seconds"),
        "query"              : query,
        "answer_file"        : str(answer_file),
    })

    # Atomic write
    tmp_path = index_path.with_suffix(".tmp")
    try:
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(index_path)
    except OSError as exc:
        print(f"[graphrag_api] WARNING: index write failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
#   uvicorn graphrag_api:app --host 0.0.0.0 --port 8000 --reload
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("graphrag_api:app", host="0.0.0.0", port=8000, reload=False)