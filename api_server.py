# HoloLearn-RAG/api_server.py

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_RAG_DIR = Path(__file__).parent.absolute()
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # always loads D:\HoloLearn-RAG\.env
print("=== GROQ KEYS DEBUG ===")
print("GENERATE:", os.getenv("GROQ_API_KEY_GENERATE", "NOT LOADED")[:15])
print("WHISPER :", os.getenv("GROQ_API_KEY_WHISPER",  "NOT LOADED")[:15])
print("GRADER  :", os.getenv("GROQ_API_KEY_GRADER",   "NOT LOADED")[:15])
print("QUERY   :", os.getenv("GROQ_API_KEY_QUERY",    "NOT LOADED")[:15])
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY",      "NOT LOADED")[:15])
print("=======================")
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from wrapper import GraphRAGWrapper

# ── Directories + keys from .env ─────────────────────────────────────────────
_WORK_DIR    = os.getenv("WORK_DIR",    "./data/rag_work")
_ANSWERS_DIR = os.getenv("ANSWERS_DIR", "./data/answers")
_INDEX_DIR   = os.getenv("INDEX_DIR",   "./data/indexes")

_rag: Optional[GraphRAGWrapper] = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    _rag = GraphRAGWrapper(
        work_dir     = _WORK_DIR,
        answers_dir  = _ANSWERS_DIR,
        index_dir    = _INDEX_DIR,
        neo4j_uri    = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687"),
        neo4j_user   = os.getenv("NEO4J_USER",     "neo4j"),
        neo4j_pass   = os.getenv("NEO4J_PASSWORD", "neo4j1234"),
        key_generate = os.getenv("GROQ_API_KEY_GENERATE", ""),
        key_whisper  = os.getenv("GROQ_API_KEY_WHISPER",  ""),
        key_query    = os.getenv("GROQ_API_KEY_QUERY",    ""),
        key_grader   = os.getenv("GROQ_API_KEY_GRADER",   ""),
        key_LLM= os.getenv("GROQ_API_KEY",""),
    )
    print("[api_server] ✅ RAG service ready on port 8002")
    yield
    print("[api_server] RAG service shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "HoloLearn RAG API",
    description = "Lecture ingestion + per-student Q&A powered by GraphRAG",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    lecture_session_id : str
    student_session_id : str
    query              : str   # plain text OR absolute path to audio file


class QueryResponse(BaseModel):
    lecture_session_id : str
    student_session_id : str
    query              : str
    answer             : str
    answer_file        : str
    timestamp          : str


class IngestResponse(BaseModel):
    lecture_session_id    : str
    chunks_indexed        : int
    nodes_written         : int
    relationships_written : int
    message               : str


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    info = _rag.health_info()
    return {"status": "ok", **info}


# ── GET /students ─────────────────────────────────────────────────────────────

@app.get("/students")
def students():
    return {
        "lecture_session_id" : _rag._lecture_state["lecture_session_id"],
        "students"           : _rag.active_students(),
        "count"              : len(_rag.active_students()),
    }


# ── POST /ingest ──────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_lecture(
    file: UploadFile = File(..., description="Lecture file: PDF / DOCX / PPTX / TXT / MD"),
):
    # Save uploaded file to a temp location first
    import tempfile
    suffix = Path(file.filename).suffix.lower() if file.filename else ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = _rag.ingest(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        lecture_session_id    = result["lecture_session_id"],
        chunks_indexed        = result["chunks_indexed"],
        nodes_written         = result["nodes_written"],
        relationships_written = result["relationships_written"],
        message               = "Lecture ingested. Students can now query.",
    )


# ── POST /query ───────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_lecture(req: QueryRequest):
    try:
        result = _rag.query(
            lecture_session_id = req.lecture_session_id,
            student_session_id = req.student_session_id,
            query_text         = req.query,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        lecture_session_id = req.lecture_session_id,
        student_session_id = req.student_session_id,
        query              = req.query,
        answer             = result["answer"],
        answer_file        = result["answer_file"],
        timestamp          = result["timestamp"],
    )