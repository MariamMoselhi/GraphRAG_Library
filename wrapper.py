# HoloLearn-RAG/wrapper.py

import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# ── Path setup ────────────────────────────────────────────────────────────────
_RAG_DIR = Path(__file__).parent.absolute()
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))
# ─────────────────────────────────────────────────────────────────────────────


class GraphRAGWrapper:
    """
    Clean interface over the GraphRAG pipeline.
    Mirrors SimpleExtractorWrapper pattern from the agent service.

    Owns:
      - Embed model       (loaded once at init, shared across all lectures)
      - Lecture state     (rebuilt on every ingest())
      - Student pipelines (created on first query per student, per lecture)
      - Answer saving     (plain .txt files)
      - JSON index        (per-lecture query log)
    """

    def __init__(
        self,
        work_dir    : str,
        answers_dir : str,
        index_dir   : str,
        neo4j_uri   : str,
        neo4j_user  : str,
        neo4j_pass  : str,
        key_generate: str,
        key_whisper : str,
        key_query   : str,
        key_grader  : str,
        key_LLM     : str,
    ):
        self.work_dir    = Path(work_dir).resolve()
        self.answers_dir = Path(answers_dir).resolve()
        self.index_dir   = Path(index_dir).resolve()

        for d in (self.work_dir, self.answers_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._neo4j_uri  = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_pass = neo4j_pass

        self._key_generate = key_generate
        self._key_whisper  = key_whisper
        self._key_query    = key_query
        self._key_grader   = key_grader
        self._key_LLM = key_LLM

        # ── Embed model (loaded once, never reset) ────────────────────────────
        self._embed_model      = None
        self._embed_model_lock = threading.Lock()

        # ── Lecture state (reset on every ingest) ─────────────────────────────
        self._lecture_state: dict = {
            "lecture_session_id" : None,
            "lecture_work_dir"   : None,
            "graph_store"        : None,
            "answer_generator"   : None,
        }

        # ── Per-student pipelines ─────────────────────────────────────────────
        self._student_pipelines: Dict[str, object] = {}
        self._students_lock = threading.Lock()
        self._index_lock    = threading.Lock()

    # ── Public: ingest ────────────────────────────────────────────────────────

    def ingest(self, file_path: str) -> dict:
        """
        Index a lecture file. Returns dict with lecture_session_id and stats.
        Called once per lecture from /ingest endpoint.
        """
        file_path = Path(file_path)
        suffix    = file_path.suffix.lower()

        lecture_session_id = f"lecture_{uuid.uuid4().hex[:12]}"
        lecture_work_dir   = self.work_dir / lecture_session_id
        lecture_work_dir.mkdir(parents=True, exist_ok=True)
        (lecture_work_dir / "students").mkdir(exist_ok=True)

        # Extract
        extracted_text = self._extract_text(str(file_path), suffix)
        if len(extracted_text.strip()) < 50:
            raise ValueError("Extracted text is too short.")

        # Chunk
        chunks = self._chunk_text(extracted_text, source=str(file_path))

        # Embed + indices
        embed_model = self._get_embed_model()
        self._build_indices(embed_model, chunks, str(lecture_work_dir))

        # Neo4j
        from graph.graph_store import GraphStore
        graph_store = GraphStore(
            uri      = self._neo4j_uri,
            user     = self._neo4j_user,
            password = self._neo4j_pass,
        )
        graph_store.init_schema()
        self._wipe_neo4j(graph_store)
        nodes_written, rels_written = self._build_graph(
            embed_model = embed_model,
            chunks      = chunks,
            graph_store = graph_store,
            api_key     = self._key_LLM,
        )

        # AnswerGenerator
        from retrieval.answer_generator import AnswerGenerator
        answer_gen = AnswerGenerator(
            llm_api_key = self._key_generate,
            output_dir  = None,
            verbose     = False,
        )

        # Update state atomically
        with self._students_lock:
            self._lecture_state["lecture_session_id"] = lecture_session_id
            self._lecture_state["lecture_work_dir"]   = lecture_work_dir
            self._lecture_state["graph_store"]        = graph_store
            self._lecture_state["answer_generator"]   = answer_gen
            self._student_pipelines = {}

        return {
            "lecture_session_id"   : lecture_session_id,
            "chunks_indexed"       : len(chunks),
            "nodes_written"        : nodes_written,
            "relationships_written": rels_written,
        }

    # ── Public: query ─────────────────────────────────────────────────────────

    def query(
        self,
        lecture_session_id : str,
        student_session_id : str,
        query_text         : str,
    ) -> dict:
        """
        Answer a student query. Returns dict with answer text and answer_file path.
        query_text can be plain text OR absolute path to an audio file.
        """
        # Guard
        if self._lecture_state["graph_store"] is None:
            raise RuntimeError("No lecture ingested. Call ingest() first.")

        if lecture_session_id != self._lecture_state["lecture_session_id"]:
            raise ValueError(
                f"lecture_session_id mismatch. "
                f"Active: '{self._lecture_state['lecture_session_id']}'."
            )

        pipeline   = self._get_or_create_student_pipeline(student_session_id)
        answer_gen = self._lecture_state["answer_generator"]
        timestamp  = datetime.utcnow()

        # Retrieval
        graded_result = pipeline.run(query_text)

        # Generation
        answer_result = None
        try:
            answer_result = answer_gen.generate(
                graded_result  = graded_result,
                pipeline       = None,
                original_query = query_text,
            )
            answer = answer_result.text
        except Exception:
            answer = (
                "I wasn't able to generate an answer at this time. "
                "Please try rephrasing your question."
            )

        # Record turn in student memory
        try:
            pipeline.record_turn(
                user_query    = query_text,
                ai_response   = answer,
                graded_result = graded_result,
            )
        except Exception:
            pass

        # Save answer .txt
        answer_file = self._save_answer_txt(
            lecture_session_id = lecture_session_id,
            student_session_id = student_session_id,
            answer             = answer,
            timestamp          = timestamp,
        )

        # Append to JSON index
        self._append_index(
            lecture_session_id = lecture_session_id,
            student_session_id = student_session_id,
            query              = query_text,
            answer_file        = answer_file,
            timestamp          = timestamp,
        )

        return {
            "answer"      : answer,
            "answer_file" : str(answer_file),
            "timestamp"   : timestamp.isoformat(timespec="seconds"),
        }

    # ── Public: health info ───────────────────────────────────────────────────

    def health_info(self) -> dict:
        return {
            "lecture_session_id" : self._lecture_state["lecture_session_id"],
            "pipeline_ready"     : self._lecture_state["graph_store"] is not None,
            "active_students"    : len(self._student_pipelines),
        }

    def active_students(self) -> list:
        return list(self._student_pipelines.keys())

    # ── Private: student pipeline factory ────────────────────────────────────

    def _get_or_create_student_pipeline(self, student_session_id: str):
        if student_session_id in self._student_pipelines:
            return self._student_pipelines[student_session_id]

        lecture_work_dir = self._lecture_state["lecture_work_dir"]
        student_dir      = lecture_work_dir / "students" / student_session_id
        student_dir.mkdir(parents=True, exist_ok=True)

        from retrieval.retrieval_pipeline import RetrievalPipeline
        new_pipeline = RetrievalPipeline(
            embedding_model   = self._get_embed_model(),
            graph_store       = self._lecture_state["graph_store"],
            faiss_dir         = str(lecture_work_dir),
            bm25_dir          = str(lecture_work_dir),
            session_id        = student_session_id,
            memory_dir        = str(student_dir),
            cache_capacity    = 64,
            cache_threshold   = 0.95,
            whisper_api_key   = self._key_whisper,
            query_llm_api_key = self._key_query,
            grader_api_key    = self._key_grader,
            enable_retry      = True,
            show_graph_viz    = False,
            verbose           = False,
        )

        with self._students_lock:
            if student_session_id not in self._student_pipelines:
                self._student_pipelines[student_session_id] = new_pipeline
            return self._student_pipelines[student_session_id]

    # ── Private: embed model ──────────────────────────────────────────────────

    def _get_embed_model(self):
        if self._embed_model is not None:
            return self._embed_model
        with self._embed_model_lock:
            if self._embed_model is None:
                from embeddings.huggingFace import HuggingFaceEmbedding
                self._embed_model = HuggingFaceEmbedding(
                    model_name = "sentence-transformers/all-MiniLM-L6-v2",
                    normalize  = True,
                )
        return self._embed_model

    # ── Private: Neo4j ────────────────────────────────────────────────────────

    def _wipe_neo4j(self, graph_store) -> None:
        with graph_store._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # ── Private: extraction ───────────────────────────────────────────────────

    def _extract_text(self, file_path: str, suffix: str) -> str:
        if suffix == ".pdf":
            from extractors.pdf_extractor import PDFExtractor
            result = PDFExtractor().extract(file_path)
        elif suffix in (".docx", ".doc"):
            from extractors.docx_extractor import DOCXExtractor
            result = DOCXExtractor().extract(file_path)
        elif suffix in (".pptx", ".ppt"):
            from extractors.pptx_extractor import PPTXExtractor
            result = PPTXExtractor().extract(file_path)
        else:
            result = {
                "success"        : True,
                "extracted_text" : Path(file_path).read_text(encoding="utf-8"),
            }
        if not result.get("success"):
            raise RuntimeError(result.get("error", "unknown extraction error"))
        return result.get("extracted_text", "")

    def _chunk_text(self, text: str, source: str) -> list:
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

    def _build_indices(self, embed_model, chunks: list, save_dir: str) -> None:
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

    def _build_graph(self, embed_model, chunks, graph_store, api_key) -> tuple:
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

    # ── Private: answer saving ────────────────────────────────────────────────

    def _save_answer_txt(
        self,
        lecture_session_id : str,
        student_session_id : str,
        answer             : str,
        timestamp          : datetime,
    ) -> Path:
        ts_str   = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{lecture_session_id}__{student_session_id}__{ts_str}.txt"
        filepath = self.answers_dir / filename
        filepath.write_text(answer, encoding="utf-8")
        return filepath.resolve()

    def _append_index(
        self,
        lecture_session_id : str,
        student_session_id : str,
        query              : str,
        answer_file        : Path,
        timestamp          : datetime,
    ) -> None:
        import json
        index_path = self.index_dir / f"{lecture_session_id}.json"
        with self._index_lock:
            data: Optional[dict] = None
            if index_path.exists():
                try:
                    data = json.loads(index_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    data = None
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
            tmp_path = index_path.with_suffix(".tmp")
            try:
                tmp_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp_path.replace(index_path)
            except OSError as exc:
                print(f"[wrapper] WARNING: index write failed: {exc}")