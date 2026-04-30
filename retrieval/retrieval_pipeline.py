"""
GraphRAG Retrieval Pipeline — full orchestrator.

Wires all retrieval modules in sequence:

    User input (voice / text)
        └─ QueryProcessor       (voice → text, memory injection)
             └─ [QueryCache]    (semantic cache check — skip retrieval if hit)
                  └─ QueryEngine     (intent, entities, Cypher, embedding)
                       ├─ BM25Retriever    (keyword)
                       ├─ VectorRetriever  (semantic FAISS)
                       └─ GraphRetriever   (Neo4j traversal)
                            └─ HybridRetriever  (RRF fusion)
                                 └─ Reranker         (cross-encoder)
                                      └─ Grader           (LLM quality gate)
                                           └─ GradedResult

Additional features
-------------------
- Adaptive retry: if Grader returns FAIL and provides a reformulation,
  the pipeline retries once with the reformulated query (max 1 retry).
- Feedback loop: the grader's failure reasons are fed back into the
  query engine on retry via the memory context slot.
- Graph visualisation: automatically printed after GraphRetriever completes.
- Full RetrievalTrace passed to retrieval_logger.print_trace() at the end.

Environment variables
---------------------
GROQ_API_KEY_WHISPER  — Groq key for voice transcription (QueryProcessor)
GROQ_API_KEY_QUERY    — Groq key for intent/Cypher/entity extraction (QueryEngine)
GROQ_API_KEY_GRADER   — Groq key for relevance grading (Grader)

Usage (minimal)
---------------
    from retrieval.retrieval_pipeline import RetrievalPipeline
    from embeddings.huggingFace import HuggingFaceEmbedding
    from graph.graph_store import GraphStore

    embed  = HuggingFaceEmbedding()
    store  = GraphStore(uri="neo4j://127.0.0.1:7687", user="neo4j", password="neo4j1234")

    pipeline = RetrievalPipeline(
        embedding_model = embed,
        graph_store     = store,
        faiss_dir       = "faiss_index",
        bm25_dir        = "bm25_index",
    )

    result = pipeline.run("What is the difference between BERT and GPT?")
    print(result.context_text)   # ready for answer generation
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .retrieval_context import (
    GradedResult,
    GraderVerdict,
    QueryRepresentation,
    RetrievalTrace,
)
from .retrieval_logger import RetrievalLogger
from .memory_store import MemoryStore
from .query_cache import QueryCache
from .faiss_store import FAISSStore
from .query_processor import QueryProcessor
from .query_engine import QueryEngine
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from .graph_visualizer import GraphVisualizer
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker
from .grader import Grader


class RetrievalPipeline:
    """
    Full GraphRAG retrieval pipeline.

    Args
    ----
    embedding_model     : HuggingFaceEmbedding (or any encode() model).
    graph_store         : Connected GraphStore instance.
    faiss_dir           : Directory containing faiss.index + faiss_meta.pkl.
                          If None, vector retrieval is disabled.
    bm25_dir            : Directory containing bm25.pkl.
                          If None, BM25 retrieval is disabled.
    session_id          : Conversation session identifier (for memory).
    memory_dir          : Directory for memory JSON files (default "memory").
    memory_max_turns    : Memory window size (default 10).
    cache_capacity      : Query cache capacity (default 128 entries).
    cache_threshold     : Cosine similarity threshold for cache hit (default 0.95).
    whisper_api_key     : Groq API key for voice transcription.
    query_llm_api_key   : Groq API key for query engine LLM.
    grader_api_key      : Groq API key for relevance grader.
    llm_model           : Groq model for query engine and grader.
    reranker_model      : Cross-encoder model name.
    top_k_bm25          : BM25 results to retrieve (default 10).
    top_k_vector        : Vector results to retrieve (default 10).
    top_k_graph         : Graph results to retrieve (default 15).
    top_k_fused         : Results after RRF fusion (default 15).
    chunk_threshold     : Grader per-chunk pass threshold (default 0.6).
    enable_retry        : Retry once with reformulated query on FAIL (default True).
    show_graph_viz      : Print graph visualisation after graph retrieval (default True).
    verbose             : Print all phase details (default True).
    """

    def __init__(
        self,
        embedding_model,
        graph_store,
        faiss_dir           : Optional[str]  = None,
        bm25_dir            : Optional[str]  = None,
        session_id          : str            = "default",
        memory_dir          : str            = "memory",
        memory_max_turns    : int            = 10,
        cache_capacity      : int            = 128,
        cache_threshold     : float          = 0.95,
        whisper_api_key     : Optional[str]  = None,
        query_llm_api_key   : Optional[str]  = None,
        grader_api_key      : Optional[str]  = None,
        llm_model           : str            = "llama-3.3-70b-versatile",
        reranker_model      : str            = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_bm25          : int            = 10,
        top_k_vector        : int            = 10,
        top_k_graph         : int            = 15,
        top_k_fused         : int            = 15,
        chunk_threshold     : float          = 0.6,
        enable_retry        : bool           = True,
        show_graph_viz      : bool           = True,
        verbose             : bool           = True,
    ):
        self.top_k_bm25    = top_k_bm25
        self.top_k_vector  = top_k_vector
        self.top_k_graph   = top_k_graph
        self.top_k_fused   = top_k_fused
        self.enable_retry  = enable_retry
        self.show_graph_viz= show_graph_viz
        self.verbose       = verbose

        # ── Shared infrastructure ─────────────────────────────────────────────
        self.logger = RetrievalLogger(verbose=verbose)

        self.memory = MemoryStore(
            session_id  = session_id,
            storage_dir = memory_dir,
            max_turns   = memory_max_turns,
            verbose     = verbose,
        )

        self.cache = QueryCache(
            capacity             = cache_capacity,
            similarity_threshold = cache_threshold,
            verbose              = verbose,
        )

        # ── Retrievers ────────────────────────────────────────────────────────
        self.query_processor = QueryProcessor(
            whisper_api_key = whisper_api_key,
            memory_store    = self.memory,
            logger          = self.logger,
            verbose         = verbose,
        )

        self.query_engine = QueryEngine(
            embedding_model = embedding_model,
            llm_api_key     = query_llm_api_key,
            llm_model       = llm_model,
            memory_store    = self.memory,
            logger          = self.logger,
            verbose         = verbose,
        )

        # BM25
        self.bm25 : Optional[BM25Retriever] = None
        if bm25_dir and os.path.exists(os.path.join(bm25_dir, "bm25.pkl")):
            try:
                self.bm25 = BM25Retriever.load(bm25_dir, verbose=verbose)
            except Exception as e:
                self.logger.warn(f"BM25 load failed ({e}) — keyword retrieval disabled")

        # Vector (FAISS)
        self.faiss : Optional[FAISSStore] = None
        if faiss_dir and os.path.exists(os.path.join(faiss_dir, "faiss.index")):
            try:
                self.faiss = FAISSStore.load(faiss_dir, embedding_model, verbose=verbose)
            except Exception as e:
                self.logger.warn(f"FAISS load failed ({e}) — vector retrieval disabled")

        self.vector_retriever = (
            VectorRetriever(self.faiss, logger=self.logger, verbose=verbose)
            if self.faiss else None
        )

        # Graph
        self.graph_retriever = GraphRetriever(
            graph_store = graph_store,
            logger      = self.logger,
            verbose     = verbose,
        )

        self.graph_visualizer = GraphVisualizer(verbose=verbose)

        # Fusion
        self.hybrid = HybridRetriever(logger=self.logger, top_k=top_k_fused, verbose=verbose)

        # Reranker
        self.reranker = Reranker(
            model_name      = reranker_model,
            embedding_model = embedding_model,
            logger          = self.logger,
            verbose         = verbose,
        )

        # Grader
        self.grader = Grader(
            llm_api_key     = grader_api_key,
            llm_model       = llm_model,
            chunk_threshold = chunk_threshold,
            memory_store    = self.memory,
            logger          = self.logger,
            verbose         = verbose,
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        user_input  : str,
        top_k_bm25  : Optional[int] = None,
        top_k_vector: Optional[int] = None,
        top_k_graph : Optional[int] = None,
    ) -> GradedResult:
        """
        Run the full retrieval pipeline.

        Args
        ----
        user_input   : Text query or path to audio file.
        top_k_bm25   : Override default BM25 top-k for this call.
        top_k_vector : Override default vector top-k for this call.
        top_k_graph  : Override default graph top-k for this call.

        Returns
        -------
        GradedResult — ready for the answer generation phase.
        If verdict is FAIL, .reformulation contains a suggested retry query.
        """
        t0    = time.time()
        trace = RetrievalTrace(query_raw=user_input)

        result = self._run_once(
            user_input   = user_input,
            trace        = trace,
            top_k_bm25   = top_k_bm25  or self.top_k_bm25,
            top_k_vector = top_k_vector or self.top_k_vector,
            top_k_graph  = top_k_graph  or self.top_k_graph,
        )

        # ── Adaptive retry on FAIL ────────────────────────────────────────────
        if (
            self.enable_retry
            and result.verdict == GraderVerdict.FAIL
            and result.reformulation
        ):
            self.logger.warn(
                f"Verdict FAIL — retrying with reformulation: "
                f"{result.reformulation[:80]}"
            )
            retry_trace = RetrievalTrace(query_raw=result.reformulation)
            result = self._run_once(
                user_input   = result.reformulation,
                trace        = retry_trace,
                top_k_bm25   = (top_k_bm25  or self.top_k_bm25)  + 5,
                top_k_vector = (top_k_vector or self.top_k_vector) + 5,
                top_k_graph  = (top_k_graph  or self.top_k_graph)  + 5,
            )
            result.trace = retry_trace

        # ── Final trace summary ────────────────────────────────────────────────
        result.trace = trace
        self.logger.print_trace(trace)

        return result

    # ── Internal single-pass run ──────────────────────────────────────────────

    def _run_once(
        self,
        user_input   : str,
        trace        : RetrievalTrace,
        top_k_bm25   : int,
        top_k_vector : int,
        top_k_graph  : int,
    ) -> GradedResult:
        """Single pass of the retrieval pipeline (called for initial + retry)."""

        # ── Stage 1: Query processing ─────────────────────────────────────────
        rep = self.query_processor.process(user_input, trace=trace)

        # ── Stage 2: Query engine (intent, entities, embedding) ───────────────
        rep = self.query_engine.process(rep, trace=trace)

        # ── Stage 3: Cache check (after embedding is available) ───────────────
        if rep.embedding is not None:
            cached = self.cache.get(rep.embedding)
            if cached is not None:
                trace.cache_hit = True
                return cached

        # ── Stage 4: Parallel retrieval ───────────────────────────────────────
        bm25_results   = []
        vector_results = []

        if self.bm25:
            bm25_results = self.bm25.search(rep, top_k=top_k_bm25, trace=trace)
        else:
            self.logger.warn("BM25 retriever not available")

        if self.vector_retriever:
            vector_results = self.vector_retriever.search(rep, top_k=top_k_vector, trace=trace)
        else:
            self.logger.warn("Vector retriever not available")

        graph_results, traversal = self.graph_retriever.search(
            rep, top_k=top_k_graph, trace=trace
        )

        # ── Stage 5: Graph visualisation ──────────────────────────────────────
        if self.show_graph_viz:
            self.graph_visualizer.print_graph(
                traversal      = traversal,
                query_entities = rep.entities,
            )

        # ── Stage 6: Hybrid fusion (RRF) ──────────────────────────────────────
        fused = self.hybrid.fuse(
            bm25_results   = bm25_results,
            vector_results = vector_results,
            graph_results  = graph_results,
            rep            = rep,
            trace          = trace,
        )

        # ── Stage 7: Reranking ────────────────────────────────────────────────
        reranked = self.reranker.rerank(
            query   = rep.normalised_text,
            results = fused,
            trace   = trace,
        )

        # ── Stage 8: Grader ───────────────────────────────────────────────────
        result = self.grader.grade(
            reranked = reranked,
            rep      = rep,
            trace    = trace,
        )

        # ── Stage 9: Cache store (only on PASS/PARTIAL) ───────────────────────
        if (
            rep.embedding is not None
            and result.verdict in (GraderVerdict.PASS, GraderVerdict.PARTIAL)
        ):
            self.cache.put(rep.embedding, result)

        return result

    # ── Memory management ─────────────────────────────────────────────────────

    def record_turn(
        self,
        user_query      : str,
        ai_response     : str,
        graded_result   : Optional[GradedResult] = None,
    ) -> None:
        """
        Record a completed conversation turn in memory.

        Call this AFTER the answer generation phase with the generated response.

        Args
        ----
        user_query    : The original user query text.
        ai_response   : The generated answer.
        graded_result : The GradedResult from this run (for chunk tracking).
        """
        chunk_ids = []
        entities  = []
        intent    = "unknown"

        if graded_result:
            chunk_ids = [
                gc.reranked.fused.chunk.chunk_id
                for gc in graded_result.passed_chunks
            ]
            entities = graded_result.query.entities
            intent   = graded_result.query.intent.value

        self.memory.add_turn(
            user_query       = user_query,
            ai_response      = ai_response,
            retrieved_chunks = chunk_ids,
            intent           = intent,
            entities         = entities,
        )

    # ── Index building helpers ────────────────────────────────────────────────

    def build_indices(self, chunks: list, save_dir: str = ".") -> None:
        """
        Build BM25 and FAISS indices from a list of Chunk objects.

        Call this once after ingestion.  Both indices are saved to disk.

        Args
        ----
        chunks   : List of Chunk objects from the chunking module.
        save_dir : Directory to save both index files.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        self.logger.phase_start("Index Builder")

        # BM25
        if self.bm25 is None:
            self.bm25 = BM25Retriever(logger=self.logger, verbose=self.verbose)
        self.bm25.build(chunks)
        self.bm25.save(save_dir)

        # FAISS
        if self.faiss is None and self.vector_retriever is None:
            emb_model = self.query_engine.embedding_model
            self.faiss = FAISSStore(emb_model, verbose=self.verbose)
            self.faiss.build_from_chunks(chunks)
            self.faiss.save(save_dir)
            self.vector_retriever = VectorRetriever(
                self.faiss, logger=self.logger, verbose=self.verbose
            )

        self.logger.success(f"Both indices saved to '{save_dir}/'")

    def cache_stats(self) -> dict:
        """Return query cache hit/miss statistics."""
        return self.cache.stats()

    def memory_summary(self) -> dict:
        """Return memory store summary."""
        return {
            "session_id" : self.memory.session_id,
            "turns"      : self.memory.turn_count,
            "entities"   : self.memory.get_recent_entities(n_turns=5),
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalPipeline(\n"
            f"  bm25={'✓' if self.bm25 else '✗'}  "
            f"vector={'✓' if self.vector_retriever else '✗'}  "
            f"graph=✓\n"
            f"  memory={self.memory}\n"
            f"  cache={self.cache}\n"
            f")"
        )