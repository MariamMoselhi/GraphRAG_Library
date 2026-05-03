"""
Vector (semantic) retriever for the GraphRAG pipeline.

Wraps FaissVectorStore (vectordb/faiss_store.py) and runs approximate
nearest-neighbour search using the query embedding produced by query_engine.py.

Why a separate wrapper module?
-------------------------------
FaissVectorStore is a storage layer; VectorRetriever is a retrieval *policy*.
Keeping them separate allows the retriever to:
  - Apply a minimum-score threshold (filter near-zero similarity results)
  - Log standardised PhaseStats and update RetrievalTrace
  - Return RetrievedChunk objects (not raw RetrievalResult objects)
  - Be swapped for a different ANN backend without touching FaissVectorStore
"""
from __future__ import annotations

import time
from typing import List, Optional

from .retrieval_context import PhaseStats, QueryRepresentation, RetrievedChunk, RetrievalTrace
from .retrieval_logger import RetrievalLogger

# Use the project's canonical FaissVectorStore (vectordb/faiss_store.py)
from vectordb.faiss_store import FaissVectorStore


class VectorRetriever:
    """
    Semantic retriever using FAISS inner-product (cosine) search.

    Args
    ----
    faiss_store : Pre-built FaissVectorStore instance (vectordb package).
    min_score   : Minimum cosine similarity to include a result (default 0.3).
    logger      : RetrievalLogger instance.
    verbose     : Print phase details.
    """

    def __init__(
        self,
        faiss_store : FaissVectorStore,
        min_score   : float = 0.3,
        logger      : Optional[RetrievalLogger] = None,
        verbose     : bool  = True,
    ):
        self.faiss_store = faiss_store
        self.min_score   = min_score
        self.logger      = logger or RetrievalLogger(verbose=verbose)
        self.verbose     = verbose

    # Search 

    def search(
        self,
        rep   : QueryRepresentation,
        top_k : int = 10,
        trace : Optional[RetrievalTrace] = None,
    ) -> List[RetrievedChunk]:
        """
        Run vector similarity search.

        Args
        ----
        rep   : QueryRepresentation with .embedding set by query_engine.py.
        top_k : Maximum number of results.
        trace : RetrievalTrace to update.

        Returns
        -------
        RetrievedChunks sorted by descending cosine similarity, above min_score.
        """
        t0 = time.time()
        self.logger.phase_start("Vector Retriever")

        if rep.embedding is None:
            self.logger.warn("No query embedding available — skipping vector search")
            return []

        # FaissVectorStore.similarity_search_by_vector returns List[RetrievalResult]
        raw_results = self.faiss_store.similarity_search_by_vector(
            rep.embedding, k=top_k
        )

        # Convert RetrievalResult → RetrievedChunk and apply min_score filter
        results: List[RetrievedChunk] = []
        for rr in raw_results:
            if rr.score < self.min_score:
                continue
            chunk = rr.chunk
            results.append(RetrievedChunk(
                chunk_id  = str(chunk.chunk_id),
                text      = chunk.text,
                source    = chunk.metadata.get("source", "unknown"),
                score     = rr.score,
                retriever = "vector",
                metadata  = dict(chunk.metadata),
            ))

        elapsed_ms = (time.time() - t0) * 1000

        if trace:
            trace.vector_count = len(results)
            trace.add_phase(PhaseStats(
                phase_name   = "Vector Retriever",
                elapsed_ms   = elapsed_ms,
                input_count  = 1,
                output_count = len(results),
                notes        = f"min_score={self.min_score}  index_size={len(self.faiss_store)}",
            ))

        self.logger.print_vector_results(results)
        self.logger.phase_end("Vector Retriever", count=len(results), elapsed_ms=elapsed_ms)
        return results