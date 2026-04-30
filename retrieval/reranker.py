"""
Cross-encoder reranker for the GraphRAG pipeline.

Scientific distinction: bi-encoder vs cross-encoder
----------------------------------------------------
The three retrievers (BM25, vector, graph) are *bi-encoders* or keyword
matchers — they encode the query and documents independently.  This is fast
but misses fine-grained query-document interaction.

A *cross-encoder* jointly encodes (query, document) as a single input,
allowing full attention across both sequences.  This is 10-50× slower per
document but produces much more accurate relevance scores — making it ideal
for reranking a small shortlist (≤ 30 candidates) rather than the full corpus.

Model
-----
cross-encoder/ms-marco-MiniLM-L-6-v2 (default)
  - Trained on the MS MARCO passage ranking benchmark
  - 6 layers, ~22M parameters — fast enough for ≤ 50 candidates
  - Outputs a relevance score; higher = more relevant

Alternative: cross-encoder/ms-marco-MiniLM-L-12-v2 for higher accuracy
  at roughly 2× the latency.

Fallback
--------
If sentence-transformers is not installed, falls back to a simple
cosine similarity reranker using the embedding model.  Much weaker but
prevents a hard failure.

Usage
-----
    reranker = Reranker()
    reranked = reranker.rerank(query="What is attention?", fused_results=[...])
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .retrieval_context import (
    FusedResult,
    PhaseStats,
    RerankedResult,
    RetrievalTrace,
)
from .retrieval_logger import RetrievalLogger

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


class Reranker:
    """
    Cross-encoder reranker.  Refines RRF-fused results for precision.

    Args
    ----
    model_name      : HuggingFace cross-encoder model identifier.
    max_length      : Max token length for (query, doc) concatenation.
    batch_size      : Inference batch size (default 16).
    embedding_model : Fallback embedding model if cross-encoder unavailable.
    logger          : RetrievalLogger instance.
    verbose         : Print phase details.
    """

    def __init__(
        self,
        model_name      : str   = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length      : int   = 512,
        batch_size      : int   = 16,
        embedding_model         = None,
        logger          : Optional[RetrievalLogger] = None,
        verbose         : bool  = True,
    ):
        self.model_name      = model_name
        self.max_length      = max_length
        self.batch_size      = batch_size
        self.embedding_model = embedding_model
        self.logger          = logger or RetrievalLogger(verbose=verbose)
        self.verbose         = verbose
        self._model          = None   # lazy-loaded

        if not _CROSS_ENCODER_AVAILABLE:
            self.logger.warn(
                "sentence-transformers not installed — using cosine fallback reranker. "
                "pip install sentence-transformers  for the cross-encoder."
            )

    # Main entry point

    def rerank(
        self,
        query   : str,
        results : List[FusedResult],
        trace   : Optional[RetrievalTrace] = None,
    ) -> List[RerankedResult]:
        """
        Rerank fused results using the cross-encoder.

        Args
        ----
        query   : Normalised query text (from QueryRepresentation.normalised_text).
        results : FusedResults from HybridRetriever.fuse().
        trace   : RetrievalTrace to update.

        Returns
        -------
        List of RerankedResult sorted by descending rerank_score.
        Each entry records the original rank and the delta for logging.
        """
        t0 = time.time()
        self.logger.phase_start("Reranker")

        if not results:
            self.logger.warn("No results to rerank")
            return []

        # Compute rerank scores
        if _CROSS_ENCODER_AVAILABLE:
            scores = self._cross_encoder_scores(query, results)
        else:
            scores = self._cosine_fallback_scores(query, results)

        # Build RerankedResult objects with original rank tracked
        paired = list(zip(results, scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        reranked: List[RerankedResult] = []
        for new_rank_0based, (fused, score) in enumerate(paired):
            orig_rank = results.index(fused) + 1    # 1-based
            new_rank  = new_rank_0based + 1
            delta     = new_rank - orig_rank

            reranked.append(RerankedResult(
                fused         = fused,
                rerank_score  = round(float(score), 4),
                original_rank = orig_rank,
                final_rank    = new_rank,
                delta         = delta,
            ))

        elapsed_ms = (time.time() - t0) * 1000

        if trace:
            trace.reranked_count = len(reranked)
            trace.add_phase(PhaseStats(
                phase_name   = "Reranker",
                elapsed_ms   = elapsed_ms,
                input_count  = len(results),
                output_count = len(reranked),
                notes        = (
                    f"model={self.model_name.split('/')[-1]}"
                    if _CROSS_ENCODER_AVAILABLE else "fallback=cosine"
                ),
            ))

        self.logger.print_reranker_results(reranked)
        self.logger.phase_end("Reranker", count=len(reranked), elapsed_ms=elapsed_ms)
        return reranked

    # Scoring implementations

    def _cross_encoder_scores(
        self,
        query  : str,
        results: List[FusedResult],
    ) -> List[float]:
        """Score (query, doc) pairs with the cross-encoder model."""
        # Lazy-load model
        if self._model is None:
            if self.verbose:
                print(f"  Loading cross-encoder: {self.model_name}…", end=" ", flush=True)
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
            if self.verbose:
                print("done")

        pairs = [(query, r.chunk.text) for r in results]

        raw_scores = self._model.predict(
            pairs,
            batch_size   = self.batch_size,
            show_progress_bar = False,
        )

        # Sigmoid-normalise to [0, 1] so scores are interpretable
        import math
        normalised = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]
        return normalised

    def _cosine_fallback_scores(
        self,
        query  : str,
        results: List[FusedResult],
    ) -> List[float]:
        """
        Fallback: rerank by cosine similarity between query embedding
        and chunk embeddings.  Much weaker than cross-encoder but never fails.
        """
        if self.embedding_model is None:
            # No embedding model either — return RRF scores as-is
            return [r.rrf_score for r in results]

        q_emb  = self.embedding_model.encode([query]).astype(np.float32)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        scores = []
        for r in results:
            c_emb  = self.embedding_model.encode([r.chunk.text]).astype(np.float32)
            c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-9)
            sim    = float(np.dot(q_norm.flatten(), c_norm.flatten()))
            # Shift from [-1,1] to [0,1]
            scores.append((sim + 1.0) / 2.0)

        return scores