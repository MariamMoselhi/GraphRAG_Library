"""
Hybrid retriever — fuses BM25, vector, and graph results using
Reciprocal Rank Fusion (RRF).

Scientific basis for RRF
-------------------------
RRF(d) = Σ_r  1 / (k + rank_r(d))

  d      = candidate document
  rank_r = rank assigned by retriever r (1-based)
  k      = smoothing constant (default 60 — empirically optimal per
           Cormack et al. 2009 "Reciprocal Rank Fusion outperforms Condorcet
           and Individual Rank Learning Methods")

Why k=60?
  A document ranked 1st contributes 1/61 ≈ 0.016.
  A document ranked 60th contributes 1/120 ≈ 0.008 — half the value.
  This gives a smooth decay that does not over-penalise mid-ranked results
  from a single retriever.

Why RRF instead of score normalisation?
  BM25 scores and cosine similarities live on incompatible scales.
  Normalising (e.g. min-max) is sensitive to outliers and changes with
  corpus size.  RRF bypasses score normalisation entirely by working only
  on rank order — making it robust and parameter-free (except k).

Adaptive weighting (optional)
------------------------------
If ``weights`` are provided, each retriever's rank list is multiplied by
its weight before fusion.  Default: equal weights {bm25:1, vector:1, graph:1}.
This is useful when you know one retriever dominates for a particular intent
(e.g. graph retrieval for RELATIONAL queries).
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

from .retrieval_context import (
    FusedResult,
    PhaseStats,
    QueryIntent,
    QueryRepresentation,
    RetrievedChunk,
    RetrievalTrace,
)
from .retrieval_logger import RetrievalLogger

# Intent-based adaptive weights: (bm25, vector, graph)
_INTENT_WEIGHTS: Dict[QueryIntent, Dict[str, float]] = {
    QueryIntent.FACTUAL    : {"bm25": 1.0, "vector": 1.2, "graph": 1.0},
    QueryIntent.RELATIONAL : {"bm25": 0.7, "vector": 1.0, "graph": 1.5},
    QueryIntent.PROCEDURAL : {"bm25": 1.2, "vector": 1.0, "graph": 0.9},
    QueryIntent.COMPARATIVE: {"bm25": 0.8, "vector": 1.1, "graph": 1.4},
    QueryIntent.FOLLOW_UP  : {"bm25": 0.9, "vector": 1.3, "graph": 1.0},
    QueryIntent.UNKNOWN    : {"bm25": 1.0, "vector": 1.0, "graph": 1.0},
}


class HybridRetriever:
    """
    Reciprocal Rank Fusion combiner for three retriever outputs.

    Args
    ----
    k            : RRF smoothing constant (default 60).
    top_k        : Number of fused results to return (default 15).
    use_adaptive : If True, apply intent-based adaptive weights (default True).
    logger       : RetrievalLogger instance.
    verbose      : Print phase details.
    """

    def __init__(
        self,
        k            : int   = 60,
        top_k        : int   = 15,
        use_adaptive : bool  = True,
        logger       : Optional[RetrievalLogger] = None,
        verbose      : bool  = True,
    ):
        self.k            = k
        self.top_k        = top_k
        self.use_adaptive = use_adaptive
        self.logger       = logger or RetrievalLogger(verbose=verbose)
        self.verbose      = verbose

    # Main entry point 

    def fuse(
        self,
        bm25_results    : List[RetrievedChunk],
        vector_results  : List[RetrievedChunk],
        graph_results   : List[RetrievedChunk],
        rep             : Optional[QueryRepresentation] = None,
        trace           : Optional[RetrievalTrace]      = None,
    ) -> List[FusedResult]:
        """
        Combine three ranked lists via RRF.

        Args
        ----
        bm25_results   : Ranked list from BM25Retriever.search()
        vector_results : Ranked list from VectorRetriever.search()
        graph_results  : Ranked list from GraphRetriever.search()
        rep            : QueryRepresentation (used for adaptive weights).
        trace          : RetrievalTrace to update.

        Returns
        -------
        List of FusedResult sorted by descending RRF score, length ≤ top_k.
        """
        t0 = time.time()
        self.logger.phase_start("Hybrid Retriever (RRF Fusion)")

        # Determine adaptive weights
        intent  = rep.intent if rep else QueryIntent.UNKNOWN
        weights = (
            _INTENT_WEIGHTS.get(intent, _INTENT_WEIGHTS[QueryIntent.UNKNOWN])
            if self.use_adaptive else
            {"bm25": 1.0, "vector": 1.0, "graph": 1.0}
        )

        self.logger.info(
            f"RRF k={self.k}  adaptive={self.use_adaptive}  intent={intent.value}  "
            f"weights=bm25×{weights['bm25']} vector×{weights['vector']} graph×{weights['graph']}"
        )

        # Accumulate per-chunk RRF scores 
        # rrf_scores    : chunk_id → accumulated RRF score
        # chunk_registry: chunk_id → RetrievedChunk (first occurrence wins)
        # contributing  : chunk_id → set of retriever names
        # indiv_ranks   : chunk_id → {retriever: rank}

        rrf_scores     : Dict[str, float]          = defaultdict(float)
        chunk_registry : Dict[str, RetrievedChunk] = {}
        contributing   : Dict[str, set]            = defaultdict(set)
        indiv_ranks    : Dict[str, dict]           = defaultdict(dict)

        retriever_lists = [
            ("bm25",   bm25_results,   weights["bm25"]),
            ("vector", vector_results, weights["vector"]),
            ("graph",  graph_results,  weights["graph"]),
        ]

        for retriever_name, ranked_list, weight in retriever_lists:
            for rank_0based, chunk in enumerate(ranked_list):
                rank          = rank_0based + 1    # 1-based
                rrf_increment = weight / (self.k + rank)

                rrf_scores[chunk.chunk_id]   += rrf_increment
                contributing[chunk.chunk_id].add(retriever_name)
                indiv_ranks[chunk.chunk_id][retriever_name] = rank

                # Register chunk (first retriever to see it owns the metadata)
                if chunk.chunk_id not in chunk_registry:
                    chunk_registry[chunk.chunk_id] = chunk

        # Deduplication: prefer the highest-score chunk copy 
        # (Different retrievers may return the same chunk_id with different scores.
        #  We already merged scores via RRF; the chunk text/source comes from
        #  chunk_registry which stores the first occurrence.)

        # Sort and cap 
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_ids    = sorted_ids[: self.top_k]

        results: List[FusedResult] = []
        for cid in top_ids:
            chunk = chunk_registry[cid]
            results.append(FusedResult(
                chunk           = chunk,
                rrf_score       = round(rrf_scores[cid], 6),
                contributing    = sorted(contributing[cid]),
                individual_ranks= indiv_ranks[cid],
            ))

        elapsed_ms = (time.time() - t0) * 1000

        # Stats
        n_multi = sum(1 for r in results if len(r.contributing) > 1)
        self.logger.info(
            f"Fused {len(results)} unique results  "
            f"(multi-retriever overlap: {n_multi})"
        )

        if trace:
            trace.fused_count = len(results)
            trace.add_phase(PhaseStats(
                phase_name   = "Hybrid Retriever (RRF)",
                elapsed_ms   = elapsed_ms,
                input_count  = (
                    len(bm25_results) + len(vector_results) + len(graph_results)
                ),
                output_count = len(results),
                notes        = f"k={self.k}  multi_overlap={n_multi}",
            ))

        self.logger.print_fusion_results(results)
        self.logger.phase_end("Hybrid Retriever (RRF)", count=len(results), elapsed_ms=elapsed_ms)
        return results