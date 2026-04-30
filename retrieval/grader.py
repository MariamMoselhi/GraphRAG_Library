"""
Relevance grader — quality gate before answer generation.

Scientific rationale
--------------------
Retrieval does not guarantee relevance.  A retriever may return chunks that
are topically adjacent but do not actually answer the query.  Without a gate,
the generation LLM is forced to answer with potentially irrelevant context,
which is a known hallucination trigger in RAG systems.

The grader implements a two-level check:
  1. Per-chunk relevance scoring — each chunk is scored 0-1.
     Chunks below ``chunk_threshold`` are marked as failed.
  2. Aggregate verdict — if fewer than ``min_passed`` chunks pass,
     the verdict is FAIL and the pipeline triggers query reformulation.

Uses Groq LLM (API key #3 — GROQ_API_KEY_GRADER) so grader calls never
compete with query engine (API key #2) or whisper (API key #1) for rate limits.

Consistency check (memory-aware)
----------------------------------
If a MemoryStore is provided, the grader also checks whether retrieved chunks
contradict facts already established in prior turns.  Contradicting chunks
are flagged (not necessarily rejected) with a note in grader_reason.

Reformulation hints
-------------------
When the verdict is FAIL, the grader LLM generates a suggested query
reformulation — a reworded version of the original that may retrieve
better results on retry.

Batching
--------
All chunks are evaluated in a single LLM call (structured as a JSON list)
to minimise API round-trips.  Groq's context window is large enough for
≤ 15 chunks with 300-char previews.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from .retrieval_context import (
    GradedChunk,
    GradedResult,
    GraderVerdict,
    PhaseStats,
    QueryRepresentation,
    RerankedResult,
    RetrievalTrace,
)
from .memory_store import MemoryStore
from .retrieval_logger import RetrievalLogger


class Grader:
    """
    LLM-powered relevance grader and consistency checker.

    Args
    ----
    llm_api_key      : Groq API key for grading (GROQ_API_KEY_GRADER).
                       Falls back to GROQ_API_KEY_GRADER then GROQ_API_KEY.
    llm_model        : Groq model (default "llama-3.3-70b-versatile").
    chunk_threshold  : Minimum per-chunk relevance score to pass (default 0.6).
    min_passed       : Minimum passing chunks for PASS verdict (default 1).
    partial_threshold: Minimum passing chunks for PARTIAL verdict (default 1).
    memory_store     : Optional MemoryStore for consistency checking.
    logger           : RetrievalLogger instance.
    verbose          : Print phase details.
    """

    def __init__(
        self,
        llm_api_key       : Optional[str]           = None,
        llm_model         : str                     = "llama-3.3-70b-versatile",
        chunk_threshold   : float                   = 0.6,
        min_passed        : int                     = 1,
        partial_threshold : int                     = 1,
        memory_store      : Optional[MemoryStore]   = None,
        logger            : Optional[RetrievalLogger] = None,
        verbose           : bool                    = True,
    ):
        from graph.llm_backend import LLMBackend

        api_key = (
             os.environ.get("GROQ_API_KEY_GRADER")
        )
        self.llm               = LLMBackend(api_key=api_key, model=llm_model, max_tokens=2000)
        self.chunk_threshold   = chunk_threshold
        self.min_passed        = min_passed
        self.partial_threshold = partial_threshold
        self.memory_store      = memory_store
        self.logger            = logger or RetrievalLogger(verbose=verbose)
        self.verbose           = verbose


    def grade(
        self,
        reranked  : List[RerankedResult],
        rep       : QueryRepresentation,
        trace     : Optional[RetrievalTrace] = None,
    ) -> GradedResult:
        """
        Grade all reranked chunks in one LLM call.

        Args
        ----
        reranked : Output from Reranker.rerank().
        rep      : QueryRepresentation with query text and intent.
        trace    : RetrievalTrace to update.

        Returns
        -------
        GradedResult with verdict, passed_chunks, failed_chunks, confidence,
        and optional reformulation suggestion.
        """
        t0 = time.time()
        self.logger.phase_start("Relevance Grader")

        if not reranked:
            self.logger.warn("No chunks to grade — returning FAIL verdict")
            result = GradedResult(
                verdict       = GraderVerdict.FAIL,
                passed_chunks = [],
                failed_chunks = [],
                query         = rep,
                confidence    = 0.0,
                reformulation = self._suggest_reformulation(rep, []),
                trace         = trace,
            )
            if trace:
                trace.verdict            = GraderVerdict.FAIL.value
                trace.overall_confidence = 0.0
            return result

        # Memory consistency context 
        consistency_context = ""
        if self.memory_store and self.memory_store.turn_count > 0:
            consistency_context = self.memory_store.build_context_string(n_turns=2)

        # Single batched LLM call 
        scores_and_reasons = self._grade_batch(
            query                 = rep.normalised_text,
            intent                = rep.intent.value,
            chunks                = reranked,
            consistency_context   = consistency_context,
        )

        # Build GradedChunk objects 
        passed : List[GradedChunk] = []
        failed : List[GradedChunk] = []

        for rr, (score, reason) in zip(reranked, scores_and_reasons):
            gc = GradedChunk(
                reranked        = rr,
                relevance_score = round(score, 3),
                passed          = score >= self.chunk_threshold,
                grader_reason   = reason,
            )
            if gc.passed:
                passed.append(gc)
            else:
                failed.append(gc)

        # Compute aggregate verdict and confidence 
        n_passed    = len(passed)
        all_scores  = [gc.relevance_score for gc in passed + failed]
        confidence  = (
            sum(gc.relevance_score for gc in passed) / max(n_passed, 1)
            if passed else 0.0
        )

        if n_passed >= self.min_passed:
            verdict = GraderVerdict.PASS
        elif n_passed >= self.partial_threshold:
            verdict = GraderVerdict.PARTIAL
        else:
            verdict = GraderVerdict.FAIL

        # Reformulation suggestion on FAIL
        reformulation: Optional[str] = None
        if verdict == GraderVerdict.FAIL:
            reformulation = self._suggest_reformulation(rep, failed)

        elapsed_ms = (time.time() - t0) * 1000

        if trace:
            trace.graded_pass_count  = n_passed
            trace.graded_fail_count  = len(failed)
            trace.verdict            = verdict.value
            trace.overall_confidence = round(confidence, 3)
            trace.reformulation      = reformulation
            trace.add_phase(PhaseStats(
                phase_name   = "Relevance Grader",
                elapsed_ms   = elapsed_ms,
                input_count  = len(reranked),
                output_count = n_passed,
                notes        = f"verdict={verdict.value}  conf={confidence:.2f}",
            ))

        self.logger.print_grader_results(
            passed    = passed,
            failed    = failed,
            verdict   = verdict,
            confidence= confidence,
        )
        self.logger.phase_end(
            "Relevance Grader",
            count      = n_passed,
            elapsed_ms = elapsed_ms,
            notes      = f"verdict={verdict.value.upper()}",
        )

        return GradedResult(
            verdict       = verdict,
            passed_chunks = passed,
            failed_chunks = failed,
            query         = rep,
            confidence    = round(confidence, 3),
            reformulation = reformulation,
            trace         = trace,
        )

    # LLM grading

    def _grade_batch(
        self,
        query               : str,
        intent              : str,
        chunks              : List[RerankedResult],
        consistency_context : str,
    ) -> List[tuple]:
        """
        Grade all chunks in one LLM call.

        Returns list of (score: float, reason: str) aligned with input chunks.
        Falls back to (0.5, "grading unavailable") on failure.
        """
        from graph.node_relation_extractor import _safe_json_parse

        # Build chunk list for the prompt (truncated for token budget)
        chunk_entries = []
        for i, rr in enumerate(chunks):
            preview = rr.fused.chunk.text[:300].replace("\n", " ")
            chunk_entries.append(
                f'  {{"index": {i}, "source": "{rr.fused.chunk.source}", '
                f'"text": "{preview}"}}'
            )

        chunks_json = "[\n" + ",\n".join(chunk_entries) + "\n]"

        consistency_block = ""
        if consistency_context:
            consistency_block = f"""
Prior conversation context (check for contradictions):
{consistency_context}
"""

        prompt = f"""You are a retrieval quality grader for a knowledge-graph RAG system.

Query: "{query}"
Query intent: {intent}
{consistency_block}
Retrieved chunks:
{chunks_json}

For each chunk, evaluate:
1. Is the chunk relevant to answering the query? (primary criterion)
2. Is the chunk consistent with prior conversation context? (secondary)

Scoring guide:
  1.0 — Directly and completely answers the query
  0.8 — Highly relevant, mostly answers the query
  0.6 — Partially relevant, contains useful information
  0.4 — Tangentially related, minor relevance
  0.2 — Topically adjacent but does not help answer
  0.0 — Irrelevant or contradicts established facts

Return ONLY a JSON array (no preamble, no markdown):
[
  {{"index": 0, "score": 0.9, "reason": "directly defines the queried concept"}},
  {{"index": 1, "score": 0.4, "reason": "related topic but doesn't address the specific question"}},
  ...
]
Produce exactly {len(chunks)} entries, one per chunk, in index order."""

        try:
            raw    = self.llm.generate(prompt)
            parsed = _safe_json_parse(raw)

            # Handle both list and {"items": [...]} wrapping
            if isinstance(parsed, dict):
                items = parsed.get("items", list(parsed.values())[0] if parsed else [])
            else:
                items = parsed if isinstance(parsed, list) else []

            # Align by index field
            score_map = {}
            for item in items:
                if isinstance(item, dict):
                    idx    = item.get("index", -1)
                    score  = min(1.0, max(0.0, float(item.get("score", 0.5))))
                    reason = str(item.get("reason", ""))[:200]
                    if 0 <= idx < len(chunks):
                        score_map[idx] = (score, reason)

            # Fill missing with 0.5
            return [
                score_map.get(i, (0.5, "no grader output for this chunk"))
                for i in range(len(chunks))
            ]

        except Exception as e:
            self.logger.warn(f"Grader LLM call failed ({e}) — defaulting to 0.5 for all chunks")
            return [(0.5, f"grading failed: {e}")] * len(chunks)

    def _suggest_reformulation(
        self,
        rep    : QueryRepresentation,
        failed : List[GradedChunk],
    ) -> Optional[str]:
        """
        Generate a reformulated query when no chunks pass grading.

        Uses the grader LLM (same API key) to suggest a clearer version
        of the original query that may retrieve better results.
        """
        failure_reasons = ""
        if failed:
            reasons = [gc.grader_reason for gc in failed[:3] if gc.grader_reason]
            if reasons:
                failure_reasons = (
                    "\n\nRejected chunk reasons:\n- " + "\n- ".join(reasons)
                )

        prompt = (
            f"A RAG retrieval query failed to find relevant documents.\n"
            f"Original query: \"{rep.normalised_text}\"\n"
            f"Detected intent: {rep.intent.value}\n"
            f"Entities found: {rep.entities}\n"
            f"{failure_reasons}\n\n"
            f"Suggest ONE improved query that is more specific and likely to "
            f"retrieve relevant results.  Return ONLY the query text, no explanation."
        )

        try:
            suggestion = self.llm.generate(prompt).strip()
            # Clean up any quotes the model added
            suggestion = suggestion.strip('"\'')
            if self.verbose:
                self.logger.info(f"Reformulation suggested: {suggestion[:80]}")
            return suggestion
        except Exception:
            return None