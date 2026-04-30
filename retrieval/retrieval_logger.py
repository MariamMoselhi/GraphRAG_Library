"""
Structured, phase-by-phase logger for the GraphRAG retrieval pipeline.

Uses ANSI escape codes for color-coded console output:
  Green  — success / passed
  Yellow — warnings / partial
  Red    — failures / rejected chunks
  Cyan   — phase headers
  White  — neutral info

All logging goes through the RetrievalLogger class, which reads from
a RetrievalTrace object populated by each pipeline phase.  Nothing in
the pipeline calls print() directly — everything routes through here.

Usage
-----
    from retrieval.retrieval_logger import RetrievalLogger
    from retrieval.retrieval_context import RetrievalTrace

    logger = RetrievalLogger(verbose=True)
    logger.print_trace(trace)           # full structured summary
    logger.phase_start("BM25 Retriever")
    logger.phase_end("BM25 Retriever", count=15, elapsed_ms=42.3)
"""
from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from typing import List, Optional

from .retrieval_context import (
    GradedChunk,
    GraderVerdict,
    RetrievalTrace,
    RetrievedChunk,
    FusedResult,
    RerankedResult,
)


# ANSI colour helpers

def _supports_color() -> bool:
    """True if the terminal supports ANSI escape codes."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _C:
    """ANSI colour codes — disabled gracefully when terminal doesn't support them."""
    _on = _supports_color()

    RESET  = "\033[0m"   if _on else ""
    BOLD   = "\033[1m"   if _on else ""
    DIM    = "\033[2m"   if _on else ""
    GREEN  = "\033[92m"  if _on else ""
    YELLOW = "\033[93m"  if _on else ""
    RED    = "\033[91m"  if _on else ""
    CYAN   = "\033[96m"  if _on else ""
    BLUE   = "\033[94m"  if _on else ""
    WHITE  = "\033[97m"  if _on else ""
    PURPLE = "\033[95m"  if _on else ""


# RetrievalLogger

class RetrievalLogger:
    """
    Structured console logger for all retrieval pipeline phases.

    Args
    ----
    verbose     : If True, print per-chunk detail (text previews, scores).
                  If False, print summary lines only.
    width       : Console column width for box-drawing (default 72).
    """

    def __init__(self, verbose: bool = True, width: int = 72):
        self.verbose = verbose
        self.width   = width

    # Public phase printers

    def phase_start(self, name: str) -> None:
        """Print a phase header banner."""
        bar = "─" * self.width
        print(f"\n{_C.CYAN}{bar}{_C.RESET}")
        print(f"{_C.CYAN}{_C.BOLD}  ▶  {name.upper()}{_C.RESET}")
        print(f"{_C.CYAN}{bar}{_C.RESET}")

    def phase_end(
        self,
        name        : str,
        count       : int,
        elapsed_ms  : float,
        notes       : str = "",
    ) -> None:
        """Print a one-line phase completion summary."""
        note_str = f"  {_C.DIM}{notes}{_C.RESET}" if notes else ""
        print(
            f"  {_C.GREEN}✓{_C.RESET} {name}: "
            f"{_C.BOLD}{count}{_C.RESET} result(s)  "
            f"{_C.DIM}[{elapsed_ms:.1f} ms]{_C.RESET}"
            f"{note_str}"
        )

    def info(self, msg: str) -> None:
        print(f"  {_C.BLUE}ℹ{_C.RESET}  {msg}")

    def warn(self, msg: str) -> None:
        print(f"  {_C.YELLOW}⚠{_C.RESET}  {msg}")

    def error(self, msg: str) -> None:
        print(f"  {_C.RED}✗{_C.RESET}  {msg}")

    def success(self, msg: str) -> None:
        print(f"  {_C.GREEN}✔{_C.RESET}  {msg}")

    # Query representation

    def print_query(
        self,
        raw_text    : str,
        intent      : str,
        entities    : List[str],
        keywords    : List[str],
        cypher      : Optional[str],
        from_memory : bool = False,
        cache_hit   : bool = False,
    ) -> None:
        """Print the parsed query representation."""
        self.phase_start("Query Representation")

        if cache_hit:
            print(f"  {_C.YELLOW}★ CACHE HIT — returning cached retrieval result{_C.RESET}")
            return

        print(f"  {_C.WHITE}Query   :{_C.RESET} {raw_text}")
        print(f"  {_C.WHITE}Intent  :{_C.RESET} {_C.PURPLE}{intent}{_C.RESET}")

        if from_memory:
            print(f"  {_C.YELLOW}Memory context injected (coreference resolution){_C.RESET}")

        if entities:
            print(f"  {_C.WHITE}Entities:{_C.RESET} {', '.join(entities)}")
        else:
            self.warn("No entities extracted — graph retrieval may be limited")

        if keywords:
            print(f"  {_C.WHITE}Keywords:{_C.RESET} {', '.join(keywords)}")

        if cypher:
            print(f"  {_C.WHITE}Cypher  :{_C.RESET}")
            for line in cypher.strip().splitlines():
                print(f"            {_C.DIM}{line}{_C.RESET}")
        else:
            self.warn("No Cypher query generated — graph traversal will use entity lookup only")

    # BM25 results
    
    def print_bm25_results(self, chunks: List[RetrievedChunk]) -> None:
        if not self.verbose or not chunks:
            return
        print(f"\n  {_C.WHITE}BM25 top results:{_C.RESET}")
        for i, c in enumerate(chunks[:5], 1):
            preview = _preview(c.text, 80)
            print(
                f"    [{i}] score={_C.YELLOW}{c.score:.3f}{_C.RESET}  "
                f"src={_C.DIM}{c.source}{_C.RESET}\n"
                f"        {preview}"
            )

    # Vector results

    def print_vector_results(self, chunks: List[RetrievedChunk]) -> None:
        if not self.verbose or not chunks:
            return
        print(f"\n  {_C.WHITE}Vector top results:{_C.RESET}")
        for i, c in enumerate(chunks[:5], 1):
            preview = _preview(c.text, 80)
            print(
                f"    [{i}] cos_sim={_C.BLUE}{c.score:.4f}{_C.RESET}  "
                f"src={_C.DIM}{c.source}{_C.RESET}\n"
                f"        {preview}"
            )

    # Graph traversal

    def print_graph_results(
        self,
        chunks          : List[RetrievedChunk],
        nodes_visited   : int,
        rels_visited    : int,
        hops            : int,
    ) -> None:
        print(
            f"  {_C.WHITE}Graph traversal:{_C.RESET} "
            f"{nodes_visited} nodes · {rels_visited} relationships · "
            f"{hops} hop(s)"
        )
        if self.verbose and chunks:
            print(f"\n  {_C.WHITE}Graph top results:{_C.RESET}")
            for i, c in enumerate(chunks[:5], 1):
                preview = _preview(c.text, 80)
                print(
                    f"    [{i}] score={_C.PURPLE}{c.score:.4f}{_C.RESET}  "
                    f"src={_C.DIM}{c.source}{_C.RESET}\n"
                    f"        {preview}"
                )

    # Fusion

    def print_fusion_results(self, results: List[FusedResult]) -> None:
        if not self.verbose or not results:
            return
        print(f"\n  {_C.WHITE}RRF fused top results:{_C.RESET}")
        for i, r in enumerate(results[:8], 1):
            contributors = "+".join(r.contributing)
            ranks_str    = "  ".join(
                f"{k}=#{v}" for k, v in r.individual_ranks.items()
            )
            preview = _preview(r.chunk.text, 80)
            print(
                f"    [{i}] rrf={_C.GREEN}{r.rrf_score:.4f}{_C.RESET}  "
                f"via=[{_C.CYAN}{contributors}{_C.RESET}]  {_C.DIM}{ranks_str}{_C.RESET}\n"
                f"        {preview}"
            )

    # Reranker

    def print_reranker_results(self, results: List[RerankedResult]) -> None:
        if not self.verbose or not results:
            return
        print(f"\n  {_C.WHITE}Reranker output:{_C.RESET}")
        for r in results[:8]:
            delta_str = ""
            if r.delta < 0:
                delta_str = f"{_C.GREEN} ↑{abs(r.delta)}{_C.RESET}"
            elif r.delta > 0:
                delta_str = f"{_C.RED} ↓{r.delta}{_C.RESET}"
            preview = _preview(r.fused.chunk.text, 80)
            print(
                f"    [{r.final_rank}] cross={_C.YELLOW}{r.rerank_score:.4f}{_C.RESET}"
                f"{delta_str}  "
                f"src={_C.DIM}{r.fused.chunk.source}{_C.RESET}\n"
                f"        {preview}"
            )

    # Grader

    def print_grader_results(
        self,
        passed   : List[GradedChunk],
        failed   : List[GradedChunk],
        verdict  : GraderVerdict,
        confidence: float,
    ) -> None:
        verdict_color = {
            GraderVerdict.PASS       : _C.GREEN,
            GraderVerdict.PARTIAL    : _C.YELLOW,
            GraderVerdict.FAIL       : _C.RED,
            GraderVerdict.REFORMULATE: _C.RED,
        }.get(verdict, _C.WHITE)

        print(
            f"\n  {_C.WHITE}Grader verdict:{_C.RESET} "
            f"{verdict_color}{_C.BOLD}{verdict.value.upper()}{_C.RESET}  "
            f"confidence={_C.WHITE}{confidence:.2f}{_C.RESET}"
        )

        if passed and self.verbose:
            print(f"\n  {_C.GREEN}Passed chunks ({len(passed)}):{_C.RESET}")
            for gc in passed:
                chunk = gc.reranked.fused.chunk
                print(
                    f"    {_C.GREEN}✔{_C.RESET} [{gc.reranked.final_rank}] "
                    f"relevance={gc.relevance_score:.2f}  "
                    f"src={_C.DIM}{chunk.source}{_C.RESET}\n"
                    f"      reason: {_C.DIM}{gc.grader_reason}{_C.RESET}\n"
                    f"      {_preview(chunk.text, 80)}"
                )

        if failed and self.verbose:
            print(f"\n  {_C.RED}Rejected chunks ({len(failed)}):{_C.RESET}")
            for gc in failed:
                chunk = gc.reranked.fused.chunk
                print(
                    f"    {_C.RED}✗{_C.RESET} [{gc.reranked.final_rank}] "
                    f"relevance={gc.relevance_score:.2f}  "
                    f"src={_C.DIM}{chunk.source}{_C.RESET}\n"
                    f"      reason: {_C.DIM}{gc.grader_reason}{_C.RESET}"
                )

    # Full trace summary

    def print_trace(self, trace: RetrievalTrace) -> None:
        """
        Print the complete pipeline summary from a finished RetrievalTrace.
        Called at the end of retrieval_pipeline.py after all phases complete.
        """
        bar  = "═" * self.width
        bar2 = "─" * self.width

        print(f"\n{_C.CYAN}{bar}{_C.RESET}")
        print(f"{_C.CYAN}{_C.BOLD}  RETRIEVAL PIPELINE — COMPLETE{_C.RESET}")
        print(f"{_C.CYAN}{bar}{_C.RESET}")

        # Query line
        print(f"  {_C.WHITE}Query  :{_C.RESET} {trace.query_raw}")
        print(f"  {_C.WHITE}Modality:{_C.RESET} {trace.input_modality.upper()}", end="")
        if trace.transcription_text:
            print(f"  →  transcribed: {_preview(trace.transcription_text, 50)}", end="")
        print()

        if trace.cache_hit:
            print(f"  {_C.YELLOW}★ Served from cache{_C.RESET}")

        if trace.memory_turns_used:
            print(f"  {_C.BLUE} Memory: {trace.memory_turns_used} prior turn(s) injected{_C.RESET}")

        print(f"\n  {_C.WHITE}Retrieval counts:{_C.RESET}")
        print(f"  {bar2}")
        _row("BM25",          trace.bm25_count,          _C.YELLOW)
        _row("Vector",        trace.vector_count,         _C.BLUE)
        _row("Graph",         trace.graph_count,          _C.PURPLE)
        _row("After fusion",  trace.fused_count,          _C.GREEN)
        _row("After rerank",  trace.reranked_count,       _C.GREEN)
        _row("Grader PASS",   trace.graded_pass_count,    _C.GREEN)
        _row("Grader FAIL",   trace.graded_fail_count,    _C.RED)

        print(f"\n  {_C.WHITE}Graph traversal:{_C.RESET}")
        print(f"  {bar2}")
        print(f"    Nodes visited    : {trace.graph_nodes_visited}")
        print(f"    Relations visited: {trace.graph_rels_visited}")
        print(f"    Hops             : {trace.graph_hops}")

        print(f"\n  {_C.WHITE}Phase timings:{_C.RESET}")
        print(f"  {bar2}")
        for p in trace.phases:
            bar_w = int((p.elapsed_ms / max(trace.total_elapsed_ms, 1)) * 20)
            bar_str = _C.CYAN + "█" * bar_w + _C.DIM + "░" * (20 - bar_w) + _C.RESET
            print(
                f"    {p.phase_name:<30} {bar_str} "
                f"{_C.DIM}{p.elapsed_ms:6.1f} ms{_C.RESET}"
            )
        print(f"    {'TOTAL':<30} {'':20}   {trace.total_elapsed_ms:.1f} ms")

        # Verdict
        verdict_color = (
            _C.GREEN  if trace.verdict == GraderVerdict.PASS.value  else
            _C.YELLOW if trace.verdict == GraderVerdict.PARTIAL.value else
            _C.RED
        )
        print(f"\n  {_C.WHITE}Final verdict:{_C.RESET} "
              f"{verdict_color}{_C.BOLD}{trace.verdict.upper()}{_C.RESET}  "
              f"confidence={trace.overall_confidence:.2f}")

        if trace.reformulation:
            print(f"  {_C.YELLOW}Suggested reformulation:{_C.RESET} {trace.reformulation}")

        print(f"{_C.CYAN}{bar}{_C.RESET}\n")

    # Memory summary

    def print_memory_summary(
        self,
        turn_count   : int,
        window_size  : int,
        entities_seen: List[str],
    ) -> None:
        self.phase_start("Memory Store")
        print(f"  Conversation turns in store : {turn_count}")
        print(f"  Active window size          : {window_size}")
        if entities_seen:
            print(
                f"  Known entities (recent)     : "
                f"{', '.join(entities_seen[:8])}"
                + (" …" if len(entities_seen) > 8 else "")
            )


# Helpers

def _preview(text: str, width: int) -> str:
    """Return a single-line preview of text, truncated to width chars."""
    flat = " ".join(text.split())
    return flat[:width] + "…" if len(flat) > width else flat


def _row(label: str, count: int, color: str) -> None:
    print(f"    {label:<20} {color}{count:>5}{_C.RESET}")