"""
Shared data models for the GraphRAG retrieval phase.

Every module in the retrieval pipeline communicates through these typed
objects — never raw dicts.  This keeps inter-module contracts explicit,
enables IDE autocomplete, and makes debugging significantly easier.

Design principles
-----------------
- Immutability by convention: fields are set at construction time.
  Downstream modules create *new* objects rather than mutating existing ones.
- RetrievalTrace is the single audit log that accumulates across all phases.
  The logger reads from it; nothing else writes to logger directly.
- Score fields use float in [0, 1] by convention.  RRF-fused scores are also
  normalised to this range before being stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# Enumerations
class QueryIntent(str, Enum):
    """
    Detected intent of the user's query.

    FACTUAL     — "What is X?" / definition or description lookup.
    RELATIONAL  — "How does X relate to Y?" / multi-hop graph question.
    PROCEDURAL  — "How do I do X?" / step-by-step.
    COMPARATIVE — "Compare X and Y" / contrast question.
    FOLLOW_UP   — References prior conversation ("elaborate on that").
    UNKNOWN     — Could not determine intent reliably.
    """
    FACTUAL     = "factual"
    RELATIONAL  = "relational"
    PROCEDURAL  = "procedural"
    COMPARATIVE = "comparative"
    FOLLOW_UP   = "follow_up"
    UNKNOWN     = "unknown"


class InputModality(str, Enum):
    """Whether the raw input was audio or text."""
    AUDIO = "audio"
    TEXT  = "text"


class GraderVerdict(str, Enum):
    """Final decision from the relevance grader."""
    PASS        = "pass"       # Sufficient relevant chunks — proceed to generation.
    PARTIAL     = "partial"    # Some chunks relevant; generation can proceed with caveats.
    FAIL        = "fail"       # No relevant chunks — trigger query reformulation.
    REFORMULATE = "reformulate"# Explicit signal to retry with a different query strategy.


# Core result objects 

@dataclass
class QueryRepresentation:
    """
    All representations of the user's query needed by the three retrievers.

    raw_text        : Original text after voice transcription (if applicable).
    normalised_text : Cleaned, lowercased query used downstream.
    intent          : Detected query intent (QueryIntent enum).
    entities        : Named entities extracted from the query
                      (e.g. ["Gradient Descent", "Adam Optimizer"]).
    keywords        : BM25 search terms (stemmed/filtered).
    cypher_query    : Neo4j Cypher query string for graph retrieval.
                      None if query engine could not produce one.
    embedding       : Dense query vector (numpy array) for FAISS search.
                      None before embedding step.
    memory_context  : Injected context from prior conversation turns
                      used for coreference resolution.
    """
    raw_text        : str
    normalised_text : str
    intent          : QueryIntent               = QueryIntent.UNKNOWN
    entities        : List[str]                 = field(default_factory=list)
    keywords        : List[str]                 = field(default_factory=list)
    cypher_query    : Optional[str]             = None
    embedding       : Optional[Any]             = None   # np.ndarray at runtime
    memory_context  : Optional[str]             = None


@dataclass
class RetrievedChunk:
    """
    A single candidate chunk returned by any of the three retrievers.

    chunk_id    : Unique identifier (matches chunk.chunk_id from ingestion).
    text        : The chunk's text content.
    source      : Source file/URL this chunk came from.
    score       : Retriever-specific raw score (BM25 score, cosine sim, etc.).
    retriever   : Which retriever produced this result ("bm25", "vector", "graph").
    metadata    : Any extra fields (page number, slide number, graph node ids, etc.).
    """
    chunk_id    : str
    text        : str
    source      : str
    score       : float
    retriever   : str
    metadata    : Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult:
    """
    A chunk after Reciprocal Rank Fusion across all three retrievers.

    rrf_score       : Fused score from RRF — higher is better.
    contributing    : Which retrievers contributed to this result.
    individual_ranks: Rank each retriever assigned (for debugging).
    chunk           : The underlying RetrievedChunk (text, source, etc.).
    """
    chunk           : RetrievedChunk
    rrf_score       : float
    contributing    : List[str]                   = field(default_factory=list)
    individual_ranks: Dict[str, int]              = field(default_factory=dict)


@dataclass
class RerankedResult:
    """
    A FusedResult after cross-encoder reranking.

    rerank_score    : Cross-encoder relevance score in [0, 1].
    original_rank   : Rank before reranking (from RRF order).
    final_rank      : Rank after reranking.
    delta           : Rank change (negative = moved up, positive = moved down).
    """
    fused           : FusedResult
    rerank_score    : float
    original_rank   : int
    final_rank      : int
    delta           : int = 0


@dataclass
class GradedChunk:
    """
    A RerankedResult after the LLM grader has evaluated its relevance.

    relevance_score : LLM-assigned relevance in [0, 1].
    passed          : Whether this chunk cleared the relevance threshold.
    grader_reason   : Short explanation from the grader LLM.
    """
    reranked        : RerankedResult
    relevance_score : float
    passed          : bool
    grader_reason   : str = ""


@dataclass
class GradedResult:
    """
    Final output of the entire retrieval pipeline.

    verdict         : Overall PASS / PARTIAL / FAIL decision.
    passed_chunks   : Chunks that cleared all filters — ready for generation.
    failed_chunks   : Chunks that did not pass — kept for diagnostics.
    query           : The full QueryRepresentation used.
    confidence      : Aggregate confidence in [0, 1] computed from grader scores.
    reformulation   : Suggested reworded query if verdict is FAIL/REFORMULATE.
    trace           : Full audit log of every phase.
    """
    verdict         : GraderVerdict
    passed_chunks   : List[GradedChunk]
    failed_chunks   : List[GradedChunk]
    query           : QueryRepresentation
    confidence      : float                       = 0.0
    reformulation   : Optional[str]               = None
    trace           : Optional["RetrievalTrace"]  = None

    @property
    def context_text(self) -> str:
        """
        Concatenate passed chunks into a single context string for the
        answer-generation phase.  Chunks are ordered by final_rank.
        """
        sorted_chunks = sorted(
            self.passed_chunks,
            key=lambda g: g.reranked.final_rank
        )
        parts = []
        for i, gc in enumerate(sorted_chunks, 1):
            chunk = gc.reranked.fused.chunk
            parts.append(
                f"[{i}] (source: {chunk.source})\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)


# Graph traversal results 
@dataclass
class GraphTraversalResult:
    """
    Structured output from the graph retriever.

    nodes           : Dicts of entity node properties from Neo4j.
    relationships   : Dicts of relationship properties from Neo4j.
    paths           : Optional list of paths (for path-finding queries).
    cypher_used     : The Cypher query that produced this result.
    traversal_depth : How many hops were traversed.
    """
    nodes           : List[Dict[str, Any]]  = field(default_factory=list)
    relationships   : List[Dict[str, Any]]  = field(default_factory=list)
    paths           : List[Any]             = field(default_factory=list)
    cypher_used     : str                   = ""
    traversal_depth : int                   = 1


# Memory

@dataclass
class ConversationTurn:
    """
    A single turn in the conversation history.

    turn_id         : Sequential integer, 0-based.
    user_query      : Raw user query text (post-transcription).
    ai_response     : The generated answer returned to the user.
    retrieved_chunks: chunk_ids that were used to generate the response.
    timestamp       : ISO-8601 timestamp string.
    query_intent    : Detected intent for this turn.
    entities        : Entities mentioned this turn (for coreference).
    """
    turn_id          : int
    user_query       : str
    ai_response      : str
    retrieved_chunks : List[str]            = field(default_factory=list)
    timestamp        : str                  = ""
    query_intent     : str                  = QueryIntent.UNKNOWN.value
    entities         : List[str]            = field(default_factory=list)


# Audit trace 

@dataclass
class PhaseStats:
    """Timing and count stats for one pipeline phase."""
    phase_name      : str
    elapsed_ms      : float                 = 0.0
    input_count     : int                   = 0
    output_count    : int                   = 0
    notes           : str                   = ""


@dataclass
class RetrievalTrace:
    """
    Full audit log accumulated across the entire retrieval pipeline.

    Every phase appends its PhaseStats here.
    The retrieval_logger reads this to print a structured summary.
    """
    query_raw           : str                       = ""
    input_modality      : str                       = InputModality.TEXT.value
    transcription_text  : Optional[str]             = None
    cache_hit           : bool                      = False

    # Per-retriever raw counts
    bm25_count          : int                       = 0
    vector_count        : int                       = 0
    graph_count         : int                       = 0
    fused_count         : int                       = 0
    reranked_count      : int                       = 0
    graded_pass_count   : int                       = 0
    graded_fail_count   : int                       = 0

    # Graph traversal detail
    graph_nodes_visited : int                       = 0
    graph_rels_visited  : int                       = 0
    graph_hops          : int                       = 0

    # Grader outcome
    verdict             : str                       = GraderVerdict.FAIL.value
    overall_confidence  : float                     = 0.0
    reformulation       : Optional[str]             = None

    # Phase timings
    phases              : List[PhaseStats]          = field(default_factory=list)

    # Memory
    memory_turns_used   : int                       = 0

    def add_phase(self, stats: PhaseStats) -> None:
        self.phases.append(stats)

    @property
    def total_elapsed_ms(self) -> float:
        return sum(p.elapsed_ms for p in self.phases)