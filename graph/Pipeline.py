# pipeline.py
"""
KG-RAG Ingestion Pipeline
─────────────────────────
Orchestrates the full ingestion flow:

    chunks
      └─ NodeExtractor          → List[ExtractedNode]
           └─ RelationshipExtractor → List[ExtractedRelationship]
                └─ GraphStore       → Neo4j

Usage (minimal)
───────────────
    from pipeline import Pipeline

    pipeline = Pipeline(
        neo4j_uri      = "bolt://localhost:7687",
        neo4j_user     = "neo4j",
        neo4j_password = "password",
        groq_api_key   = "gsk_...",
        embedding_fn   = my_embed_fn,   # Callable[[List[str]], np.ndarray]
    )

    stats = pipeline.run(chunks)
    print(stats)

Usage (advanced — bring your own components)
────────────────────────────────────────────
    from graph import NodeExtractor, RelationshipExtractor, GraphStore, LLMBackend
    from pipeline import Pipeline

    llm   = LLMBackend(api_key="gsk_...", model="llama-3.3-70b-versatile", batch_size=4)
    store = GraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
    store.init_schema()

    pipeline = Pipeline.from_components(
        node_extractor         = NodeExtractor(llm, embedding_fn),
        relationship_extractor = RelationshipExtractor(llm, mode="constrained"),
        graph_store            = store,
    )

    stats = pipeline.run(chunks)

Incremental ingestion (single chunk)
─────────────────────────────────────
    stats = pipeline.ingest_chunk(chunk)

Environment variables
─────────────────────
    GROQ_API_KEY   — Groq API key (used when api_key is not passed directly)
    NEO4J_URI      — bolt://localhost:7687
    NEO4J_USER     — neo4j
    NEO4J_PASSWORD — (your password)
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

import numpy as np

from graph import (
    ExtractedNode,
    ExtractedRelationship,
    GraphStore,
    LLMBackend,
    NodeExtractor,
    RelationshipExtractor,
)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    """
    Summary of a single pipeline.run() call.

    Attributes
    ----------
    chunks_processed        : Number of input chunks processed.
    nodes_extracted         : Unique nodes extracted (after deduplication).
    nodes_written           : Nodes successfully written to Neo4j.
    relationships_extracted : Relationships extracted (before dedup).
    relationships_written   : Relationships successfully written to Neo4j.
    cross_doc_relationships : Cross-document relationships written.
    elapsed_seconds         : Wall-clock time for the full run.
    errors                  : List of non-fatal warning messages.
    """
    chunks_processed        : int   = 0
    nodes_extracted         : int   = 0
    nodes_written           : int   = 0
    relationships_extracted : int   = 0
    relationships_written   : int   = 0
    cross_doc_relationships : int   = 0
    elapsed_seconds         : float = 0.0
    errors                  : List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'─' * 50}\n"
            f"  Pipeline run complete\n"
            f"{'─' * 50}\n"
            f"  Chunks processed        : {self.chunks_processed}\n"
            f"  Nodes extracted         : {self.nodes_extracted}\n"
            f"  Nodes written to Neo4j  : {self.nodes_written}\n"
            f"  Relationships extracted : {self.relationships_extracted}\n"
            f"  Relationships written   : {self.relationships_written}\n"
            f"  Cross-doc relationships : {self.cross_doc_relationships}\n"
            f"  Elapsed time            : {self.elapsed_seconds:.1f}s\n"
            + (f"  Warnings                : {len(self.errors)}\n" if self.errors else "")
            + f"{'─' * 50}"
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """
    End-to-end KG-RAG ingestion pipeline.

    Wires together NodeExtractor → RelationshipExtractor → GraphStore
    and exposes a single .run(chunks) entry point.

    Construction
    ────────────
    Use Pipeline(...) for the default configuration, or
    Pipeline.from_components(...) to inject pre-built components.

    Args (default constructor)
    ──────────────────────────
    neo4j_uri           : Neo4j Bolt URI. Falls back to NEO4J_URI env var.
    neo4j_user          : Neo4j username. Falls back to NEO4J_USER env var.
    neo4j_password      : Neo4j password. Falls back to NEO4J_PASSWORD env var.
    neo4j_database      : Neo4j database (default "neo4j").
    groq_api_key        : Groq API key. Falls back to GROQ_API_KEY env var.
    groq_model          : Groq model string (default llama-3.3-70b-versatile).
    embedding_fn        : Callable[[List[str]], np.ndarray].
                          Must accept a list of strings and return a 2-D numpy
                          array of shape (len(texts), embedding_dim).
    embedding_dim       : Dimensionality of embedding_fn output (default 384).
    llm_batch_size      : Chunks to combine per Groq API call (default 1).
    llm_max_tokens      : Max tokens per LLM call (default 1024).
    rel_mode            : "constrained" | "unconstrained" (default "constrained").
    rel_confidence      : Minimum confidence to keep a relationship (default 0.6).
    node_max_workers    : Thread pool size for NodeExtractor (default 2).
    graph_batch_size    : Neo4j write batch size (default 256).
    extract_cross_doc   : Whether to run cross-document relationship pass (default True).
    show_progress       : Print per-step progress to stdout (default True).
    """

    def __init__(
        self,
        # Neo4j
        neo4j_uri          : Optional[str]      = None,
        neo4j_user         : Optional[str]      = None,
        neo4j_password     : Optional[str]      = None,
        neo4j_database     : str                = "neo4j",
        # LLM
        groq_api_key       : Optional[str]      = None,
        groq_model         : Optional[str]      = None,
        # Embeddings
        embedding_fn       : Optional[Callable] = None,
        embedding_dim      : int                = 384,
        # Tuning
        llm_batch_size     : int                = 1,
        llm_max_tokens     : int                = 1024,
        rel_mode           : Literal["constrained", "unconstrained"] = "constrained",
        rel_confidence     : float              = 0.6,
        node_max_workers   : int                = 2,
        graph_batch_size   : int                = 256,
        extract_cross_doc  : bool               = True,
        show_progress      : bool               = True,
    ):
        if embedding_fn is None:
            raise ValueError(
                "embedding_fn is required. "
                "Pass a callable that accepts List[str] and returns np.ndarray."
            )

        # Resolve connection params from env if not provided
        uri      = neo4j_uri      or os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
        user     = neo4j_user     or os.environ.get("NEO4J_USER",     "neo4j")
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "")

        llm = LLMBackend(
            api_key    = groq_api_key,
            model      = groq_model,
            max_tokens = llm_max_tokens,
            batch_size = llm_batch_size,
        )

        node_extractor = NodeExtractor(
            llm         = llm,
            embedding_fn= embedding_fn,
            max_workers = node_max_workers,
        )

        relationship_extractor = RelationshipExtractor(
            llm                  = llm,
            mode                 = rel_mode,
            confidence_threshold = rel_confidence,
        )

        store = GraphStore(
            uri          = uri,
            user         = user,
            password     = password,
            database     = neo4j_database,
            embedding_dim= embedding_dim,
            batch_size   = graph_batch_size,
        )
        store.init_schema()

        self._node_extractor         = node_extractor
        self._relationship_extractor = relationship_extractor
        self._store                  = store
        self._extract_cross_doc      = extract_cross_doc
        self._show_progress          = show_progress

    # ── Alternative constructor ───────────────────────────────────────────────

    @classmethod
    def from_components(
        cls,
        node_extractor        : NodeExtractor,
        relationship_extractor: RelationshipExtractor,
        graph_store           : GraphStore,
        extract_cross_doc     : bool = True,
        show_progress         : bool = True,
    ) -> "Pipeline":
        """
        Construct a Pipeline from pre-built component instances.

        Useful when you want fine-grained control over each component,
        or when you already hold an open GraphStore connection.

        Note: graph_store.init_schema() is NOT called automatically here —
        call it yourself before passing the store in, or ensure the schema
        already exists.
        """
        instance = object.__new__(cls)
        instance._node_extractor         = node_extractor
        instance._relationship_extractor = relationship_extractor
        instance._store                  = graph_store
        instance._extract_cross_doc      = extract_cross_doc
        instance._show_progress          = show_progress
        return instance

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, chunks: list) -> PipelineStats:
        """
        Run the full ingestion pipeline over a list of chunks.

        Steps
        ─────
        1. Extract nodes from all chunks (parallel, with deduplication + embedding).
        2. Build a global node_id_map (name → node_id) for relationship resolution.
        3. Write nodes to Neo4j.
        4. Extract relationships per chunk (batched LLM calls).
        5. Deduplicate relationships.
        6. Optionally extract cross-document relationships.
        7. Write all relationships to Neo4j.

        Args
        ────
        chunks : List of Chunk objects. Each chunk must have:
                 - chunk.text         (str)
                 - chunk.chunk_id     (str or int)
                 - chunk.metadata     (dict) with key "source" set.

        Returns
        ───────
        PipelineStats with counts and elapsed time.
        """
        if not chunks:
            warnings.warn("pipeline.run() called with an empty chunk list.")
            return PipelineStats()

        stats   = PipelineStats(chunks_processed=len(chunks))
        t_start = time.time()

        # ── Step 1: Node extraction ───────────────────────────────────────────
        self._log("─" * 50)
        self._log(f"[1/5] Extracting nodes from {len(chunks)} chunk(s)...")

        try:
            nodes: List[ExtractedNode] = self._node_extractor.extract_from_chunks(
                chunks,
                show_progress=self._show_progress,
            )
        except RuntimeError as exc:
            # LLM is down — abort rather than write a partial graph
            raise RuntimeError(
                f"Pipeline aborted: node extraction failed. Reason: {exc}"
            ) from exc

        stats.nodes_extracted = len(nodes)
        self._log(f"    → {len(nodes)} unique nodes extracted.")

        # ── Step 2: Build global node_id_map ─────────────────────────────────
        #    name.lower() → node_id  (for relationship resolution)
        node_id_map: Dict[str, str] = {
            node.name.lower(): node.node_id for node in nodes
        }
        # Also index aliases for fuzzy resolution
        for node in nodes:
            for alias in node.aliases:
                node_id_map.setdefault(alias.lower(), node.node_id)

        # ── Step 3: Write nodes to Neo4j ──────────────────────────────────────
        self._log("[2/5] Writing nodes to Neo4j...")
        try:
            stats.nodes_written = self._store.upsert_nodes(nodes)
            self._log(f"    → {stats.nodes_written} node(s) written.")
        except Exception as exc:
            msg = f"Node write failed: {exc}"
            warnings.warn(msg)
            stats.errors.append(msg)

        # ── Step 4: Build per-chunk node index ───────────────────────────────
        #    chunk_index → List[ExtractedNode] that came from that chunk
        self._log("[3/5] Extracting relationships...")

        nodes_per_chunk: Dict[int, List[ExtractedNode]] = {}
        for idx, chunk in enumerate(chunks):
            chunk_id = str(getattr(chunk, "chunk_id", idx))
            nodes_per_chunk[idx] = [
                n for n in nodes if n.source_chunk == chunk_id
            ]

        # ── Step 5: Relationship extraction (batched) ─────────────────────────
        all_relationships: List[ExtractedRelationship] = []

        try:
            raw_rels = self._relationship_extractor.extract_from_chunks(
                chunks          = chunks,
                nodes_per_chunk = nodes_per_chunk,
                node_id_map     = node_id_map,
                show_progress   = self._show_progress,
            )
            all_relationships.extend(raw_rels)
        except Exception as exc:
            msg = f"Relationship extraction failed: {exc}"
            warnings.warn(msg)
            stats.errors.append(msg)

        # ── Step 6 (optional): Cross-document relationships ───────────────────
        cross_doc_rels: List[ExtractedRelationship] = []

        if self._extract_cross_doc and len(chunks) > 1:
            self._log("[4/5] Extracting cross-document relationships...")

            # Build source → nodes mapping required by the extractor
            all_document_nodes: Dict[str, List[ExtractedNode]] = {}
            for node in nodes:
                all_document_nodes.setdefault(node.source, []).append(node)

            for chunk in chunks:
                try:
                    cross = self._relationship_extractor.extract_cross_document_references(
                        chunk              = chunk,
                        node_id_map        = node_id_map,
                        all_document_nodes = all_document_nodes,
                    )
                    cross_doc_rels.extend(cross)
                except Exception as exc:
                    chunk_id = getattr(chunk, "chunk_id", "?")
                    msg = f"Cross-doc extraction failed for chunk {chunk_id}: {exc}"
                    warnings.warn(msg)
                    stats.errors.append(msg)

            all_relationships.extend(cross_doc_rels)
            self._log(f"    → {len(cross_doc_rels)} cross-document relationship(s) found.")
        else:
            self._log("[4/5] Skipping cross-document relationships.")

        # Deduplicate all relationships before writing
        all_relationships = self._relationship_extractor.deduplicate_relationships(
            all_relationships
        )
        stats.relationships_extracted = len(all_relationships)
        stats.cross_doc_relationships = len(cross_doc_rels)
        self._log(f"    → {len(all_relationships)} unique relationship(s) after deduplication.")

        # ── Step 7: Write relationships to Neo4j ─────────────────────────────
        self._log("[5/5] Writing relationships to Neo4j...")
        try:
            stats.relationships_written = self._store.upsert_relationships(all_relationships)
            self._log(f"    → {stats.relationships_written} relationship(s) written.")
        except Exception as exc:
            msg = f"Relationship write failed: {exc}"
            warnings.warn(msg)
            stats.errors.append(msg)

        # ── Done ──────────────────────────────────────────────────────────────
        stats.elapsed_seconds = time.time() - t_start
        self._log(str(stats))
        return stats

    # ── Incremental ingestion ─────────────────────────────────────────────────

    def ingest_chunk(self, chunk) -> PipelineStats:
        """
        Incrementally ingest a single new chunk into the graph.

        Useful for real-time or streaming pipelines where documents arrive
        one at a time after the initial bulk ingestion.

        The chunk is expected to have chunk.metadata["source"] set.

        Args
        ────
        chunk : A single Chunk object.

        Returns
        ───────
        PipelineStats for this single-chunk run.
        """
        return self.run([chunk])

    # ── Graph stats ───────────────────────────────────────────────────────────

    def graph_stats(self) -> dict:
        """
        Return high-level stats about the current Neo4j graph.

        Returns
        ───────
        dict with keys:
            "nodes"         : total Entity node count
            "relationships" : total relationship count
        """
        return {
            "nodes"        : self._store.count_nodes(),
            "relationships": self._store.count_relationships(),
        }

    # ── Context manager support ───────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying Neo4j driver connection pool."""
        self._store.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Pipeline(\n"
            f"  node_extractor={self._node_extractor!r},\n"
            f"  relationship_extractor={self._relationship_extractor!r},\n"
            f"  store={self._store!r},\n"
            f"  extract_cross_doc={self._extract_cross_doc},\n"
            f")"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self._show_progress:
            print(msg)