"""
Graph retriever for the GraphRAG pipeline.

Wraps your existing GraphStore (graph/graph_store.py) and orchestrates:

  1. Entity vector search — finds the best-matching graph nodes for each
     extracted entity using Neo4j's vector index (ANN on node embeddings).
  2. Neighbourhood traversal — for each matched node, fetches 1-2 hop
     subgraph (nodes + relationships) via get_neighbourhood().
  3. Path finding — when the query contains ≥ 2 entities, calls
     get_paths_between() on every entity pair. This is graph retrieval's
     killer feature for relational / comparative queries.
  4. Cypher execution — if query_engine.py produced a Cypher query, runs
     it directly for maximum precision.
  5. Converts all graph results into RetrievedChunk objects (text = entity
     description + relationship context) for uniform downstream handling.

Design note: graph traversal depth
-----------------------------------
We use hops=1 by default.  hops=2 gives richer context but can return
hundreds of nodes for highly connected entities, drowning the reranker.
The adaptive strategy: if hops=1 returns fewer than ``min_graph_results``
chunks, automatically retry with hops=2.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .retrieval_context import (
    GraphTraversalResult,
    PhaseStats,
    QueryIntent,
    QueryRepresentation,
    RetrievedChunk,
    RetrievalTrace,
)
from .retrieval_logger import RetrievalLogger


class GraphRetriever:
    """
    Graph traversal retriever using Neo4j GraphStore.

    Args
    ----
    graph_store     : GraphStore instance (already connected + schema initialised).
    hops            : Starting traversal depth (default 1; auto-expands to 2).
    min_graph_results: If fewer chunks returned at hops=1, retry with hops=2.
    top_k_entities  : Number of entity nodes to anchor traversal on (default 5).
    logger          : RetrievalLogger instance.
    verbose         : Print phase details.
    """

    def __init__(
        self,
        graph_store,
        hops                : int   = 1,
        min_graph_results   : int   = 3,
        top_k_entities      : int   = 5,
        logger              : Optional[RetrievalLogger] = None,
        verbose             : bool  = True,
    ):
        self.graph_store       = graph_store
        self.hops              = hops
        self.min_graph_results = min_graph_results
        self.top_k_entities    = top_k_entities
        self.logger            = logger or RetrievalLogger(verbose=verbose)
        self.verbose           = verbose

    # ── Main entry point ──────────────────────────────────────────────────────

    def search(
        self,
        rep   : QueryRepresentation,
        top_k : int = 15,
        trace : Optional[RetrievalTrace] = None,
    ) -> Tuple[List[RetrievedChunk], GraphTraversalResult]:
        """
        Run the full graph retrieval strategy.

        Returns
        -------
        (chunks, traversal_result) where chunks are RetrievedChunks ready for
        fusion and traversal_result is the raw graph data for visualisation.
        """
        t0 = time.time()
        self.logger.phase_start("Graph Retriever")

        traversal  = GraphTraversalResult()
        all_chunks : List[RetrievedChunk] = []

        # Strategy 1: Direct Cypher (if available)
        if rep.cypher_query:
            cypher_chunks, cypher_data = self._run_cypher(rep.cypher_query)
            all_chunks.extend(cypher_chunks)
            traversal.nodes.extend(cypher_data.get("nodes", []))
            traversal.relationships.extend(cypher_data.get("relationships", []))
            traversal.cypher_used = rep.cypher_query

        # Strategy 2: Entity vector search + neighbourhood 
        entity_chunks, entity_nodes = self._entity_traversal(rep, top_k)
        all_chunks.extend(entity_chunks)
        traversal.nodes.extend(entity_nodes)

        # Auto-expand depth if insufficient results
        if len(all_chunks) < self.min_graph_results and self.hops < 2:
            self.logger.info(
                f"Only {len(all_chunks)} graph results at hops=1 — "
                "retrying with hops=2"
            )
            extra_chunks, extra_nodes = self._entity_traversal(rep, top_k, hops=2)
            all_chunks.extend(extra_chunks)
            traversal.nodes.extend(extra_nodes)
            traversal.traversal_depth = 2
        else:
            traversal.traversal_depth = self.hops

        # Strategy 3: Path finding (relational / comparative queries) 
        if rep.intent in (QueryIntent.RELATIONAL, QueryIntent.COMPARATIVE) \
                and len(rep.entities) >= 2:
            path_chunks, paths = self._path_finding(rep.entities[:4])
            all_chunks.extend(path_chunks)
            traversal.paths.extend(paths)

        # Deduplicate by chunk_id
        seen : set             = set()
        deduped               : List[RetrievedChunk] = []
        for chunk in all_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                deduped.append(chunk)

        # Sort by score descending, take top_k
        deduped.sort(key=lambda c: c.score, reverse=True)
        deduped = deduped[:top_k]

        # Deduplicate traversal nodes
        node_ids  = set()
        uniq_nodes = []
        for n in traversal.nodes:
            nid = n.get("node_id", "")
            if nid not in node_ids:
                node_ids.add(nid)
                uniq_nodes.append(n)
        traversal.nodes = uniq_nodes

        # Traversal stats
        traversal.relationships = list({
            (r.get("source_id"), r.get("target_id"), r.get("relation_type")): r
            for r in traversal.relationships
        }.values())

        elapsed_ms = (time.time() - t0) * 1000

        if trace:
            trace.graph_count        = len(deduped)
            trace.graph_nodes_visited= len(traversal.nodes)
            trace.graph_rels_visited = len(traversal.relationships)
            trace.graph_hops         = traversal.traversal_depth
            trace.add_phase(PhaseStats(
                phase_name   = "Graph Retriever",
                elapsed_ms   = elapsed_ms,
                input_count  = len(rep.entities),
                output_count = len(deduped),
                notes        = (
                    f"nodes={len(traversal.nodes)}  "
                    f"rels={len(traversal.relationships)}  "
                    f"hops={traversal.traversal_depth}"
                ),
            ))

        self.logger.print_graph_results(
            chunks        = deduped,
            nodes_visited = len(traversal.nodes),
            rels_visited  = len(traversal.relationships),
            hops          = traversal.traversal_depth,
        )
        self.logger.phase_end("Graph Retriever", count=len(deduped), elapsed_ms=elapsed_ms)
        return deduped, traversal

    # Strategy implementations

    def _entity_traversal(
        self,
        rep  : QueryRepresentation,
        top_k: int,
        hops : Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], List[Dict]]:
        """
        For each query entity: vector-search the graph, then traverse neighbourhood.
        Returns (chunks, nodes).
        """
        hops     = hops or self.hops
        chunks   : List[RetrievedChunk] = []
        all_nodes: List[Dict]           = []

        # Entity vector search (uses Neo4j vector index on node embeddings)
        if rep.embedding is not None:
            try:
                vector_nodes = self.graph_store.similarity_search(
                    rep.embedding, top_k=self.top_k_entities
                )
                for vn in vector_nodes:
                    nid   = vn.get("node_id", "")
                    score = float(vn.get("score", 0.0))
                    chunk = self._node_to_chunk(vn, score=score)
                    chunks.append(chunk)

                    # Neighbourhood traversal around this node
                    nbhd = self.graph_store.get_neighbourhood(nid, hops=hops)
                    for node in nbhd.get("nodes", []):
                        all_nodes.append(node)
                        nchunk = self._node_to_chunk(node, score=score * 0.8)
                        chunks.append(nchunk)

                    for rel in nbhd.get("relationships", []):
                        rchunk = self._rel_to_chunk(rel, score=score * 0.6)
                        if rchunk:
                            chunks.append(rchunk)

            except Exception as e:
                self.logger.warn(f"Entity vector search failed: {e}")

        # Exact name lookup fallback
        for entity_name in rep.entities[:self.top_k_entities]:
            try:
                node = self.graph_store.get_node_by_name(entity_name)
                if node:
                    all_nodes.append(node)
                    chunks.append(self._node_to_chunk(node, score=0.85))
                    nbhd = self.graph_store.get_neighbourhood(
                        node["node_id"], hops=hops
                    )
                    for n in nbhd.get("nodes", []):
                        all_nodes.append(n)
                        chunks.append(self._node_to_chunk(n, score=0.7))
            except Exception as e:
                self.logger.warn(f"Name lookup failed for '{entity_name}': {e}")

        return chunks, all_nodes

    def _run_cypher(self, cypher: str) -> Tuple[List[RetrievedChunk], Dict]:
        """Execute a Cypher query directly and convert results to chunks."""
        chunks = []
        data   = {"nodes": [], "relationships": []}
        try:
            with self.graph_store._session() as session:
                result = session.run(cypher)
                records = list(result)

            for record in records:
                record_dict = dict(record)
                # Extract any node-like values
                for key, value in record_dict.items():
                    if hasattr(value, "items"):
                        node_dict = dict(value)
                        data["nodes"].append(node_dict)
                        chunks.append(self._node_to_chunk(node_dict, score=0.9))
                    elif isinstance(value, str) and value.strip():
                        # Plain text result (e.g. description field)
                        chunks.append(RetrievedChunk(
                            chunk_id  = f"cypher_{hash(value)}",
                            text      = value,
                            source    = "graph_cypher",
                            score     = 0.85,
                            retriever = "graph",
                            metadata  = {"cypher_key": key},
                        ))
        except Exception as e:
            self.logger.warn(f"Cypher execution failed: {e}")

        return chunks, data

    def _path_finding(
        self,
        entities: List[str],
    ) -> Tuple[List[RetrievedChunk], List]:
        """Find paths between entity pairs and convert to chunks."""
        chunks = []
        paths  = []

        # Get node_ids for each entity name
        node_ids = {}
        for name in entities:
            try:
                node = self.graph_store.get_node_by_name(name)
                if node:
                    node_ids[name] = node["node_id"]
            except Exception:
                pass

        names = list(node_ids.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                src = node_ids[names[i]]
                tgt = node_ids[names[j]]
                try:
                    found = self.graph_store.get_paths_between(src, tgt, max_hops=3)
                    paths.extend(found)
                    for path in found:
                        path_text = self._path_to_text(path, names[i], names[j])
                        if path_text:
                            chunks.append(RetrievedChunk(
                                chunk_id  = f"path_{src}_{tgt}_{len(chunks)}",
                                text      = path_text,
                                source    = "graph_path",
                                score     = 0.88,
                                retriever = "graph",
                                metadata  = {
                                    "source_entity": names[i],
                                    "target_entity": names[j],
                                    "path_length"  : len(path) // 2 + 1,
                                },
                            ))
                except Exception as e:
                    self.logger.warn(f"Path finding failed ({names[i]} → {names[j]}): {e}")

        return chunks, paths

    # Conversion helpers

    @staticmethod
    def _node_to_chunk(node: Dict, score: float = 0.75) -> RetrievedChunk:
        """Convert a Neo4j node dict to a RetrievedChunk."""
        name        = node.get("name", "Unknown")
        entity_type = node.get("entity_type", "Entity")
        description = node.get("description", "")
        aliases     = node.get("aliases", [])
        source      = node.get("source", "graph")

        text = f"[{entity_type}] {name}"
        if aliases:
            text += f" (also: {', '.join(aliases[:3])})"
        if description:
            text += f"\n{description}"

        return RetrievedChunk(
            chunk_id  = node.get("node_id", f"node_{hash(name)}"),
            text      = text,
            source    = source,
            score     = score,
            retriever = "graph",
            metadata  = {
                "node_id"    : node.get("node_id", ""),
                "entity_type": entity_type,
                "name"       : name,
            },
        )

    @staticmethod
    def _rel_to_chunk(rel: Dict, score: float = 0.65) -> Optional[RetrievedChunk]:
        """Convert a Neo4j relationship dict to a RetrievedChunk."""
        src_id  = rel.get("source_id", "")
        tgt_id  = rel.get("target_id", "")
        rel_type= rel.get("relation_type", "RELATED")
        desc    = rel.get("description", "")
        source  = rel.get("source", "graph")

        if not (src_id and tgt_id):
            return None

        text = f"[Relationship: {rel_type}]\n{desc}" if desc else f"[Relationship: {rel_type}]"

        return RetrievedChunk(
            chunk_id  = f"rel_{src_id}_{tgt_id}_{rel_type}",
            text      = text,
            source    = source,
            score     = score,
            retriever = "graph",
            metadata  = {
                "source_id"   : src_id,
                "target_id"   : tgt_id,
                "relation_type": rel_type,
                "confidence"  : rel.get("confidence", 1.0),
            },
        )

    @staticmethod
    def _path_to_text(path: List[Dict], src_name: str, tgt_name: str) -> str:
        """Convert a path (list of alternating node/rel dicts) to readable text."""
        if not path:
            return ""
        parts = []
        for item in path:
            t = item.get("type", "")
            if t == "node":
                parts.append(item.get("name", item.get("node_id", "?")))
            elif t == "relationship":
                parts.append(f"--[{item.get('relation_type', '?')}]-->")
        return (
            f"Path from '{src_name}' to '{tgt_name}':\n"
            + " ".join(parts)
        )