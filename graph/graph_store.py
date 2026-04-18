# graph/graph_store.py
"""
Neo4j persistence layer for the KG-RAG pipeline.

Responsibilities
----------------
- Write ExtractedNode and ExtractedRelationship objects to Neo4j
- Maintain a vector index on node embeddings for fuzzy entity lookup
- Expose query helpers used by the retriever:
    · exact entity lookup by name / node_id
    · vector similarity search (ANN)
    · neighbourhood traversal (1- and 2-hop)
    · full subgraph retrieval for a list of node_ids
- Graceful deduplication: MERGE on node_id / (source_id, target_id, relation_type)
  so repeated ingestion runs are idempotent.

Dependencies
------------
    pip install neo4j numpy

Usage
-----
    from graph.graph_store import GraphStore
    from graph import NodeExtractor, RelationshipExtractor

    store = GraphStore(uri="bolt://localhost:7687", user="neo4j", password="...")
    store.init_schema()                 # create constraints + vector index once

    store.upsert_nodes(nodes)           # List[ExtractedNode]
    store.upsert_relationships(rels)    # List[ExtractedRelationship]

    results = store.similarity_search(query_vector, top_k=5)   # query_vector: np.ndarray
    subgraph = store.get_neighbourhood("a3f1b2c4d5e6f7a8", hops=2)
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, ClientError
except ImportError as exc:
    raise ImportError(
        "neo4j driver is required.\n"
        "pip install neo4j"
    ) from exc

from .base import ExtractedNode, ExtractedRelationship


# ── Constants ──────────────────────────────────────────────────────────────────

# Name of the Neo4j vector index created on Entity nodes
VECTOR_INDEX_NAME = "entity_embeddings"

# Default embedding dimensionality — must match your embedding model
DEFAULT_EMBEDDING_DIM = 384


# ── GraphStore ────────────────────────────────────────────────────────────────

class GraphStore:
    """
    Neo4j-backed persistence and retrieval store for KG-RAG.

    All nodes are stored as (:Entity) labels.
    All relationships keep their semantic type (e.g. :USES, :BASED_ON).

    Args
    ----
    uri           : Neo4j Bolt URI  (e.g. "bolt://localhost:7687")
    user          : Neo4j username  (default "neo4j")
    password      : Neo4j password
    database      : Neo4j database  (default "neo4j")
    embedding_dim : Dimensionality of node embeddings (default 384)
    batch_size    : Nodes / relationships written per transaction (default 256)
    """

    def __init__(
        self,
        uri          : str,
        user         : str  = "neo4j",
        password     : str  = "",
        database     : str  = "neo4j",
        embedding_dim: int  = DEFAULT_EMBEDDING_DIM,
        batch_size   : int  = 256,
    ):
        self.uri           = uri
        self.database      = database
        self.embedding_dim = embedding_dim
        self.batch_size    = batch_size

        try:
            self._driver: Driver = GraphDatabase.driver(
                uri, auth=(user, password)
            )
            self._driver.verify_connectivity()
        except ServiceUnavailable as exc:
            raise ConnectionError(
                f"Cannot connect to Neo4j at {uri}. "
                "Make sure Neo4j is running and the URI / credentials are correct."
            ) from exc

    # ── Schema initialisation ─────────────────────────────────────────────────

    def init_schema(self) -> None:
        """
        Idempotent setup — safe to call on every app start.

        Creates:
        - Uniqueness constraint on Entity.node_id
        - Index on Entity.name  (fast exact lookup)
        - Neo4j vector index on Entity.embedding
        """
        with self._session() as session:
            # Uniqueness constraint (also creates a backing B-tree index)
            session.run("""
                CREATE CONSTRAINT entity_node_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.node_id IS UNIQUE
            """)

            # B-tree index for exact name lookups
            session.run("""
                CREATE INDEX entity_name IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)

            # Vector index — requires Neo4j 5.11+ with GDS or built-in vector support
            try:
                session.run(f"""
                    CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
            except ClientError as exc:
                warnings.warn(
                    f"Vector index creation failed (Neo4j < 5.11 or GDS not installed): {exc}\n"
                    "Similarity search will fall back to brute-force cosine."
                )

        print("✓ Neo4j schema initialised.")

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert_nodes(self, nodes: List[ExtractedNode]) -> int:
        """
        MERGE-upsert a list of ExtractedNode into Neo4j.

        Existing nodes are updated (description, aliases) only when the
        incoming description is longer — mirrors NodeExtractor dedup logic.
        Embeddings are set unconditionally so they stay fresh.

        Returns the number of nodes written.
        """
        if not nodes:
            return 0

        written = 0
        for batch in self._batches(nodes):
            # Pre-merge aliases per node_id within the batch (avoids APOC dependency)
            merged: Dict[str, dict] = {}
            for node in batch:
                props = node.to_neo4j_properties()
                props["embedding"] = (
                    node.embedding.tolist()
                    if node.embedding is not None
                    else None
                )
                if props["node_id"] in merged:
                    existing = merged[props["node_id"]]
                    existing["aliases"] = list(set(existing["aliases"] + props["aliases"]))
                    if len(props["description"]) > len(existing["description"]):
                        existing["description"] = props["description"]
                else:
                    merged[props["node_id"]] = props
            records = list(merged.values())

            with self._session() as session:
                session.run(
                    """
                    UNWIND $records AS r
                    MERGE (e:Entity {node_id: r.node_id})
                    ON CREATE SET
                        e.name         = r.name,
                        e.entity_type  = r.entity_type,
                        e.description  = r.description,
                        e.source_chunk = r.source_chunk,
                        e.source       = r.source,
                        e.aliases      = r.aliases,
                        e.embedding    = r.embedding
                    ON MATCH SET
                        e.embedding    = r.embedding,
                        e.aliases      = r.aliases,
                        e.description  = CASE
                            WHEN size(r.description) > size(e.description)
                            THEN r.description
                            ELSE e.description
                        END
                    """,
                    records=records,
                )
            # Count de-duplicated records, not the raw batch.
            # len(batch) over-counts when duplicate node_ids appear in
            # the same batch (those are merged into one record before write).
            written += len(records)

        return written

    def upsert_relationships(self, relationships: List[ExtractedRelationship]) -> int:
        """
        MERGE-upsert a list of ExtractedRelationship into Neo4j.

        Each relationship is typed dynamically (e.g. :USES, :BASED_ON).
        Because Neo4j does not allow parameterised relationship types,
        relationships are grouped by type and one Cypher statement is
        issued per type — this keeps batching efficient while staying safe.

        Returns the number of relationships written.
        """
        if not relationships:
            return 0

        # Group by relation_type so we can use a typed MERGE
        by_type: Dict[str, List] = {}
        for rel in relationships:
            by_type.setdefault(rel.relation_type, []).append(rel)

        written = 0
        for rel_type, rels in by_type.items():
            # Sanitise type: only allow UPPER_SNAKE_CASE to prevent injection
            safe_type = _sanitise_rel_type(rel_type)
            if not safe_type:
                warnings.warn(f"Skipped relationship with unsafe type: '{rel_type}'")
                continue

            for batch in self._batches(rels):
                records = [r.to_neo4j_properties() | {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                } for r in batch]

                cypher = f"""
                    UNWIND $records AS r
                    MATCH (src:Entity {{node_id: r.source_id}})
                    MATCH (tgt:Entity {{node_id: r.target_id}})
                    MERGE (src)-[rel:{safe_type}]->(tgt)
                    ON CREATE SET
                        rel.relation_type = r.relation_type,
                        rel.description   = r.description,
                        rel.source_chunk  = r.source_chunk,
                        rel.source        = r.source,
                        rel.confidence    = r.confidence,
                        rel.mode          = r.mode
                    ON MATCH SET
                        rel.confidence = CASE
                            WHEN r.confidence > rel.confidence
                            THEN r.confidence
                            ELSE rel.confidence
                        END,
                        rel.description = CASE
                            WHEN r.confidence > rel.confidence
                            THEN r.description
                            ELSE rel.description
                        END
                """
                with self._session() as session:
                    session.run(cypher, records=records)
                written += len(batch)

        return written

    def delete_by_source(self, source: str) -> Tuple[int, int]:
        """
        Remove all nodes and relationships that came from a given source file.

        Useful for re-ingesting an updated document without stale data.

        Returns (nodes_deleted, relationships_deleted).
        """
        with self._session() as session:
            # Count relationships before deletion
            rel_result = session.run(
                """
                MATCH (e:Entity {source: $source})-[r]-()
                RETURN count(DISTINCT r) AS rels
                """,
                source=source,
            )
            rel_row = rel_result.single()
            rels_deleted = rel_row["rels"] if rel_row else 0

            # Count nodes before deletion
            node_result = session.run(
                "MATCH (e:Entity {source: $source}) RETURN count(e) AS nodes",
                source=source,
            )
            node_row = node_result.single()
            nodes_deleted = node_row["nodes"] if node_row else 0

            # Delete nodes and all their relationships
            session.run(
                "MATCH (e:Entity {source: $source}) DETACH DELETE e",
                source=source,
            )

        return (nodes_deleted, rels_deleted)

    # ── Read / retrieval operations ───────────────────────────────────────────

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return a single Entity node dict by node_id, or None."""
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity {node_id: $node_id}) RETURN e",
                node_id=node_id,
            )
            record = result.single()
            return dict(record["e"]) if record else None

    def get_node_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Case-insensitive exact name lookup.
        Returns the first match or None.
        """
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) = toLower($name) RETURN e LIMIT 1",
                name=name,
            )
            record = result.single()
            return dict(record["e"]) if record else None

    def similarity_search(
        self,
        query_vector: np.ndarray,
        top_k       : int = 5,
    ) -> List[Dict[str, Any]]:
        """
        ANN similarity search using the Neo4j vector index.

        Falls back to brute-force cosine if the vector index is unavailable.

        Args
        ----
        query_vector : 1-D numpy array of the query embedding
        top_k        : number of nearest neighbours to return

        Returns
        -------
        List of dicts with keys: node_id, name, entity_type, description,
        source, score (cosine similarity 0–1).
        """
        vec = query_vector.tolist()
        try:
            return self._vector_index_search(vec, top_k)
        except ClientError:
            warnings.warn("Vector index unavailable — falling back to brute-force cosine search.")
            return self._brute_force_similarity(query_vector, top_k)

    def get_neighbourhood(
        self,
        node_id: str,
        hops   : int = 1,
    ) -> Dict[str, Any]:
        """
        Return all nodes and relationships within `hops` of the given node.

        Args
        ----
        node_id : Starting node
        hops    : Number of hops (1 or 2 recommended; >3 can be very large)

        Returns
        -------
        {
            "nodes":         [{ node properties }, ...],
            "relationships": [{ rel properties + source_id + target_id }, ...],
        }
        """
        with self._session() as session:
            result = session.run(
                f"""
                MATCH path = (start:Entity {{node_id: $node_id}})-[*1..{hops}]-(neighbour:Entity)
                WITH nodes(path) AS ns, relationships(path) AS rs
                UNWIND ns AS n
                WITH collect(DISTINCT n) AS all_nodes, rs
                UNWIND rs AS r
                RETURN all_nodes, collect(DISTINCT r) AS all_rels
                """,
                node_id=node_id,
            )
            record = result.single()
            if not record:
                # node exists but has no neighbours
                node = self.get_node_by_id(node_id)
                return {"nodes": [node] if node else [], "relationships": []}

            nodes = [dict(n) for n in record["all_nodes"]]
            rels  = []
            for r in record["all_rels"]:
                rel_dict = dict(r)
                rel_dict["source_id"]     = r.start_node["node_id"]
                rel_dict["target_id"]     = r.end_node["node_id"]
                rel_dict["relation_type"] = r.type
                rels.append(rel_dict)

            return {"nodes": nodes, "relationships": rels}

    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Return all nodes in node_ids and every relationship between them.

        Useful for rendering a focused subgraph after vector retrieval.
        """
        with self._session() as session:
            # Fetch nodes
            n_result = session.run(
                "MATCH (e:Entity) WHERE e.node_id IN $ids RETURN e",
                ids=node_ids,
            )
            nodes = [dict(r["e"]) for r in n_result]

            # Fetch relationships between those nodes
            r_result = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.node_id IN $ids AND b.node_id IN $ids
                RETURN a.node_id AS src, b.node_id AS tgt,
                       type(r) AS rel_type, properties(r) AS props
                """,
                ids=node_ids,
            )
            rels = []
            for row in r_result:
                rel = dict(row["props"])
                rel["source_id"]     = row["src"]
                rel["target_id"]     = row["tgt"]
                rel["relation_type"] = row["rel_type"]
                rels.append(rel)

        return {"nodes": nodes, "relationships": rels}

    def get_paths_between(
        self,
        source_id: str,
        target_id: str,
        max_hops : int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """
        Find all shortest paths between two nodes up to max_hops.

        Returns a list of paths; each path is an ordered list of alternating
        node and relationship dicts:
            [node, rel, node, rel, node, ...]
        """
        with self._session() as session:
            result = session.run(
                f"""
                MATCH path = allShortestPaths(
                    (a:Entity {{node_id: $src}})-[*1..{max_hops}]-(b:Entity {{node_id: $tgt}})
                )
                RETURN path
                LIMIT 10
                """,
                src=source_id,
                tgt=target_id,
            )
            paths = []
            for record in result:
                path = record["path"]
                items: List[Dict] = []
                for i, node in enumerate(path.nodes):
                    items.append({"type": "node", **dict(node)})
                    if i < len(path.relationships):
                        rel = path.relationships[i]
                        items.append({
                            "type"         : "relationship",
                            "relation_type": rel.type,
                            **dict(rel),
                        })
                paths.append(items)
        return paths

    def count_nodes(self) -> int:
        """Return total number of Entity nodes in the graph."""
        with self._session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS n")
            return result.single()["n"]

    def count_relationships(self) -> int:
        """Return total number of relationships in the graph."""
        with self._session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS n")
            return result.single()["n"]

    def close(self) -> None:
        """Close the Neo4j driver connection pool."""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"GraphStore(uri={self.uri!r}, database={self.database!r})"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _session(self) -> Session:
        return self._driver.session(database=self.database)

    def _batches(self, items: list) -> List[list]:
        """Split a list into batches of self.batch_size."""
        return [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

    def _vector_index_search(
        self,
        vec  : list,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Query the Neo4j built-in vector index (Neo4j 5.11+)."""
        with self._session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes(
                    '{VECTOR_INDEX_NAME}', $top_k, $vec
                )
                YIELD node AS e, score
                RETURN e.node_id    AS node_id,
                       e.name       AS name,
                       e.entity_type AS entity_type,
                       e.description AS description,
                       e.source      AS source,
                       score
                """,
                top_k=top_k,
                vec=vec,
            )
            return [dict(r) for r in result]

    def _brute_force_similarity(
        self,
        query_vector: np.ndarray,
        top_k       : int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback: pull all embeddings from Neo4j and compute cosine similarity
        in Python. Slow for large graphs — use only when vector index is absent.
        """
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN e.node_id    AS node_id,
                       e.name       AS name,
                       e.entity_type AS entity_type,
                       e.description AS description,
                       e.source      AS source,
                       e.embedding   AS embedding
                """
            )
            rows = [dict(r) for r in result]

        if not rows:
            return []

        q = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        scored = []
        for row in rows:
            vec = np.array(row.pop("embedding"), dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            score = float(np.dot(q, vec))
            scored.append({**row, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ── Utility ───────────────────────────────────────────────────────────────────

def _sanitise_rel_type(rel_type: str) -> str:
    """
    Ensure a relationship type string is safe to interpolate into Cypher.

    Allows only: A-Z, 0-9, underscore.
    Returns empty string if the input is invalid.
    """
    import re
    cleaned = rel_type.strip().upper().replace(" ", "_").replace("-", "_")
    return cleaned if re.fullmatch(r"[A-Z][A-Z0-9_]*", cleaned) else ""