# graph/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Shared data models ────────────────────────────────────────────────────────

@dataclass
class ExtractedNode:
    """
    A single entity node ready to be stored in Neo4j.

    node_id is a stable hash of (name + entity_type).
    Same entity always produces the same ID regardless of which
    chunk or document it appears in — enables deduplication in Neo4j.

    source is read from chunk.metadata["source"] — can be a filename,
    document ID, URL, or any string the caller sets before extraction.

    embedding is stored as a Neo4j node property so fuzzy entity
    matching works at query time when exact name lookup fails.
    """
    node_id     : str
    name        : str
    entity_type : str
    description : str
    source_chunk: str                          # chunk.chunk_id
    source      : str                          # chunk.metadata["source"]
    embedding   : Optional[np.ndarray] = None
    aliases     : List[str] = field(default_factory=list)

    def to_neo4j_properties(self) -> dict:
        """
        Serialize to a flat dict for Neo4j property storage.
        Excludes embedding — handled separately via Neo4j vector index API.
        """
        return {
            "node_id"     : self.node_id,
            "name"        : self.name,
            "entity_type" : self.entity_type,
            "description" : self.description,
            "source_chunk": self.source_chunk,
            "source"      : self.source,
            "aliases"     : self.aliases,
        }


@dataclass
class ExtractedRelationship:
    """
    A directed semantic relationship between two entity nodes.

    Direction matters:
        BERT  -[USES]->  Attention Mechanism
    is different from:
        Attention Mechanism  -[USED_BY]->  BERT

    source → target always follows the logical direction
    of the relationship as stated in the text.

    confidence is used during ingestion to filter noisy
    LLM-extracted relationships. Edges below threshold
    are discarded before writing to Neo4j.
    """
    source_id    : str
    target_id    : str
    source_name  : str
    target_name  : str
    relation_type: str
    description  : str
    source_chunk : str                         # chunk.chunk_id
    source       : str                         # chunk.metadata["source"]
    confidence   : float = 1.0
    mode         : str   = "constrained"   # "constrained" | "unconstrained"

    def to_neo4j_properties(self) -> dict:
        """Serialize to a flat dict for Neo4j relationship properties."""
        return {
            "relation_type": self.relation_type,
            "description"  : self.description,
            "source_chunk" : self.source_chunk,
            "source"       : self.source,
            "confidence"   : self.confidence,
            "mode"         : self.mode,
        }


# ── Abstract base classes ─────────────────────────────────────────────────────

class BaseNodeExtractor(ABC):
    """
    Abstract contract for all node extractor implementations.

    graph_builder.py depends on this interface, not on the concrete
    implementation — swap any backend without touching graph_builder.py.
    """

    @abstractmethod
    def extract_from_chunks(
        self,
        chunks: list,
        show_progress: bool = True,
    ) -> List[ExtractedNode]:
        """
        Extract and deduplicate entity nodes from a list of chunks.

        Must return a deduplicated list — same entity appearing in
        multiple chunks should produce exactly ONE ExtractedNode.

        Args:
            chunks:        Chunk objects from the chunking module.
                           Each chunk must have chunk.metadata["source"] set.
            show_progress: Whether to print progress to stdout.

        Returns:
            Deduplicated list of ExtractedNode, each with embedding set.
        """
        ...

    @abstractmethod
    def extract_from_single_chunk_public(
        self,
        chunk,
    ) -> List[ExtractedNode]:
        """
        Extract nodes from a single chunk (incremental ingestion).

        Args:
            chunk: A single Chunk object with chunk.metadata["source"] set.

        Returns:
            List of ExtractedNode with embeddings set.
        """
        ...


class BaseRelationshipExtractor(ABC):
    """
    Abstract contract for all relationship extractor implementations.
    """

    @abstractmethod
    def extract_from_chunk(
        self,
        chunk,
        nodes_in_chunk: List[ExtractedNode],
        node_id_map: dict,
    ) -> List[ExtractedRelationship]:
        """
        Extract directed relationships between entity pairs in one chunk.

        Args:
            chunk:          The source Chunk object.
            nodes_in_chunk: Nodes already extracted from this chunk.
            node_id_map:    Global mapping of entity_name.lower() → node_id.

        Returns:
            List of ExtractedRelationship above confidence threshold.
        """
        ...

    @abstractmethod
    def extract_cross_document_references(
        self,
        chunk,
        node_id_map: dict,
        all_document_nodes: dict,
    ) -> List[ExtractedRelationship]:
        """
        Detect explicit references to concepts from other documents.

        Args:
            chunk:               Source Chunk object.
            node_id_map:         Global name → node_id map.
            all_document_nodes:  source → List[ExtractedNode] mapping.

        Returns:
            List of cross-document ExtractedRelationship objects.
        """
        ...

    @abstractmethod
    def deduplicate_relationships(
        self,
        relationships: List[ExtractedRelationship],
    ) -> List[ExtractedRelationship]:
        """
        Remove duplicates keeping highest confidence per unique triple.

        Args:
            relationships: Raw list potentially containing duplicates.

        Returns:
            Deduplicated list, one entry per unique (source_id, target_id, relation_type).
        """
        ...