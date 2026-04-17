# graph/__init__.py
"""
graph/
─────
Public API
──────────
Data models:
    ExtractedNode           — entity node ready for Neo4j
    ExtractedRelationship   — directed edge ready for Neo4j

Abstract base classes:
    BaseNodeExtractor       — contract for node extractors
    BaseRelationshipExtractor — contract for relationship extractors

Concrete implementations:
    NodeExtractor           — spaCy + LLM entity extraction
    RelationshipExtractor   — LLM relationship extraction

Usage
─────
    from graph import NodeExtractor, RelationshipExtractor
    from graph import ExtractedNode, ExtractedRelationship

    # graph_builder.py uses these directly
    # Everything else in the project imports from here, not
    # from the individual submodules, keeping imports clean.
"""
from .base import (
    ExtractedNode,
    ExtractedRelationship,
    BaseNodeExtractor,
    BaseRelationshipExtractor,
)
from .llm_backend import LLMBackend
from .node_extractor import NodeExtractor
from .relationships_extractor import RelationshipExtractor
from .graph_store import GraphStore

__all__ = [
    # Data models
    "ExtractedNode",
    "ExtractedRelationship",
    # Abstract bases
    "BaseNodeExtractor",
    "BaseRelationshipExtractor",
    # LLM backend
    "LLMBackend",
    # Implementations
    "NodeExtractor",
    "RelationshipExtractor",
    # Storage
    "GraphStore",
]