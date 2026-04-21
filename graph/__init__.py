# # graph/__init__.py
# """
# graph/
# ─────
# Public API
# ──────────
# Data models:
#     ExtractedNode           — entity node ready for Neo4j
#     ExtractedRelationship   — directed edge ready for Neo4j

# Abstract base classes:
#     BaseNodeExtractor       — contract for node extractors
#     BaseRelationshipExtractor — contract for relationship extractors

# Concrete implementations:
#     NodeExtractor           — spaCy + LLM entity extraction
#     RelationshipExtractor   — LLM relationship extraction

# Usage
# ─────
#     from graph import NodeExtractor, RelationshipExtractor
#     from graph import ExtractedNode, ExtractedRelationship

#     # graph_builder.py uses these directly
#     # Everything else in the project imports from here, not
#     # from the individual submodules, keeping imports clean.
# """
# from .base import (
#     ExtractedNode,
#     ExtractedRelationship,
#     BaseNodeExtractor,
#     BaseRelationshipExtractor,
# )
# from .llm_backend import LLMBackend
# from .node_extractor import NodeExtractor
# from .relationships_extractor import RelationshipExtractor

# # GraphStore is intentionally NOT imported here at module load time.
# # graph_store.py imports neo4j at the top level, so an eager import would
# # make neo4j a hard dependency for every `from graph import ...` statement,
# # even in extraction-only environments that have no Neo4j.
# #
# # Instead, GraphStore is exposed via __getattr__ so it is only imported
# # (and neo4j validated) when the caller actually accesses it:
# #
# #     from graph import GraphStore        # triggers lazy import
# #     from graph import NodeExtractor     # neo4j never touched

# def __getattr__(name: str):
#     if name == "GraphStore":
#         from .graph_store import GraphStore  # noqa: PLC0415
#         return GraphStore
#     raise AttributeError(f"module 'graph' has no attribute {name!r}")


# __all__ = [
#     # Data models
#     "ExtractedNode",
#     "ExtractedRelationship",
#     # Abstract bases
#     "BaseNodeExtractor",
#     "BaseRelationshipExtractor",
#     # LLM backend
#     "LLMBackend",
#     # Implementations
#     "NodeExtractor",
#     "RelationshipExtractor",
#     # Storage (lazy — neo4j imported only on first access)
#     "GraphStore",
# ]
# graph/__init__.py
"""
graph/
─────
Public API
──────────
Data models:
    ExtractedNode               — entity node ready for Neo4j
    ExtractedRelationship       — directed edge ready for Neo4j

Abstract base classes:
    BaseNodeExtractor           — contract for node extractors
    BaseRelationshipExtractor   — contract for relationship extractors

Concrete implementations:
    CombinedExtractor           — single LLM call per batch extracts both
                                  entity nodes AND relationships simultaneously.
                                  Satisfies both BaseNodeExtractor and
                                  BaseRelationshipExtractor — pass the same
                                  instance for both roles in Pipeline.from_components().

Usage
─────
    from graph import CombinedExtractor, LLMBackend
    from graph import ExtractedNode, ExtractedRelationship

    llm  = LLMBackend(api_key="gsk_...", model="llama-3.3-70b-versatile")
    ext  = CombinedExtractor(llm=llm, embedding_fn=my_embed_fn)

    # Pipeline.from_components() uses the same instance for both roles:
    #   node_extractor         = ext
    #   relationship_extractor = ext
"""
from .base import (
    ExtractedNode,
    ExtractedRelationship,
    BaseNodeExtractor,
    BaseRelationshipExtractor,
)
from .llm_backend import LLMBackend
from .node_relation_extractor import CombinedExtractor

# GraphStore is intentionally NOT imported here at module load time.
# graph_store.py imports neo4j at the top level, so an eager import would
# make neo4j a hard dependency for every `from graph import ...` statement,
# even in extraction-only environments that have no Neo4j.
#
# Instead, GraphStore is exposed via __getattr__ so it is only imported
# (and neo4j validated) when the caller actually accesses it:
#
#     from graph import GraphStore        # triggers lazy import
#     from graph import CombinedExtractor # neo4j never touched

def __getattr__(name: str):
    if name == "GraphStore":
        from .graph_store import GraphStore  # noqa: PLC0415
        return GraphStore
    raise AttributeError(f"module 'graph' has no attribute {name!r}")


__all__ = [
    # Data models
    "ExtractedNode",
    "ExtractedRelationship",
    # Abstract bases
    "BaseNodeExtractor",
    "BaseRelationshipExtractor",
    # LLM backend
    "LLMBackend",
    # Combined extractor (node + relationship roles in one class)
    "CombinedExtractor",
    # Storage (lazy — neo4j imported only on first access)
    "GraphStore",
]