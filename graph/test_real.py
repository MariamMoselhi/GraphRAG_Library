"""
test_real.py — Full KG-RAG Test Suite
======================================
Tests every layer of the stack using YOUR real components:
    graph/base.py                      — ExtractedNode, ExtractedRelationship, ABCs
    graph/llm_backend.py               — LLMBackend (real Groq API calls)
    graph/node_extractor.py            — NodeExtractor (spaCy + real LLM)
    graph/relationships_extractor.py   — RelationshipExtractor (constrained / unconstrained)
    graph/graph_store.py               — Neo4j persistence + retrieval
    Pipeline.py                        — End-to-end orchestrator

Nothing is mocked. The Groq API is called for real. Neo4j is written for real.
The only "fake" inputs are short academic text strings used as chunk.text so
you don't need actual PDF files.

Requirements
------------
    GROQ_API_KEY set in environment or .env file
    Neo4j running (for --neo4j flag)

Run modes
---------
    # Groq API only, skip Neo4j
    python test_real.py

    # Groq API + real Neo4j (nodes and rels actually written to DB)
    python test_real.py --neo4j --password neo4j1234

    # Verbose output
    python test_real.py --neo4j --password neo4j1234 -v

Environment variables
---------------------
    GROQ_API_KEY   — Groq API key (required)
    NEO4J_URI      — neo4j://127.0.0.1:7687
    NEO4J_USER     — neo4j
    NEO4J_PASSWORD — neo4j1234
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Load .env before anything else ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

# ── YOUR Chunk class ──────────────────────────────────────────────────────────
try:
    from chunking.chunk_base import Chunk
    print("✓ Chunking module loaded")
except ImportError as e:
    print(f"ERROR: Could not import Chunk — {e}")
    sys.exit(1)

# ── YOUR graph module ─────────────────────────────────────────────────────────
try:
    from graph.base import (
        ExtractedNode,
        ExtractedRelationship,
        BaseNodeExtractor,
        BaseRelationshipExtractor,
    )
    from graph.llm_backend import LLMBackend
    from graph.node_extractor import NodeExtractor
    from graph.relationships_extractor import (
        RelationshipExtractor,
        ALLOWED_RELATIONS,
        CROSS_DOC_RELATIONS,
    )
    from graph.graph_store import GraphStore, _sanitise_rel_type
    print("✓ Graph module loaded")
except ImportError as e:
    print(f"ERROR: Could not import graph module — {e}")
    sys.exit(1)

# ── YOUR Pipeline ─────────────────────────────────────────────────────────────
try:
    from Pipeline import Pipeline, PipelineStats
    print("✓ Pipeline module loaded")
except ImportError as e:
    print(f"ERROR: Could not import Pipeline — {e}")
    sys.exit(1)

# ── YOUR embeddings (optional) ───────────────────────────────────────────────
try:
    from embeddings import HuggingFaceEmbedding
    _EMBEDDINGS_AVAILABLE = True
    print("✓ Embeddings module loaded")
except ImportError:
    _EMBEDDINGS_AVAILABLE = False
    print("WARNING: Embeddings module not found — using deterministic dummy")

# ── Test counters ─────────────────────────────────────────────────────────────
_total  = 0
_passed = 0

# =============================================================================
# SAMPLE TEXT INPUTS
# Short academic paragraphs used as chunk.text — no real PDFs needed.
# =============================================================================

TEXT_1 = """
BERT, developed by Google, is a transformer-based model that uses
the Attention Mechanism for natural language understanding tasks.
Vaswani et al. introduced the Transformer architecture in 2017.
BERT is evaluated on the SQuAD dataset and achieves high F1 scores.
Dropout is used as a regularisation method to prevent overfitting.
"""

TEXT_2 = """
Building on the Attention Mechanism introduced earlier, GPT-4 extends
BERT by using a decoder-only transformer architecture. Recall that
backpropagation is required to train all neural network models.
GPT-4 was developed by OpenAI and evaluated on multiple benchmarks.
The BLEU metric is commonly used to evaluate language generation tasks.
"""

TEXT_3 = """
Gradient Descent is the optimisation method used to minimise the loss
function during neural network training. Stochastic Gradient Descent
is a variant of Gradient Descent that uses mini-batches of data.
ResNet was developed by Microsoft Research and solves the vanishing
gradient problem using skip connections.
"""

# =============================================================================
# HELPERS
# =============================================================================

def _dummy_embedding_fn(texts: List[str]) -> np.ndarray:
    """Deterministic fallback when HuggingFaceEmbedding is not available."""
    rng = np.random.default_rng(seed=42)
    return rng.random((len(texts), 384)).astype(np.float32)

def _make_chunk(text: str, chunk_id: int, source: str = "test_doc.pdf") -> Chunk:
    t = text.strip()
    return Chunk(text=t, chunk_id=chunk_id, start_char=0, end_char=len(t),
                 metadata={"source": source})

def header(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")

def run_test(description: str, condition: bool) -> None:
    global _total, _passed
    _total += 1
    if condition:
        _passed += 1
    print(f"  {'✓ PASS' if condition else '✗ FAIL'}  {description}")

# =============================================================================
# BASE — dataclasses + ABCs
# =============================================================================

def test_extracted_node_dataclass(verbose=False):
    header("ExtractedNode dataclass  [base.py]")
    node = ExtractedNode(
        node_id="abc123", name="BERT", entity_type="Model",
        description="A transformer-based model.",
        source_chunk="0", source="test_doc.pdf", aliases=["bert"],
    )
    run_test("Instantiates without error",           node is not None)
    run_test("node_id set",                          node.node_id == "abc123")
    run_test("name set",                             node.name == "BERT")
    run_test("entity_type set",                      node.entity_type == "Model")
    run_test("source set",                           node.source == "test_doc.pdf")
    run_test("source_chunk set",                     node.source_chunk == "0")
    run_test("embedding is None by default",         node.embedding is None)
    run_test("aliases is a list",                    isinstance(node.aliases, list))
    props = node.to_neo4j_properties()
    run_test("to_neo4j_properties() returns dict",   isinstance(props, dict))
    run_test("props contains node_id",               "node_id" in props)
    run_test("props contains source",                "source" in props)
    run_test("props does NOT contain embedding",     "embedding" not in props)
    node.embedding = np.ones(384, dtype=np.float32)
    run_test("embedding can be set as ndarray",      isinstance(node.embedding, np.ndarray))
    if verbose:
        print(f"    props keys: {list(props.keys())}")


def test_extracted_relationship_dataclass(verbose=False):
    header("ExtractedRelationship dataclass  [base.py]")
    rel = ExtractedRelationship(
        source_id="abc123", target_id="def456",
        source_name="BERT", target_name="Attention Mechanism",
        relation_type="USES", description="BERT uses Attention.",
        source_chunk="0", source="test_doc.pdf",
        confidence=0.95, mode="constrained",
    )
    run_test("Instantiates without error",           rel is not None)
    run_test("source_id set",                        rel.source_id == "abc123")
    run_test("target_id set",                        rel.target_id == "def456")
    run_test("relation_type set",                    rel.relation_type == "USES")
    run_test("confidence set",                       rel.confidence == 0.95)
    run_test("mode set",                             rel.mode == "constrained")
    props = rel.to_neo4j_properties()
    run_test("to_neo4j_properties() returns dict",   isinstance(props, dict))
    run_test("props contains relation_type",         "relation_type" in props)
    run_test("props contains confidence",            "confidence" in props)
    run_test("props contains mode",                  "mode" in props)
    if verbose:
        print(f"    props keys: {list(props.keys())}")


def test_abstract_base_classes(verbose=False):
    header("Abstract base classes  [base.py]")
    try:
        BaseNodeExtractor()
        run_test("BaseNodeExtractor not directly instantiable", False)
    except TypeError:
        run_test("BaseNodeExtractor not directly instantiable", True)
    try:
        BaseRelationshipExtractor()
        run_test("BaseRelationshipExtractor not directly instantiable", False)
    except TypeError:
        run_test("BaseRelationshipExtractor not directly instantiable", True)
    run_test("NodeExtractor subclasses BaseNodeExtractor",
             issubclass(NodeExtractor, BaseNodeExtractor))
    run_test("RelationshipExtractor subclasses BaseRelationshipExtractor",
             issubclass(RelationshipExtractor, BaseRelationshipExtractor))


# =============================================================================
# LLM BACKEND — real Groq call
# =============================================================================

def test_llm_backend(llm: LLMBackend, verbose=False):
    header("LLMBackend — real Groq API  [llm_backend.py]")
    run_test("Instance created",                     llm is not None)
    run_test("generate() is callable",               callable(llm.generate))
    run_test("generate_batch() is callable",         callable(llm.generate_batch))
    run_test("model is set",                         bool(llm.model))
    run_test("max_tokens is set",                    llm.max_tokens > 0)
    run_test("temperature is set",                   llm.temperature >= 0.0)

    response = llm.generate("Reply with exactly: OK")
    run_test("generate() returns a non-empty string",
             isinstance(response, str) and len(response) > 0)

    batch = llm.generate_batch(["Say: A", "Say: B"])
    run_test("generate_batch() returns a list",      isinstance(batch, list))
    run_test("generate_batch() length matches",      len(batch) == 2)
    run_test("generate_batch() each item is string", all(isinstance(r, str) for r in batch))

    if verbose:
        print(f"    model     : {llm.model}")
        print(f"    response  : {response!r}")
        print(f"    batch[0]  : {batch[0]!r}")
        print(f"    batch[1]  : {batch[1]!r}")


# =============================================================================
# RELATION TAXONOMY
# =============================================================================

def test_allowed_relations(verbose=False):
    header("Relation taxonomy  [relationships_extractor.py]")
    run_test("ALLOWED_RELATIONS is non-empty",        len(ALLOWED_RELATIONS) > 0)
    run_test("At least 20 types",                    len(ALLOWED_RELATIONS) >= 20)
    for r in ["USES", "EXTENDS", "IMPLEMENTS", "IMPROVES_UPON", "REPLACES", "COMBINES",
              "DEPENDS_ON", "PART_OF", "BASED_ON", "DERIVED_FROM",
              "EVALUATED_ON", "TRAINED_ON", "BENCHMARKED_AGAINST", "OUTPERFORMS",
              "INTRODUCES", "PROPOSED_BY", "DEVELOPED_BY", "PUBLISHED_IN",
              "COMPARES_WITH", "CONTRASTS_WITH", "EQUIVALENT_TO",
              "CITES", "BUILDS_ON", "MOTIVATED_BY", "ADDRESSES", "SOLVES",
              "APPLIED_TO", "GENERALIZES", "SPECIALIZES"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)
    run_test("CROSS_DOC_RELATIONS ⊆ ALLOWED_RELATIONS",
             all(r in ALLOWED_RELATIONS for r in CROSS_DOC_RELATIONS))
    if verbose:
        print(f"    Total: {len(ALLOWED_RELATIONS)}")


# =============================================================================
# NODE EXTRACTOR — real Groq calls, real spaCy
# =============================================================================

def test_node_extractor_init(llm, embedding_fn, verbose=False) -> NodeExtractor:
    header("NodeExtractor init  [node_extractor.py]")
    extractor = NodeExtractor(llm=llm, embedding_fn=embedding_fn, max_workers=2)
    run_test("Instantiates without error",           extractor is not None)
    run_test("Is BaseNodeExtractor subclass",        isinstance(extractor, BaseNodeExtractor))
    run_test("llm attached",                         extractor.llm is llm)
    run_test("spaCy model loaded",                   extractor._nlp is not None)
    run_test("allowed_types is a list",              isinstance(extractor.allowed_types, list))
    run_test("Cache starts empty",                   extractor._cache == {})
    run_test("'Model' in allowed_types",             "Model"     in extractor.allowed_types)
    run_test("'Method' in allowed_types",            "Method"    in extractor.allowed_types)
    run_test("'Theory' in allowed_types",            "Theory"    in extractor.allowed_types)
    run_test("'Framework' in allowed_types",         "Framework" in extractor.allowed_types)
    if verbose:
        print(f"    allowed_types: {extractor.allowed_types}")
    return extractor


def test_node_extractor_spacy(extractor: NodeExtractor, verbose=False):
    header("NodeExtractor — spaCy NER  [node_extractor.py]")
    text  = "Google developed BERT, a transformer model. Vaswani published the paper."
    items = extractor._extract_with_spacy(text)
    run_test("Returns a list",                       isinstance(items, list))
    run_test("At least 1 entity detected",           len(items) >= 1)
    run_test("All items have required keys",
             all("name" in i and "entity_type" in i and "description" in i for i in items))
    run_test("entity_type from spaCy label map",
             all(i["entity_type"] in ["Author", "Institution", "Model", "Framework"] for i in items))
    if verbose:
        for i in items:
            print(f"    spaCy: {i['name']} ({i['entity_type']})")


def test_node_extractor_llm(extractor: NodeExtractor, verbose=False):
    header("NodeExtractor — real Groq LLM extraction  [node_extractor.py]")
    items = extractor._extract_with_llm(TEXT_1)
    run_test("Returns a list",                       isinstance(items, list))
    run_test("At least 1 entity extracted",          len(items) >= 1)
    run_test("All items have required keys",
             all("name" in i and "entity_type" in i and "description" in i for i in items))
    run_test("All entity_types are strings",
             all(isinstance(i["entity_type"], str) and len(i["entity_type"]) > 0 for i in items))
    run_test("All entity_types are from allowed list",
             all(i["entity_type"] in extractor.allowed_types for i in items))
    if verbose:
        for i in items:
            print(f"    LLM: {i['name']} ({i['entity_type']}) — {i['description'][:60]}")


def test_node_extractor_merge(extractor: NodeExtractor, verbose=False):
    header("NodeExtractor — merge  [node_extractor.py]")
    spacy_items = [
        {"name": "Google", "entity_type": "Institution", "description": "Google.", "aliases": ["google"]},
        {"name": "BERT",   "entity_type": "Model",       "description": "BERT.",   "aliases": ["bert"]},
    ]
    llm_items = [
        {"name": "BERT",               "entity_type": "Model",  "description": "Transformer for NLU.", "aliases": ["bert"]},
        {"name": "Attention Mechanism","entity_type": "Method", "description": "Attention.",            "aliases": ["attention"]},
    ]
    merged = extractor._merge_entity_sets(spacy_items, llm_items)
    names  = [i["name"] for i in merged]
    run_test("Merged list is non-empty",             len(merged) > 0)
    run_test("BERT not duplicated",                  names.count("BERT") == 1)
    run_test("spaCy-only Google included",           "Google" in names)
    run_test("LLM Attention Mechanism included",     "Attention Mechanism" in names)
    if verbose:
        print(f"    Merged names: {names}")


def test_node_extractor_deduplication(extractor: NodeExtractor, verbose=False):
    header("NodeExtractor — deduplication  [node_extractor.py]")
    nid = extractor._make_node_id("BERT", "Model")
    nodes = [
        ExtractedNode(node_id=nid, name="BERT", entity_type="Model",
                      description="Short.", source_chunk="0", source="doc.pdf", aliases=["bert"]),
        ExtractedNode(node_id=nid, name="BERT", entity_type="Model",
                      description="A longer and more detailed description of BERT.",
                      source_chunk="1", source="doc.pdf", aliases=["bert", "BERT model"]),
        ExtractedNode(node_id=extractor._make_node_id("Dropout", "Method"),
                      name="Dropout", entity_type="Method",
                      description="Regularisation.", source_chunk="0", source="doc.pdf",
                      aliases=["dropout"]),
    ]
    deduped = extractor._deduplicate_nodes(nodes)
    bert    = next((n for n in deduped if n.name == "BERT"), None)
    run_test("2 unique nodes after dedup",           len(deduped) == 2)
    run_test("BERT node present",                    bert is not None)
    run_test("Longer description kept",              bert and "longer" in bert.description)
    run_test("Aliases merged",                       bert and "BERT model" in bert.aliases)
    if verbose:
        for n in deduped:
            print(f"    {n.name} | desc: {n.description[:50]} | aliases: {n.aliases}")


def test_node_extractor_embedding(extractor: NodeExtractor, verbose=False):
    header("NodeExtractor — _embed_nodes()  [node_extractor.py]")
    nodes = [
        ExtractedNode(node_id="a1", name="BERT",    entity_type="Model",
                      description="Transformer.", source_chunk="0", source="doc.pdf"),
        ExtractedNode(node_id="b2", name="Dropout", entity_type="Method",
                      description="Regularisation.", source_chunk="0", source="doc.pdf"),
    ]
    embedded = extractor._embed_nodes(nodes)
    run_test("Returns a list",                       isinstance(embedded, list))
    run_test("All nodes have embeddings",            all(n.embedding is not None for n in embedded))
    run_test("Embeddings are np.ndarray",            all(isinstance(n.embedding, np.ndarray) for n in embedded))
    run_test("Embeddings are 1-D",                   all(n.embedding.ndim == 1 for n in embedded))
    if verbose:
        for n in embedded:
            print(f"    {n.name}: shape {n.embedding.shape}")


def test_node_extractor_caching(extractor: NodeExtractor, verbose=False):
    header("NodeExtractor — caching  [node_extractor.py]")
    extractor._cache.clear()
    chunk = _make_chunk(TEXT_1, chunk_id=0)

    # First call — hits Groq
    extractor._extract_from_single_chunk(chunk)
    run_test("Cache populated after first call",     chunk.text in extractor._cache)

    # Second call — must serve from cache (no new Groq call)
    cached = extractor._cache[chunk.text]
    result = extractor._extract_from_single_chunk(chunk)
    run_test("Second call returns same nodes from cache",
             len(result) == len(cached))
    if verbose:
        print(f"    Cached nodes: {[n.name for n in cached]}")


def test_node_extractor_single_chunk(extractor: NodeExtractor, verbose=False):
    header("NodeExtractor — extract_from_single_chunk_public()  [node_extractor.py]")
    extractor._cache.clear()
    chunk = _make_chunk(TEXT_1, chunk_id=0, source="paper.pdf")
    nodes = extractor.extract_from_single_chunk_public(chunk)
    run_test("Returns a list",                       isinstance(nodes, list))
    run_test("At least 1 node",                      len(nodes) >= 1)
    run_test("All are ExtractedNode",                all(isinstance(n, ExtractedNode) for n in nodes))
    run_test("source == paper.pdf",                  all(n.source == "paper.pdf" for n in nodes))
    run_test("source_chunk == '0'",                  all(n.source_chunk == "0" for n in nodes))
    run_test("All have embeddings",                  all(n.embedding is not None for n in nodes))
    if verbose:
        for n in nodes:
            print(f"    {n.name} ({n.entity_type})")


def test_node_extractor_full_batch(extractor: NodeExtractor, verbose=False) -> List[ExtractedNode]:
    header("NodeExtractor — extract_from_chunks()  [node_extractor.py]")
    extractor._cache.clear()
    chunks = [
        _make_chunk(TEXT_1, chunk_id=0, source="doc.pdf"),
        _make_chunk(TEXT_2, chunk_id=1, source="doc.pdf"),
        _make_chunk(TEXT_3, chunk_id=2, source="doc.pdf"),
    ]
    nodes = extractor.extract_from_chunks(chunks, show_progress=True)
    run_test("Returns a list",                       isinstance(nodes, list))
    run_test("At least 1 node",                      len(nodes) >= 1)
    run_test("All are ExtractedNode",                all(isinstance(n, ExtractedNode) for n in nodes))
    run_test("All have embeddings",                  all(n.embedding is not None for n in nodes))
    run_test("No duplicate node_ids",                len({n.node_id for n in nodes}) == len(nodes))
    if verbose:
        print(f"    Total unique nodes: {len(nodes)}")
        for n in nodes:
            print(f"    {n.name} ({n.entity_type}) — {n.description[:60]}")
    return nodes


# =============================================================================
# RELATIONSHIP EXTRACTOR — real Groq calls
# =============================================================================

def test_rel_extractor_init(llm, verbose=False):
    header("RelationshipExtractor init  [relationships_extractor.py]")
    c = RelationshipExtractor(llm=llm, mode="constrained",   confidence_threshold=0.6, max_entity_pairs=20)
    u = RelationshipExtractor(llm=llm, mode="unconstrained", confidence_threshold=0.6)
    run_test("Constrained instantiates",             c is not None)
    run_test("Unconstrained instantiates",           u is not None)
    run_test("Is BaseRelationshipExtractor",         isinstance(c, BaseRelationshipExtractor))
    run_test("mode='constrained' set",               c.mode == "constrained")
    run_test("mode='unconstrained' set",             u.mode == "unconstrained")
    run_test("confidence_threshold set",             c.confidence_threshold == 0.6)
    run_test("max_entity_pairs set",                 c.max_entity_pairs == 20)
    try:
        RelationshipExtractor(llm=llm, mode="bad_mode")
        run_test("Invalid mode raises ValueError",   False)
    except ValueError:
        run_test("Invalid mode raises ValueError",   True)
    return c, u


def test_rel_extractor_constrained(extractor, nodes, verbose=False):
    header("RelationshipExtractor — constrained (real Groq)  [relationships_extractor.py]")
    chunk       = _make_chunk(TEXT_1, chunk_id=0, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}
    rels = extractor.extract_from_chunk(chunk, nodes_in_chunk=nodes[:6], node_id_map=node_id_map)
    run_test("Returns a list",                       isinstance(rels, list))
    run_test("All are ExtractedRelationship",        all(isinstance(r, ExtractedRelationship) for r in rels))
    run_test("All above confidence threshold",       all(r.confidence >= extractor.confidence_threshold for r in rels))
    run_test("All rel_types in ALLOWED_RELATIONS",   all(r.relation_type in ALLOWED_RELATIONS for r in rels))
    run_test("All have mode='constrained'",          all(r.mode == "constrained" for r in rels))
    run_test("All have source='doc.pdf'",            all(r.source == "doc.pdf" for r in rels))
    run_test("source_id and target_id set",          all(r.source_id and r.target_id for r in rels))
    if verbose:
        for r in rels:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name} ({r.confidence:.2f})")
            print(f"      {r.description}")
    return rels


def test_rel_extractor_unconstrained(extractor, nodes, verbose=False):
    header("RelationshipExtractor — unconstrained (real Groq)  [relationships_extractor.py]")
    chunk       = _make_chunk(TEXT_1, chunk_id=0, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}
    rels = extractor.extract_from_chunk(chunk, nodes_in_chunk=nodes[:6], node_id_map=node_id_map)
    run_test("Returns a list",                       isinstance(rels, list))
    run_test("All have mode='unconstrained'",         all(r.mode == "unconstrained" for r in rels))
    run_test("All rel_types are non-empty strings",
             all(isinstance(r.relation_type, str) and len(r.relation_type) > 0 for r in rels))
    run_test("All rel_types are UPPER_SNAKE_CASE",
             all(r.relation_type == r.relation_type.upper() for r in rels))
    if verbose:
        for r in rels:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name} ({r.confidence:.2f})")


def test_rel_extractor_cross_document(extractor, nodes, verbose=False):
    header("RelationshipExtractor — cross-document (real Groq)  [relationships_extractor.py]")
    chunk       = _make_chunk(TEXT_2, chunk_id=1, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}
    cross = extractor.extract_cross_document_references(
        chunk=chunk, node_id_map=node_id_map, all_document_nodes={"doc.pdf": nodes},
    )
    run_test("Returns a list",                       isinstance(cross, list))
    run_test("All are ExtractedRelationship",        all(isinstance(r, ExtractedRelationship) for r in cross))

    # Chunk without reference markers — should skip Groq entirely
    plain_chunk = _make_chunk("The sky is blue. Water is wet.", chunk_id=99, source="doc.pdf")
    no_refs = extractor.extract_cross_document_references(
        chunk=plain_chunk, node_id_map=node_id_map, all_document_nodes={"doc.pdf": nodes},
    )
    run_test("Chunk without reference markers returns []", no_refs == [])
    if verbose:
        print(f"    Cross-doc refs found: {len(cross)}")
        for r in cross:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name}")


def test_rel_extractor_deduplication(extractor, verbose=False):
    header("RelationshipExtractor — deduplication  [relationships_extractor.py]")
    rels = [
        ExtractedRelationship("a","b","BERT","Attention Mechanism","USES","First.",
                              "0","doc.pdf",confidence=0.70),
        ExtractedRelationship("a","b","BERT","Attention Mechanism","USES","Second.",
                              "1","doc.pdf",confidence=0.95),
        ExtractedRelationship("a","c","BERT","SQuAD","EVALUATED_ON","On SQuAD.",
                              "0","doc.pdf",confidence=0.90),
    ]
    deduped  = extractor.deduplicate_relationships(rels)
    uses_rel = next((r for r in deduped if r.relation_type == "USES"), None)
    run_test("2 unique rels after dedup",            len(deduped) == 2)
    run_test("USES rel kept",                        uses_rel is not None)
    run_test("Higher confidence (0.95) kept",        uses_rel and uses_rel.confidence == 0.95)
    if verbose:
        for r in deduped:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name} ({r.confidence})")


def test_rel_extractor_pair_priority(extractor, nodes, verbose=False):
    header("RelationshipExtractor — pair prioritisation  [relationships_extractor.py]")
    from itertools import combinations
    pairs = list(combinations(nodes[:6], 2))
    if len(pairs) < 2:
        print("  SKIP  Not enough nodes.")
        return
    prioritised = extractor._prioritise_pairs(pairs, max_pairs=5)
    run_test("Returns a list",                       isinstance(prioritised, list))
    run_test("Respects max_pairs cap",               len(prioritised) <= 5)
    run_test("At least 1 pair returned",             len(prioritised) >= 1)
    if verbose:
        for a, b in prioritised:
            print(f"    {a.name} ({a.entity_type}) ↔ {b.name} ({b.entity_type})")


# =============================================================================
# GRAPH STORE — utility (no Neo4j needed)
# =============================================================================

def test_sanitise_rel_type(verbose=False):
    header("_sanitise_rel_type  [graph_store.py]")
    run_test("'USES' → 'USES'",                      _sanitise_rel_type("USES") == "USES")
    run_test("'uses' → 'USES'",                      _sanitise_rel_type("uses") == "USES")
    run_test("'Based On' → 'BASED_ON'",              _sanitise_rel_type("Based On") == "BASED_ON")
    run_test("'with-hyphen' → 'WITH_HYPHEN'",        _sanitise_rel_type("with-hyphen") == "WITH_HYPHEN")
    run_test("empty string → ''",                    _sanitise_rel_type("") == "")
    run_test("'123bad' → '' (starts with digit)",    _sanitise_rel_type("123bad") == "")
    if verbose:
        for s in ["USES", "uses", "Based On", "with-hyphen", "", "123bad"]:
            print(f"    '{s}' → '{_sanitise_rel_type(s)}'")


# =============================================================================
# GRAPH STORE — Neo4j integration
# Uses YOUR real GraphStore. Nodes have real embeddings from YOUR embedding_fn.
# =============================================================================

def _make_store_test_nodes(embedding_fn) -> List[ExtractedNode]:
    import hashlib
    texts = [
        ("BERT",               "Model",   "Transformer-based NLU model."),
        ("Attention Mechanism","Method",  "Allows focus on relevant input parts."),
        ("SQuAD",              "Dataset", "Stanford QA dataset."),
        ("Dropout",            "Method",  "Regularisation technique."),
    ]
    # Get real embeddings for all nodes in one batch
    embed_texts = [f"{name}: {desc}" for name, _, desc in texts]
    vectors     = embedding_fn(embed_texts)

    nodes = []
    for (name, etype, desc), vec in zip(texts, vectors):
        nid = hashlib.md5(f"{name.lower()}:{etype.lower()}".encode()).hexdigest()[:16]
        nodes.append(ExtractedNode(
            node_id=nid, name=name, entity_type=etype,
            description=desc, source_chunk="chunk_0",
            source="gs_test.pdf", aliases=[name.lower()],
            embedding=vec,
        ))
    return nodes


def _make_store_test_rels(nodes: List[ExtractedNode]) -> List[ExtractedRelationship]:
    m = {n.name: n for n in nodes}
    return [
        ExtractedRelationship(
            source_id=m["BERT"].node_id, target_id=m["Attention Mechanism"].node_id,
            source_name="BERT", target_name="Attention Mechanism",
            relation_type="USES", description="BERT uses Attention Mechanism.",
            source_chunk="chunk_0", source="gs_test.pdf", confidence=0.95,
        ),
        ExtractedRelationship(
            source_id=m["BERT"].node_id, target_id=m["SQuAD"].node_id,
            source_name="BERT", target_name="SQuAD",
            relation_type="EVALUATED_ON", description="BERT evaluated on SQuAD.",
            source_chunk="chunk_0", source="gs_test.pdf", confidence=0.92,
        ),
        ExtractedRelationship(
            source_id=m["BERT"].node_id, target_id=m["Dropout"].node_id,
            source_name="BERT", target_name="Dropout",
            relation_type="USES", description="BERT uses Dropout.",
            source_chunk="chunk_0", source="gs_test.pdf", confidence=0.88,
        ),
    ]


def test_graph_store_connection(store: GraphStore, verbose=False):
    header("GraphStore — connection  [graph_store.py]")
    run_test("GraphStore instance created",          store is not None)
    run_test("uri is set",                           bool(store.uri))
    run_test("database is set",                      bool(store.database))
    if verbose:
        print(f"    uri={store.uri}  db={store.database}")


def test_graph_store_schema(store: GraphStore, verbose=False):
    header("GraphStore — init_schema()  [graph_store.py]")
    try:
        store.init_schema()
        run_test("init_schema() runs without error", True)
    except Exception as e:
        run_test(f"init_schema() failed: {e}",       False)


def test_graph_store_upsert_nodes(store: GraphStore, nodes: List[ExtractedNode], verbose=False):
    header("GraphStore — upsert_nodes()  [graph_store.py]")
    written = store.upsert_nodes(nodes)
    run_test("Returns count > 0",                    written > 0)
    run_test(f"count == {len(nodes)}",               written == len(nodes))
    for node in nodes:
        fetched = store.get_node_by_id(node.node_id)
        run_test(f"'{node.name}' retrievable by node_id",
                 fetched is not None and fetched["node_id"] == node.node_id)
        run_test(f"'{node.name}' name matches",      fetched and fetched["name"] == node.name)
        run_test(f"'{node.name}' entity_type matches",
                 fetched and fetched["entity_type"] == node.entity_type)
        run_test(f"'{node.name}' source matches",    fetched and fetched["source"] == "gs_test.pdf")
    by_name = store.get_node_by_name("BERT")
    run_test("get_node_by_name('BERT') returns result", by_name is not None)
    run_test("result entity_type == 'Model'",        by_name and by_name["entity_type"] == "Model")
    if verbose:
        print(f"    Nodes written: {written}")


def test_graph_store_idempotent(store: GraphStore, nodes: List[ExtractedNode], verbose=False):
    header("GraphStore — upsert idempotency  [graph_store.py]")
    count_before = store.count_nodes()
    store.upsert_nodes(nodes)
    count_after  = store.count_nodes()
    run_test("Node count unchanged after re-upsert", count_before == count_after)
    if verbose:
        print(f"    before={count_before}  after={count_after}")


def test_graph_store_upsert_rels(store, nodes, rels, verbose=False):
    header("GraphStore — upsert_relationships()  [graph_store.py]")
    written  = store.upsert_relationships(rels)
    run_test("Returns count > 0",                    written > 0)
    run_test(f"count == {len(rels)}",                written == len(rels))
    node_ids = [n.node_id for n in nodes]
    subgraph = store.get_subgraph(node_ids)
    run_test("get_subgraph() has 'nodes' + 'relationships'",
             "nodes" in subgraph and "relationships" in subgraph)
    run_test("Subgraph has all 4 nodes",             len(subgraph["nodes"]) == 4)
    run_test("Subgraph has 3 relationships",         len(subgraph["relationships"]) == 3)
    rel_types = {r["relation_type"] for r in subgraph["relationships"]}
    run_test("USES present in subgraph",             "USES" in rel_types)
    run_test("EVALUATED_ON present in subgraph",     "EVALUATED_ON" in rel_types)
    if verbose:
        print(f"    Written: {written}  rel_types: {rel_types}")


def test_graph_store_counts(store: GraphStore, verbose=False):
    header("GraphStore — count_nodes / count_relationships  [graph_store.py]")
    n = store.count_nodes()
    r = store.count_relationships()
    run_test("count_nodes() is int >= 0",            isinstance(n, int) and n >= 0)
    run_test("count_relationships() is int >= 0",    isinstance(r, int) and r >= 0)
    run_test("At least 4 nodes in graph",            n >= 4)
    run_test("At least 3 relationships in graph",    r >= 3)
    if verbose:
        print(f"    nodes={n}  rels={r}")


def test_graph_store_neighbourhood(store: GraphStore, nodes: List[ExtractedNode], verbose=False):
    header("GraphStore — get_neighbourhood()  [graph_store.py]")
    bert = next(n for n in nodes if n.name == "BERT")
    nbr  = store.get_neighbourhood(bert.node_id, hops=1)
    run_test("Returns dict with 'nodes' and 'relationships'",
             "nodes" in nbr and "relationships" in nbr)
    run_test("More than 1 node (BERT + neighbours)", len(nbr["nodes"]) > 1)
    run_test("At least 1 relationship",              len(nbr["relationships"]) >= 1)
    names = {n.get("name") for n in nbr["nodes"]}
    run_test("BERT itself in neighbourhood",         "BERT" in names)
    run_test("At least one connected entity present",
             bool(names & {"Attention Mechanism", "SQuAD", "Dropout"}))
    if verbose:
        print(f"    Neighbours of BERT: {names}")


def test_graph_store_similarity_search(store: GraphStore, nodes: List[ExtractedNode], verbose=False):
    header("GraphStore — similarity_search()  [graph_store.py]")
    bert    = next(n for n in nodes if n.name == "BERT")
    results = store.similarity_search(bert.embedding, top_k=3)
    run_test("Returns a list",                       isinstance(results, list))
    run_test("At most top_k=3 results",              len(results) <= 3)
    run_test("Each result has 'node_id'",            all("node_id" in r for r in results))
    run_test("Each result has 'score'",              all("score" in r for r in results))
    run_test("Scores are floats in [0, 1]",
             all(isinstance(r["score"], float) and 0.0 <= r["score"] <= 1.0 for r in results))
    if results:
        run_test("Top result is BERT (self-similarity)", results[0]["node_id"] == bert.node_id)
    if verbose:
        for r in results:
            print(f"    {r['name']} ({r['entity_type']}) — score: {r['score']:.4f}")


def test_graph_store_paths_between(store: GraphStore, nodes: List[ExtractedNode], verbose=False):
    header("GraphStore — get_paths_between()  [graph_store.py]")
    bert  = next(n for n in nodes if n.name == "BERT")
    squad = next(n for n in nodes if n.name == "SQuAD")
    paths = store.get_paths_between(bert.node_id, squad.node_id, max_hops=2)
    run_test("Returns a list of paths",              isinstance(paths, list))
    run_test("At least one path found",              len(paths) >= 1)
    if paths:
        path  = paths[0]
        types = {item.get("type") for item in path}
        run_test("Path contains 'node' items",       "node" in types)
        run_test("Path contains 'relationship' items","relationship" in types)
        run_test("Path has at least 3 items (n-r-n)", len(path) >= 3)
    if verbose:
        for i, path in enumerate(paths):
            summary = " → ".join(
                item.get("name") or item.get("relation_type", "?") for item in path
            )
            print(f"    Path {i+1}: {summary}")


def test_graph_store_delete_by_source(store: GraphStore, verbose=False):
    header("GraphStore — delete_by_source()  [graph_store.py]")
    import hashlib
    tmp = ExtractedNode(
        node_id     = hashlib.md5(b"tmp_delete_test_node").hexdigest()[:16],
        name        = "TmpDeleteTestNode",
        entity_type = "Concept",
        description = "Temporary node for delete test.",
        source_chunk= "tmp_chunk",
        source      = "tmp_delete_source.pdf",
        embedding   = np.random.default_rng(99).random(384).astype(np.float32),
    )
    store.upsert_nodes([tmp])
    run_test("Temp node inserted before delete",
             store.get_node_by_id(tmp.node_id) is not None)
    n_del, r_del = store.delete_by_source("tmp_delete_source.pdf")
    run_test("Returns (int, int)",                   isinstance(n_del, int) and isinstance(r_del, int))
    run_test("At least 1 node deleted",              n_del >= 1)
    run_test("Temp node gone from graph",            store.get_node_by_id(tmp.node_id) is None)
    if verbose:
        print(f"    Deleted — nodes: {n_del}  rels: {r_del}")


# =============================================================================
# PIPELINE — real Groq + real Neo4j end-to-end
# =============================================================================

def test_pipeline_stats_dataclass(verbose=False):
    header("PipelineStats dataclass  [Pipeline.py]")
    stats = PipelineStats(
        chunks_processed=3, nodes_extracted=10, nodes_written=10,
        relationships_extracted=8, relationships_written=8,
        cross_doc_relationships=2, elapsed_seconds=1.5,
    )
    run_test("Instantiates",                         stats is not None)
    run_test("chunks_processed set",                 stats.chunks_processed == 3)
    run_test("nodes_extracted set",                  stats.nodes_extracted == 10)
    run_test("nodes_written set",                    stats.nodes_written == 10)
    run_test("relationships_extracted set",          stats.relationships_extracted == 8)
    run_test("relationships_written set",            stats.relationships_written == 8)
    run_test("cross_doc_relationships set",          stats.cross_doc_relationships == 2)
    run_test("elapsed_seconds set",                  stats.elapsed_seconds == 1.5)
    run_test("errors is [] by default",              stats.errors == [])
    s = str(stats)
    run_test("__str__() contains chunk count",       "3" in s)
    run_test("__str__() contains node count",        "10" in s)
    if verbose:
        print(stats)


def test_pipeline_from_components(llm, embedding_fn, store: GraphStore, verbose=False) -> Pipeline:
    header("Pipeline.from_components()  [Pipeline.py]")
    node_ext = NodeExtractor(llm=llm, embedding_fn=embedding_fn, max_workers=2)
    rel_ext  = RelationshipExtractor(llm=llm, mode="constrained", confidence_threshold=0.6)
    pipeline = Pipeline.from_components(
        node_extractor        = node_ext,
        relationship_extractor= rel_ext,
        graph_store           = store,
        extract_cross_doc     = True,
        show_progress         = True,
        schema_already_exists = True,
    )
    run_test("Pipeline.from_components() instantiates", pipeline is not None)
    run_test("_node_extractor is NodeExtractor",     isinstance(pipeline._node_extractor, NodeExtractor))
    run_test("_relationship_extractor is RelationshipExtractor",
             isinstance(pipeline._relationship_extractor, RelationshipExtractor))
    run_test("_store is GraphStore",                 isinstance(pipeline._store, GraphStore))
    run_test("_extract_cross_doc is True",           pipeline._extract_cross_doc is True)
    run_test("repr() is a string",                   isinstance(repr(pipeline), str))
    if verbose:
        print(f"    {pipeline!r}")
    return pipeline


def test_pipeline_empty_chunks(pipeline: Pipeline, verbose=False):
    header("Pipeline — empty chunk list  [Pipeline.py]")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stats = pipeline.run([])
    run_test("run([]) returns PipelineStats",        isinstance(stats, PipelineStats))
    run_test("chunks_processed == 0",                stats.chunks_processed == 0)
    run_test("Warning issued for empty chunks",
             any("empty" in str(x.message).lower() for x in w))


def test_pipeline_run(pipeline: Pipeline, verbose=False) -> PipelineStats:
    """
    Full end-to-end run: Groq extracts nodes + rels, GraphStore writes to Neo4j,
    then we query Neo4j directly to confirm everything is actually stored.
    """
    header("Pipeline.run() — full ingestion + Neo4j verification  [Pipeline.py]")

    pipeline._node_extractor._cache.clear()

    chunks = [
        _make_chunk(TEXT_1, chunk_id=0, source="pipeline_test.pdf"),
        _make_chunk(TEXT_2, chunk_id=1, source="pipeline_test.pdf"),
        _make_chunk(TEXT_3, chunk_id=2, source="pipeline_test.pdf"),
    ]

    stats = pipeline.run(chunks)

    run_test("run() returns PipelineStats",          isinstance(stats, PipelineStats))
    run_test("chunks_processed == 3",                stats.chunks_processed == 3)
    run_test("nodes_extracted > 0",                  stats.nodes_extracted > 0)
    run_test("nodes_written > 0",                    stats.nodes_written > 0)
    run_test("nodes_written == nodes_extracted",     stats.nodes_written == stats.nodes_extracted)
    run_test("relationships_written >= 0",           stats.relationships_written >= 0)
    run_test("elapsed_seconds > 0",                  stats.elapsed_seconds > 0)

    # ── Verify data is ACTUALLY in Neo4j ──────────────────────────────────────
    store = pipeline._store

    run_test("Neo4j node count > 0",                 store.count_nodes() > 0)

    bert = store.get_node_by_name("BERT")
    run_test("'BERT' node stored in Neo4j",          bert is not None)
    run_test("'BERT' entity_type == 'Model'",        bert and bert["entity_type"] == "Model")
    run_test("'BERT' source == 'pipeline_test.pdf'", bert and bert["source"] == "pipeline_test.pdf")
    run_test("'BERT' has embedding in Neo4j",        bert and bert.get("embedding") is not None)

    run_test("Neo4j relationship count > 0",         store.count_relationships() > 0)

    if bert:
        nbr = store.get_neighbourhood(bert["node_id"], hops=1)
        run_test("BERT has at least 1 neighbour in Neo4j",
                 len(nbr["nodes"]) > 1)

    if verbose:
        print(stats)
        print(f"    Neo4j — nodes: {store.count_nodes()}  rels: {store.count_relationships()}")

    return stats


def test_pipeline_graph_stats(pipeline: Pipeline, verbose=False):
    header("Pipeline.graph_stats()  [Pipeline.py]")
    gs = pipeline.graph_stats()
    run_test("Returns a dict",                       isinstance(gs, dict))
    run_test("'nodes' key present",                  "nodes" in gs)
    run_test("'relationships' key present",          "relationships" in gs)
    run_test("nodes count > 0",                      gs["nodes"] > 0)
    run_test("relationships count > 0",              gs["relationships"] > 0)
    if verbose:
        print(f"    graph_stats: {gs}")


def test_pipeline_ingest_chunk(pipeline: Pipeline, verbose=False):
    header("Pipeline.ingest_chunk() — incremental ingestion  [Pipeline.py]")
    pipeline._node_extractor._cache.clear()
    chunk = _make_chunk(TEXT_1, chunk_id=99, source="incremental_test.pdf")
    stats = pipeline.ingest_chunk(chunk)

    run_test("Returns PipelineStats",                isinstance(stats, PipelineStats))
    run_test("chunks_processed == 1",                stats.chunks_processed == 1)
    run_test("nodes_extracted > 0",                  stats.nodes_extracted > 0)
    run_test("nodes_written > 0",                    stats.nodes_written > 0)

    bert = pipeline._store.get_node_by_name("BERT")
    run_test("'BERT' in Neo4j after incremental ingest", bert is not None)
    if verbose:
        print(stats)


def test_pipeline_idempotent(pipeline: Pipeline, verbose=False):
    header("Pipeline — idempotent re-ingestion  [Pipeline.py]")
    pipeline._node_extractor._cache.clear()
    store        = pipeline._store
    nodes_before = store.count_nodes()
    rels_before  = store.count_relationships()

    chunks = [
        _make_chunk(TEXT_1, chunk_id=0, source="pipeline_test.pdf"),
        _make_chunk(TEXT_2, chunk_id=1, source="pipeline_test.pdf"),
    ]
    pipeline.run(chunks)

    nodes_after = store.count_nodes()
    rels_after  = store.count_relationships()
    run_test("Node count unchanged after re-run",    nodes_after == nodes_before)
    run_test("Rel count unchanged after re-run",     rels_after  == rels_before)
    if verbose:
        print(f"    nodes: {nodes_before}→{nodes_after}  rels: {rels_before}→{rels_after}")


def test_pipeline_context_manager(llm, embedding_fn, neo4j_args: dict, verbose=False):
    header("Pipeline — context manager  [Pipeline.py]")
    node_ext = NodeExtractor(llm=llm, embedding_fn=embedding_fn, max_workers=1)
    rel_ext  = RelationshipExtractor(llm=llm, mode="constrained")
    store    = GraphStore(**neo4j_args, embedding_dim=384)
    store.init_schema()
    with Pipeline.from_components(
        node_extractor        = node_ext,
        relationship_extractor= rel_ext,
        graph_store           = store,
        schema_already_exists = True,
        show_progress         = False,
    ) as p:
        run_test("Pipeline usable inside 'with' block", p is not None)
    try:
        p.graph_stats()
        run_test("Driver closed after context manager exits", False)
    except Exception:
        run_test("Driver closed after context manager exits", True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KG-RAG Full Test Suite")
    parser.add_argument("-v", "--verbose",  action="store_true")
    parser.add_argument("--neo4j",          action="store_true",
                        help="Run Neo4j integration + pipeline tests")
    parser.add_argument("--uri",     default=os.environ.get("NEO4J_URI",      "neo4j://127.0.0.1:7687"))
    parser.add_argument("--user",    default=os.environ.get("NEO4J_USER",     "neo4j"))
    parser.add_argument("--password",default=os.environ.get("NEO4J_PASSWORD", "neo4j1234"))
    parser.add_argument("--model",   default="llama-3.3-70b-versatile",
                        help="Groq model to use")
    args = parser.parse_args()
    v = args.verbose

    # ── YOUR Groq LLM ─────────────────────────────────────────────────────────
    print("\nConnecting to Groq...")
    try:
        llm = LLMBackend(model=args.model, max_tokens=1024, temperature=0.0)
        print(f"  ✓ {llm}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Make sure GROQ_API_KEY is set in your environment or .env file.")
        sys.exit(1)

    # ── YOUR embedding function ───────────────────────────────────────────────
    if _EMBEDDINGS_AVAILABLE:
        print("Loading embedding model...")
        embedding_fn = HuggingFaceEmbedding().encode
        print("  ✓ HuggingFaceEmbedding ready")
    else:
        embedding_fn = _dummy_embedding_fn
        print("  Using deterministic dummy embedding (dim=384)")

    # ── YOUR Neo4j GraphStore ─────────────────────────────────────────────────
    store = None
    neo4j_args = {
        "uri"     : args.uri,
        "user"    : args.user,
        "password": args.password,
        "database": "neo4j",
    }

    if args.neo4j:
        print(f"\nConnecting to Neo4j at {args.uri}...")
        try:
            store = GraphStore(**neo4j_args, embedding_dim=384)
            for src in ["gs_test.pdf", "pipeline_test.pdf",
                        "incremental_test.pdf", "tmp_delete_source.pdf"]:
                store.delete_by_source(src)
            print("  ✓ Neo4j connected and cleaned")
        except Exception as e:
            print(f"  ERROR: {e}\n  Neo4j tests will be skipped.")
            store = None

    print(f"\n{'=' * 70}")
    print(f"  KG-RAG Full Test Suite  (everything real, nothing mocked)")
    print(f"  LLM       : {llm.model}")
    print(f"  Embedding : {'HuggingFaceEmbedding' if _EMBEDDINGS_AVAILABLE else 'dummy (dim=384)'}")
    print(f"  Neo4j     : {args.uri if store else 'Skipped (add --neo4j)'}")
    print(f"{'=' * 70}")

    # ── BASE ──────────────────────────────────────────────────────────────────
    test_extracted_node_dataclass(verbose=v)
    test_extracted_relationship_dataclass(verbose=v)
    test_abstract_base_classes(verbose=v)

    # ── LLM BACKEND ───────────────────────────────────────────────────────────
    test_llm_backend(llm, verbose=v)

    # ── RELATION TAXONOMY ─────────────────────────────────────────────────────
    test_allowed_relations(verbose=v)

    # ── NODE EXTRACTOR ────────────────────────────────────────────────────────
    node_extractor = test_node_extractor_init(llm, embedding_fn, verbose=v)
    test_node_extractor_spacy(node_extractor, verbose=v)
    test_node_extractor_llm(node_extractor, verbose=v)
    test_node_extractor_caching(node_extractor, verbose=v)
    test_node_extractor_merge(node_extractor, verbose=v)
    test_node_extractor_deduplication(node_extractor, verbose=v)
    test_node_extractor_embedding(node_extractor, verbose=v)
    test_node_extractor_single_chunk(node_extractor, verbose=v)
    nodes = test_node_extractor_full_batch(node_extractor, verbose=v)

    # ── RELATIONSHIP EXTRACTOR ────────────────────────────────────────────────
    constrained_ext, unconstrained_ext = test_rel_extractor_init(llm, verbose=v)
    test_rel_extractor_constrained(constrained_ext,    nodes, verbose=v)
    test_rel_extractor_unconstrained(unconstrained_ext, nodes, verbose=v)
    test_rel_extractor_cross_document(constrained_ext, nodes, verbose=v)
    test_rel_extractor_deduplication(constrained_ext, verbose=v)
    test_rel_extractor_pair_priority(constrained_ext, nodes, verbose=v)

    # ── GRAPH STORE — utility ─────────────────────────────────────────────────
    test_sanitise_rel_type(verbose=v)

    # ── GRAPH STORE — Neo4j ───────────────────────────────────────────────────
    if store is not None:
        test_nodes = _make_store_test_nodes(embedding_fn)
        test_rels  = _make_store_test_rels(test_nodes)
        test_graph_store_connection(store, verbose=v)
        test_graph_store_schema(store, verbose=v)
        test_graph_store_upsert_nodes(store, test_nodes, verbose=v)
        test_graph_store_idempotent(store, test_nodes, verbose=v)
        test_graph_store_upsert_rels(store, test_nodes, test_rels, verbose=v)
        test_graph_store_counts(store, verbose=v)
        test_graph_store_neighbourhood(store, test_nodes, verbose=v)
        test_graph_store_similarity_search(store, test_nodes, verbose=v)
        test_graph_store_paths_between(store, test_nodes, verbose=v)
        # test_graph_store_delete_by_source(store, verbose=v)
        # store.delete_by_source("gs_test.pdf")
    else:
        header("GraphStore Neo4j tests — SKIPPED")
        print("  Run with: python test_real.py --neo4j --password <pwd>")

    # ── PIPELINE ──────────────────────────────────────────────────────────────
    test_pipeline_stats_dataclass(verbose=v)

    if store is not None:
        pipeline = test_pipeline_from_components(llm, embedding_fn, store, verbose=v)
        test_pipeline_empty_chunks(pipeline, verbose=v)
        test_pipeline_run(pipeline, verbose=v)
        test_pipeline_graph_stats(pipeline, verbose=v)
        test_pipeline_ingest_chunk(pipeline, verbose=v)
        test_pipeline_idempotent(pipeline, verbose=v)
        test_pipeline_context_manager(llm, embedding_fn, neo4j_args, verbose=v)
        # store.delete_by_source("pipeline_test.pdf")
        # store.delete_by_source("incremental_test.pdf")
        store.close()
    else:
        header("Pipeline Neo4j tests — SKIPPED")
        print("  Run with: python test_real.py --neo4j --password <pwd>")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    failed = _total - _passed
    print(f"\n{'=' * 70}")
    if failed == 0:
        print(f"  ALL {_total} tests passed ✓")
    else:
        print(f"  {_passed} / {_total} passed   ({failed} FAILED ✗)")
    print(f"{'=' * 70}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()