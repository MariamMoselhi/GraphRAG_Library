"""
graph_test.py — Test suite for GraphRAG Graph Module
=====================================================
Tests the graph layer (base.py, llm_backend.py, node_extractor.py,
relationships_extractor.py) using a mock LLMBackend by default so
no API key or network call is needed.

Run
---
    python graph_test.py              # mock mode — no model needed
    python graph_test.py --mock       # explicit mock mode
    python graph_test.py -v           # verbose output
    python graph_test.py --mock -v    # verbose mock

Module structure being tested
------------------------------
    graph/__init__.py                  <- public re-exports
        graph/base.py                  <- ExtractedNode, ExtractedRelationship, ABCs
        graph/llm_backend.py           <- LLMBackend (Groq HTTP wrapper)
        graph/node_extractor.py        <- NodeExtractor (spaCy + LLM hybrid)
        graph/relationships_extractor.py <- RelationshipExtractor (constrained / unconstrained)

Dependencies
------------
    chunking/chunk_base.py  — Chunk dataclass
    embeddings/             — HuggingFaceEmbedding (optional)
    spacy en_core_web_sm    — always needed (local NER)
"""

from __future__ import annotations

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Load .env file (must happen before any LLMBackend instantiation)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set externally

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # graph/
_ROOT = _HERE.parent                      # project_root/
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Import Chunk
# ---------------------------------------------------------------------------
try:
    from chunking.chunk_base import Chunk
    print("✓ Chunking module loaded successfully")
except ImportError as e:
    print(f"ERROR: Could not import Chunk — {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import graph module
# ---------------------------------------------------------------------------
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
    print("✓ Graph module loaded successfully")
except ImportError as e:
    print(f"ERROR: Could not import graph module — {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optional: embeddings
# ---------------------------------------------------------------------------
try:
    from embeddings import HuggingFaceEmbedding
    _EMBEDDINGS_AVAILABLE = True
    print("✓ Embeddings module loaded successfully")
except ImportError:
    _EMBEDDINGS_AVAILABLE = False
    print("WARNING: Embeddings module not found — using dummy embedding_fn")

# ---------------------------------------------------------------------------
# Global test counters
# ---------------------------------------------------------------------------
PASS = "✓ PASS"
FAIL = "✗ FAIL"
_total  = 0
_passed = 0

# ---------------------------------------------------------------------------
# Mock texts
# ---------------------------------------------------------------------------
MOCK_TEXT_1 = """
BERT, developed by Google, is a transformer-based model that uses
the Attention Mechanism for natural language understanding tasks.
Vaswani et al. introduced the Transformer architecture in 2017.
BERT is evaluated on the SQuAD dataset and achieves high F1 scores.
Dropout is used as a regularisation method to prevent overfitting.
"""

MOCK_TEXT_2 = """
Building on the Attention Mechanism introduced earlier, GPT-4 extends
BERT by using a decoder-only transformer architecture. Recall that
backpropagation is required to train all neural network models.
GPT-4 was developed by OpenAI and evaluated on multiple benchmarks.
The BLEU metric is commonly used to evaluate language generation tasks.
"""

MOCK_TEXT_3 = """
Gradient Descent is the optimisation method used to minimise the loss
function during neural network training. Stochastic Gradient Descent
is a variant of Gradient Descent that uses mini-batches of data.
ResNet was developed by Microsoft Research and solves the vanishing
gradient problem using skip connections.
"""

# ---------------------------------------------------------------------------
# Mock LLM JSON responses
# ---------------------------------------------------------------------------
MOCK_NODE_RESPONSE_1 = json.dumps({
    "entities": [
        {"name": "BERT", "entity_type": "Model",
         "description": "A transformer-based model for NLU developed by Google.",
         "aliases": ["bert"]},
        {"name": "Attention Mechanism", "entity_type": "Method",
         "description": "Allows models to focus on relevant input parts.",
         "aliases": ["attention", "self-attention"]},
        {"name": "Transformer", "entity_type": "Model",
         "description": "Neural network architecture by Vaswani et al. 2017.",
         "aliases": ["transformer architecture"]},
        {"name": "SQuAD", "entity_type": "Dataset",
         "description": "Stanford Question Answering Dataset.",
         "aliases": ["squad"]},
        {"name": "Dropout", "entity_type": "Method",
         "description": "Regularisation by randomly deactivating neurons.",
         "aliases": ["dropout regularisation"]},
        {"name": "F1", "entity_type": "Metric",
         "description": "Harmonic mean of precision and recall.",
         "aliases": ["f1 score"]},
    ]
})

MOCK_NODE_RESPONSE_2 = json.dumps({
    "entities": [
        {"name": "GPT-4", "entity_type": "Model",
         "description": "Decoder-only transformer model by OpenAI.",
         "aliases": ["gpt4"]},
        {"name": "Backpropagation", "entity_type": "Method",
         "description": "Algorithm to compute gradients for training neural networks.",
         "aliases": ["backprop"]},
        {"name": "BLEU", "entity_type": "Metric",
         "description": "Metric for evaluating machine-generated text quality.",
         "aliases": ["bleu score"]},
    ]
})

MOCK_NODE_RESPONSE_3 = json.dumps({
    "entities": [
        {"name": "Gradient Descent", "entity_type": "Method",
         "description": "Optimisation method to minimise a loss function.",
         "aliases": ["gd"]},
        {"name": "Stochastic Gradient Descent", "entity_type": "Method",
         "description": "Mini-batch variant of Gradient Descent.",
         "aliases": ["SGD", "sgd"]},
        {"name": "ResNet", "entity_type": "Model",
         "description": "Deep residual network using skip connections.",
         "aliases": ["residual network"]},
    ]
})

MOCK_RELATIONSHIP_RESPONSE_CONSTRAINED = json.dumps({
    "relationships": [
        {"source": "BERT", "target": "Attention Mechanism",
         "relation_type": "USES",
         "description": "BERT uses Attention Mechanism for NLU.",
         "confidence": 0.95},
        {"source": "BERT", "target": "SQuAD",
         "relation_type": "EVALUATED_ON",
         "description": "BERT is evaluated on SQuAD.",
         "confidence": 0.92},
        {"source": "BERT", "target": "Dropout",
         "relation_type": "USES",
         "description": "BERT uses Dropout for regularisation.",
         "confidence": 0.88},
    ]
})

MOCK_RELATIONSHIP_RESPONSE_UNCONSTRAINED = json.dumps({
    "relationships": [
        {"source": "BERT", "target": "Attention Mechanism",
         "relation_type": "LEVERAGES_FOR_ENCODING",
         "description": "BERT leverages attention for contextual encoding.",
         "confidence": 0.95},
        {"source": "BERT", "target": "SQuAD",
         "relation_type": "ACHIEVES_SOTA_ON",
         "description": "BERT achieves state-of-the-art on SQuAD.",
         "confidence": 0.90},
    ]
})

MOCK_CROSS_DOC_RESPONSE = json.dumps({
    "relationships": [
        {"source": "GPT-4", "target": "Attention Mechanism",
         "relation_type": "EXTENDS",
         "description": "Building on the Attention Mechanism introduced earlier.",
         "confidence": 0.91},
        {"source": "GPT-4", "target": "Backpropagation",
         "relation_type": "DEPENDS_ON",
         "description": "Backpropagation is required to train GPT-4.",
         "confidence": 0.89},
    ]
})

MOCK_INVALID_RELATION_RESPONSE = json.dumps({
    "relationships": [
        {"source": "BERT", "target": "SQuAD",
         "relation_type": "INVENTED_RELATION_XYZ",
         "description": "Invalid relation type.",
         "confidence": 0.95},
        {"source": "BERT", "target": "Attention Mechanism",
         "relation_type": "USES",
         "description": "Valid relation.",
         "confidence": 0.90},
    ]
})

MOCK_LOW_CONFIDENCE_RESPONSE = json.dumps({
    "relationships": [
        {"source": "BERT", "target": "SQuAD",
         "relation_type": "EVALUATED_ON",
         "description": "High confidence.",
         "confidence": 0.95},
        {"source": "BERT", "target": "Dropout",
         "relation_type": "USES",
         "description": "Low confidence — should be filtered.",
         "confidence": 0.3},
    ]
})

MOCK_EMPTY_RESPONSE = json.dumps({"entities": [], "relationships": []})

# ---------------------------------------------------------------------------
# Mock LLM call counter
# ---------------------------------------------------------------------------
_mock_call_count = 0
_mock_responses: list = []

def _reset_mock(responses: list) -> None:
    global _mock_call_count, _mock_responses
    _mock_call_count = 0
    _mock_responses  = responses

def _mock_generate(prompt: str) -> str:
    global _mock_call_count
    if not _mock_responses:
        return MOCK_EMPTY_RESPONSE
    resp = _mock_responses[_mock_call_count % len(_mock_responses)]
    _mock_call_count += 1
    return resp

# ---------------------------------------------------------------------------
# Build a mock LLMBackend — no real model loaded
# ---------------------------------------------------------------------------

def _make_mock_llm() -> LLMBackend:
    """
    Returns a LLMBackend whose .generate() and .generate_batch() are replaced
    with mock functions.  No API key or network call is needed.
    """
    llm = LLMBackend.__new__(LLMBackend)
    llm.api_key     = "mock-key"
    llm.model       = "mock-model"
    llm.max_tokens  = 1024
    llm.temperature = 0.0
    llm.batch_size  = 1
    llm.site_url    = ""
    llm.site_name   = ""
    llm.generate    = _mock_generate

    def _mock_generate_batch(prompts: list) -> list:
        return [_mock_generate(p) for p in prompts]

    llm.generate_batch = _mock_generate_batch
    return llm


def _load_real_llm(model_name: str = "llama-3.3-70b-versatile") -> LLMBackend:
    """
    Create a real LLMBackend using Groq.
    Requires GROQ_API_KEY to be set in the environment.
    Returns None if initialisation fails (e.g. missing API key).
    """
    print(f"\nConnecting to Groq model: {model_name}")
    try:
        llm = LLMBackend(model=model_name, max_tokens=512, temperature=0.0)
        print(f"  ✓ LLMBackend ready: {llm}")
        return llm
    except Exception as e:
        print(f"  ERROR initialising LLMBackend: {e}")
        return None

# ---------------------------------------------------------------------------
# Dummy embedding function
# ---------------------------------------------------------------------------

def _dummy_embedding_fn(texts: List[str]) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.random((len(texts), 384)).astype(np.float32)

# ---------------------------------------------------------------------------
# Build mock Chunk objects
# ---------------------------------------------------------------------------

def _make_chunk(text: str, chunk_id: int, source: str = "test_doc.pdf") -> Chunk:
    return Chunk(
        text       = text.strip(),
        chunk_id   = chunk_id,
        start_char = 0,
        end_char   = len(text.strip()),
        metadata   = {"source": source},
    )

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")

def run_test(description: str, condition: bool) -> None:
    global _total, _passed
    _total += 1
    status = PASS if condition else FAIL
    if condition:
        _passed += 1
    print(f"  {status}  {description}")

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_extracted_node_dataclass(verbose: bool = False) -> None:
    header("ExtractedNode dataclass  [base.py]")

    node = ExtractedNode(
        node_id="abc123", name="BERT", entity_type="Model",
        description="A transformer-based model.",
        source_chunk="0", source="test_doc.pdf", aliases=["bert"],
    )

    run_test("ExtractedNode instantiates without error", node is not None)
    run_test("node.node_id is set", node.node_id == "abc123")
    run_test("node.name is set", node.name == "BERT")
    run_test("node.entity_type is set", node.entity_type == "Model")
    run_test("node.source is set", node.source == "test_doc.pdf")
    run_test("node.source_chunk is set", node.source_chunk == "0")
    run_test("node.embedding is None by default", node.embedding is None)
    run_test("node.aliases is a list", isinstance(node.aliases, list))

    props = node.to_neo4j_properties()
    run_test("to_neo4j_properties() returns a dict", isinstance(props, dict))
    run_test("to_neo4j_properties() contains node_id", "node_id" in props)
    run_test("to_neo4j_properties() contains source", "source" in props)
    run_test("to_neo4j_properties() does NOT contain embedding", "embedding" not in props)

    node.embedding = np.ones(384, dtype=np.float32)
    run_test("embedding can be attached as np.ndarray", isinstance(node.embedding, np.ndarray))

    if verbose:
        print(f"    Neo4j properties: {list(props.keys())}")


def test_extracted_relationship_dataclass(verbose: bool = False) -> None:
    header("ExtractedRelationship dataclass  [base.py]")

    rel = ExtractedRelationship(
        source_id="abc123", target_id="def456",
        source_name="BERT", target_name="Attention Mechanism",
        relation_type="USES", description="BERT uses Attention.",
        source_chunk="0", source="test_doc.pdf",
        confidence=0.95, mode="constrained",
    )

    run_test("ExtractedRelationship instantiates without error", rel is not None)
    run_test("rel.source_id is set", rel.source_id == "abc123")
    run_test("rel.target_id is set", rel.target_id == "def456")
    run_test("rel.relation_type is set", rel.relation_type == "USES")
    run_test("rel.confidence is set", rel.confidence == 0.95)
    run_test("rel.source is set", rel.source == "test_doc.pdf")
    run_test("rel.mode is set", rel.mode == "constrained")

    props = rel.to_neo4j_properties()
    run_test("to_neo4j_properties() returns a dict", isinstance(props, dict))
    run_test("to_neo4j_properties() contains relation_type", "relation_type" in props)
    run_test("to_neo4j_properties() contains source", "source" in props)
    run_test("to_neo4j_properties() contains confidence", "confidence" in props)
    run_test("to_neo4j_properties() contains mode", "mode" in props)

    if verbose:
        print(f"    Neo4j properties: {list(props.keys())}")


def test_abstract_base_classes(verbose: bool = False) -> None:
    header("Abstract base classes  [base.py]")

    try:
        BaseNodeExtractor()
        run_test("BaseNodeExtractor cannot be instantiated directly", False)
    except TypeError:
        run_test("BaseNodeExtractor cannot be instantiated directly", True)

    try:
        BaseRelationshipExtractor()
        run_test("BaseRelationshipExtractor cannot be instantiated directly", False)
    except TypeError:
        run_test("BaseRelationshipExtractor cannot be instantiated directly", True)

    run_test("NodeExtractor is subclass of BaseNodeExtractor",
             issubclass(NodeExtractor, BaseNodeExtractor))
    run_test("RelationshipExtractor is subclass of BaseRelationshipExtractor",
             issubclass(RelationshipExtractor, BaseRelationshipExtractor))


def test_llm_backend_mock(verbose: bool = False) -> LLMBackend:
    header("LLMBackend mock  [llm_backend.py]")

    llm = _make_mock_llm()

    run_test("LLMBackend mock created without error", llm is not None)
    run_test("llm.generate is callable", callable(llm.generate))
    run_test("llm.max_tokens is set", llm.max_tokens == 1024)
    run_test("llm.temperature is set", llm.temperature == 0.0)
    run_test("llm.api_key is set", llm.api_key == "mock-key")
    run_test("llm.batch_size is set", llm.batch_size == 1)

    _reset_mock([MOCK_NODE_RESPONSE_1])
    result = llm.generate("test prompt")
    run_test("generate() returns a string", isinstance(result, str))
    run_test("generate() returns non-empty string", len(result) > 0)
    parsed = json.loads(result)
    run_test("generate() returns valid JSON", "entities" in parsed)

    if verbose:
        print(f"    Mock response length: {len(result)}")

    return llm


def test_allowed_relations(verbose: bool = False) -> None:
    header("Relation taxonomy  [relationships_extractor.py]")

    run_test("ALLOWED_RELATIONS is a non-empty list", isinstance(ALLOWED_RELATIONS, list) and len(ALLOWED_RELATIONS) > 0)
    run_test("ALLOWED_RELATIONS has >= 20 types", len(ALLOWED_RELATIONS) >= 20)

    # Methodological
    for r in ["USES", "EXTENDS", "IMPLEMENTS", "IMPROVES_UPON", "REPLACES", "COMBINES"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)

    # Structural
    for r in ["DEPENDS_ON", "PART_OF", "BASED_ON", "DERIVED_FROM"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)

    # Evaluation
    for r in ["EVALUATED_ON", "TRAINED_ON", "BENCHMARKED_AGAINST", "OUTPERFORMS"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)

    # Attribution
    for r in ["INTRODUCES", "PROPOSED_BY", "DEVELOPED_BY", "PUBLISHED_IN"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)

    # Comparison
    for r in ["COMPARES_WITH", "CONTRASTS_WITH", "EQUIVALENT_TO"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)

    # Knowledge flow
    for r in ["CITES", "BUILDS_ON", "MOTIVATED_BY", "ADDRESSES", "SOLVES"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)

    # Domain
    for r in ["APPLIED_TO", "GENERALIZES", "SPECIALIZES"]:
        run_test(f"{r} in ALLOWED_RELATIONS", r in ALLOWED_RELATIONS)

    run_test("CROSS_DOC_RELATIONS is a subset of ALLOWED_RELATIONS",
             all(r in ALLOWED_RELATIONS for r in CROSS_DOC_RELATIONS))

    if verbose:
        print(f"    Total relations: {len(ALLOWED_RELATIONS)}")
        print(f"    Relations: {ALLOWED_RELATIONS}")


def test_node_extractor_init(
    llm: LLMBackend,
    embedding_fn,
    verbose: bool = False,
) -> NodeExtractor:
    header("NodeExtractor initialisation  [node_extractor.py]")

    extractor = NodeExtractor(
        llm          = llm,
        embedding_fn = embedding_fn,
        max_workers  = 2,
    )

    run_test("NodeExtractor instantiates without error", extractor is not None)
    run_test("isinstance(extractor, BaseNodeExtractor)",
             isinstance(extractor, BaseNodeExtractor))
    run_test("extractor.llm is the mock LLMBackend", extractor.llm is llm)
    run_test("extractor._nlp is loaded", extractor._nlp is not None)
    run_test("extractor.allowed_types is a list",
             isinstance(extractor.allowed_types, list))
    run_test("extractor._cache is empty dict", extractor._cache == {})
    run_test("Default allowed_types includes 'Model'", "Model" in extractor.allowed_types)
    run_test("Default allowed_types includes 'Method'", "Method" in extractor.allowed_types)
    run_test("Default allowed_types includes 'Theory'", "Theory" in extractor.allowed_types)
    run_test("Default allowed_types includes 'Framework'", "Framework" in extractor.allowed_types)

    if verbose:
        print(f"    Allowed types : {extractor.allowed_types}")
        print(f"    Max workers   : {extractor.max_workers}")

    return extractor


def test_node_extractor_spacy(
    extractor: NodeExtractor,
    verbose: bool = False,
) -> None:
    header("NodeExtractor spaCy NER  [node_extractor.py → _extract_with_spacy()]")

    text  = "Google developed BERT, a transformer model. Vaswani published the paper."
    items = extractor._extract_with_spacy(text)

    run_test("_extract_with_spacy() returns a list", isinstance(items, list))
    run_test("spaCy returns at least 1 entity", len(items) >= 1)
    run_test("Each item has required keys",
             all("name" in i and "entity_type" in i and "description" in i for i in items))
    run_test("entity_type values are from spaCy label map",
             all(i["entity_type"] in ["Author", "Institution", "Model", "Framework"]
                 for i in items))

    if verbose:
        for item in items:
            print(f"    spaCy: {item['name']} ({item['entity_type']})")


def test_node_extractor_llm(
    extractor: NodeExtractor,
    verbose: bool = False,
) -> None:
    header("NodeExtractor LLM extraction (mock)  [node_extractor.py → _extract_with_llm()]")

    _reset_mock([MOCK_NODE_RESPONSE_1])
    items = extractor._extract_with_llm(MOCK_TEXT_1)

    run_test("_extract_with_llm() returns a list", isinstance(items, list))
    run_test("LLM returns at least 1 entity", len(items) >= 1)
    run_test("Each item has name, entity_type, description",
             all("name" in i and "entity_type" in i and "description" in i for i in items))
    names = [i["name"] for i in items]
    run_test("BERT is extracted", "BERT" in names)
    run_test("Attention Mechanism is extracted", "Attention Mechanism" in names)

    # Test fail-fast behavior on bad JSON
    _reset_mock(["this is not valid json"])
    with warnings.catch_warnings(record=True):
        try:
            extractor._extract_with_llm(MOCK_TEXT_1)
            run_test("Bad LLM JSON raises RuntimeError", False)
        except RuntimeError:
            run_test("Bad LLM JSON raises RuntimeError", True)

    if verbose:
        for item in items:
            print(f"    LLM: {item['name']} ({item['entity_type']})")


def test_node_extractor_llm_real(
    extractor: NodeExtractor,
    verbose: bool = False,
) -> None:
    header("NodeExtractor LLM extraction (REAL)  [node_extractor.py → _extract_with_llm()]")

    items = extractor._extract_with_llm(MOCK_TEXT_1)

    run_test("_extract_with_llm() returns a list", isinstance(items, list))
    run_test("LLM returns at least 1 entity", len(items) >= 1)
    run_test("Each item has name, entity_type, description",
             all("name" in i and "entity_type" in i and "description" in i for i in items))
    run_test("Each entity_type is a string", all(isinstance(i["entity_type"], str) for i in items))

    if verbose:
        for item in items:
            print(f"    LLM (real): {item['name']} ({item['entity_type']})")


def test_node_extractor_merge(
    extractor: NodeExtractor,
    verbose: bool = False,
) -> None:
    header("NodeExtractor merge  [node_extractor.py → _merge_entity_sets()]")

    spacy_items = [
        {"name": "Google", "entity_type": "Institution",
         "description": "Google — mentioned.", "aliases": ["google"]},
        {"name": "BERT", "entity_type": "Model",
         "description": "BERT — spaCy.", "aliases": ["bert"]},
    ]
    llm_items = [
        {"name": "BERT", "entity_type": "Model",
         "description": "Transformer model for NLU.", "aliases": ["bert"]},
        {"name": "Attention Mechanism", "entity_type": "Method",
         "description": "Attention-based method.", "aliases": ["attention"]},
    ]

    merged = extractor._merge_entity_sets(spacy_items, llm_items)
    names  = [i["name"] for i in merged]

    run_test("Merged list is non-empty", len(merged) > 0)
    run_test("LLM BERT not duplicated", names.count("BERT") == 1)
    run_test("spaCy-only Google is included", "Google" in names)
    run_test("LLM Attention Mechanism is included", "Attention Mechanism" in names)

    if verbose:
        print(f"    Merged: {names}")


def test_node_extractor_deduplication(
    extractor: NodeExtractor,
    verbose: bool = False,
) -> None:
    header("NodeExtractor deduplication  [node_extractor.py → _deduplicate_nodes()]")

    node_id = extractor._make_node_id("BERT", "Model")
    nodes = [
        ExtractedNode(node_id=node_id, name="BERT", entity_type="Model",
                      description="Short desc.", source_chunk="0",
                      source="doc.pdf", aliases=["bert"]),
        ExtractedNode(node_id=node_id, name="BERT", entity_type="Model",
                      description="A longer and more detailed description of BERT.",
                      source_chunk="1", source="doc.pdf",
                      aliases=["bert", "BERT model"]),
        ExtractedNode(node_id=extractor._make_node_id("Dropout", "Method"),
                      name="Dropout", entity_type="Method",
                      description="Regularisation.", source_chunk="0",
                      source="doc.pdf", aliases=["dropout"]),
    ]

    deduped = extractor._deduplicate_nodes(nodes)
    bert    = next((n for n in deduped if n.name == "BERT"), None)

    run_test("2 unique nodes after deduplication", len(deduped) == 2)
    run_test("BERT node exists", bert is not None)
    run_test("Longer description kept",
             "longer" in bert.description if bert else False)
    run_test("Aliases merged",
             "BERT model" in bert.aliases if bert else False)

    if verbose:
        for n in deduped:
            print(f"    {n.name} | aliases: {n.aliases} | desc: {n.description[:50]}")


def test_node_extractor_embedding(
    extractor: NodeExtractor,
    verbose: bool = False,
) -> None:
    header("NodeExtractor embedding  [node_extractor.py → _embed_nodes()]")

    nodes = [
        ExtractedNode(node_id="a1", name="BERT", entity_type="Model",
                      description="Transformer.", source_chunk="0", source="doc.pdf"),
        ExtractedNode(node_id="b2", name="Dropout", entity_type="Method",
                      description="Regularisation.", source_chunk="0", source="doc.pdf"),
    ]
    embedded = extractor._embed_nodes(nodes)

    run_test("_embed_nodes() returns a list", isinstance(embedded, list))
    run_test("All nodes have embeddings",
             all(n.embedding is not None for n in embedded))
    run_test("Embeddings are np.ndarray",
             all(isinstance(n.embedding, np.ndarray) for n in embedded))
    run_test("Embeddings are 1D",
             all(n.embedding.ndim == 1 for n in embedded))

    if verbose:
        for n in embedded:
            print(f"    {n.name}: shape {n.embedding.shape}")


def test_node_extractor_single_chunk(
    extractor: NodeExtractor,
    use_real: bool = False,
    verbose: bool = False,
) -> None:
    header("NodeExtractor single chunk  [node_extractor.py → extract_from_single_chunk_public()]")

    extractor._cache.clear()
    if not use_real:
        _reset_mock([MOCK_NODE_RESPONSE_1])
    chunk = _make_chunk(MOCK_TEXT_1, chunk_id=0, source="paper.pdf")
    nodes = extractor.extract_from_single_chunk_public(chunk)

    run_test("Returns a list", isinstance(nodes, list))
    run_test("Returns at least 1 node", len(nodes) >= 1)
    run_test("All are ExtractedNode",
             all(isinstance(n, ExtractedNode) for n in nodes))
    run_test("All have source == paper.pdf",
             all(n.source == "paper.pdf" for n in nodes))
    run_test("All have source_chunk == '0'",
             all(n.source_chunk == "0" for n in nodes))
    run_test("All have embeddings",
             all(n.embedding is not None for n in nodes))

    if verbose:
        for n in nodes:
            print(f"    {n.name} ({n.entity_type}) | source: {n.source}")


def test_node_extractor_caching(
    extractor: NodeExtractor,
    verbose: bool = False,
) -> None:
    header("NodeExtractor caching  [node_extractor.py → _cache]")

    extractor._cache.clear()
    _reset_mock([MOCK_NODE_RESPONSE_1])

    chunk = _make_chunk(MOCK_TEXT_1, chunk_id=0, source="doc.pdf")
    nodes_first = extractor._extract_from_single_chunk(chunk)

    call_count_after_first = _mock_call_count

    # Second call with same text — should hit cache, no LLM call
    nodes_second = extractor._extract_from_single_chunk(chunk)
    call_count_after_second = _mock_call_count

    run_test("First call populates cache", chunk.text in extractor._cache)
    run_test("Second call uses cache — no new LLM call",
             call_count_after_second == call_count_after_first)
    run_test("Cached result matches original",
             len(nodes_first) == len(nodes_second))

    if verbose:
        print(f"    LLM calls after 1st: {call_count_after_first}")
        print(f"    LLM calls after 2nd: {call_count_after_second}")


def test_node_extractor_full_batch(
    extractor: NodeExtractor,
    use_real: bool = False,
    verbose: bool = False,
) -> List[ExtractedNode]:
    header("NodeExtractor full batch  [node_extractor.py → extract_from_chunks()]")

    extractor._cache.clear()
    if not use_real:
        _reset_mock([MOCK_NODE_RESPONSE_1, MOCK_NODE_RESPONSE_2, MOCK_NODE_RESPONSE_3])

    chunks = [
        _make_chunk(MOCK_TEXT_1, chunk_id=0, source="doc.pdf"),
        _make_chunk(MOCK_TEXT_2, chunk_id=1, source="doc.pdf"),
        _make_chunk(MOCK_TEXT_3, chunk_id=2, source="doc.pdf"),
    ]
    nodes = extractor.extract_from_chunks(chunks, show_progress=False)

    run_test("Returns a list", isinstance(nodes, list))
    run_test("Returns at least 1 node", len(nodes) >= 1)
    run_test("All are ExtractedNode",
             all(isinstance(n, ExtractedNode) for n in nodes))
    run_test("All have embeddings",
             all(n.embedding is not None for n in nodes))
    run_test("No duplicate node_ids",
             len({n.node_id for n in nodes}) == len(nodes))

    if verbose:
        print(f"    Total unique nodes: {len(nodes)}")
        for n in nodes:
            print(f"    {n.name} ({n.entity_type})")

    return nodes


def test_rel_extractor_init_constrained(
    llm: LLMBackend = None,
    verbose: bool = False,
) -> RelationshipExtractor:
    header("RelationshipExtractor init — constrained  [relationships_extractor.py]")

    llm = llm or _make_mock_llm()
    extractor = RelationshipExtractor(
        llm                  = llm,
        mode                 = "constrained",
        confidence_threshold = 0.6,
        max_entity_pairs     = 20,
    )

    run_test("Instantiates without error", extractor is not None)
    run_test("isinstance(extractor, BaseRelationshipExtractor)",
             isinstance(extractor, BaseRelationshipExtractor))
    run_test("mode == 'constrained'", extractor.mode == "constrained")
    run_test("confidence_threshold == 0.6", extractor.confidence_threshold == 0.6)
    run_test("max_entity_pairs == 20", extractor.max_entity_pairs == 20)

    if verbose:
        print(f"    Mode: {extractor.mode}")

    return extractor


def test_rel_extractor_init_unconstrained(
    llm: LLMBackend = None,
    verbose: bool = False,
) -> RelationshipExtractor:
    header("RelationshipExtractor init — unconstrained  [relationships_extractor.py]")

    llm = llm or _make_mock_llm()
    extractor = RelationshipExtractor(llm=llm, mode="unconstrained")

    run_test("Instantiates without error", extractor is not None)
    run_test("mode == 'unconstrained'", extractor.mode == "unconstrained")

    # Invalid mode should raise ValueError
    try:
        RelationshipExtractor(llm=_make_mock_llm(), mode="invalid_mode")
        run_test("Invalid mode raises ValueError", False)
    except ValueError:
        run_test("Invalid mode raises ValueError", True)

    if verbose:
        print(f"    Mode: {extractor.mode}")

    return extractor


def test_rel_extractor_constrained_extraction(
    extractor: RelationshipExtractor,
    nodes    : List[ExtractedNode],
    use_real : bool = False,
    verbose  : bool = False,
) -> List[ExtractedRelationship]:
    header("RelationshipExtractor constrained extraction  [relationships_extractor.py]")

    if not use_real:
        _reset_mock([MOCK_RELATIONSHIP_RESPONSE_CONSTRAINED])
    chunk       = _make_chunk(MOCK_TEXT_1, chunk_id=0, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}

    rels = extractor.extract_from_chunk(
        chunk          = chunk,
        nodes_in_chunk = nodes[:6],
        node_id_map    = node_id_map,
    )

    run_test("Returns a list", isinstance(rels, list))
    run_test("All are ExtractedRelationship",
             all(isinstance(r, ExtractedRelationship) for r in rels))
    run_test("All confidences above threshold",
             all(r.confidence >= extractor.confidence_threshold for r in rels))
    run_test("All relation_types in ALLOWED_RELATIONS",
             all(r.relation_type in ALLOWED_RELATIONS for r in rels))
    run_test("All have mode == 'constrained'",
             all(r.mode == "constrained" for r in rels))
    run_test("All have source == 'doc.pdf'",
             all(r.source == "doc.pdf" for r in rels))

    if verbose:
        for r in rels:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name} ({r.confidence:.2f})")

    return rels


def test_rel_extractor_unconstrained_extraction(
    extractor: RelationshipExtractor,
    nodes    : List[ExtractedNode],
    use_real : bool = False,
    verbose  : bool = False,
) -> None:
    header("RelationshipExtractor unconstrained extraction  [relationships_extractor.py]")

    if not use_real:
        _reset_mock([MOCK_RELATIONSHIP_RESPONSE_UNCONSTRAINED])
    chunk       = _make_chunk(MOCK_TEXT_1, chunk_id=0, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}

    rels = extractor.extract_from_chunk(
        chunk          = chunk,
        nodes_in_chunk = nodes[:6],
        node_id_map    = node_id_map,
    )

    run_test("Returns a list", isinstance(rels, list))
    run_test("All have mode == 'unconstrained'",
             all(r.mode == "unconstrained" for r in rels))
    if use_real:
        run_test("Unconstrained output has non-empty relation_type",
                 all(isinstance(r.relation_type, str) and len(r.relation_type) > 0 for r in rels))
    else:
        # In mock mode, verify custom labels are preserved.
        run_test("Custom relation types are kept",
                 any(r.relation_type not in ALLOWED_RELATIONS for r in rels))
    # Normalised to UPPER_SNAKE_CASE
    run_test("All relation_types are UPPER_SNAKE_CASE",
             all(r.relation_type == r.relation_type.upper() for r in rels))

    if verbose:
        for r in rels:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name}")


def test_rel_extractor_invalid_relation_filtered(
    extractor: RelationshipExtractor,
    nodes    : List[ExtractedNode],
    verbose  : bool = False,
) -> None:
    header("RelationshipExtractor invalid relation filtering  [relationships_extractor.py]")

    _reset_mock([MOCK_INVALID_RELATION_RESPONSE])
    chunk       = _make_chunk(MOCK_TEXT_1, chunk_id=0, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}

    rels = extractor.extract_from_chunk(
        chunk          = chunk,
        nodes_in_chunk = nodes[:6],
        node_id_map    = node_id_map,
    )

    run_test("INVENTED_RELATION_XYZ is filtered out in constrained mode",
             all(r.relation_type != "INVENTED_RELATION_XYZ" for r in rels))
    run_test("Valid USES relation is kept",
             any(r.relation_type == "USES" for r in rels))

    if verbose:
        print(f"    Remaining rels: {[r.relation_type for r in rels]}")


def test_rel_extractor_confidence_filtering(
    extractor: RelationshipExtractor,
    nodes    : List[ExtractedNode],
    verbose  : bool = False,
) -> None:
    header("RelationshipExtractor confidence filtering  [relationships_extractor.py]")

    _reset_mock([MOCK_LOW_CONFIDENCE_RESPONSE])
    chunk       = _make_chunk(MOCK_TEXT_1, chunk_id=0, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}

    rels = extractor.extract_from_chunk(
        chunk          = chunk,
        nodes_in_chunk = nodes[:6],
        node_id_map    = node_id_map,
    )

    run_test("Low confidence (0.3) relationship filtered out",
             all(r.confidence >= extractor.confidence_threshold for r in rels))
    run_test("High confidence (0.95) relationship kept",
             any(r.confidence >= 0.9 for r in rels))

    if verbose:
        print(f"    Threshold: {extractor.confidence_threshold}")
        print(f"    Kept: {[(r.relation_type, r.confidence) for r in rels]}")


def test_rel_extractor_cross_document(
    extractor: RelationshipExtractor,
    nodes    : List[ExtractedNode],
    use_real : bool = False,
    verbose  : bool = False,
) -> None:
    header("RelationshipExtractor cross-document  [relationships_extractor.py]")
    if not use_real:
        _reset_mock([MOCK_CROSS_DOC_RESPONSE])
    chunk       = _make_chunk(MOCK_TEXT_2, chunk_id=1, source="doc.pdf")
    node_id_map = {n.name.lower(): n.node_id for n in nodes}

    cross_refs = extractor.extract_cross_document_references(
        chunk              = chunk,
        node_id_map        = node_id_map,
        all_document_nodes = {"doc.pdf": nodes},
    )

    run_test("Returns a list", isinstance(cross_refs, list))
    run_test("All are ExtractedRelationship",
             all(isinstance(r, ExtractedRelationship) for r in cross_refs))

    # Chunk without markers → [] without LLM call
    call_before = _mock_call_count
    plain_chunk = _make_chunk(MOCK_TEXT_3, chunk_id=2, source="doc.pdf")
    no_refs = extractor.extract_cross_document_references(
        chunk              = plain_chunk,
        node_id_map        = node_id_map,
        all_document_nodes = {"doc.pdf": nodes},
    )
    run_test("Chunk without markers returns [] (no LLM call)",
             no_refs == [] and _mock_call_count == call_before)

    if verbose:
        print(f"    Cross-doc refs: {len(cross_refs)}")
        for r in cross_refs:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name}")


def test_rel_extractor_deduplication(
    extractor: RelationshipExtractor,
    verbose  : bool = False,
) -> None:
    header("RelationshipExtractor deduplication  [relationships_extractor.py]")

    rels = [
        ExtractedRelationship(source_id="a", target_id="b",
            source_name="BERT", target_name="Attention Mechanism",
            relation_type="USES", description="First.",
            source_chunk="0", source="doc.pdf", confidence=0.7),
        ExtractedRelationship(source_id="a", target_id="b",
            source_name="BERT", target_name="Attention Mechanism",
            relation_type="USES", description="Second.",
            source_chunk="1", source="doc.pdf", confidence=0.95),
        ExtractedRelationship(source_id="a", target_id="c",
            source_name="BERT", target_name="SQuAD",
            relation_type="EVALUATED_ON", description="BERT on SQuAD.",
            source_chunk="0", source="doc.pdf", confidence=0.9),
    ]

    deduped  = extractor.deduplicate_relationships(rels)
    uses_rel = next((r for r in deduped if r.relation_type == "USES"), None)

    run_test("2 unique relationships after dedup", len(deduped) == 2)
    run_test("USES relationship kept", uses_rel is not None)
    run_test("Higher confidence (0.95) kept",
             uses_rel.confidence == 0.95 if uses_rel else False)

    if verbose:
        for r in deduped:
            print(f"    {r.source_name} -[{r.relation_type}]-> {r.target_name} ({r.confidence})")


def test_rel_extractor_priority_pairs(
    extractor: RelationshipExtractor,
    nodes    : List[ExtractedNode],
    verbose  : bool = False,
) -> None:
    header("RelationshipExtractor pair prioritisation  [relationships_extractor.py]")

    from itertools import combinations
    pairs = list(combinations(nodes[:6], 2))

    if len(pairs) < 2:
        print("  SKIP  Not enough nodes.")
        return

    prioritised = extractor._prioritise_pairs(pairs, max_pairs=5)

    run_test("Returns a list", isinstance(prioritised, list))
    run_test("Returns at most max_pairs", len(prioritised) <= 5)
    run_test("Returns at least 1 pair", len(prioritised) >= 1)

    if verbose:
        for a, b in prioritised:
            print(f"    {a.name} ({a.entity_type}) ↔ {b.name} ({b.entity_type})")


def test_end_to_end(
    node_extractor    : NodeExtractor,
    constrained_extractor  : RelationshipExtractor,
    unconstrained_extractor: RelationshipExtractor,
    use_real: bool = False,
    verbose: bool = False,
) -> None:
    header("End-to-end pipeline  [chunk → nodes → constrained & unconstrained rels]")

    node_extractor._cache.clear()

    chunks = [
        _make_chunk(MOCK_TEXT_1, chunk_id=0, source="paper.pdf"),
        _make_chunk(MOCK_TEXT_2, chunk_id=1, source="paper.pdf"),
    ]

    # Step 1 — extract nodes
    if not use_real:
        _reset_mock([MOCK_NODE_RESPONSE_1, MOCK_NODE_RESPONSE_2])
    nodes = node_extractor.extract_from_chunks(chunks, show_progress=False)
    run_test("Step 1: nodes extracted", len(nodes) > 0)

    # Step 2 — build node_id_map
    node_id_map = {n.name.lower(): n.node_id for n in nodes}
    run_test(
        "Step 2: node_id_map built",
        all(n.name.lower() in node_id_map for n in nodes)
    )
    # Step 3a — constrained relationships
    if not use_real:
        _reset_mock([MOCK_RELATIONSHIP_RESPONSE_CONSTRAINED])
    constrained_rels = []
    for chunk in chunks:
        chunk_nodes = nodes[:4]
        if not use_real:
            _reset_mock([MOCK_RELATIONSHIP_RESPONSE_CONSTRAINED])
        constrained_rels.extend(
            constrained_extractor.extract_from_chunk(chunk, chunk_nodes, node_id_map)
        )
    run_test("Step 3a: constrained rels extracted", isinstance(constrained_rels, list))
    run_test("Step 3a: all constrained rel types in ALLOWED_RELATIONS",
             all(r.relation_type in ALLOWED_RELATIONS for r in constrained_rels))

    # Step 3b — unconstrained relationships
    unconstrained_rels = []
    for chunk in chunks:
        chunk_nodes = nodes[:4]
        _reset_mock([MOCK_RELATIONSHIP_RESPONSE_UNCONSTRAINED])
        unconstrained_rels.extend(
            unconstrained_extractor.extract_from_chunk(chunk, chunk_nodes, node_id_map)
        )
    run_test("Step 3b: unconstrained rels extracted", isinstance(unconstrained_rels, list))
    run_test("Step 3b: unconstrained rels have mode='unconstrained'",
             all(r.mode == "unconstrained" for r in unconstrained_rels))

    # Step 4 — deduplicate
    deduped_c = constrained_extractor.deduplicate_relationships(constrained_rels)
    deduped_u = unconstrained_extractor.deduplicate_relationships(unconstrained_rels)
    run_test("Step 4: constrained dedup — no duplicate triples",
             len({(r.source_id, r.target_id, r.relation_type) for r in deduped_c}) == len(deduped_c))
    run_test("Step 4: unconstrained dedup — no duplicate triples",
             len({(r.source_id, r.target_id, r.relation_type) for r in deduped_u}) == len(deduped_u))

    if verbose:
        print(f"    Nodes              : {len(nodes)}")
        print(f"    Constrained rels   : {len(deduped_c)}")
        print(f"    Unconstrained rels : {len(deduped_u)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphRAG Graph Module Test Suite",
        epilog=(
            "Examples:\n"
            "  python graph_test.py        # mock mode\n"
            "  python graph_test.py --mock # explicit mock\n"
            "  python graph_test.py -v     # verbose"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (default)")
    parser.add_argument("--real", action="store_true",
                        help="Use real Groq API (requires GROQ_API_KEY)")
    parser.add_argument("--model", type=str,
                        default="llama-3.3-70b-versatile",
                        help="Groq model string for --real mode")
    args = parser.parse_args()
    v = args.verbose
    use_real = args.real and not args.mock   # --mock always wins

    # Set up embedding_fn
    if _EMBEDDINGS_AVAILABLE:
        print("\nLoading embedding model...")
        emb_model    = HuggingFaceEmbedding()
        embedding_fn = emb_model.encode
    else:
        embedding_fn = _dummy_embedding_fn

    print(f"\n{'=' * 70}")
    print(f"  GraphRAG — Graph Module Test Suite")
    print(f"  Mode     : {'REAL MODEL via Groq (' + args.model + ')' if use_real else 'MOCK (no model needed)'}")
    print(f"  Embedding: {'HuggingFaceEmbedding' if _EMBEDDINGS_AVAILABLE else 'dummy (random)'}")
    print(f"{'=' * 70}")

    if use_real:
        active_llm = _load_real_llm(args.model)
        if active_llm is None:
            print("ERROR: Could not load real model — aborting.")
            sys.exit(1)
        print("  NOTE: Real-mode skips mock-specific tests (call counting, bad-JSON fallback).")
    else:
        active_llm = _make_mock_llm()

    # ── Run all tests ──────────────────────────────────────────────────────
    test_extracted_node_dataclass(verbose=v)
    test_extracted_relationship_dataclass(verbose=v)
    test_abstract_base_classes(verbose=v)
    test_llm_backend_mock(verbose=v)
    test_allowed_relations(verbose=v)

    node_extractor = test_node_extractor_init(active_llm, embedding_fn, verbose=v)
    test_node_extractor_spacy(node_extractor, verbose=v)
    if not use_real:
        test_node_extractor_llm(node_extractor, verbose=v)
        test_node_extractor_caching(node_extractor, verbose=v)
    else:
        test_node_extractor_llm_real(node_extractor, verbose=v)
    test_node_extractor_merge(node_extractor, verbose=v)
    test_node_extractor_deduplication(node_extractor, verbose=v)
    test_node_extractor_embedding(node_extractor, verbose=v)
    test_node_extractor_single_chunk(node_extractor, use_real=use_real, verbose=v)
    nodes = test_node_extractor_full_batch(node_extractor, use_real=use_real, verbose=v)

    constrained_extractor   = test_rel_extractor_init_constrained(active_llm, verbose=v)
    unconstrained_extractor = test_rel_extractor_init_unconstrained(active_llm, verbose=v)
    test_rel_extractor_constrained_extraction(constrained_extractor, nodes, use_real=use_real, verbose=v)
    test_rel_extractor_unconstrained_extraction(unconstrained_extractor, nodes, use_real=use_real, verbose=v)
    if not use_real:
        test_rel_extractor_invalid_relation_filtered(constrained_extractor, nodes, verbose=v)
        test_rel_extractor_confidence_filtering(constrained_extractor, nodes, verbose=v)
    test_rel_extractor_cross_document(constrained_extractor, nodes, use_real=use_real, verbose=v)
    test_rel_extractor_deduplication(constrained_extractor, verbose=v)
    test_rel_extractor_priority_pairs(constrained_extractor, nodes, verbose=v)
    test_end_to_end(node_extractor, constrained_extractor, unconstrained_extractor,
                    use_real=use_real, verbose=v)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if _passed == _total:
        print(f"  {_passed} / {_total} tests passed")
    else:
        failed = _total - _passed
        print(f"  {_passed} / {_total} tests passed   ({failed} failed)")
    print(f"{'=' * 70}\n")

    sys.exit(0 if _passed == _total else 1)


if __name__ == "__main__":
    main()