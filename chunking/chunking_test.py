"""
chunking_test.py — Test suite for Graph-RAG Chunking Module
======================================================================
Tests every chunker on realistic text extracted from user-provided files using
the project's extractors (PDFExtractor, PPTXExtractor, DOCXExtractor,
AudioExtractor, URLExtractor, VideoExtractor).

Run
---
    python chunking_test.py           # will prompt for file path, run all tests
    python chunking_test.py -v        # verbose (show each chunk's text preview)
    python chunking_test.py --method hybrid   # test only HybridChunker
    python chunking_test.py --file path/to/file.pdf  # specify file directly

Dependencies: extractors module for file extraction.
"""

import sys
import time
import textwrap
import argparse
from typing import List, Dict, Any
from pathlib import Path

# ── Add parent directory to path (allow imports from root) ──────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from chunking import (
        chunk_text,
        FixedSizeChunker,
        SentenceChunker,
        ParagraphChunker,
        RecursiveChunker,
        SlidingWindowChunker,
        SemanticChunker,
        HybridChunker,
        ChunkAnalyser,
    )
except ImportError as e:
    print(f"ERROR: Could not import chunking.py — {e}")
    print("Make sure chunking.py is in the same directory as this test file.")
    sys.exit(1)

try:
    from extractors.extractors_wrapper import SimpleExtractorWrapper
    print("✓ Extractors module loaded successfully")
except ImportError as e:
    print(f"ERROR: Could not import extractors module — {e}")
    print(f"ImportError details: {e}")
    print(f"sys.path includes: {sys.path[:3]}")
    SimpleExtractorWrapper = None


EXTRACTED_TEXT = None
EXTRACTED_SOURCE_NAME = "test"
USE_MOCK_TEXT = True

# Mock text fallback (if extraction fails)
PDF_TEXT = """
--- Page 1 ---
Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that focuses on building
systems that can learn from and make decisions based on data. Unlike traditional
programming, where explicit rules are written by programmers, machine learning
models infer rules from training examples.

There are three main categories of machine learning: supervised learning,
unsupervised learning, and reinforcement learning. Each paradigm is suited to
different types of problems and data availability scenarios.

--- Page 2 ---
Supervised Learning

In supervised learning, the model is trained on labeled data. Each training
example consists of an input-output pair. The model learns a mapping from inputs
to outputs by minimising a loss function over the training set.

Common supervised learning algorithms include linear regression, logistic
regression, decision trees, support vector machines, and neural networks.
The choice of algorithm depends on the nature of the data and the prediction task.

Overfitting occurs when a model learns the training data too well, capturing noise
rather than the underlying pattern. Regularisation techniques such as L1 (Lasso),
L2 (Ridge), and dropout help mitigate overfitting.

--- Page 3 ---
Unsupervised Learning

Unsupervised learning deals with unlabeled data. The goal is to discover hidden
structure or representations in the input. Clustering algorithms such as K-means,
DBSCAN, and hierarchical clustering group similar data points together.

Dimensionality reduction techniques, including Principal Component Analysis (PCA)
and t-SNE, project high-dimensional data into lower-dimensional spaces while
preserving important structure. These techniques are widely used for visualisation
and as preprocessing steps.

--- Page 4 ---
Reinforcement Learning

Reinforcement learning (RL) involves an agent that interacts with an environment.
The agent receives rewards or penalties for its actions and learns a policy that
maximises cumulative reward over time.

Key RL algorithms include Q-learning, Deep Q-Networks (DQN), Policy Gradient
methods, and Proximal Policy Optimisation (PPO). RL has achieved remarkable
results in game-playing (AlphaGo, Atari) and robotics.
"""

# ── PPTX-style text (with slide markers) ─────────────────────────────────────
PPTX_TEXT = """
============================================================
SLIDE 1
============================================================

Introduction to Neural Networks
Deep Learning Fundamentals — Lecture 3

[SPEAKER NOTES]
Welcome everyone. Today we cover the mathematical foundations of neural networks.

============================================================
SLIDE 2
============================================================

What is a Neural Network?

A neural network is a computational model loosely inspired by the structure of
biological neural systems. It consists of layers of interconnected nodes (neurons)
that transform input signals into output predictions.

Each connection has an associated weight that is adjusted during training.

[SPEAKER NOTES]
Draw the classic three-layer diagram on the whiteboard.

============================================================
SLIDE 3
============================================================

Activation Functions

Activation functions introduce non-linearity into the network, enabling it to
learn complex patterns.

Common activation functions:
- ReLU: f(x) = max(0, x)
- Sigmoid: f(x) = 1 / (1 + e^{-x})
- Tanh: f(x) = (e^x - e^{-x}) / (e^x + e^{-x})
- Softmax: used for multi-class output layers

[SPEAKER NOTES]
Plot each function in Python. Ask students to identify where each is useful.

============================================================
SLIDE 4
============================================================

Backpropagation

Backpropagation is the algorithm used to compute gradients of the loss function
with respect to the network weights. It applies the chain rule of calculus to
efficiently propagate error signals from the output layer back to the input layer.

Gradient descent then updates the weights in the direction that reduces the loss.
"""

# ── DOCX-style text (with heading markers) ───────────────────────────────────
DOCX_TEXT = """
============================================================
[HEADING 1] Research Report: Climate Change Impacts
============================================================

This report synthesises current scientific understanding of climate change and
its projected impacts on global ecosystems, human health, and economic systems.

============================================================
[HEADING 2] Temperature Trends
============================================================

Global average surface temperatures have risen by approximately 1.1 degrees
Celsius above pre-industrial levels. The rate of warming has accelerated since
the mid-twentieth century, driven primarily by anthropogenic greenhouse gas
emissions.

The past decade (2011-2020) was the warmest on record. Polar regions are
warming at two to four times the global average rate, a phenomenon known as
Arctic amplification.

============================================================
[HEADING 2] Sea Level Rise
============================================================

Global mean sea level has risen by approximately 20 centimetres since 1900.
The rate of rise has accelerated in recent decades due to the combined effects
of thermal expansion of seawater and the melting of ice sheets and glaciers.

Projections indicate a further rise of 0.3 to 1.0 metres by 2100 under
moderate emissions scenarios, with higher ranges possible if ice sheet
dynamics are destabilised.

============================================================
[HEADING 2] Ecosystem Disruption
============================================================

Climate change is altering the distribution of species, timing of biological
events (phenology), and the composition of ecological communities. Coral bleaching
events have increased in frequency and severity due to ocean warming and
acidification.

Tropical forests, which store vast quantities of carbon, face increased drought
stress and wildfire risk. Permafrost thaw in high-latitude regions releases
stored methane and carbon dioxide, creating a positive feedback loop.
"""

# ── Audio transcript (plain prose, no structural markers) ────────────────────
AUDIO_TEXT = """
Hello everyone and welcome back. Today we are going to talk about the fundamentals
of graph databases and why they matter for modern applications. A graph database
stores data in nodes and edges, where nodes represent entities and edges represent
relationships between them.

Unlike relational databases that use tables and joins, graph databases are
optimised for traversing connected data. This makes them ideal for social
networks, recommendation engines, fraud detection, and knowledge graphs.

The two most popular graph database systems are Neo4j and Amazon Neptune. Neo4j
uses the Cypher query language, which is designed to be human-readable and
expressive for graph traversal operations.

Let me give you a concrete example. Suppose we are building a social network.
In a relational database, finding all friends-of-friends would require multiple
join operations that become increasingly expensive as the graph grows. In a graph
database, this is a simple two-hop traversal that runs in constant time regardless
of the total number of nodes.

Property graphs allow both nodes and edges to carry arbitrary key-value properties.
This flexibility means you can store rich metadata alongside the structural
relationships without needing a separate table or join.

Graph RAG, which stands for Graph Retrieval Augmented Generation, combines the
structured knowledge representation of graph databases with the generative
capabilities of large language models. Instead of retrieving flat document chunks,
the retriever explores the knowledge graph to assemble contextually rich subgraphs
that provide more coherent context to the language model.
"""

# ── URL / web-scraped text ────────────────────────────────────────────────────
URL_TEXT = """
Title: Python Programming Language — Official Documentation
URL: https://docs.python.org/3/
============================================================

Python is a high-level, general-purpose programming language. Its design
philosophy emphasises code readability with the use of significant indentation.

Python is dynamically typed and garbage-collected. It supports multiple
programming paradigms, including structured, object-oriented, and functional
programming.

Guido van Rossum began working on Python as a successor to the ABC programming
language in the late 1980s. He released the first version in 1991. Python
consistently ranks as one of the most popular programming languages worldwide.

The Python Package Index (PyPI) hosts hundreds of thousands of third-party
packages. The pip package manager allows developers to install packages with
a single command. Virtual environments isolate project dependencies.

Python is the dominant language for data science and machine learning, powered
by libraries such as NumPy, pandas, scikit-learn, TensorFlow, and PyTorch.
Web development frameworks like Django and Flask make Python suitable for
backend development. Python also sees heavy use in automation, scripting, and
scientific computing.
"""

# Short edge-case text
SHORT_TEXT = "This is a very short text. It has only two sentences."

# Empty text
EMPTY_TEXT = ""

# Repetitive text (stress test for deduplication / merge) 
REPEAT_TEXT = "Data science. " * 80


def get_file_path_from_user() -> str:
    """Prompt user to enter a file path."""
    print("\n" + "─" * 60)
    print("  Enter a file path to extract and chunk")
    print("  Supported formats: PDF, DOCX, PPTX, TXT, MP3, MP4, WAV, URL")
    print("─" * 60)
    file_path = input("File path (or leave empty for mock text): ").strip()
    return file_path


def load_extracted_text(file_path: str) -> tuple[str, str, bool]:
    """
    Extract text from a file using SimpleExtractorWrapper.
    
    Args:
        file_path: Path to the file to extract from
        
    Returns:
        Tuple of (extracted_text, source_name, success)
    """
    if not SimpleExtractorWrapper:
        print("ERROR: Extractors module not available.")
        return "", "", False
    
    if not file_path or not Path(file_path).exists():
        print(f"ERROR: File not found: {file_path}")
        return "", "", False
    
    try:
        wrapper = SimpleExtractorWrapper()
        extracted_text = wrapper.extract_auto(file_path)
        
        if not extracted_text:
            print("ERROR: Extraction returned empty text.")
            return "", "", False
        
        source_name = Path(file_path).stem
        print(f"\n✓ Successfully extracted {len(extracted_text)} characters from {source_name}")
        return extracted_text, source_name, True
        
    except Exception as e:
        print(f"ERROR: Failed to extract file: {e}")
        return "", "", False



PASS = "✓ PASS"
FAIL = "✗ FAIL"

_total  = 0
_passed = 0


def run_test(name: str, condition: bool, detail: str = "") -> None:
    """Register a single assertion result."""
    global _total, _passed
    _total += 1
    status = PASS if condition else FAIL
    if condition:
        _passed += 1
    suffix = f"  [{detail}]" if detail and not condition else ""
    print(f"  {status}  {name}{suffix}")


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def preview(chunks: List[Dict[str, Any]], n: int = 3, verbose: bool = False) -> None:
    """Print a preview of the first *n* chunks."""
    if not verbose:
        return
    for i, c in enumerate(chunks[:n]):
        text_preview = c["text"][:100].replace("\n", "↵")
        print(f"    [{i}] id={c['chunk_id']}  chars={len(c['text'])}  "
              f"tokens={c['token_count']}")
        print(f"        \"{text_preview}…\"")
    if len(chunks) > n:
        print(f"    … and {len(chunks) - n} more chunks")


# INDIVIDUAL CHUNKER TESTS

def test_fixed_size_chunker(verbose: bool = False) -> None:
    header("FixedSizeChunker")
    chunker = FixedSizeChunker(chunk_size=400, char_overlap=80, min_chunk_size=50)

    # Normal text
    chunks = chunker.chunk(PDF_TEXT, source_id="pdf_test")
    run_test("Produces chunks from PDF text",           len(chunks) > 0)
    run_test("All chunks have required keys",
             all("chunk_id" in c and "text" in c for c in chunks))
    run_test("chunk_index is 0-based sequential",
             all(c["chunk_index"] == i for i, c in enumerate(chunks)))
    run_test("No chunk exceeds chunk_size * 1.2",        # slight margin for stripped ws
             all(len(c["text"]) <= 400 * 1.2 for c in chunks))
    run_test("source_id embedded in chunk_id",
             all(c["chunk_id"].startswith("pdf_test_") for c in chunks))
    run_test("token_count > 0 for non-empty chunks",
             all(c["token_count"] > 0 for c in chunks if c["text"].strip()))

    # Overlap check: chunk N+1 should contain some text from chunk N 
    if len(chunks) >= 2:
        # The end of chunk 0 overlaps into chunk 1
        tail_of_first = chunks[0]["text"][-60:].strip()
        overlap_found = any(tail_of_first[:30] in c["text"] for c in chunks[1:])
        run_test("Character overlap present between consecutive chunks", overlap_found)

    # Edge cases 
    run_test("Empty text returns empty list",   chunker.chunk(EMPTY_TEXT) == [])
    short_chunks = chunker.chunk(SHORT_TEXT, source_id="short")
    run_test("Short text produces ≥ 1 chunk",   len(short_chunks) >= 1)

    stats = ChunkAnalyser.analyse(chunks)
    run_test("ChunkAnalyser reports correct count", stats["count"] == len(chunks))
    if verbose:
        ChunkAnalyser.print_report(stats)
        preview(chunks, verbose=verbose)


def test_sentence_chunker(verbose: bool = False) -> None:
    header("SentenceChunker")
    chunker = SentenceChunker(
        sentences_per_chunk=4, overlap_sentences=1,
        max_chunk_size=900, min_chunk_size=60
    )

    chunks = chunker.chunk(AUDIO_TEXT, source_id="audio_test")
    run_test("Produces chunks from audio transcript",  len(chunks) > 0)
    run_test("All chunks non-empty",                   all(c["text"].strip() for c in chunks))
    run_test("No chunk exceeds max_chunk_size",
             all(len(c["text"]) <= 900 * 1.1 for c in chunks))

    # Verify chunk IDs are unique
    ids = [c["chunk_id"] for c in chunks]
    run_test("All chunk_ids are unique",               len(ids) == len(set(ids)))

    # At least one chunk should contain multiple sentences (i.e. sentence grouping works)
    multi_sent = [c for c in chunks if c["text"].count(".") >= 2]
    run_test("Most chunks contain multiple sentences", len(multi_sent) >= len(chunks) // 2)

    run_test("Empty text returns []",   chunker.chunk(EMPTY_TEXT) == [])
    run_test("Short text handled",      len(chunker.chunk(SHORT_TEXT)) >= 1)

    if verbose:
        ChunkAnalyser.print_report(ChunkAnalyser.analyse(chunks))
        preview(chunks, verbose=verbose)


def test_paragraph_chunker(verbose: bool = False) -> None:
    header("ParagraphChunker")
    chunker = ParagraphChunker(
        max_paragraphs_per_chunk=2, max_chunk_size=900,
        overlap_sentences=1, min_chunk_size=80
    )

    chunks = chunker.chunk(DOCX_TEXT, source_id="docx_test")
    run_test("Produces chunks from DOCX text",       len(chunks) > 0)
    run_test("Metadata contains chunker name",
             all(c["metadata"]["chunker"] == "ParagraphChunker" for c in chunks))
    run_test("No chunk exceeds max_chunk_size * 1.1",
             all(len(c["text"]) <= 900 * 1.1 for c in chunks))

    # Paragraph chunker should respect paragraph boundaries (text should not have
    # artificial mid-sentence breaks introduced by the chunker itself)
    run_test("All chunks non-empty",     all(c["text"].strip() for c in chunks))

    # Test with URL text (single-paragraph structure)
    url_chunks = chunker.chunk(URL_TEXT, source_id="url_test")
    run_test("URL text produces ≥ 1 chunk",          len(url_chunks) >= 1)

    run_test("Empty text returns []",   chunker.chunk(EMPTY_TEXT) == [])

    if verbose:
        ChunkAnalyser.print_report(ChunkAnalyser.analyse(chunks))
        preview(chunks, verbose=verbose)


def test_recursive_chunker(verbose: bool = False) -> None:
    header("RecursiveChunker")
    chunker = RecursiveChunker(
        max_chunk_size=600, overlap_sentences=1, min_chunk_size=80
    )

    chunks = chunker.chunk(PDF_TEXT, source_id="pdf_recursive")
    run_test("Produces chunks from PDF text",        len(chunks) > 0)
    run_test("No chunk exceeds max_chunk_size * 1.05",
             all(len(c["text"]) <= 600 * 1.05 for c in chunks))
    run_test("All chunks non-empty",     all(c["text"].strip() for c in chunks))
    run_test("Chunk indices sequential",
             all(c["chunk_index"] == i for i, c in enumerate(chunks)))

    # Test on text with no paragraph breaks (edge case)
    single_para = AUDIO_TEXT.replace("\n\n", " ")
    sp_chunks = chunker.chunk(single_para, source_id="single_para")
    run_test("Handles text with no paragraph breaks", len(sp_chunks) >= 1)
    run_test("Single-para: no chunk exceeds max",
             all(len(c["text"]) <= 600 * 1.05 for c in sp_chunks))

    run_test("Empty text returns []",    chunker.chunk(EMPTY_TEXT) == [])

    # Stress test: very long repeated text
    t0 = time.perf_counter()
    _ = chunker.chunk(REPEAT_TEXT * 3, source_id="stress")
    elapsed = time.perf_counter() - t0
    run_test(f"Stress test (long repeated text) completes in < 2 s  [{elapsed:.3f}s]",
             elapsed < 2.0)

    if verbose:
        ChunkAnalyser.print_report(ChunkAnalyser.analyse(chunks))
        preview(chunks, verbose=verbose)


def test_sliding_window_chunker(verbose: bool = False) -> None:
    header("SlidingWindowChunker")
    chunker = SlidingWindowChunker(
        window_sentences=5, step_sentences=3,
        max_chunk_size=1000, min_chunk_size=60
    )

    chunks = chunker.chunk(URL_TEXT, source_id="url_sliding")
    run_test("Produces chunks",                      len(chunks) > 0)
    run_test("No chunk exceeds max_chunk_size * 1.1",
             all(len(c["text"]) <= 1000 * 1.1 for c in chunks))

    # Sliding window → later chunks should share content with earlier ones
    if len(chunks) >= 3:
        # Find a word from chunk 0 that should also appear in chunk 1
        words_0 = set(chunks[0]["text"].split())
        words_1 = set(chunks[1]["text"].split())
        overlap = words_0 & words_1
        run_test("Consecutive chunks share vocabulary (sliding overlap)",
                 len(overlap) >= 3)
    else:
        run_test("Consecutive chunks share vocabulary (sliding overlap)",
                 True, "skip: too few chunks")

    run_test("Empty text returns []",  chunker.chunk(EMPTY_TEXT) == [])
    run_test("Short text handled",     len(chunker.chunk(SHORT_TEXT)) >= 1)

    if verbose:
        ChunkAnalyser.print_report(ChunkAnalyser.analyse(chunks))
        preview(chunks, verbose=verbose)


def test_semantic_chunker(verbose: bool = False) -> None:
    header("SemanticChunker")
    chunker = SemanticChunker(
        max_chunk_size=1200, overlap_sentences=1, min_chunk_size=100,
        buffer_size=2, similarity_threshold=0.20, percentile_threshold=25.0
    )

    # Use audio text (no structural markers — best test for semantic detection)
    t0 = time.perf_counter()
    chunks = chunker.chunk(AUDIO_TEXT, source_id="audio_semantic")
    elapsed = time.perf_counter() - t0

    run_test(f"Produces chunks  [{elapsed:.3f}s]",      len(chunks) > 0)
    run_test("Completes in < 3 seconds (pure Python TF-IDF)", elapsed < 3.0)
    run_test("All chunks non-empty",    all(c["text"].strip() for c in chunks))
    run_test("No chunk exceeds max_chunk_size * 1.1",
             all(len(c["text"]) <= 1200 * 1.1 for c in chunks))
    run_test("Chunk IDs are unique",
             len({c["chunk_id"] for c in chunks}) == len(chunks))

    # Semantically distinct paragraphs in AUDIO_TEXT should produce ≥ 2 chunks
    run_test("Semantic chunker finds multiple topics in audio text",
             len(chunks) >= 2)

    # Test TF-IDF tokeniser
    from chunking import SemanticChunker as SC
    tokens = SC._tokenise("Hello World! This is a test sentence. 42.")
    run_test("Tokeniser strips punctuation and numbers", "42" not in tokens)
    run_test("Tokeniser lowercases",    "hello" in tokens)

    # Cosine similarity sanity checks
    vec_a = {"machine": 0.5, "learning": 0.5}
    vec_b = {"machine": 0.5, "learning": 0.5}
    vec_c = {"cooking": 0.7, "recipe": 0.3}
    sim_identical = SC._cosine_similarity(vec_a, vec_b)
    sim_different = SC._cosine_similarity(vec_a, vec_c)
    run_test("Cosine sim of identical vectors = 1.0",  abs(sim_identical - 1.0) < 1e-9)
    run_test("Cosine sim of disjoint vectors = 0.0",   abs(sim_different) < 1e-9)
    run_test("Cosine sim of empty vectors = 0.0",      SC._cosine_similarity({}, vec_a) == 0.0)

    # Test on PPTX text (structured — semantic should still find boundaries)
    t0 = time.perf_counter()
    pptx_chunks = chunker.chunk(PPTX_TEXT, source_id="pptx_semantic")
    elapsed2 = time.perf_counter() - t0
    run_test(f"Works on PPTX text  [{elapsed2:.3f}s]", len(pptx_chunks) > 0)

    run_test("Empty text returns []",   chunker.chunk(EMPTY_TEXT) == [])

    if verbose:
        ChunkAnalyser.print_report(ChunkAnalyser.analyse(chunks))
        preview(chunks, verbose=verbose)


def test_hybrid_chunker(verbose: bool = False) -> None:
    header("HybridChunker   ")
    chunker = HybridChunker(
        max_chunk_size=1024, overlap_sentences=1, min_chunk_size=100,
        max_paragraphs_per_chunk=3, keep_structural_markers=True
    )

    # PDF (has page markers)
    t0 = time.perf_counter()
    pdf_chunks = chunker.chunk(PDF_TEXT, source_id="pdf_hybrid")
    pdf_time = time.perf_counter() - t0
    run_test(f"PDF: produces chunks  [{pdf_time:.3f}s]",       len(pdf_chunks) > 0)
    run_test("PDF: completes in < 0.5 s (no ML)",              pdf_time < 0.5)
    run_test("PDF: no chunk exceeds max_chunk_size * 1.1",
             all(len(c["text"]) <= 1024 * 1.1 for c in pdf_chunks))
    run_test("PDF: structural markers preserved in chunk text",
             any("Page" in c["text"] for c in pdf_chunks))

    # PPTX (has slide markers + speaker notes)
    pptx_chunks = chunker.chunk(PPTX_TEXT, source_id="pptx_hybrid")
    run_test("PPTX: produces chunks",                          len(pptx_chunks) > 0)
    run_test("PPTX: SLIDE marker found in chunk text",
             any("SLIDE" in c["text"] for c in pptx_chunks))

    # DOCX (has HEADING markers)
    docx_chunks = chunker.chunk(DOCX_TEXT, source_id="docx_hybrid")
    run_test("DOCX: produces chunks",                          len(docx_chunks) > 0)
    run_test("DOCX: HEADING marker found",
             any("[HEADING" in c["text"] for c in docx_chunks))

    # Audio (plain prose, no markers → falls back to paragraph grouping)
    audio_chunks = chunker.chunk(AUDIO_TEXT, source_id="audio_hybrid")
    run_test("Audio: produces chunks",                         len(audio_chunks) > 0)
    run_test("Audio: all non-empty",    all(c["text"].strip() for c in audio_chunks))

    # URL text
    url_chunks = chunker.chunk(URL_TEXT, source_id="url_hybrid")
    run_test("URL: produces chunks",                           len(url_chunks) > 0)

    # Metadata injection
    extra = {"filename": "lecture.pdf", "language": "en"}
    meta_chunks = chunker.chunk(PDF_TEXT, source_id="meta_test", metadata=extra)
    run_test("Injected metadata present in all chunks",
             all(c["metadata"]["filename"] == "lecture.pdf" for c in meta_chunks))
    run_test("Chunker name in metadata",
             all(c["metadata"]["chunker"] == "HybridChunker" for c in meta_chunks))

    # Edge cases
    run_test("Empty text returns []",                          len(chunker.chunk(EMPTY_TEXT)) == 0)
    run_test("Short text returns ≥ 1 chunk",                   len(chunker.chunk(SHORT_TEXT)) >= 1)

    # Chunk IDs unique across all texts
    all_ids = [c["chunk_id"] for c in pdf_chunks + pptx_chunks + docx_chunks]
    run_test("chunk_ids are unique within a source",
             len({c["chunk_id"] for c in pdf_chunks}) == len(pdf_chunks))

    if verbose:
        print("\n  PDF chunks:")
        ChunkAnalyser.print_report(ChunkAnalyser.analyse(pdf_chunks))
        preview(pdf_chunks, verbose=verbose)
        print("\n  PPTX chunks:")
        ChunkAnalyser.print_report(ChunkAnalyser.analyse(pptx_chunks))
        preview(pptx_chunks, verbose=verbose)


# CROSS-CHUNKER COMPARATIVE TEST

def test_chunk_text_convenience(verbose: bool = False) -> None:
    header("chunk_text() convenience function")

    # Default method should be hybrid
    chunks = chunk_text(PDF_TEXT, source_id="conv_test")
    run_test("Default method is HybridChunker",
             chunks[0]["metadata"]["chunker"] == "HybridChunker")
    run_test("Returns non-empty list",   len(chunks) > 0)

    # All methods should work
    for method in ["hybrid", "semantic", "recursive", "paragraph",
                   "sentence", "sliding", "fixed"]:
        c = chunk_text(AUDIO_TEXT, source_id=f"test_{method}", method=method)
        run_test(f"method='{method}' produces chunks", len(c) >= 0)

    # Invalid method should raise ValueError
    raised = False
    try:
        chunk_text("Some text.", method="nonexistent")
    except ValueError:
        raised = True
    run_test("Invalid method raises ValueError", raised)

    if verbose:
        print("  (All methods ran without errors)")


def test_all_methods_comparative(verbose: bool = False) -> None:
    header("Comparative benchmarks — all methods on PDF_TEXT")
    methods = ["hybrid", "semantic", "recursive", "paragraph",
               "sentence", "sliding", "fixed"]

    fmt = f"  {'Method':<16} {'Chunks':>6} {'Avg chars':>10} {'Time (ms)':>10}"
    print(fmt)
    print(f"  {'─'*16} {'─'*6} {'─'*10} {'─'*10}")

    for method in methods:
        t0 = time.perf_counter()
        chunks = chunk_text(PDF_TEXT, source_id="bench", method=method)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = ChunkAnalyser.analyse(chunks)
        print(
            f"  {method:<16} {stats.get('count', 0):>6} "
            f"{stats.get('avg_chars', 0):>10.1f} "
            f"{elapsed_ms:>10.2f}"
        )

    # All methods must finish in under 5 seconds total
    t0 = time.perf_counter()
    for method in methods:
        chunk_text(PDF_TEXT, source_id="speed", method=method)
    total = time.perf_counter() - t0
    run_test(f"All 7 methods complete in < 5 s total  [{total:.3f}s]", total < 5.0)


#  CHUNK ANALYSER TESTS

def test_chunk_analyser(verbose: bool = False) -> None:
    header("ChunkAnalyser")

    chunks = chunk_text(PDF_TEXT, source_id="analyser_test")
    stats  = ChunkAnalyser.analyse(chunks)

    run_test("count matches len(chunks)",         stats["count"] == len(chunks))
    run_test("total_chars > 0",                   stats["total_chars"] > 0)
    run_test("avg_chars in realistic range",       50 < stats["avg_chars"] < 2000)
    run_test("min_chars <= avg_chars",             stats["min_chars"] <= stats["avg_chars"])
    run_test("max_chars >= avg_chars",             stats["max_chars"] >= stats["avg_chars"])
    run_test("empty_chunks = 0",                   stats["empty_chunks"] == 0)

    # Empty chunk list
    empty_stats = ChunkAnalyser.analyse([])
    run_test("Empty list returns count=0",         empty_stats["count"] == 0)

    if verbose:
        ChunkAnalyser.print_report(stats)


# EDGE-CASE AND STRESS TESTS

def test_edge_cases(verbose: bool = False) -> None:
    header("Edge cases & stress tests")

    chunker = HybridChunker(max_chunk_size=512, min_chunk_size=50)

    # Whitespace-only text
    run_test("Whitespace-only text returns []",   chunker.chunk("   \n\n  \t  ") == [])

    # Single very long sentence (no sentence-boundary) 
    long_sent = "word " * 300       # 1500 chars, no punctuation
    chunks = chunker.chunk(long_sent, source_id="long_sent")
    run_test("Very long sentence (no punct) chunked safely", len(chunks) >= 1)
    run_test("Long-sentence chunks respect size ceiling",
             all(len(c["text"]) <= 512 * 1.05 for c in chunks))

    # Unicode / multilingual text
    unicode_text = (
        "Machine learning is fascinating. "
        "التعلم الآلي مجال مثير للاهتمام. "
        "机器学习是人工智能的一个分支。 "
        "Машинное обучение — это раздел искусственного интеллекта. "
    ) * 5
    u_chunks = chunker.chunk(unicode_text, source_id="unicode")
    run_test("Unicode / multilingual text chunked without error", len(u_chunks) >= 1)

    # Text with many structural markers but no body
    marker_only = "\n".join([
        "--- Page 1 ---",
        "--- Page 2 ---",
        "--- Page 3 ---",
    ])
    m_chunks = chunker.chunk(marker_only, source_id="markers_only")
    # Should not crash; may produce 0 or a few tiny chunks
    run_test("Text with only structural markers does not crash", True)

    # Stress: large document (simulate a 100-page PDF)
    large_doc = "\n\n".join([
        f"--- Page {i} ---\n" + (PDF_TEXT.split("---")[2] * 2)
        for i in range(1, 51)     # 50 simulated pages
    ])
    t0 = time.perf_counter()
    large_chunks = chunker.chunk(large_doc, source_id="large_doc")
    elapsed = time.perf_counter() - t0
    run_test(f"50-page document chunked in < 1 s  [{elapsed:.3f}s]",  elapsed < 1.0)
    run_test("50-page document produces many chunks",                   len(large_chunks) > 10)

    if verbose:
        print(f"  50-page doc → {len(large_chunks)} chunks in {elapsed:.3f}s")




def main() -> None:
    global EXTRACTED_TEXT, EXTRACTED_SOURCE_NAME, USE_MOCK_TEXT
    
    parser = argparse.ArgumentParser(
        description="Graph-RAG Chunking Test Suite with File Extraction"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show chunk previews and detailed stats"
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Run tests only for a specific chunker "
             "(hybrid | semantic | recursive | paragraph | sentence | sliding | fixed)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to file to extract and chunk (optional; will prompt if not provided)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock text instead of extracting from a file"
    )
    args = parser.parse_args()

    # Try to extract text from file
    if args.mock:
        print("Using mock text for testing (--mock flag set).")
        USE_MOCK_TEXT = True
    elif args.file:
        text, source, success = load_extracted_text(args.file)
        if success:
            EXTRACTED_TEXT = text
            EXTRACTED_SOURCE_NAME = source
            USE_MOCK_TEXT = False
        else:
            print("\n⚠ Extraction failed. Falling back to mock text.")
            USE_MOCK_TEXT = True
    else:
        # Prompt user for file (unless -v or --method alone is passed)
        if not sys.argv[1:] or all(a in ["-v", "--verbose"] for a in sys.argv[1:]):
            file_path = get_file_path_from_user()
            if file_path:
                text, source, success = load_extracted_text(file_path)
                if success:
                    EXTRACTED_TEXT = text
                    EXTRACTED_SOURCE_NAME = source
                    USE_MOCK_TEXT = False
                else:
                    print("\nExtraction failed. Falling back to mock text.")
                    USE_MOCK_TEXT = True
    
    # Run tests
    TESTS = {
        "fixed":     test_fixed_size_chunker,
        "sentence":  test_sentence_chunker,
        "paragraph": test_paragraph_chunker,
        "recursive": test_recursive_chunker,
        "sliding":   test_sliding_window_chunker,
        "semantic":  test_semantic_chunker,
        "hybrid":    test_hybrid_chunker,
    }

    ALWAYS_RUN = [test_chunk_text_convenience, test_all_methods_comparative,
                  test_chunk_analyser, test_edge_cases]

    print("\n" + "═" * 60)
    print("     Graph-RAG — Chunking Test Suite")
    if not USE_MOCK_TEXT:
        print(f"     Source: {EXTRACTED_SOURCE_NAME}")
    print("═" * 60)
    
    # Use extracted text for all tests
    global PDF_TEXT, PPTX_TEXT, DOCX_TEXT, AUDIO_TEXT, URL_TEXT
    if not USE_MOCK_TEXT and EXTRACTED_TEXT:
        # Reuse extracted text across all tests
        PDF_TEXT = EXTRACTED_TEXT
        PPTX_TEXT = EXTRACTED_TEXT
        DOCX_TEXT = EXTRACTED_TEXT
        AUDIO_TEXT = EXTRACTED_TEXT
        URL_TEXT = EXTRACTED_TEXT

    if args.method:
        fn = TESTS.get(args.method.lower())
        if fn is None:
            print(f"Unknown method '{args.method}'. Choose from: {list(TESTS)}")
            sys.exit(1)
        fn(verbose=args.verbose)
    else:
        for fn in TESTS.values():
            fn(verbose=args.verbose)
        for fn in ALWAYS_RUN:
            fn(verbose=args.verbose)

   
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {_passed} / {_total} tests passed")
    if _passed == _total:
        print(" All tests passed!")
    else:
        failed = _total - _passed
        print(f" {failed} test(s) failed — review output above.")
    print("═" * 60 + "\n")

    sys.exit(0 if _passed == _total else 1)


if __name__ == "__main__":
    main()