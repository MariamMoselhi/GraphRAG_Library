"""
chunking_test.py — Test suite for GraphRAG Chunking Module
===========================================================
Tests every chunker on text extracted from user-provided files using
the project's extractors (PDFExtractor, PPTXExtractor, DOCXExtractor,
AudioExtractor, URLExtractor, VideoExtractor).

The workflow is:
1. Accept input path from command line argument or interactive prompt
2. Extract text using SimpleExtractorWrapper.extract_auto()
3. Run all chunker tests on the extracted text
4. Display results in standardized format (with optional fallback to mock data)

Run
---
    python chunking_test.py                           # will prompt for file path, run all tests
    python chunking_test.py -v                        # verbose (show detailed stats)
    python chunking_test.py --file path/to/file.pdf   # specify file directly
    python chunking_test.py --mock                    # use mock data instead of extracting

Module structure being tested
------------------------------
    chunking.py              <- public API  (the only file imported here)
        chunk_implementations.py  <- all six chunker classes
            chunk_base.py         <- Chunk dataclass + BaseChunker
                text_cleaner.py   <- TextCleaner

Dependencies: extractors module for file extraction.
"""

import sys
import time
import argparse
from typing import List
from pathlib import Path

# Add parent directory to path so the chunking package and extractors are found
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


try:
    from chunking import (
        Chunk,
        chunk_text,
        get_chunker,
        DEFAULT_CHUNKER,
        FixedSizeChunker,
        SentenceChunker,
        ParagraphChunker,
        RecursiveChunker,
        SlidingWindowChunker,
        SemanticChunker,
    )
    from text_cleaner import TextCleaner   # internal module — imported directly
except ImportError as e:
    print(f"ERROR: Could not import chunking package — {e}")
    print("Make sure chunking.py, chunk_implementations.py, chunk_base.py,")
    print("and text_cleaner.py are all in the same directory as this test file.")
    sys.exit(1)


try:
    from extractors.extractors_wrapper import SimpleExtractorWrapper
    print("✓ Extractors module loaded successfully")
except ImportError as e:
    print(f"WARNING: Could not import extractors module — {e}")
    print(f"  sys.path includes: {sys.path[:3]}")
    SimpleExtractorWrapper = None


EXTRACTED_TEXT        = None
EXTRACTED_SOURCE_NAME = "test"
USE_MOCK_TEXT         = True

PASS = "✓ PASS"
FAIL = "✗ FAIL"

_total  = 0
_passed = 0



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

SHORT_TEXT  = "This is a very short text. It has only two sentences."
EMPTY_TEXT  = ""
REPEAT_TEXT = "Data science. " * 80


def get_file_path_from_user() -> str:
    """Prompt user to enter a file path."""
    print("\n" + "─" * 60)
    print("  Enter a file path to extract and chunk")
    print("  Supported formats: PDF, DOCX, PPTX, TXT, MP3, MP4, WAV, URL")
    print("─" * 60)
    file_path = input("File path (or leave empty for mock text): ").strip()
    return file_path


def load_extracted_text(file_path: str) -> tuple:
    """
    Extract text from a file using SimpleExtractorWrapper.

    Returns
    -------
    Tuple of (extracted_text: str, source_name: str, success: bool)
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



def run_test(name: str, condition: bool, detail: str = "") -> None:
    """Register and print a single assertion result."""
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


def preview(chunks: List[Chunk], n: int = 3, verbose: bool = False) -> None:
    """Print a preview of the first *n* chunks."""
    if not verbose or not chunks:
        return
    for i, c in enumerate(chunks[:n]):
        text_preview = c.text[:100].replace("\n", "↵")
        print(f"    [{i}] id={c.chunk_id}  chars={c.char_count}")
        print(f"        \"{text_preview}…\"")
    if len(chunks) > n:
        print(f"    … and {len(chunks) - n} more chunks")


def test_fixed_size_chunker(text: str, verbose: bool = False) -> None:
    """
    Tests FixedSizeChunker from chunk_implementations.py.
    Instantiated via get_chunker() from chunking.py (the public API).
    """
    header("FixedSizeChunker  [chunk_implementations.py]")

    # Instantiate through the public factory — confirms the registry in chunking.py is wired correctly
    chunker = get_chunker("fixed_size", chunk_size=256, overlap=30)

    # Verify the factory returned the right concrete class
    run_test(
        "get_chunker('fixed_size') returns FixedSizeChunker",
        isinstance(chunker, FixedSizeChunker),
    )

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))

    if len(chunks) >= 2:
        overlap_region = chunks[0].text[-30:]
        run_test("Overlap region present in next chunk", overlap_region in chunks[1].text)

    # Chunk dataclass properties — defined in chunk_base.py
    if chunks:
        run_test("Chunk.char_count matches len(text)", chunks[0].char_count == len(chunks[0].text))
        run_test("Chunk.word_count > 0", chunks[0].word_count > 0)
        run_test("Chunk.start_char >= 0", chunks[0].start_char >= 0)
        run_test("Chunk.end_char > start_char", chunks[0].end_char > chunks[0].start_char)
        run_test("metadata contains 'chunker' key", "chunker" in chunks[0].metadata)
        run_test("metadata['chunker'] == 'fixed_size'", chunks[0].metadata["chunker"] == "fixed_size")

    # Edge cases
    run_test("Empty text returns []", len(chunker.chunk("")) == 0)
    run_test("Short text produces >= 1 chunk", len(chunker.chunk(SHORT_TEXT)) >= 1)

    print(f"  {len(chunks)} chunks produced")
    preview(chunks, verbose=verbose)


def test_sentence_chunker(text: str, verbose: bool = False) -> None:
    """
    Tests SentenceChunker from chunk_implementations.py.
    Also indirectly tests BaseChunker._split_sentences() from chunk_base.py.
    """
    header("SentenceChunker  [chunk_implementations.py -> chunk_base.py]")

    chunker = get_chunker("sentence", sentences_per_chunk=3, overlap_sentences=1)
    run_test(
        "get_chunker('sentence') returns SentenceChunker",
        isinstance(chunker, SentenceChunker),
    )

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test(
        "chunk_id is sequential from 0",
        all(chunks[i].chunk_id == i for i in range(len(chunks))),
    )

    if chunks:
        run_test(
            "metadata contains 'sentence_count'",
            "sentence_count" in chunks[0].metadata,
        )
        run_test(
            "metadata['chunker'] == 'sentence'",
            chunks[0].metadata["chunker"] == "sentence",
        )

    run_test("Empty text returns []", len(chunker.chunk("")) == 0)

    print(f"  {len(chunks)} chunks produced")
    preview(chunks, verbose=verbose)


def test_paragraph_chunker(text: str, verbose: bool = False) -> None:
    """
    Tests ParagraphChunker from chunk_implementations.py.
    Also verifies TextCleaner (text_cleaner.py) strips page markers
    before the paragraph splitter sees them.
    """
    header("ParagraphChunker  [chunk_implementations.py -> text_cleaner.py]")

    chunker = get_chunker("paragraph", min_chars=50)
    run_test(
        "get_chunker('paragraph') returns ParagraphChunker",
        isinstance(chunker, ParagraphChunker),
    )

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test(
        "chunk_id is sequential from 0",
        all(chunks[i].chunk_id == i for i in range(len(chunks))),
    )

    if chunks:
        run_test(
            "metadata['chunker'] == 'paragraph'",
            chunks[0].metadata["chunker"] == "paragraph",
        )
        avg_chars = sum(c.char_count for c in chunks) // len(chunks)
        run_test(f"Average chunk size > 0  (avg={avg_chars} chars)", avg_chars > 0)

    run_test("Empty text returns []", len(chunker.chunk("")) == 0)

    print(f"  {len(chunks)} chunks produced")
    preview(chunks, verbose=verbose)


def test_recursive_chunker(text: str, verbose: bool = False) -> None:
    """
    Tests RecursiveChunker from chunk_implementations.py.
    Verifies the hard max_chunk_size guarantee that the recursive _split()
    method is designed to enforce.
    """
    header("RecursiveChunker  [chunk_implementations.py]")

    chunker = get_chunker("recursive", max_chunk_size=300)
    run_test(
        "get_chunker('recursive') returns RecursiveChunker",
        isinstance(chunker, RecursiveChunker),
    )

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test(
        "No chunk exceeds max_chunk_size (300 chars)",
        all(c.char_count <= 300 for c in chunks),
    )

    if chunks:
        run_test(
            "metadata['chunker'] == 'recursive'",
            chunks[0].metadata["chunker"] == "recursive",
        )
        max_seen = max(c.char_count for c in chunks)
        run_test(
            f"Largest chunk is within limit  (max={max_seen} chars)",
            max_seen <= 300,
        )

    run_test("Empty text returns []", len(chunker.chunk("")) == 0)

    # Stress: very long word should not cause an infinite loop
    very_long_word = "a" * 1000 + " normal text here."
    vl_chunks = chunker.chunk(very_long_word)
    run_test("Very long word does not cause infinite loop", len(vl_chunks) >= 1)

    print(f"  {len(chunks)} chunks produced")
    preview(chunks, verbose=verbose)


def test_sliding_window_chunker(text: str, verbose: bool = False) -> None:
    """
    Tests SlidingWindowChunker from chunk_implementations.py.
    Verifies the overlap metadata that downstream graph-builders rely on.
    """
    header("SlidingWindowChunker  [chunk_implementations.py -> chunk_base.py]")

    chunker = get_chunker("sliding_window", window_size=5, step_size=3)
    run_test(
        "get_chunker('sliding_window') returns SlidingWindowChunker",
        isinstance(chunker, SlidingWindowChunker),
    )

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test(
        "chunk_id is sequential from 0",
        all(chunks[i].chunk_id == i for i in range(len(chunks))),
    )

    if chunks:
        run_test(
            "metadata contains 'overlap_sentences'",
            "overlap_sentences" in chunks[0].metadata,
        )
        # overlap = window_size - step_size = 5 - 3 = 2
        run_test(
            "overlap_sentences == window_size - step_size  (2)",
            chunks[0].metadata.get("overlap_sentences") == 2,
        )
        run_test(
            "metadata['chunker'] == 'sliding_window'",
            chunks[0].metadata["chunker"] == "sliding_window",
        )

    run_test("Empty text returns []", len(chunker.chunk("")) == 0)

    print(f"  {len(chunks)} chunks produced")
    preview(chunks, verbose=verbose)


def test_semantic_chunker(text: str, verbose: bool = False) -> None:
    """
    Tests SemanticChunker from chunk_implementations.py.
    Skips gracefully if sentence-transformers is not installed.
    Verifies the avg_internal_similarity metadata that the graph-builder
    uses to weight edges between nodes.
    """
    header("SemanticChunker  [chunk_implementations.py — DEFAULT chunker]")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("  [SKIP] sentence-transformers not installed")
        print("  Install with:  pip install sentence-transformers")
        return

    chunker = get_chunker(
        "semantic",
        threshold=0.5,
        max_sentences_per_chunk=15,
        min_sentences_per_chunk=2,
    )
    run_test(
        "get_chunker('semantic') returns SemanticChunker",
        isinstance(chunker, SemanticChunker),
    )

    t0 = time.perf_counter()
    chunks = chunker.chunk(text)
    elapsed = time.perf_counter() - t0

    run_test(f"Produces chunks  [{elapsed:.3f}s]", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test("Completes in reasonable time  (< 10s)", elapsed < 10.0)

    if chunks:
        run_test(
            "metadata contains 'avg_internal_similarity'",
            "avg_internal_similarity" in chunks[0].metadata,
        )
        for c in chunks:
            sim = c.metadata.get("avg_internal_similarity", -1)
            run_test(
                f"Chunk {c.chunk_id} similarity in [0.0, 1.0]  (got {sim:.4f})",
                0.0 <= sim <= 1.0,
            )
        run_test(
            "metadata['chunker'] == 'semantic'",
            chunks[0].metadata["chunker"] == "semantic",
        )

    run_test("Empty text returns []", len(chunker.chunk("")) == 0)

    print(f"  {len(chunks)} chunks produced")
    preview(chunks, verbose=verbose)


def test_text_cleaner(text: str, verbose: bool = False) -> None:
    """
    Tests TextCleaner from text_cleaner.py directly.
    Verifies that page markers, control characters, and whitespace noise
    are stripped before the chunkers receive the text.
    """
    header("TextCleaner  [text_cleaner.py — internal module]")

    cleaner = TextCleaner()

    # Page markers inserted by PDFExtractor
    noisy = "--- Page 1 ---\nSome text.\n\n--- Page 2 ---\nMore text."
    cleaned = cleaner.clean(noisy)
    run_test("Page markers are removed", "--- Page" not in cleaned)
    run_test("Clean output is non-empty", len(cleaned) > 0)

    # Null bytes and control characters
    with_nulls = "Hello\x00World\x01This\x02is\x03text"
    cleaned_nulls = cleaner.clean(with_nulls)
    run_test("Null bytes are removed", "\x00" not in cleaned_nulls)
    run_test("Control chars are removed", "\x01" not in cleaned_nulls)

    # Multiple spaces and tabs
    with_spaces = "word1   word2\t\tword3"
    cleaned_spaces = cleaner.clean(with_spaces)
    run_test("Multiple spaces collapsed to one", "   " not in cleaned_spaces)
    run_test("Tabs collapsed", "\t" not in cleaned_spaces)

    # Triple newlines
    with_newlines = "line1\n\n\n\nline2"
    cleaned_newlines = cleaner.clean(with_newlines)
    run_test("Triple newlines collapsed to double", "\n\n\n" not in cleaned_newlines)

    # Empty input guard
    run_test("Empty string returns empty string", cleaner.clean("") == "")

    # Applied to the live text (PDF mock has page markers — clean should remove them)
    cleaned_live = cleaner.clean(text)
    run_test("Live text is cleaned without error", isinstance(cleaned_live, str))

    if verbose:
        print(f"    Raw length: {len(text)}  ->  Cleaned length: {len(cleaned_live)}")


def test_default_chunker(text: str, verbose: bool = False) -> None:
    """
    Verifies DEFAULT_CHUNKER is 'semantic' and that get_chunker() with no
    arguments returns a SemanticChunker — as defined in chunking.py.
    """
    header("Default chunker & factory  [chunking.py]")

    run_test("DEFAULT_CHUNKER == 'semantic'", DEFAULT_CHUNKER == "semantic")

    default_chunker = get_chunker()   # no arguments -> should use DEFAULT_CHUNKER
    run_test(
        "get_chunker() with no args returns SemanticChunker",
        isinstance(default_chunker, SemanticChunker),
    )

    # chunk_text() convenience function defined in chunking.py
    result = chunk_text(text, chunker_type="paragraph")
    run_test("chunk_text() returns a list", isinstance(result, list))
    run_test("chunk_text() returns Chunk objects", all(isinstance(c, Chunk) for c in result))

    # Unknown chunker type raises ValueError (registry guard in chunking.py)
    try:
        get_chunker("nonexistent_chunker")
        run_test("Unknown chunker type raises ValueError", False)
    except ValueError:
        run_test("Unknown chunker type raises ValueError", True)

    # All six registry keys resolve without error
    for key in ["fixed_size", "sentence", "paragraph", "recursive", "sliding_window", "semantic"]:
        c = get_chunker(key)
        run_test(f"Registry key '{key}' instantiates without error", c is not None)


def test_edge_cases(text: str, verbose: bool = False) -> None:
    """
    Edge cases and stress tests exercised through the public API (chunking.py).
    All assertions ultimately test behaviour defined in chunk_implementations.py
    and chunk_base.py.
    """
    header("Edge cases & stress tests  [all layers]")

    # Empty string — every chunker should return []
    for key in ["fixed_size", "sentence", "paragraph", "recursive", "sliding_window"]:
        result = chunk_text("", chunker_type=key)
        run_test(f"{key}: empty string returns []", result == [])

    # Very long word — RecursiveChunker must not loop infinitely
    very_long_word = "a" * 1000 + " normal text here."
    long_chunks = chunk_text(very_long_word, chunker_type="recursive", max_chunk_size=300)
    run_test("RecursiveChunker: very long word doesn't cause infinite loop", len(long_chunks) >= 1)

    # Single sentence
    single_sent = "This is one sentence."
    sent_chunks = chunk_text(single_sent, chunker_type="sentence",
                             sentences_per_chunk=5, overlap_sentences=1)
    run_test("SentenceChunker: single sentence produces >= 1 chunk", len(sent_chunks) >= 1)

    # chunk_id always starts at 0
    for key in ["fixed_size", "sentence", "paragraph", "recursive", "sliding_window"]:
        chunks = chunk_text(text, chunker_type=key)
        if chunks:
            run_test(
                f"{key}: first chunk has chunk_id == 0",
                chunks[0].chunk_id == 0,
            )

    # Chunk offsets are internally consistent (Chunk dataclass from chunk_base.py)
    para_chunks = chunk_text(text, chunker_type="fixed_size", chunk_size=200, overlap=20)
    for c in para_chunks:
        run_test(
            f"FixedSize chunk {c.chunk_id}: end_char > start_char",
            c.end_char > c.start_char,
        )
        break   # verify the first one to keep output concise

    if verbose:
        print(f"    Long word produced {len(long_chunks)} chunks")
        print(f"    Single sentence produced {len(sent_chunks)} chunk(s)")


def main() -> None:
    global EXTRACTED_TEXT, EXTRACTED_SOURCE_NAME, USE_MOCK_TEXT
    global PDF_TEXT, PPTX_TEXT, DOCX_TEXT, AUDIO_TEXT, URL_TEXT

    parser = argparse.ArgumentParser(
        description="GraphRAG Chunking Test Suite (four-file module structure)",
        epilog=(
            "Examples:\n"
            "  python chunking_test.py                       # interactive input\n"
            "  python chunking_test.py --file lecture.pdf    # single file\n"
            "  python chunking_test.py --mock                # use mock data\n"
            "  python chunking_test.py -v                    # verbose output"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show chunk previews and detailed statistics",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a file to extract and chunk",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock text instead of extracting from files",
    )
    args = parser.parse_args()


    if args.mock:
        print("Using mock text for testing (--mock flag set).")
        USE_MOCK_TEXT = True

    elif args.file:
        text, source, success = load_extracted_text(args.file)
        if success:
            EXTRACTED_TEXT        = text
            EXTRACTED_SOURCE_NAME = source
            USE_MOCK_TEXT         = False
        else:
            print("\n Warning: Extraction failed. Falling back to mock text.")
            USE_MOCK_TEXT = True

    else:
        # No flag given — prompt interactively, but only if no other args were passed
        only_verbose = all(a in ["-v", "--verbose"] for a in sys.argv[1:])
        if not sys.argv[1:] or only_verbose:
            file_path = get_file_path_from_user()
            if file_path:
                text, source, success = load_extracted_text(file_path)
                if success:
                    EXTRACTED_TEXT        = text
                    EXTRACTED_SOURCE_NAME = source
                    USE_MOCK_TEXT         = False
                else:
                    print("\nExtraction failed. Falling back to mock text.")
                    USE_MOCK_TEXT = True


    if not USE_MOCK_TEXT and EXTRACTED_TEXT:
        PDF_TEXT   = EXTRACTED_TEXT
        PPTX_TEXT  = EXTRACTED_TEXT
        DOCX_TEXT  = EXTRACTED_TEXT
        AUDIO_TEXT = EXTRACTED_TEXT
        URL_TEXT   = EXTRACTED_TEXT


    print(f"\n{'=' * 70}")
    print(f"  GraphRAG — Chunking Test Suite (four-file module structure)")
    if not USE_MOCK_TEXT:
        print(f"  Source: {EXTRACTED_SOURCE_NAME}")
    else:
        print(f"  Source: mock text")
    print(f"{'=' * 70}")


    TESTS = [
        ("TextCleaner",          test_text_cleaner),
        ("Default chunker",      test_default_chunker),
        ("FixedSizeChunker",     test_fixed_size_chunker),
        ("SentenceChunker",      test_sentence_chunker),
        ("ParagraphChunker",     test_paragraph_chunker),
        ("RecursiveChunker",     test_recursive_chunker),
        ("SlidingWindowChunker", test_sliding_window_chunker),
        ("SemanticChunker",      test_semantic_chunker),
        ("Edge cases",           test_edge_cases),
    ]

    for name, test_fn in TESTS:
        try:
            test_fn(PDF_TEXT, verbose=args.verbose)
        except Exception as e:
            header(name)
            print(f"  FAIL  Unexpected error: {str(e)[:120]}")
 

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