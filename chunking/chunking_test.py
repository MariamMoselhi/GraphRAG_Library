"""
chunking_test.py — Test suite for Graph-RAG Chunking Module
======================================================================
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
sys.path.insert(0, str(Path(__file__).parent / "__pycache__"))

try:
    from chunking import (
        Chunk,
        TextCleaner,
        chunk_text,
        get_chunker,
        FixedSizeChunker,
        SentenceChunker,
        ParagraphChunker,
        RecursiveChunker,
        SlidingWindowChunker,
        SemanticChunker,
    )
except ImportError as e:
    print(f"ERROR: Could not import chunking.py — {e}")
    print("Make sure chunking.py is in the same directory or __pycache__ subfolder.")
    sys.exit(1)

try:
    from extractors.extractors_wrapper import SimpleExtractorWrapper
    print("✓ Extractors module loaded successfully")
except ImportError as e:
    print(f"ERROR: Could not import extractors module — {e}")
    print(f"ImportError details: {e}")
    print(f"sys.path includes: {sys.path[:3]}")
    SimpleExtractorWrapper = None


# ── GLOBAL STATE ─────────────────────────────────────────────────────────────
EXTRACTED_TEXT = None
EXTRACTED_SOURCE_NAME = "test"
USE_MOCK_TEXT = True

PASS = "✓ PASS"
FAIL = "✗ FAIL"

_total  = 0
_passed = 0


# ── MOCK TEXT DATA (fallback if extraction fails) ────────────────────────────
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

SHORT_TEXT = "This is a very short text. It has only two sentences."
EMPTY_TEXT = ""
REPEAT_TEXT = "Data science. " * 80
# ── FILE INPUT FUNCTIONS ─────────────────────────────────────────────────────

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


# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def run_test(name: str, condition: bool, detail: str = "") -> None:
    """Register a single assertion result."""
    global _total, _passed
    _total += 1
    status = PASS if condition else FAIL
    if condition:
        _passed += 1
    suffix = f"  [{detail}]" if detail and not condition else ""
    info_prefix = "  " if detail and not condition else "  "
    print(f"{info_prefix}{status}  {name}{suffix}")


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def preview(chunks: List[Dict[str, Any]], n: int = 3, verbose: bool = False) -> None:
    """Print a preview of the first *n* chunks."""
    if not verbose or not chunks:
        return
    for i, c in enumerate(chunks[:n]):
        text_preview = c.get("text", "")[:100].replace("\n", "↵")
        print(f"    [{i}] id={c.get('chunk_id', 'N/A')}  chars={len(c.get('text', ''))}  ")
        print(f"        \"{text_preview}…\"")
    if len(chunks) > n:
        print(f"    … and {len(chunks) - n} more chunks")


# ── CHUNKER TESTS ────────────────────────────────────────────────────────────

def test_fixed_size_chunker(text: str, verbose: bool = False) -> None:
    header("FixedSizeChunker")
    chunker = FixedSizeChunker(chunk_size=256, overlap=30)

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))

    if len(chunks) >= 2:
        overlap_region = chunks[0].text[-30:]
        overlap_found = overlap_region in chunks[1].text
        run_test("Overlap region present in next chunk", overlap_found)

    run_test("Empty text returns empty list", len(chunker.chunk("")) == 0)
    short_chunks = chunker.chunk("This is a very short text. It has only two sentences.")
    run_test("Short text produces ≥ 1 chunk", len(short_chunks) >= 1)

    print(f"{len(chunks)} chunks produced")
    if verbose and chunks:
        print(f"    Sample: Chunk 0 has {chunks[0].char_count} chars")


def test_sentence_chunker(text: str, verbose: bool = False) -> None:
    header("SentenceChunker")
    chunker = SentenceChunker(sentences_per_chunk=3, overlap_sentences=1)

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test("chunk_id is sequential", all(chunks[i].chunk_id == i for i in range(len(chunks))))

    run_test("Empty text returns empty list", len(chunker.chunk("")) == 0)

    print(f"{len(chunks)} chunks produced")
    if verbose and chunks:
        print(f"    Sample: Chunk 0 has {chunks[0].char_count} chars")


def test_paragraph_chunker(text: str, verbose: bool = False) -> None:
    header("ParagraphChunker")
    chunker = ParagraphChunker(min_chars=50)

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test("chunk_id is sequential", all(chunks[i].chunk_id == i for i in range(len(chunks))))

    run_test("Empty text returns empty list", len(chunker.chunk("")) == 0)

    print(f"{len(chunks)} chunks produced")
    if verbose and chunks:
        print(f"    Sample: Avg {sum(c.char_count for c in chunks) // len(chunks)} chars per chunk")


def test_recursive_chunker(text: str, verbose: bool = False) -> None:
    header("RecursiveChunker")
    chunker = RecursiveChunker(max_chunk_size=300)

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test("No chunk exceeds max_chunk_size", all(c.char_count <= 300 for c in chunks))

    run_test("Empty text returns empty list", len(chunker.chunk("")) == 0)

    print(f"{len(chunks)} chunks produced")
    if verbose and chunks:
        print(f"    Sample: Max chunk size {max(c.char_count for c in chunks)} chars")


def test_sliding_window_chunker(text: str, verbose: bool = False) -> None:
    header("SlidingWindowChunker")
    chunker = SlidingWindowChunker(window_size=5, step_size=3)

    chunks = chunker.chunk(text)
    run_test("Produces chunks from text", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test("chunk_id is sequential", all(chunks[i].chunk_id == i for i in range(len(chunks))))

    run_test("Empty text returns empty list", len(chunker.chunk("")) == 0)

    print(f"{len(chunks)} chunks produced")
    if verbose and chunks:
        print(f"    Sample: Sliding window overlaps applied")


def test_semantic_chunker(text: str, verbose: bool = False) -> None:
    header("SemanticChunker")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("  [SKIP] sentence-transformers not installed")
        return

    chunker = SemanticChunker(
        threshold=0.5,
        max_sentences_per_chunk=15,
        min_sentences_per_chunk=2,
    )

    t0 = time.perf_counter()
    chunks = chunker.chunk(text)
    elapsed = time.perf_counter() - t0

    run_test(f"Produces chunks  [{elapsed:.3f}s]", len(chunks) >= 1)
    run_test("All chunks are Chunk objects", all(isinstance(c, Chunk) for c in chunks))
    run_test("Completes in reasonable time", elapsed < 10.0)

    run_test("Empty text returns empty list", len(chunker.chunk("")) == 0)

    print(f"{len(chunks)} chunks produced")
    if verbose and chunks:
        print(f"    Sample: Processing time: {elapsed:.3f}s")


def test_edge_cases(text: str, verbose: bool = False) -> None:
    header("Edge cases & stress tests")

    run_test("Empty string returns []", len(chunk_text("")) == 0)

    very_long_word = "a" * 1000 + " normal text here."
    long_chunks = chunk_text(very_long_word)
    run_test("Very long word doesn't cause infinite loop", len(long_chunks) >= 1)

    single_sent = "This is one sentence."
    sent_chunks = chunk_text(single_sent)
    run_test("Single sentence produces ≥ 1 chunk", len(sent_chunks) >= 1)

    print(f"Long word produced {len(long_chunks)} chunks, single sentence produced {len(sent_chunks)} chunk(s)")
    if verbose:
        print(f"    Edge cases handled correctly")


# ── MAIN TEST RUNNER ────────────────────────────────────────────────────────



def main() -> None:
    global EXTRACTED_TEXT, EXTRACTED_SOURCE_NAME, USE_MOCK_TEXT
    global PDF_TEXT, PPTX_TEXT, DOCX_TEXT, AUDIO_TEXT, URL_TEXT
    
    parser = argparse.ArgumentParser(
        description="Graph-RAG Chunking Test Suite with File Extraction",
        epilog="Examples:\n"
               "  python chunking_test.py                                    # interactive input\n"
               "  python chunking_test.py --file path/to/file.pdf            # single file\n"
               "  python chunking_test.py --mock                             # use mock data\n"
               "  python chunking_test.py -v                                 # verbose output",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show chunk previews and detailed statistics"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a file to extract and chunk"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock text instead of extracting from files"
    )
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # TRY TO EXTRACT TEXT FROM FILE
    # ─────────────────────────────────────────────────────────────────────────
    
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
        # Prompt user for file (unless running with just -v flag)
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # RUN TESTS
    # ─────────────────────────────────────────────────────────────────────────
    
    TESTS = [
        ("FixedSizeChunker", test_fixed_size_chunker),
        ("SentenceChunker", test_sentence_chunker),
        ("ParagraphChunker", test_paragraph_chunker),
        ("RecursiveChunker", test_recursive_chunker),
        ("SlidingWindowChunker", test_sliding_window_chunker),
        ("SemanticChunker", test_semantic_chunker),
    ]

    MISC = [test_edge_cases]

    print(f"\n{'═' * 70}")
    print(f"     Graph-RAG — Chunking Test Suite")
    if not USE_MOCK_TEXT:
        print(f"     Source: {EXTRACTED_SOURCE_NAME}")
    print(f"{'═' * 70}")

    # Use extracted text for all tests if available
    if not USE_MOCK_TEXT and EXTRACTED_TEXT:
        # Reuse extracted text across all tests
        PDF_TEXT = EXTRACTED_TEXT
        PPTX_TEXT = EXTRACTED_TEXT
        DOCX_TEXT = EXTRACTED_TEXT
        AUDIO_TEXT = EXTRACTED_TEXT
        URL_TEXT = EXTRACTED_TEXT

    for name, test_fn in TESTS:
        try:
            test_fn(PDF_TEXT, verbose=args.verbose)
        except Exception as e:
            header(name)
            print(f"  ✗ Error: {str(e)[:100]}")

    for test_fn in MISC:
        try:
            test_fn(PDF_TEXT, verbose=args.verbose)
        except Exception as e:
            print(f"  ✗ Error in {test_fn.__name__}: {str(e)[:100]}")

    print(f"\n{'═' * 70}")
    if _passed == _total:
        print(f"  ✓ {_passed} / {_total} tests passed")
    else:
        failed = _total - _passed
        print(f"  ✓ {_passed} / {_total} tests passed  ({failed} failed)")
    print(f"{'═' * 70}\n")

    sys.exit(0 if _passed == _total else 1)


if __name__ == "__main__":
    main()
