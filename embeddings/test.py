"""
embedding_test.py — Test suite for GraphRAG Embedding Module
=============================================================
Tests the embedding layer (BaseEmbedding + HuggingFaceEmbedding) on raw
text, on chunks produced by the chunking module, and on edge-case inputs.

The workflow is:
1. Accept optional input text from command line or interactive prompt
2. Optionally extract text using SimpleExtractorWrapper.extract_auto()
3. Optionally chunk the text using the chunking module
4. Run all embedding tests
5. Display results in the same standardised format as chunking_test.py

Run
---
    python embedding_test.py                           # will prompt for file path
    python embedding_test.py -v                        # verbose (show vector previews)
    python embedding_test.py --file path/to/file.pdf   # specify file directly
    python embedding_test.py --mock                    # use mock text directly
    python embedding_test.py --model BAAI/bge-small-en-v1.5   # custom model

Module structure being tested
------------------------------
    embeddings/__init__.py        <- public re-exports
        embeddings/base.py        <- BaseEmbedding ABC
        embeddings/huggingFace.py <- HuggingFaceEmbedding
    vectordb/__init__.py          <- public re-exports
        vectordb/base.py          <- InMemoryVectorStore, RetrievalResult
        vectordb/faiss_store.py   <- FaissVectorStore

Dependencies: chunking module (optional, for chunk-level & vector store tests),
              extractors module (optional, for file extraction).
"""

from __future__ import annotations

import sys
import time
import argparse
import warnings
from typing import List, Optional
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# test.py sits under embeddings/.
# Add project_root so absolute package imports like `from embeddings import ...`
# resolve regardless of whether this file is run from root or directly.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import embedding module
# ---------------------------------------------------------------------------
try:
    from embeddings import BaseEmbedding, HuggingFaceEmbedding
    print("✓ Embeddings module loaded successfully")
except ImportError as e:
    print(f"ERROR: Could not import embeddings package — {e}")
    print("Make sure embeddings/__init__.py, base.py, and huggingFace.py exist.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optional: chunking module (used for chunk-level embedding tests)
# ---------------------------------------------------------------------------
try:
    from chunking import chunk_text, Chunk
    _CHUNKING_AVAILABLE = True
    print("✓ Chunking module loaded successfully")
except ImportError as e:
    _CHUNKING_AVAILABLE = False
    print(f"WARNING: Chunking module not found ({e}) — chunk-level tests will be skipped.")

# ---------------------------------------------------------------------------
# Optional: extractors
# ---------------------------------------------------------------------------
try:
    from extractors.extractors_wrapper import SimpleExtractorWrapper
    _EXTRACTORS_AVAILABLE = True
    print("✓ Extractors module loaded successfully")
except ImportError:
    SimpleExtractorWrapper = None
    _EXTRACTORS_AVAILABLE = False
    print("WARNING: Extractors module not found — file extraction tests will be skipped.")

# ---------------------------------------------------------------------------
# Optional: vectordb (FaissVectorStore + InMemoryVectorStore)
# ---------------------------------------------------------------------------
try:
    from vectordb import FaissVectorStore, InMemoryVectorStore, RetrievalResult
    _VECTORDB_AVAILABLE = True
    print("✓ VectorDB module loaded successfully")
except ImportError as e:
    FaissVectorStore       = None
    InMemoryVectorStore    = None
    RetrievalResult        = None
    _VECTORDB_AVAILABLE    = False
    print(f"WARNING: VectorDB module not found ({e}) — vector store tests will be skipped.")


# ---------------------------------------------------------------------------
# Global test counters
# ---------------------------------------------------------------------------
PASS = "✓ PASS"
FAIL = "✗ FAIL"

_total  = 0
_passed = 0


# ---------------------------------------------------------------------------
# Mock text (same topic as chunking_test.py for consistency)
# ---------------------------------------------------------------------------
MOCK_TEXT = """
Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that focuses on building
systems that can learn from and make decisions based on data. Unlike traditional
programming, where explicit rules are written by programmers, machine learning
models infer rules from training examples.

There are three main categories of machine learning: supervised learning,
unsupervised learning, and reinforcement learning. Each paradigm is suited to
different types of problems and data availability scenarios.

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

Unsupervised Learning

Unsupervised learning deals with unlabeled data. The goal is to discover hidden
structure or representations in the input. Clustering algorithms such as K-means,
DBSCAN, and hierarchical clustering group similar data points together.

Dimensionality reduction techniques, including Principal Component Analysis (PCA)
and t-SNE, project high-dimensional data into lower-dimensional spaces while
preserving important structure. These techniques are widely used for visualisation
and as preprocessing steps.

Reinforcement Learning

Reinforcement learning (RL) involves an agent that interacts with an environment.
The agent receives rewards or penalties for its actions and learns a policy that
maximises cumulative reward over time.

Key RL algorithms include Q-learning, Deep Q-Networks (DQN), Policy Gradient
methods, and Proximal Policy Optimisation (PPO). RL has achieved remarkable
results in game-playing (AlphaGo, Atari) and robotics.
"""

# Semantically similar pair — should score high similarity
SIMILAR_PAIR = (
    "Machine learning models learn patterns from data.",
    "Deep learning algorithms discover representations in datasets.",
)

# Semantically dissimilar pair — should score low similarity
DISSIMILAR_PAIR = (
    "Machine learning models learn patterns from data.",
    "The Eiffel Tower is located in Paris, France.",
)


# ---------------------------------------------------------------------------
# Test helpers (identical style to chunking_test.py)
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


def get_file_path_from_user() -> Optional[str]:
    print("\nEnter a file path to extract text from (or press Enter to use mock text):")
    path = input("  > ").strip()
    return path if path else None


def load_extracted_text(file_path: str):
    """Extract text from a file using SimpleExtractorWrapper."""
    if not _EXTRACTORS_AVAILABLE:
        print("  Extractors not available — cannot extract from file.")
        return "", file_path, False

    print(f"\nExtracting text from: {file_path}")
    wrapper = SimpleExtractorWrapper()
    text = wrapper.extract_auto(file_path)
    if text and text.strip():
        print(f"  ✓ Extracted {len(text):,} characters from {Path(file_path).name}")
        return text, Path(file_path).name, True
    else:
        print(f"  ✗ Extraction returned empty text.")
        return "", file_path, False


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_model_initialization(model_name: str, verbose: bool = False) -> HuggingFaceEmbedding:
    """
    Verifies that HuggingFaceEmbedding initialises correctly and that
    the dimension property is set.
    """
    header("Model initialisation  [huggingFace.py]")

    start = time.time()
    model = HuggingFaceEmbedding(model_name=model_name, normalize=True)
    elapsed = time.time() - start

    run_test("Model loads without exception", model is not None)
    run_test("isinstance(model, BaseEmbedding)", isinstance(model, BaseEmbedding))
    run_test("isinstance(model, HuggingFaceEmbedding)", isinstance(model, HuggingFaceEmbedding))
    run_test("model.dimension is an int", isinstance(model.dimension, int))
    run_test("model.dimension > 0", (model.dimension or 0) > 0)
    run_test("__repr__ returns a string", isinstance(repr(model), str))

    if verbose:
        print(f"    Model: {model_name}")
        print(f"    Dimension: {model.dimension}")
        print(f"    Device: {model.model.device}")
        print(f"    Load time: {elapsed:.2f}s")

    return model


def test_encode_single(model: HuggingFaceEmbedding, verbose: bool = False) -> None:
    """
    Tests encode_one() — the single-text convenience method from base.py.
    """
    header("Single-text encoding  [base.py → encode_one()]")

    text = "Machine learning models learn patterns from data."

    vec = model.encode_one(text)

    run_test("encode_one() returns np.ndarray", isinstance(vec, np.ndarray))
    run_test("encode_one() returns 1D array", vec.ndim == 1)
    run_test("encode_one() shape matches model.dimension", vec.shape[0] == model.dimension)
    run_test("encode_one() dtype is float32", vec.dtype == np.float32)
    run_test("encode_one() vector has no NaN values", not np.isnan(vec).any())
    run_test("encode_one() vector has no Inf values", not np.isinf(vec).any())

    if model.normalize:
        norm = float(np.linalg.norm(vec))
        run_test(f"Normalised vector has unit norm (got {norm:.4f})", abs(norm - 1.0) < 1e-4)

    if verbose:
        print(f"    Vector shape : {vec.shape}")
        print(f"    Vector dtype : {vec.dtype}")
        print(f"    Vector norm  : {np.linalg.norm(vec):.6f}")
        print(f"    First 5 dims : {vec[:5]}")


def test_encode_batch(model: HuggingFaceEmbedding, verbose: bool = False) -> None:
    """
    Tests encode() on a batch of texts — the primary method in huggingFace.py.
    """
    header("Batch encoding  [huggingFace.py → encode()]")

    texts = [
        "Supervised learning uses labeled data.",
        "Unsupervised learning finds hidden structure.",
        "Reinforcement learning maximises cumulative reward.",
        "Neural networks are inspired by the brain.",
        "Gradient descent minimises the loss function.",
    ]

    start  = time.time()
    matrix = model.encode(texts)
    elapsed = time.time() - start

    run_test("encode() returns np.ndarray", isinstance(matrix, np.ndarray))
    run_test("encode() returns 2D array", matrix.ndim == 2)
    run_test("encode() row count matches input length", matrix.shape[0] == len(texts))
    run_test("encode() column count matches model.dimension", matrix.shape[1] == model.dimension)
    run_test("encode() dtype is float32", matrix.dtype == np.float32)
    run_test("encode() matrix has no NaN values", not np.isnan(matrix).any())
    run_test("encode() matrix has no Inf values", not np.isinf(matrix).any())

    if verbose:
        print(f"    Batch size : {len(texts)}")
        print(f"    Matrix shape: {matrix.shape}")
        print(f"    Encode time : {elapsed:.3f}s  ({elapsed/len(texts)*1000:.1f} ms/text)")


def test_encode_batch_progress(model: HuggingFaceEmbedding, verbose: bool = False) -> None:
    """
    Tests encode_batch() (the progress-bar variant from huggingFace.py / base.py).
    """
    header("encode_batch() with progress flag  [huggingFace.py]")

    texts = [
        "Deep Q-Networks extend Q-learning with neural networks.",
        "Policy gradient methods optimise the policy directly.",
        "PCA reduces dimensionality while preserving variance.",
    ]

    matrix = model.encode_batch(texts, show_progress=False)

    run_test("encode_batch() returns np.ndarray", isinstance(matrix, np.ndarray))
    run_test("encode_batch() returns 2D array", matrix.ndim == 2)
    run_test("encode_batch() shape[0] == len(texts)", matrix.shape[0] == len(texts))
    run_test("encode_batch() shape[1] == model.dimension", matrix.shape[1] == model.dimension)
    run_test("encode_batch() dtype is float32", matrix.dtype == np.float32)


def test_similarity(model: HuggingFaceEmbedding, verbose: bool = False) -> None:
    """
    Tests model.similarity() — the cosine similarity helper in huggingFace.py.
    Verifies that semantically close sentences score higher than dissimilar ones.
    """
    header("Cosine similarity  [huggingFace.py → similarity()]")

    v1 = model.encode_one(SIMILAR_PAIR[0])
    v2 = model.encode_one(SIMILAR_PAIR[1])
    v3 = model.encode_one(DISSIMILAR_PAIR[1])

    sim_close = model.similarity(v1, v2)
    sim_far   = model.similarity(v1, v3)

    run_test("similarity() returns a float", isinstance(sim_close, float))
    run_test("similarity score is in [-1, 1]", -1.0 <= sim_close <= 1.0)
    run_test("similar pair scores higher than dissimilar pair", sim_close > sim_far)

    # Self-similarity should be ~1.0 when normalised
    sim_self = model.similarity(v1, v1)
    run_test("Self-similarity ≈ 1.0 (normalised)", abs(sim_self - 1.0) < 1e-4)

    if verbose:
        print(f"    Similar pair   : {sim_close:.4f}  ({SIMILAR_PAIR[0][:40]}…)")
        print(f"    Dissimilar pair: {sim_far:.4f}  ({DISSIMILAR_PAIR[1][:40]}…)")
        print(f"    Self-similarity: {sim_self:.4f}")


def test_edge_cases(model: HuggingFaceEmbedding, verbose: bool = False) -> None:
    """
    Edge-case inputs: empty list, empty string, None-like strings, very long text,
    mixed valid/invalid batch.
    """
    header("Edge cases  [huggingFace.py + base.py]")

    # Empty list → shape (0, dim)
    result_empty_list = model.encode([])
    run_test("encode([]) returns ndarray", isinstance(result_empty_list, np.ndarray))
    run_test("encode([]) shape is (0, dim)", result_empty_list.shape == (0, model.dimension))

    # encode_batch([]) → shape (0, dim)
    result_empty_batch = model.encode_batch([])
    run_test("encode_batch([]) returns ndarray", isinstance(result_empty_batch, np.ndarray))
    run_test("encode_batch([]) shape is (0, dim)", result_empty_batch.shape == (0, model.dimension))

    # Single whitespace-only string → zero vector (with warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_ws = model.encode(["   "])
    run_test("Whitespace-only string returns shape (1, dim)", result_ws.shape == (1, model.dimension))
    run_test("Whitespace-only string returns float32", result_ws.dtype == np.float32)

    # Mixed batch with one invalid entry → full output shape preserved
    mixed = ["Valid sentence here.", "   ", "Another valid sentence."]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_mixed = model.encode(mixed)
    run_test("Mixed valid/invalid batch: shape[0] == len(input)", result_mixed.shape[0] == len(mixed))
    run_test("Mixed valid/invalid batch: dtype is float32", result_mixed.dtype == np.float32)

    # Very long text — should not crash
    long_text = " ".join(["machine learning"] * 500)
    try:
        result_long = model.encode_one(long_text)
        run_test("Very long text: encodes without exception", True)
        run_test("Very long text: returns 1D float32 array", result_long.ndim == 1 and result_long.dtype == np.float32)
    except Exception as exc:
        run_test(f"Very long text: encodes without exception (got {exc})", False)
        run_test("Very long text: returns 1D float32 array", False)

    if verbose:
        print(f"    Empty-list result shape : {result_empty_list.shape}")
        print(f"    Mixed-batch result shape: {result_mixed.shape}")
        print(f"    Long-text vector norm   : {np.linalg.norm(result_long):.4f}")


def test_chunk_embedding(
    model: HuggingFaceEmbedding,
    text: str,
    verbose: bool = False,
) -> None:
    """
    End-to-end test: chunk the text, embed every chunk, validate the
    resulting embedding matrix.  Skipped when chunking module is absent.
    """
    header("Chunk-level embedding (end-to-end)  [chunking → embeddings]")

    if not _CHUNKING_AVAILABLE:
        print("  SKIP  Chunking module not available.")
        return

    # Use paragraph chunker — deterministic and fast
    chunks = chunk_text(text, chunker_type="paragraph")

    if not chunks:
        print("  SKIP  No chunks produced from the provided text.")
        return

    chunk_texts = [c.text for c in chunks]

    start  = time.time()
    matrix = model.encode_batch(chunk_texts, show_progress=False)
    elapsed = time.time() - start

    run_test("Chunk texts list is non-empty", len(chunk_texts) > 0)
    run_test("embed(chunks) returns 2D ndarray", matrix.ndim == 2)
    run_test("embed(chunks) row count == number of chunks", matrix.shape[0] == len(chunks))
    run_test("embed(chunks) column count == model.dimension", matrix.shape[1] == model.dimension)
    run_test("embed(chunks) dtype is float32", matrix.dtype == np.float32)
    run_test("embed(chunks) has no NaN values", not np.isnan(matrix).any())
    run_test("embed(chunks) has no Inf values", not np.isinf(matrix).any())

    # Each row should be a distinct vector (no duplicate all-zero rows)
    zero_rows = int((matrix == 0).all(axis=1).sum())
    run_test("No all-zero embedding rows", zero_rows == 0)

    if verbose:
        print(f"    Number of chunks : {len(chunks)}")
        print(f"    Matrix shape     : {matrix.shape}")
        print(f"    Encode time      : {elapsed:.3f}s")
        print(f"    ms per chunk     : {elapsed / len(chunks) * 1000:.1f}")
        # Show similarity between first two chunks
        if len(chunks) >= 2:
            s = model.similarity(matrix[0], matrix[1])
            print(f"    Chunk[0] vs Chunk[1] similarity: {s:.4f}")


def test_determinism(model: HuggingFaceEmbedding, verbose: bool = False) -> None:
    """
    Encodes the same text twice and checks that the outputs are identical
    (deterministic inference).
    """
    header("Determinism  [huggingFace.py]")

    text = "Reinforcement learning agents maximise cumulative reward."

    v1 = model.encode_one(text)
    v2 = model.encode_one(text)

    run_test("Two encode_one() calls on same text return equal arrays",
             np.allclose(v1, v2, atol=1e-6))

    batch1 = model.encode([text, SIMILAR_PAIR[0]])
    batch2 = model.encode([text, SIMILAR_PAIR[0]])
    run_test("Two encode() calls on same batch return equal arrays",
             np.allclose(batch1, batch2, atol=1e-6))

    if verbose:
        diff = float(np.max(np.abs(v1 - v2)))
        print(f"    Max abs difference between two runs: {diff:.2e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def test_faiss_store(
    model: "HuggingFaceEmbedding",
    text: str,
    verbose: bool = False,
) -> None:
    """
    End-to-end test: chunk text → embed → store in FaissVectorStore → search.
    Skipped when vectordb or chunking modules are absent.
    """
    header("FaissVectorStore  [vectordb/faiss_store.py]")

    if not _VECTORDB_AVAILABLE:
        print("  SKIP  VectorDB module not available.")
        return
    if not _CHUNKING_AVAILABLE:
        print("  SKIP  Chunking module not available (needed to produce chunks).")
        return

    # --- build chunks and attach embeddings into metadata ---
    chunks = chunk_text(text, chunker_type="paragraph")
    if not chunks:
        print("  SKIP  No chunks produced from the provided text.")
        return

    chunk_texts_list = [c.text for c in chunks]
    embeddings = model.encode_batch(chunk_texts_list, show_progress=False)
    for c, emb in zip(chunks, embeddings):
        c.metadata["embedding"] = emb

    # --- instantiate store and add chunks ---
    store = FaissVectorStore(dim=model.dimension)

    run_test("FaissVectorStore instantiates without error", store is not None)
    run_test("Initial store length is 0", len(store) == 0)

    store.add_chunks(chunks)
    run_test("add_chunks() stores all chunks", len(store) == len(chunks))

    # --- similarity search ---
    query_text = "supervised learning labeled data loss function"
    query_vec  = model.encode_one(query_text)

    results = store.similarity_search_by_vector(query_vec, k=3)

    run_test("similarity_search_by_vector() returns a list", isinstance(results, list))
    run_test("Search returns <= k results", len(results) <= 3)
    run_test("Search returns at least 1 result", len(results) >= 1)
    run_test(
        "Each result is a RetrievalResult",
        all(isinstance(r, RetrievalResult) for r in results),
    )
    run_test(
        "Each result has a float score",
        all(isinstance(r.score, float) for r in results),
    )
    run_test(
        "Each result has a Chunk with text",
        all(isinstance(r.chunk.text, str) and r.chunk.text for r in results),
    )
    # scores should be descending
    scores = [r.score for r in results]
    run_test(
        "Results are sorted by descending score",
        all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
    )
    # top result should be more relevant than a random dissimilar query
    dissimilar_vec     = model.encode_one("The Eiffel Tower is located in Paris.")
    dissimilar_results = store.similarity_search_by_vector(dissimilar_vec, k=1)
    relevant_results   = store.similarity_search_by_vector(query_vec, k=1)
    if relevant_results and dissimilar_results:
        run_test(
            "Relevant query scores higher than dissimilar query",
            relevant_results[0].score >= dissimilar_results[0].score,
        )

    # k > store size should not crash
    big_k_results = store.similarity_search_by_vector(query_vec, k=9999)
    run_test("k > store size returns all chunks without error", len(big_k_results) == len(chunks))

    # empty store search
    empty_store = FaissVectorStore(dim=model.dimension)
    run_test(
        "Search on empty store returns []",
        empty_store.similarity_search_by_vector(query_vec, k=5) == [],
    )

    if verbose:
        print(f"    Chunks in store : {len(store)}")
        print(f"    Query           : '{query_text}'")
        for i, r in enumerate(results):
            preview = r.chunk.text[:60].replace("\n", " ")
            print(f"    Result {i+1}: score={r.score:.4f}  '{preview}…'")


def test_in_memory_store(
    model: "HuggingFaceEmbedding",
    text: str,
    verbose: bool = False,
) -> None:
    """
    Same pipeline as test_faiss_store but using InMemoryVectorStore (numpy).
    Skipped when vectordb or chunking modules are absent.
    """
    header("InMemoryVectorStore  [vectordb/base.py]")

    if not _VECTORDB_AVAILABLE:
        print("  SKIP  VectorDB module not available.")
        return
    if not _CHUNKING_AVAILABLE:
        print("  SKIP  Chunking module not available.")
        return

    chunks = chunk_text(text, chunker_type="paragraph")
    if not chunks:
        print("  SKIP  No chunks produced from the provided text.")
        return

    embeddings = model.encode_batch([c.text for c in chunks], show_progress=False)
    for c, emb in zip(chunks, embeddings):
        c.metadata["embedding"] = emb

    store = InMemoryVectorStore()
    run_test("InMemoryVectorStore instantiates without error", store is not None)
    run_test("Initial store length is 0", len(store) == 0)

    store.add_chunks(chunks)
    run_test("add_chunks() stores all chunks", len(store) == len(chunks))

    query_vec = model.encode_one("unsupervised clustering dimensionality reduction")
    results   = store.similarity_search_by_vector(query_vec, k=3)

    run_test("similarity_search_by_vector() returns a list", isinstance(results, list))
    run_test("Search returns <= k results", len(results) <= 3)
    run_test("Search returns at least 1 result", len(results) >= 1)
    run_test(
        "Each result is a RetrievalResult",
        all(isinstance(r, RetrievalResult) for r in results),
    )
    scores = [r.score for r in results]
    run_test(
        "Results are sorted by descending score",
        all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
    )

    # empty store
    empty_store = InMemoryVectorStore()
    run_test(
        "Search on empty store returns []",
        empty_store.similarity_search_by_vector(query_vec, k=5) == [],
    )

    if verbose:
        print(f"    Chunks in store : {len(store)}")
        for i, r in enumerate(results):
            preview = r.chunk.text[:60].replace("\n", " ")
            print(f"    Result {i+1}: score={r.score:.4f}  '{preview}…'")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphRAG Embedding Test Suite",
        epilog=(
            "Examples:\n"
            "  python embedding_test.py                         # interactive input\n"
            "  python embedding_test.py --file lecture.pdf      # single file\n"
            "  python embedding_test.py --mock                  # use mock data\n"
            "  python embedding_test.py -v                      # verbose output\n"
            "  python embedding_test.py --model BAAI/bge-small-en-v1.5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show vector previews and detailed statistics")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a file to extract and embed")
    parser.add_argument("--mock", action="store_true",
                        help="Use built-in mock text instead of extracting from a file")
    parser.add_argument("--model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace model name (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Determine source text
    # ------------------------------------------------------------------
    use_mock  = True
    test_text = MOCK_TEXT
    source    = "mock text"

    if args.mock:
        print("Using mock text for testing (--mock flag set).")

    elif args.file:
        text, src, ok = load_extracted_text(args.file)
        if ok:
            test_text = text
            source    = src
            use_mock  = False
        else:
            print("\nWarning: Extraction failed. Falling back to mock text.")

    else:
        only_verbose = all(a in ["-v", "--verbose"] for a in sys.argv[1:])
        if not sys.argv[1:] or only_verbose:
            path = get_file_path_from_user()
            if path:
                text, src, ok = load_extracted_text(path)
                if ok:
                    test_text = text
                    source    = src
                    use_mock  = False
                else:
                    print("\nExtraction failed. Falling back to mock text.")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  GraphRAG — Embedding Test Suite")
    print(f"  Model : {args.model}")
    print(f"  Source: {source}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Initialise model (shared across all tests)
    # ------------------------------------------------------------------
    model = test_model_initialization(args.model, verbose=args.verbose)

    # ------------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------------
    TESTS = [
        ("Single-text encoding",   lambda t, v: test_encode_single(model, v)),
        ("Batch encoding",         lambda t, v: test_encode_batch(model, v)),
        ("encode_batch() variant", lambda t, v: test_encode_batch_progress(model, v)),
        ("Cosine similarity",      lambda t, v: test_similarity(model, v)),
        ("Determinism",            lambda t, v: test_determinism(model, v)),
        ("Edge cases",             lambda t, v: test_edge_cases(model, v)),
        ("Chunk-level embedding",  lambda t, v: test_chunk_embedding(model, t, v)),
        ("FaissVectorStore",       lambda t, v: test_faiss_store(model, t, v)),
        ("InMemoryVectorStore",    lambda t, v: test_in_memory_store(model, t, v)),
    ]

    for name, test_fn in TESTS:
        try:
            test_fn(test_text, args.verbose)
        except Exception as exc:
            header(name)
            print(f"  {FAIL}  Unexpected error: {str(exc)[:120]}")
            _total  # keep counter consistent (run_test was not called)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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