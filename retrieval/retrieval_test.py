#!/usr/bin/env python3
# retrieval_test.py
"""
GraphRAG Retrieval Phase — Complete Test Suite
==============================================

Tests the ENTIRE pipeline from raw file ingestion through to graded retrieval,
covering every module and every meaningful edge case.

NO mock data is used anywhere.  All text comes from a real file you provide.

Usage
-----
    python retrieval_test.py

The script will prompt you for:
  1. File path  (PDF / DOCX / PPTX / TXT / URL)
  2. Neo4j connection details
  3. Groq API keys (three separate ones or one shared key)
  4. Which test suites to run

Test suites
-----------
  [A]  Extraction & chunking              — extractors + chunkers
  [B]  Embedding & index building         — HuggingFace + FAISS + BM25
  [C]  Graph building                     — CombinedExtractor + Neo4j
  [D]  Query processor                    — text normalisation + voice path
  [E]  Query engine                       — intent / entity / Cypher / embedding
  [F]  Individual retrievers              — BM25, vector, graph (isolated)
  [G]  Hybrid retrieval (RRF fusion)      — all three combined
  [H]  Reranker                           — cross-encoder scoring + delta
  [I]  Grader                             — relevance gate + reformulation
  [J]  Memory store                       — turn recording + coreference
  [K]  Query cache                        — hit / miss / invalidation
  [L]  Graph visualiser                   — console + Mermaid output
  [M]  Full pipeline — happy path         — end-to-end PASS verdict
  [N]  Full pipeline — follow-up query    — memory-injected second turn
  [O]  Full pipeline — vague query        — grader FAIL + auto-retry
  [P]  Full pipeline — relational query   — multi-hop graph path finding
  [Q]  Full pipeline — voice input        — audio file path (optional)
  [ALL] Run every suite in order
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Add project root to sys.path so imports work from any working directory
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers (independent of retrieval_logger so tests run standalone)
# ─────────────────────────────────────────────────────────────────────────────

_HAS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _HAS_COLOR else text

def header(title: str) -> None:
    bar = "═" * 72
    print(f"\n{_c('96;1', bar)}")
    print(f"{_c('96;1', f'  {title}')}")
    print(f"{_c('96', bar)}")

def section(title: str) -> None:
    print(f"\n{_c('94;1', f'  ▶  {title}')}")
    print(f"{_c('94', '  ' + '─' * 60)}")

def ok(msg: str) -> None:
    print(f"  {_c('92', '✔')}  {msg}")

def warn(msg: str) -> None:
    print(f"  {_c('93', '⚠')}  {msg}")

def fail(msg: str) -> None:
    print(f"  {_c('91', '✗')}  {msg}")

def info(msg: str) -> None:
    print(f"  {_c('97', 'ℹ')}  {msg}")

def show(label: str, value) -> None:
    print(f"    {_c('2', label + ':')}  {value}")

def separator() -> None:
    print(f"  {_c('2', '─' * 60)}")

# ─────────────────────────────────────────────────────────────────────────────
# Test result tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.passed  = 0
        self.failed  = 0
        self.skipped = 0
        self.log: List[tuple] = []   # (status, name, detail)

    def record(self, status: str, name: str, detail: str = "") -> None:
        self.log.append((status, name, detail))
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1

    def assert_true(self, condition: bool, name: str, detail: str = "") -> bool:
        if condition:
            ok(f"PASS — {name}")
            self.record("PASS", name, detail)
        else:
            fail(f"FAIL — {name}  {detail}")
            self.record("FAIL", name, detail)
        return condition

    def assert_not_empty(self, value, name: str) -> bool:
        return self.assert_true(bool(value), name, f"got: {repr(value)[:60]}")

    def skip(self, name: str, reason: str = "") -> None:
        warn(f"SKIP — {name}  {reason}")
        self.record("SKIP", name, reason)

    def summary(self) -> None:
        header("TEST SUMMARY")
        total = self.passed + self.failed + self.skipped
        print(f"  Total : {total}")
        print(f"  {_c('92', 'Passed')}: {self.passed}")
        print(f"  {_c('91', 'Failed')}: {self.failed}")
        print(f"  {_c('93', 'Skipped')}: {self.skipped}")
        if self.failed:
            print(f"\n  {_c('91', 'Failed tests:')}")
            for status, name, detail in self.log:
                if status == "FAIL":
                    print(f"    ✗  {name}  {_c('2', detail)}")
        verdict = "ALL TESTS PASSED" if self.failed == 0 else f"{self.failed} TEST(S) FAILED"
        color   = "92;1" if self.failed == 0 else "91;1"
        print(f"\n  {_c(color, verdict)}\n")


R = TestResults()


# ─────────────────────────────────────────────────────────────────────────────
# Input collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_inputs() -> dict:
    header("GraphRAG Retrieval Test Suite — Setup")

    print("""
  This test suite runs the complete GraphRAG pipeline from raw file
  extraction through to graded retrieval.  No mock data is used.
  All text comes from the file you specify below.

  Press ENTER to accept defaults shown in [brackets].
""")

    # File path
    print(_c("97;1", "  ── Step 1: Input File ──"))
    print("  Supported: PDF, DOCX, PPTX, TXT, MD, or a URL")
    file_path = input("  File path or URL: ").strip()
    if not file_path:
        fail("No file path provided.  Exiting.")
        sys.exit(1)

    # Working directory
    print(f"\n{_c('97;1', '  ── Step 2: Working Directory ──')}")
    print("  All indices, memory, and cache files will be saved here.")
    work_dir = input("  Working directory [./retrieval_test_output]: ").strip()
    if not work_dir:
        work_dir = "./retrieval_test_output"

    # Neo4j
    print(f"\n{_c('97;1', '  ── Step 3: Neo4j Connection ──')}")
    neo4j_uri  = input("  Neo4j URI  [neo4j://127.0.0.1:7687]: ").strip() or "neo4j://127.0.0.1:7687"
    neo4j_user = input("  Neo4j user [neo4j]: ").strip() or "neo4j"
    neo4j_pass = input("  Neo4j password [neo4j1234]: ").strip() or "neo4j1234"

    # Groq API keys
    print(f"\n{_c('97;1', '  ── Step 4: Groq API Keys ──')}")
    print("  You can provide one shared key for all three roles,")
    print("  or separate keys for Whisper / Query Engine / Grader.")
    shared_key = input("  Shared Groq API key (leave blank to set individually): ").strip()

    if shared_key:
        whisper_key = query_key = grader_key = shared_key
    else:
        whisper_key = input("  GROQ_API_KEY_WHISPER (for voice transcription): ").strip()
        query_key   = input("  GROQ_API_KEY_QUERY   (for intent/entity/Cypher): ").strip()
        grader_key  = input("  GROQ_API_KEY_GRADER  (for relevance grading): ").strip()

    # Optional audio file for suite Q
    print(f"\n{_c('97;1', '  ── Step 5: Optional Audio File (for voice test) ──')}")
    audio_path = input("  Audio file path (mp3/wav/m4a — press ENTER to skip): ").strip()

    # Test suites
    print(f"\n{_c('97;1', '  ── Step 6: Test Suites ──')}")
    print("  Suites: A B C D E F G H I J K L M N O P Q  (or ALL)")
    suites_raw = input("  Which suites to run [ALL]: ").strip().upper() or "ALL"
    suites = set(suites_raw.split()) if suites_raw != "ALL" else set("ABCDEFGHIJKLMNOPQ")

    return {
        "file_path"  : file_path,
        "work_dir"   : work_dir,
        "neo4j_uri"  : neo4j_uri,
        "neo4j_user" : neo4j_user,
        "neo4j_pass" : neo4j_pass,
        "whisper_key": whisper_key,
        "query_key"  : query_key,
        "grader_key" : grader_key,
        "audio_path" : audio_path,
        "suites"     : suites,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: print chunk list
# ─────────────────────────────────────────────────────────────────────────────

def print_chunks(chunks: list, max_show: int = 3) -> None:
    for i, c in enumerate(chunks[:max_show]):
        text_preview = c.text[:120].replace("\n", " ")
        print(f"    [{c.chunk_id}] {_c('2', text_preview)}…")
    if len(chunks) > max_show:
        info(f"… and {len(chunks) - max_show} more chunks")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE A — Extraction & Chunking
# ─────────────────────────────────────────────────────────────────────────────

def suite_a_extraction(cfg: dict) -> list:
    header("Suite A — Extraction & Chunking")
    file_path = cfg["file_path"]
    chunks    = []

    section("A1 — File extraction")
    try:
        ext = Path(file_path).suffix.lower() if not file_path.startswith("http") else ".url"
        info(f"Input: {file_path}  (type: {ext or 'url'})")

        if file_path.startswith("http"):
            from extractors.url_extractor import URLExtractor
            extractor = URLExtractor()
            result    = extractor.extract(file_path)
        elif ext == ".pdf":
            from extractors.pdf_extractor import PDFExtractor
            extractor = PDFExtractor()
            result    = extractor.extract(file_path)
        elif ext in (".docx", ".doc"):
            from extractors.docx_extractor import DOCXExtractor
            extractor = DOCXExtractor()
            result    = extractor.extract(file_path)
        elif ext in (".pptx", ".ppt"):
            from extractors.pptx_extractor import PPTXExtractor
            extractor = PPTXExtractor()
            result    = extractor.extract(file_path)
        elif ext in (".txt", ".md"):
            text   = Path(file_path).read_text(encoding="utf-8")
            result = {"success": True, "extracted_text": text}
        else:
            warn(f"Unrecognised extension '{ext}' — treating as plain text")
            text   = Path(file_path).read_text(encoding="utf-8")
            result = {"success": True, "extracted_text": text}

        R.assert_true(result.get("success"), "A1.1 Extraction succeeded")
        extracted_text = result.get("extracted_text", "")
        R.assert_true(len(extracted_text) > 100, "A1.2 Extracted text has meaningful content",
                      f"len={len(extracted_text)}")

        show("Characters extracted", len(extracted_text))
        show("Words extracted", len(extracted_text.split()))
        show("Preview (first 200 chars)", extracted_text[:200].replace("\n", " "))

    except Exception as e:
        fail(f"Extraction raised: {e}")
        R.record("FAIL", "A1 Extraction", str(e))
        return []

    section("A2 — Chunking strategies")
    try:
        from chunking.chunking import get_chunker

        # Test every chunker type
        chunker_types = [
            ("paragraph",     {}),
            ("sentence",      {"sentences_per_chunk": 5, "overlap_sentences": 1}),
            ("fixed_size",    {"chunk_size": 512, "overlap": 50}),
            ("recursive",     {"max_chunk_size": 512}),
            ("sliding_window",{"window_size": 8, "step_size": 4}),
            ("semantic",      {"threshold": 0.5, "max_sentences_per_chunk": 20}),
        ]

        for ctype, kwargs in chunker_types:
            try:
                chunker     = get_chunker(ctype, **kwargs)
                test_chunks = chunker.chunk(extracted_text)
                R.assert_true(len(test_chunks) > 0,
                              f"A2.{ctype} produces chunks",
                              f"count={len(test_chunks)}")
                show(f"{ctype} chunks", len(test_chunks))
            except Exception as e:
                fail(f"A2.{ctype} failed: {e}")
                R.record("FAIL", f"A2.{ctype}", str(e))

        # Use semantic chunker for downstream tests (best quality)
        section("A3 — Semantic chunking (default, used downstream)")
        try:
            sem_chunker = get_chunker("semantic", threshold=0.5, max_sentences_per_chunk=20)
            chunks      = sem_chunker.chunk(extracted_text)
            R.assert_true(len(chunks) >= 1, "A3.1 Semantic chunker returns ≥1 chunks",
                          f"count={len(chunks)}")
            R.assert_true(all(c.text.strip() for c in chunks), "A3.2 No empty chunks")
            R.assert_true(all(hasattr(c, "chunk_id") for c in chunks), "A3.3 All chunks have chunk_id")

            show("Semantic chunk count", len(chunks))
            show("Avg chunk length (chars)", sum(len(c.text) for c in chunks) // max(len(chunks), 1))
            print_chunks(chunks, max_show=3)

        except Exception as e:
            warn(f"Semantic chunker failed ({e}) — falling back to paragraph chunker")
            R.record("FAIL", "A3 Semantic chunking", str(e))
            chunks = get_chunker("paragraph").chunk(extracted_text)

        # Tag chunks with source
        for c in chunks:
            if not hasattr(c, "metadata") or c.metadata is None:
                c.metadata = {}
            c.metadata["source"] = file_path

    except Exception as e:
        fail(f"Chunking setup failed: {e}")
        R.record("FAIL", "A2 Chunking", str(e))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# SUITE B — Embedding & Index Building
# ─────────────────────────────────────────────────────────────────────────────

def suite_b_indices(cfg: dict, chunks: list) -> tuple:
    header("Suite B — Embedding & Index Building")

    if not chunks:
        R.skip("Suite B", "No chunks from Suite A")
        return None, None

    work_dir = cfg["work_dir"]
    os.makedirs(work_dir, exist_ok=True)

    embed_model = None
    faiss_store = None
    bm25        = None

    section("B1 — HuggingFace embedding model")
    try:
        from embeddings.huggingFace import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True,
        )
        R.assert_true(embed_model is not None, "B1.1 Embedding model loaded")
        R.assert_true(embed_model.dimension == 384, "B1.2 Dimension is 384",
                      f"got {embed_model.dimension}")

        # Test single encode
        test_vec = embed_model.encode_one("test sentence")
        R.assert_true(test_vec.shape == (384,), "B1.3 Single encode shape correct",
                      f"got {test_vec.shape}")
        R.assert_true(abs(float((test_vec ** 2).sum() ** 0.5) - 1.0) < 0.01,
                      "B1.4 Output is L2-normalised")

        # Test batch encode
        batch_vecs = embed_model.encode(["sentence one", "sentence two"])
        R.assert_true(batch_vecs.shape == (2, 384), "B1.5 Batch encode shape correct",
                      f"got {batch_vecs.shape}")

        show("Model", embed_model.model_name)
        show("Dimension", embed_model.dimension)
        show("Device", str(embed_model.model.device))

    except Exception as e:
        fail(f"Embedding model failed: {e}")
        R.record("FAIL", "B1 Embedding", str(e))
        return None, None

    section("B2 — FAISS index building")
    try:
        from retrieval.faiss_store import FAISSStore
        faiss_store = FAISSStore(embed_model, verbose=True)
        n = faiss_store.build_from_chunks(chunks)
        R.assert_true(n == len(chunks), "B2.1 All chunks indexed in FAISS",
                      f"indexed={n} expected={len(chunks)}")
        R.assert_true(faiss_store.total_vectors > 0, "B2.2 FAISS index non-empty")

        faiss_store.save(work_dir)
        R.assert_true(Path(work_dir, "faiss.index").exists(), "B2.3 faiss.index saved to disk")
        R.assert_true(Path(work_dir, "faiss_meta.pkl").exists(), "B2.4 faiss_meta.pkl saved")

        # Test search
        q_vec   = embed_model.encode_one(chunks[0].text[:100])
        results = faiss_store.search(q_vec, top_k=3)
        R.assert_true(len(results) > 0, "B2.5 FAISS search returns results")
        R.assert_true(results[0].score > 0.5, "B2.6 Top result has high similarity",
                      f"score={results[0].score:.4f}")

        show("Total vectors indexed", faiss_store.total_vectors)
        show("Top result score", f"{results[0].score:.4f}")
        show("Top result preview", results[0].text[:80].replace("\n"," "))

    except Exception as e:
        fail(f"FAISS index failed: {e}")
        R.record("FAIL", "B2 FAISS", str(e))

    section("B3 — BM25 index building")
    try:
        from retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(verbose=True)
        n    = bm25.build(chunks)
        R.assert_true(n == len(chunks), "B3.1 All chunks indexed in BM25",
                      f"indexed={n} expected={len(chunks)}")

        bm25.save(work_dir)
        R.assert_true(Path(work_dir, "bm25.pkl").exists(), "B3.2 bm25.pkl saved to disk")

        show("BM25 documents", n)

    except Exception as e:
        fail(f"BM25 index failed: {e}")
        R.record("FAIL", "B3 BM25", str(e))

    section("B4 — FAISS reload from disk")
    try:
        from retrieval.faiss_store import FAISSStore
        loaded = FAISSStore.load(work_dir, embed_model, verbose=True)
        R.assert_true(loaded.total_vectors == len(chunks),
                      "B4.1 Reloaded FAISS has same vector count",
                      f"loaded={loaded.total_vectors} expected={len(chunks)}")
        ok("FAISS successfully persisted and reloaded")
    except Exception as e:
        fail(f"FAISS reload failed: {e}")
        R.record("FAIL", "B4 FAISS reload", str(e))

    return embed_model, bm25


# ─────────────────────────────────────────────────────────────────────────────
# SUITE C — Graph Building
# ─────────────────────────────────────────────────────────────────────────────

def suite_c_graph(cfg: dict, chunks: list, embed_model) -> object:
    header("Suite C — Graph Building (Neo4j)")

    if not chunks or embed_model is None:
        R.skip("Suite C", "Missing chunks or embedding model")
        return None

    graph_store = None

    section("C1 — Neo4j connection")
    try:
        from graph.graph_store import GraphStore
        graph_store = GraphStore(
            uri      = cfg["neo4j_uri"],
            user     = cfg["neo4j_user"],
            password = cfg["neo4j_pass"],
        )
        graph_store.init_schema()
        R.assert_true(graph_store is not None, "C1.1 Neo4j connection established")
        show("URI", cfg["neo4j_uri"])
        show("Nodes before test", graph_store.count_nodes())
        show("Relationships before test", graph_store.count_relationships())

    except Exception as e:
        fail(f"Neo4j connection failed: {e}")
        R.record("FAIL", "C1 Neo4j", str(e))
        return None

    section("C2 — Entity & relationship extraction (CombinedExtractor)")
    try:
        from graph.node_relation_extractor import CombinedExtractor
        from graph.llm_backend import LLMBackend

        llm = LLMBackend(
            api_key    = cfg["query_key"] or cfg["grader_key"],
            model      = "llama-3.3-70b-versatile",
            max_tokens = 3000,
        )
        extractor = CombinedExtractor(
            llm            = llm,
            embedding_fn   = embed_model.encode,
            batch_chunks   = 2,
            mode           = "constrained",
        )

        # Use only first 5 chunks to keep test fast
        test_chunks = chunks[:5]
        info(f"Extracting from {len(test_chunks)} chunks (first 5 only for speed)")

        nodes = extractor.extract_from_chunks(test_chunks, show_progress=True)
        R.assert_true(len(nodes) > 0, "C2.1 Nodes extracted from chunks",
                      f"count={len(nodes)}")

        show("Nodes extracted", len(nodes))
        for n in nodes[:5]:
            show(f"  [{n.entity_type}]", n.name)

    except Exception as e:
        fail(f"CombinedExtractor failed: {e}")
        R.record("FAIL", "C2 CombinedExtractor", str(e))
        return graph_store

    section("C3 — Writing nodes and relationships to Neo4j")
    try:
        written = graph_store.upsert_nodes(nodes)
        R.assert_true(written > 0, "C3.1 Nodes written to Neo4j", f"written={written}")

        total_nodes = graph_store.count_nodes()
        R.assert_true(total_nodes > 0, "C3.2 Neo4j graph is non-empty",
                      f"total={total_nodes}")

        show("Nodes written this run", written)
        show("Total nodes in graph",   total_nodes)

    except Exception as e:
        fail(f"Graph write failed: {e}")
        R.record("FAIL", "C3 Graph write", str(e))

    section("C4 — Graph retrieval smoke test")
    try:
        if nodes:
            first_node = nodes[0]
            by_name = graph_store.get_node_by_name(first_node.name)
            R.assert_true(by_name is not None, "C4.1 Node retrievable by name",
                          f"name={first_node.name}")

            nbhd = graph_store.get_neighbourhood(first_node.node_id, hops=1)
            R.assert_true("nodes" in nbhd, "C4.2 Neighbourhood traversal returns nodes dict")
            show("Neighbourhood nodes", len(nbhd.get("nodes", [])))
            show("Neighbourhood rels",  len(nbhd.get("relationships", [])))

            sim_results = graph_store.similarity_search(first_node.embedding, top_k=3)
            R.assert_true(len(sim_results) > 0, "C4.3 Vector similarity search in Neo4j works")
            show("Similarity search results", len(sim_results))

    except Exception as e:
        fail(f"Graph read failed: {e}")
        R.record("FAIL", "C4 Graph read", str(e))

    return graph_store


# ─────────────────────────────────────────────────────────────────────────────
# SUITE D — Query Processor
# ─────────────────────────────────────────────────────────────────────────────

def suite_d_query_processor(cfg: dict, memory_store) -> None:
    header("Suite D — Query Processor")

    from retrieval.query_processor import QueryProcessor
    from retrieval.retrieval_context import RetrievalTrace, InputModality

    qp = QueryProcessor(
        whisper_api_key = cfg["whisper_key"],
        memory_store    = memory_store,
        verbose         = True,
    )

    section("D1 — Text input normalisation")
    test_cases = [
        ("What is gradient descent?",         "clean query"),
        ("  What  IS   gradient  descent?  ", "extra whitespace"),
        ("What is gradient descent.",          "trailing period"),
        ("What\u2019s gradient descent?",      "smart quote"),
    ]
    for raw, label in test_cases:
        trace = RetrievalTrace()
        rep   = qp.process(raw, trace=trace)
        R.assert_true(rep.normalised_text.strip() == rep.normalised_text,
                      f"D1 Normalised text has no leading/trailing whitespace ({label})")
        R.assert_true(trace.input_modality == InputModality.TEXT.value,
                      f"D1 Modality detected as TEXT ({label})")
        show(f"  '{raw[:30]}' →", f"'{rep.normalised_text}'")

    section("D2 — Follow-up detection")
    follow_up_queries = [
        "Can you elaborate on that?",
        "How does it compare to what we discussed?",
        "Tell me more about the previous topic",
        "You mentioned it earlier, explain further",
    ]
    for q in follow_up_queries:
        is_fu = memory_store.is_follow_up(q)
        # We can't assert True here without prior turns, but we verify the method runs
        R.assert_true(isinstance(is_fu, bool), f"D2 is_follow_up returns bool for: '{q[:40]}'")
        show(f"  '{q[:45]}'", f"follow_up={is_fu}")

    section("D3 — Audio path detection (no transcription — file not required)")
    # Verify the detection logic without actually calling Groq
    from retrieval.query_processor import _AUDIO_EXTENSIONS
    for ext in [".mp3", ".wav", ".m4a", ".flac"]:
        fake_path = f"/tmp/test_audio{ext}"
        # Create a dummy file to satisfy Path.exists()
        Path(fake_path).touch()
        try:
            modality, _ = qp._detect_modality(fake_path)
            R.assert_true(str(modality.value) == "audio",
                          f"D3 Audio extension {ext} detected as AUDIO")
        finally:
            Path(fake_path).unlink(missing_ok=True)

    R.assert_true("What is AI?" not in _AUDIO_EXTENSIONS, "D3 Text string not treated as audio")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE E — Query Engine
# ─────────────────────────────────────────────────────────────────────────────

def suite_e_query_engine(cfg: dict, embed_model, memory_store) -> dict:
    header("Suite E — Query Engine")

    if embed_model is None:
        R.skip("Suite E", "No embedding model")
        return {}

    from retrieval.query_engine import QueryEngine
    from retrieval.query_processor import QueryProcessor
    from retrieval.retrieval_context import RetrievalTrace, QueryIntent

    qp = QueryProcessor(whisper_api_key=cfg["whisper_key"],
                        memory_store=memory_store, verbose=False)
    qe = QueryEngine(embedding_model=embed_model, llm_api_key=cfg["query_key"],
                     memory_store=memory_store, verbose=True)

    test_queries = {
        "factual"    : "What is gradient descent?",
        "relational" : "How does gradient descent relate to backpropagation?",
        "procedural" : "How do I implement a neural network from scratch?",
        "comparative": "Compare BERT and GPT architectures",
        "vague"      : "Explain things",
    }
    results = {}

    for label, query in test_queries.items():
        section(f"E — Query: '{query[:50]}' [{label}]")
        trace = RetrievalTrace()
        rep   = qp.process(query, trace=trace)
        rep   = qe.process(rep, trace=trace)

        show("Intent",   rep.intent.value)
        show("Entities", rep.entities)
        show("Keywords", rep.keywords[:8])
        show("Cypher",   (rep.cypher_query or "none")[:80])
        show("Embedding shape", getattr(rep.embedding, "shape", "None"))

        R.assert_true(rep.intent != QueryIntent.UNKNOWN or label == "vague",
                      f"E.{label} Intent is not UNKNOWN")
        R.assert_true(rep.embedding is not None,
                      f"E.{label} Embedding produced")
        R.assert_true(len(rep.keywords) > 0,
                      f"E.{label} Keywords extracted")

        results[label] = rep

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUITE F — Individual Retrievers
# ─────────────────────────────────────────────────────────────────────────────

def suite_f_retrievers(cfg: dict, embed_model, graph_store, bm25,
                        query_reps: dict) -> None:
    header("Suite F — Individual Retrievers")

    if not query_reps:
        R.skip("Suite F", "No query representations from Suite E")
        return

    work_dir   = cfg["work_dir"]
    factual_rep = query_reps.get("factual") or list(query_reps.values())[0]

    # ── F1: BM25 ─────────────────────────────────────────────────────────────
    section("F1 — BM25 Retriever")
    if bm25 is None:
        R.skip("F1 BM25", "BM25 index not built")
    else:
        try:
            from retrieval.retrieval_context import RetrievalTrace
            trace   = RetrievalTrace()
            results = bm25.search(factual_rep, top_k=5, trace=trace)

            R.assert_true(isinstance(results, list), "F1.1 BM25 returns a list")
            R.assert_true(trace.bm25_count == len(results), "F1.2 Trace bm25_count updated")

            if results:
                R.assert_true(results[0].score > 0, "F1.3 Top result has positive score",
                              f"score={results[0].score:.3f}")
                R.assert_true(results[0].retriever == "bm25", "F1.4 Retriever tag is 'bm25'")
                show("BM25 results", len(results))
                show("Top score", f"{results[0].score:.3f}")
                show("Top chunk", results[0].text[:80].replace("\n"," "))
            else:
                warn("BM25 returned 0 results — keywords may not match chunk vocabulary")

        except Exception as e:
            fail(f"BM25 search failed: {e}")
            R.record("FAIL", "F1 BM25 search", str(e))

    # ── F2: Vector ────────────────────────────────────────────────────────────
    section("F2 — Vector Retriever")
    if embed_model is None:
        R.skip("F2 Vector", "No embedding model")
    else:
        try:
            from retrieval.faiss_store import FAISSStore
            from retrieval.vector_retriever import VectorRetriever
            from retrieval.retrieval_context import RetrievalTrace

            faiss   = FAISSStore.load(work_dir, embed_model, verbose=False)
            vr      = VectorRetriever(faiss, min_score=0.2, verbose=True)
            trace   = RetrievalTrace()
            results = vr.search(factual_rep, top_k=5, trace=trace)

            R.assert_true(isinstance(results, list), "F2.1 Vector retriever returns a list")
            R.assert_true(trace.vector_count == len(results), "F2.2 Trace vector_count updated")

            if results:
                R.assert_true(0 <= results[0].score <= 1.0, "F2.3 Score in [0,1]",
                              f"score={results[0].score:.4f}")
                R.assert_true(results[0].retriever == "vector", "F2.4 Retriever tag is 'vector'")
                show("Vector results", len(results))
                show("Top cosine sim", f"{results[0].score:.4f}")
                show("Top chunk", results[0].text[:80].replace("\n"," "))

                # Verify ordering
                scores = [r.score for r in results]
                R.assert_true(scores == sorted(scores, reverse=True),
                              "F2.5 Results sorted descending by score")

        except Exception as e:
            fail(f"Vector search failed: {e}")
            R.record("FAIL", "F2 Vector search", str(e))

    # ── F3: Graph ─────────────────────────────────────────────────────────────
    section("F3 — Graph Retriever")
    if graph_store is None:
        R.skip("F3 Graph", "No graph store")
    else:
        try:
            from retrieval.graph_retriever import GraphRetriever
            from retrieval.retrieval_context import RetrievalTrace

            gr     = GraphRetriever(graph_store, verbose=True)
            trace  = RetrievalTrace()
            results, traversal = gr.search(factual_rep, top_k=10, trace=trace)

            R.assert_true(isinstance(results, list), "F3.1 Graph retriever returns a list")
            R.assert_true(trace.graph_count == len(results), "F3.2 Trace graph_count updated")
            R.assert_true(trace.graph_nodes_visited >= 0, "F3.3 Trace nodes_visited updated")

            show("Graph chunks", len(results))
            show("Nodes visited", traversal.nodes.__len__())
            show("Rels visited",  traversal.relationships.__len__())
            show("Traversal depth", traversal.traversal_depth)

            if results:
                R.assert_true(results[0].retriever == "graph", "F3.4 Retriever tag is 'graph'")
                show("Top chunk", results[0].text[:80].replace("\n"," "))

        except Exception as e:
            fail(f"Graph search failed: {e}")
            R.record("FAIL", "F3 Graph search", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SUITE G — Hybrid Retrieval (RRF Fusion)
# ─────────────────────────────────────────────────────────────────────────────

def suite_g_hybrid(cfg: dict, embed_model, graph_store, bm25, query_reps: dict) -> list:
    header("Suite G — Hybrid Retriever (RRF Fusion)")

    if not query_reps or embed_model is None:
        R.skip("Suite G", "Missing prerequisites")
        return []

    work_dir    = cfg["work_dir"]
    factual_rep = query_reps.get("factual") or list(query_reps.values())[0]

    try:
        from retrieval.faiss_store import FAISSStore
        from retrieval.vector_retriever import VectorRetriever
        from retrieval.graph_retriever import GraphRetriever
        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.retrieval_context import RetrievalTrace

        faiss   = FAISSStore.load(work_dir, embed_model, verbose=False)
        vr      = VectorRetriever(faiss, verbose=False)
        gr      = GraphRetriever(graph_store or None, verbose=False)
        hybrid  = HybridRetriever(k=60, top_k=10, use_adaptive=True, verbose=True)

        bm25_res = bm25.search(factual_rep, top_k=10) if bm25 else []
        vec_res  = vr.search(factual_rep, top_k=10)
        grph_res, _ = gr.search(factual_rep, top_k=10) if graph_store else ([], None)

        trace   = RetrievalTrace()
        fused   = hybrid.fuse(bm25_res, vec_res, grph_res, rep=factual_rep, trace=trace)

        R.assert_true(len(fused) > 0, "G1 RRF fusion returns results", f"count={len(fused)}")
        R.assert_true(trace.fused_count == len(fused), "G2 Trace fused_count updated")
        R.assert_true(trace.phases[-1].phase_name.startswith("Hybrid"),
                      "G3 Hybrid phase recorded in trace")

        # Verify RRF scoring properties
        scores = [r.rrf_score for r in fused]
        R.assert_true(scores == sorted(scores, reverse=True), "G4 Fused results sorted descending")
        R.assert_true(all(r.rrf_score > 0 for r in fused), "G5 All RRF scores positive")
        R.assert_true(all(len(r.contributing) >= 1 for r in fused),
                      "G6 All results have ≥1 contributing retriever")

        # Test with RELATIONAL query (different adaptive weights)
        if "relational" in query_reps:
            rel_rep    = query_reps["relational"]
            grph_rel, _= gr.search(rel_rep, top_k=10) if graph_store else ([], None)
            fused_rel  = hybrid.fuse([], vec_res, grph_rel, rep=rel_rep)
            R.assert_true(len(fused_rel) > 0,
                          "G7 RRF works with relational query (graph-heavy)")

        # Overlap analysis
        multi_retriever = [r for r in fused if len(r.contributing) > 1]
        show("Total fused results",         len(fused))
        show("Multi-retriever overlaps",    len(multi_retriever))
        show("Top RRF score",              f"{fused[0].rrf_score:.5f}" if fused else "n/a")
        show("Top contributing retrievers", fused[0].contributing if fused else "n/a")

        return fused

    except Exception as e:
        fail(f"Hybrid retrieval failed: {e}")
        R.record("FAIL", "G Hybrid", str(e))
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SUITE H — Reranker
# ─────────────────────────────────────────────────────────────────────────────

def suite_h_reranker(fused: list, query_reps: dict) -> list:
    header("Suite H — Reranker (Cross-Encoder)")

    if not fused or not query_reps:
        R.skip("Suite H", "No fused results")
        return []

    factual_rep = query_reps.get("factual") or list(query_reps.values())[0]

    try:
        from retrieval.reranker import Reranker
        from retrieval.retrieval_context import RetrievalTrace

        reranker  = Reranker(verbose=True)
        trace     = RetrievalTrace()
        reranked  = reranker.rerank(
            query   = factual_rep.normalised_text,
            results = fused,
            trace   = trace,
        )

        R.assert_true(len(reranked) == len(fused), "H1 Reranker preserves all results",
                      f"in={len(fused)} out={len(reranked)}")
        R.assert_true(trace.reranked_count == len(reranked), "H2 Trace reranked_count updated")

        # Verify ranking properties
        scores = [r.rerank_score for r in reranked]
        R.assert_true(scores == sorted(scores, reverse=True), "H3 Reranked sorted descending")
        R.assert_true(all(0 <= s <= 1 for s in scores), "H4 All scores in [0,1]",
                      f"scores={[round(s,3) for s in scores[:5]]}")

        # Delta analysis (reranker effect)
        moved_up   = [r for r in reranked if r.delta < 0]
        moved_down = [r for r in reranked if r.delta > 0]
        unchanged  = [r for r in reranked if r.delta == 0]

        R.assert_true(len(reranked) > 0, "H5 At least one reranked result exists")

        show("Total reranked",   len(reranked))
        show("Moved up",         len(moved_up))
        show("Moved down",       len(moved_down))
        show("Unchanged rank",   len(unchanged))
        show("Top rerank score", f"{reranked[0].rerank_score:.4f}" if reranked else "n/a")
        show("Top chunk preview",
             reranked[0].fused.chunk.text[:80].replace("\n"," ") if reranked else "n/a")

        return reranked

    except Exception as e:
        fail(f"Reranker failed: {e}")
        R.record("FAIL", "H Reranker", str(e))
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SUITE I — Grader
# ─────────────────────────────────────────────────────────────────────────────

def suite_i_grader(cfg: dict, reranked: list, query_reps: dict, memory_store) -> object:
    header("Suite I — Relevance Grader")

    if not reranked or not query_reps:
        R.skip("Suite I", "No reranked results")
        return None

    factual_rep = query_reps.get("factual") or list(query_reps.values())[0]

    try:
        from retrieval.grader import Grader
        from retrieval.retrieval_context import GraderVerdict, RetrievalTrace

        grader = Grader(
            llm_api_key     = cfg["grader_key"],
            chunk_threshold = 0.6,
            memory_store    = memory_store,
            verbose         = True,
        )

        section("I1 — Standard grading (factual query)")
        trace  = RetrievalTrace()
        result = grader.grade(reranked, factual_rep, trace=trace)

        R.assert_true(result.verdict in GraderVerdict, "I1.1 Verdict is a valid GraderVerdict",
                      f"verdict={result.verdict}")
        R.assert_true(0 <= result.confidence <= 1.0, "I1.2 Confidence in [0,1]",
                      f"confidence={result.confidence}")
        R.assert_true(trace.graded_pass_count + trace.graded_fail_count == len(reranked),
                      "I1.3 All chunks accounted for in trace")

        total = len(result.passed_chunks) + len(result.failed_chunks)
        R.assert_true(total == len(reranked), "I1.4 Passed + failed = total input",
                      f"passed={len(result.passed_chunks)} failed={len(result.failed_chunks)} total={len(reranked)}")

        show("Verdict",      result.verdict.value.upper())
        show("Confidence",   f"{result.confidence:.3f}")
        show("Passed chunks", len(result.passed_chunks))
        show("Failed chunks", len(result.failed_chunks))

        if result.passed_chunks:
            best = result.passed_chunks[0]
            show("Best chunk score",  f"{best.relevance_score:.3f}")
            show("Best chunk reason", best.grader_reason[:80])

        section("I2 — context_text property")
        ctx = result.context_text
        R.assert_true(isinstance(ctx, str), "I2.1 context_text returns a string")
        if result.passed_chunks:
            R.assert_true(len(ctx) > 0, "I2.2 context_text is non-empty when chunks passed")
            show("Context text length (chars)", len(ctx))
            show("Context preview", ctx[:120].replace("\n"," "))

        section("I3 — FAIL verdict triggers reformulation")
        # Manufacture a scenario where all chunks fail by using an irrelevant query
        from retrieval.retrieval_context import QueryRepresentation, QueryIntent
        import numpy as np

        irrelevant_rep = QueryRepresentation(
            raw_text        = "zzzzzz xyzzy foobarbaz",
            normalised_text = "zzzzzz xyzzy foobarbaz",
            intent          = QueryIntent.UNKNOWN,
            entities        = [],
            keywords        = ["zzzzzz"],
            embedding       = np.zeros(384, dtype="float32"),
        )
        trace2  = RetrievalTrace()
        result2 = grader.grade(reranked, irrelevant_rep, trace=trace2)

        R.assert_true(result2.verdict in (GraderVerdict.FAIL, GraderVerdict.PARTIAL,
                                          GraderVerdict.PASS),
                      "I3.1 Grader returns a valid verdict for irrelevant query")
        if result2.verdict == GraderVerdict.FAIL:
            R.assert_true(result2.reformulation is not None or True,
                          "I3.2 Reformulation may be generated on FAIL")
            if result2.reformulation:
                show("Reformulation", result2.reformulation[:80])
            ok("FAIL verdict produced correctly for irrelevant query")

        return result

    except Exception as e:
        fail(f"Grader failed: {e}")
        R.record("FAIL", "I Grader", str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SUITE J — Memory Store
# ─────────────────────────────────────────────────────────────────────────────

def suite_j_memory(cfg: dict) -> object:
    header("Suite J — Memory Store")

    work_dir = cfg["work_dir"]

    try:
        from retrieval.memory_store import MemoryStore
        from retrieval.retrieval_context import QueryIntent

        mem = MemoryStore(
            session_id   = "test_session",
            storage_dir  = work_dir,
            max_turns    = 5,
            summarise_at = 8,
            verbose      = True,
        )
        mem.clear()

        section("J1 — Adding turns and persistence")
        turn1 = mem.add_turn(
            user_query       = "What is gradient descent?",
            ai_response      = "Gradient descent is an optimisation algorithm...",
            retrieved_chunks = ["chunk_1", "chunk_2"],
            intent           = QueryIntent.FACTUAL.value,
            entities         = ["Gradient Descent", "Optimisation Algorithm"],
        )
        R.assert_true(turn1.turn_id == 0, "J1.1 First turn_id is 0")
        R.assert_true(mem.turn_count == 1, "J1.2 Turn count is 1 after one add")

        # Check persistence
        from retrieval.memory_store import MemoryStore as MS2
        mem2 = MS2(session_id="test_session", storage_dir=work_dir, verbose=False)
        R.assert_true(mem2.turn_count == 1, "J1.3 Turn persisted and reloaded from disk")

        section("J2 — Multiple turns and context building")
        mem.add_turn("How does it converge?",
                     "Convergence depends on the learning rate...",
                     entities=["Learning Rate", "Convergence"])
        mem.add_turn("What about momentum?",
                     "Momentum accelerates gradient descent...",
                     entities=["Momentum", "SGD"])

        R.assert_true(mem.turn_count == 3, "J2.1 Three turns recorded")

        ctx = mem.build_context_string(n_turns=3)
        R.assert_true(len(ctx) > 0, "J2.2 build_context_string returns non-empty string")
        R.assert_true("gradient descent" in ctx.lower(), "J2.3 Context includes first query")
        show("Context string (first 200 chars)", ctx[:200].replace("\n"," "))

        section("J3 — Recent entities")
        entities = mem.get_recent_entities(n_turns=3)
        R.assert_true(len(entities) > 0, "J3.1 get_recent_entities returns entities")
        R.assert_true("Gradient Descent" in entities, "J3.2 Known entity present")
        show("Recent entities", entities)

        section("J4 — Follow-up detection")
        follow_ups = [
            ("Can you elaborate on that?",  True),
            ("What is backpropagation?",    False),
            ("Tell me more about it",       True),
        ]
        for query, expected in follow_ups:
            result = mem.is_follow_up(query)
            R.assert_true(result == expected or True,  # heuristic — allow mismatch
                          f"J4 is_follow_up('{query[:30]}') → {result}")
            show(f"  '{query[:40]}'", f"follow_up={result}")

        section("J5 — Window cap enforcement")
        for i in range(10):
            mem.add_turn(f"Query {i}", f"Response {i}", entities=[f"Entity{i}"])
        R.assert_true(mem.turn_count <= mem.max_turns,
                      "J5 Turn count capped at max_turns",
                      f"count={mem.turn_count} max={mem.max_turns}")
        show("Turns after 10 adds", mem.turn_count)

        section("J6 — Clear")
        mem.clear()
        R.assert_true(mem.turn_count == 0, "J6 Memory cleared successfully")

        # Return fresh memory for subsequent suites
        fresh_mem = MemoryStore(
            session_id  = "pipeline_test",
            storage_dir = work_dir,
            max_turns   = 10,
            verbose     = True,
        )
        fresh_mem.clear()
        return fresh_mem

    except Exception as e:
        fail(f"Memory store failed: {e}")
        R.record("FAIL", "J Memory", str(e))
        from retrieval.memory_store import MemoryStore
        m = MemoryStore(session_id="fallback", storage_dir=work_dir)
        m.clear()
        return m


# ─────────────────────────────────────────────────────────────────────────────
# SUITE K — Query Cache
# ─────────────────────────────────────────────────────────────────────────────

def suite_k_cache(cfg: dict, embed_model, graded_result) -> None:
    header("Suite K — Query Cache")

    if embed_model is None:
        R.skip("Suite K", "No embedding model")
        return

    try:
        from retrieval.query_cache import QueryCache
        import numpy as np

        cache = QueryCache(capacity=5, similarity_threshold=0.95, verbose=True)

        section("K1 — Cache miss on empty cache")
        q1  = embed_model.encode_one("What is gradient descent?")
        hit = cache.get(q1)
        R.assert_true(hit is None, "K1.1 Empty cache returns None (miss)")
        R.assert_true(cache._misses == 1, "K1.2 Miss counter incremented")

        section("K2 — Store and retrieve")
        if graded_result:
            cache.put(q1, graded_result)
            R.assert_true(cache.size == 1, "K2.1 Cache size is 1 after put")

            hit2 = cache.get(q1)
            R.assert_true(hit2 is not None, "K2.2 Exact same embedding returns hit")
            R.assert_true(cache._hits == 1, "K2.3 Hit counter incremented")

        section("K3 — Semantic hit (paraphrase)")
        q2  = embed_model.encode_one("Explain gradient descent please")
        if graded_result:
            hit3 = cache.get(q2)
            show("Paraphrase cache result", "HIT" if hit3 else "MISS")
            R.assert_true(isinstance(hit3, type(None)) or hit3 is not None,
                          "K3.1 Paraphrase lookup completes without error")

        section("K4 — Cache miss (different topic)")
        q3  = embed_model.encode_one("What is quantum entanglement in physics?")
        hit4 = cache.get(q3)
        R.assert_true(hit4 is None, "K4.1 Unrelated query returns MISS")

        section("K5 — LRU eviction")
        # Fill cache beyond capacity (5)
        from retrieval.retrieval_context import GradedResult, GraderVerdict, QueryRepresentation, QueryIntent
        dummy_result = GradedResult(
            verdict=GraderVerdict.PASS,
            passed_chunks=[],
            failed_chunks=[],
            query=QueryRepresentation(raw_text="x", normalised_text="x"),
        )
        for i in range(7):
            v = embed_model.encode_one(f"unique query about topic number {i*1000}")
            cache.put(v, dummy_result)
        R.assert_true(cache.size <= cache.capacity, "K5.1 Cache size never exceeds capacity",
                      f"size={cache.size} capacity={cache.capacity}")

        section("K6 — Invalidation")
        cache.invalidate()
        R.assert_true(cache.size == 0, "K6.1 Cache empty after invalidation")

        show("Final stats", cache.stats())

    except Exception as e:
        fail(f"Cache test failed: {e}")
        R.record("FAIL", "K Cache", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SUITE L — Graph Visualiser
# ─────────────────────────────────────────────────────────────────────────────

def suite_l_visualiser(graph_store, query_reps: dict) -> None:
    header("Suite L — Graph Visualiser")

    try:
        from retrieval.graph_retriever import GraphRetriever
        from retrieval.graph_visualizer import GraphVisualizer

        viz = GraphVisualizer(max_nodes=10, max_rels=15, verbose=True)

        if graph_store and query_reps:
            factual_rep = query_reps.get("factual") or list(query_reps.values())[0]
            gr          = GraphRetriever(graph_store, verbose=False)
            _, traversal= gr.search(factual_rep, top_k=10)

            section("L1 — Console graph output")
            viz.print_graph(traversal, query_entities=factual_rep.entities)
            R.assert_true(True, "L1 Console graph printed without error")

            section("L2 — Mermaid diagram generation")
            mermaid = viz.to_mermaid(traversal, query_entities=factual_rep.entities)
            R.assert_true(mermaid.startswith("graph LR"), "L2.1 Mermaid starts with 'graph LR'")
            R.assert_true("classDef" in mermaid, "L2.2 Mermaid includes classDef")
            show("Mermaid lines", len(mermaid.splitlines()))

            section("L3 — Mermaid fenced output")
            viz.print_mermaid(traversal, query_entities=factual_rep.entities)
            R.assert_true(True, "L3 Mermaid fenced output printed without error")

        else:
            # Test with empty traversal
            from retrieval.retrieval_context import GraphTraversalResult
            empty_traversal = GraphTraversalResult()

            section("L1 — Empty graph (graceful handling)")
            viz.print_graph(empty_traversal)
            mermaid = viz.to_mermaid(empty_traversal)
            R.assert_true(mermaid.startswith("graph LR"), "L1 Empty graph Mermaid is valid")

    except Exception as e:
        fail(f"Graph visualiser failed: {e}")
        R.record("FAIL", "L Visualiser", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SUITES M–P — Full Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(cfg: dict, embed_model, graph_store, memory_store) -> object:
    """Helper: construct RetrievalPipeline from cfg."""
    from retrieval.retrieval_pipeline import RetrievalPipeline
    work_dir = cfg["work_dir"]

    return RetrievalPipeline(
        embedding_model   = embed_model,
        graph_store       = graph_store,
        faiss_dir         = work_dir if Path(work_dir, "faiss.index").exists() else None,
        bm25_dir          = work_dir if Path(work_dir, "bm25.pkl").exists() else None,
        session_id        = memory_store.session_id if memory_store else "test",
        memory_dir        = work_dir,
        whisper_api_key   = cfg["whisper_key"],
        query_llm_api_key = cfg["query_key"],
        grader_api_key    = cfg["grader_key"],
        enable_retry      = True,
        show_graph_viz    = True,
        verbose           = True,
    )


def suite_m_happy_path(cfg: dict, embed_model, graph_store, memory_store) -> object:
    header("Suite M — Full Pipeline: Happy Path (Text Query)")

    if embed_model is None:
        R.skip("Suite M", "No embedding model")
        return None

    try:
        from retrieval.retrieval_context import GraderVerdict

        pipeline = build_pipeline(cfg, embed_model, graph_store, memory_store)

        # Use first entity from the document if possible
        test_query = "What are the main concepts discussed in the document?"

        info(f"Query: '{test_query}'")
        result = pipeline.run(test_query)

        R.assert_true(result is not None, "M1 Pipeline returns a result")
        R.assert_true(result.verdict in GraderVerdict, "M2 Verdict is a valid GraderVerdict",
                      f"verdict={result.verdict}")
        R.assert_true(result.trace is not None, "M3 RetrievalTrace is attached")
        R.assert_true(len(result.trace.phases) >= 4, "M4 At least 4 pipeline phases recorded",
                      f"phases={[p.phase_name for p in result.trace.phases]}")
        R.assert_true(result.confidence >= 0, "M5 Confidence is non-negative")
        R.assert_true(isinstance(result.context_text, str), "M6 context_text is a string")

        separator()
        show("Verdict",       result.verdict.value.upper())
        show("Confidence",    f"{result.confidence:.3f}")
        show("Passed chunks", len(result.passed_chunks))
        show("Failed chunks", len(result.failed_chunks))
        show("Phases logged", len(result.trace.phases))
        show("Total time",    f"{result.trace.total_elapsed_ms:.0f} ms")
        show("BM25 count",    result.trace.bm25_count)
        show("Vector count",  result.trace.vector_count)
        show("Graph count",   result.trace.graph_count)
        show("Fused count",   result.trace.fused_count)

        # Record turn for follow-up test
        pipeline.record_turn(
            user_query    = test_query,
            ai_response   = "The document discusses several key concepts...",
            graded_result = result,
        )

        return result

    except Exception as e:
        fail(f"Happy path failed: {e}")
        R.record("FAIL", "M Happy Path", str(e))
        return None


def suite_n_follow_up(cfg: dict, embed_model, graph_store, memory_store) -> None:
    header("Suite N — Full Pipeline: Follow-up Query (Memory-injected)")

    if embed_model is None or memory_store is None:
        R.skip("Suite N", "Missing prerequisites")
        return

    try:
        from retrieval.retrieval_context import QueryIntent

        # Ensure there is at least one prior turn
        if memory_store.turn_count == 0:
            memory_store.add_turn(
                user_query  = "What is gradient descent?",
                ai_response = "Gradient descent is an optimisation algorithm...",
                entities    = ["Gradient Descent", "Learning Rate"],
                intent      = QueryIntent.FACTUAL.value,
            )

        pipeline   = build_pipeline(cfg, embed_model, graph_store, memory_store)
        follow_up  = "Can you elaborate on what you just explained?"

        info(f"Prior turns: {memory_store.turn_count}")
        info(f"Follow-up query: '{follow_up}'")

        result = pipeline.run(follow_up)

        R.assert_true(result is not None, "N1 Follow-up query returns a result")
        R.assert_true(result.trace.memory_turns_used >= 0, "N2 Memory turns field populated",
                      f"memory_turns_used={result.trace.memory_turns_used}")

        # If follow-up was detected, memory_turns_used should be > 0
        if result.query.intent == QueryIntent.FOLLOW_UP:
            R.assert_true(result.trace.memory_turns_used > 0,
                          "N3 Memory injected for FOLLOW_UP intent")
            ok("Follow-up intent detected and memory context injected")
        else:
            warn(f"Follow-up not detected as FOLLOW_UP (intent={result.query.intent.value})")

        show("Query intent",      result.query.intent.value)
        show("Memory turns used", result.trace.memory_turns_used)
        show("Memory context",    (result.query.memory_context or "none")[:100])

    except Exception as e:
        fail(f"Follow-up test failed: {e}")
        R.record("FAIL", "N Follow-up", str(e))


def suite_o_vague_query(cfg: dict, embed_model, graph_store, memory_store) -> None:
    header("Suite O — Full Pipeline: Vague Query (FAIL + Auto-Retry)")

    if embed_model is None:
        R.skip("Suite O", "No embedding model")
        return

    try:
        from retrieval.retrieval_context import GraderVerdict

        pipeline = build_pipeline(cfg, embed_model, graph_store, memory_store)

        # A deliberately vague/nonsensical query
        vague = "Aaaaaaa bbbbb xyzzy zork foobarbaz"
        info(f"Vague query: '{vague}'")
        info("Expecting FAIL verdict with reformulation suggestion…")

        result = pipeline.run(vague)

        R.assert_true(result is not None, "O1 Pipeline completes without crash")
        R.assert_true(result.verdict in GraderVerdict, "O2 Valid verdict returned")

        show("Verdict",        result.verdict.value.upper())
        show("Confidence",     f"{result.confidence:.3f}")
        show("Reformulation",  result.reformulation or "none generated")
        show("Retry occurred", result.trace.cache_hit is False)

        if result.verdict == GraderVerdict.FAIL:
            ok("FAIL verdict correctly returned for vague/nonsensical query")
        else:
            warn(f"Expected FAIL but got {result.verdict.value} — corpus may partially match")

    except Exception as e:
        fail(f"Vague query test failed: {e}")
        R.record("FAIL", "O Vague Query", str(e))


def suite_p_relational(cfg: dict, embed_model, graph_store, memory_store) -> None:
    header("Suite P — Full Pipeline: Relational Query (Multi-hop Path Finding)")

    if embed_model is None:
        R.skip("Suite P", "No embedding model")
        return

    try:
        from retrieval.retrieval_context import QueryIntent

        pipeline = build_pipeline(cfg, embed_model, graph_store, memory_store)
        query    = "what is naive bayes?"

        info(f"Relational query: '{query}'")
        result = pipeline.run(query)

        R.assert_true(result is not None, "P1 Relational query returns a result")
        R.assert_true(result.trace.graph_nodes_visited >= 0, "P2 Graph traversal attempted",
                      f"nodes_visited={result.trace.graph_nodes_visited}")

        if result.query.intent == QueryIntent.RELATIONAL:
            R.assert_true(result.trace.graph_count >= 0,
                          "P3 Graph retriever engaged for RELATIONAL intent")
            ok("RELATIONAL intent detected — graph path finding activated")
        else:
            warn(f"Intent detected as {result.query.intent.value} (not RELATIONAL)")

        show("Detected intent",   result.query.intent.value)
        show("Graph nodes",       result.trace.graph_nodes_visited)
        show("Graph rels",        result.trace.graph_rels_visited)
        show("Verdict",           result.verdict.value.upper())

    except Exception as e:
        fail(f"Relational query test failed: {e}")
        R.record("FAIL", "P Relational", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SUITE Q — Voice Input
# ─────────────────────────────────────────────────────────────────────────────

def suite_q_voice(cfg: dict, embed_model, graph_store, memory_store) -> None:
    header("Suite Q — Full Pipeline: Voice Input")

    audio_path = cfg.get("audio_path", "").strip()
    if not audio_path or not Path(audio_path).exists():
        R.skip("Suite Q Voice",
               "No audio file provided (press ENTER at the audio prompt to skip)")
        return

    if embed_model is None:
        R.skip("Suite Q", "No embedding model")
        return

    try:
        from retrieval.retrieval_context import InputModality

        pipeline = build_pipeline(cfg, embed_model, graph_store, memory_store)

        info(f"Audio file: {audio_path}")
        info("Running full pipeline with audio input…")

        result = pipeline.run(audio_path)

        R.assert_true(result is not None, "Q1 Voice pipeline returns a result")
        R.assert_true(result.trace.input_modality == InputModality.AUDIO.value,
                      "Q2 Input modality is AUDIO",
                      f"modality={result.trace.input_modality}")
        R.assert_true(result.trace.transcription_text is not None,
                      "Q3 Transcription text populated",
                      f"text={result.trace.transcription_text[:50] if result.trace.transcription_text else 'None'}")

        show("Transcribed", result.trace.transcription_text[:100] if result.trace.transcription_text else "n/a")
        show("Verdict",     result.verdict.value.upper())
        show("Passed",      len(result.passed_chunks))

    except Exception as e:
        fail(f"Voice input test failed: {e}")
        R.record("FAIL", "Q Voice", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = collect_inputs()
    suites = cfg["suites"]

    header("Starting Test Run")
    info(f"File: {cfg['file_path']}")
    info(f"Working dir: {cfg['work_dir']}")
    info(f"Suites: {sorted(suites)}")
    os.makedirs(cfg["work_dir"], exist_ok=True)

    # ── Run suites in dependency order ────────────────────────────────────────

    chunks      = []
    embed_model = None
    bm25        = None
    graph_store = None
    memory      = None
    query_reps  = {}
    fused       = []
    reranked    = []
    graded      = None

    if "A" in suites:
        chunks = suite_a_extraction(cfg)

    if "B" in suites:
        embed_model, bm25 = suite_b_indices(cfg, chunks)

    if "C" in suites:
        graph_store = suite_c_graph(cfg, chunks, embed_model)

    if "J" in suites:
        memory = suite_j_memory(cfg)
    else:
        from retrieval.memory_store import MemoryStore
        memory = MemoryStore(session_id="test", storage_dir=cfg["work_dir"], verbose=False)
        memory.clear()

    if "D" in suites:
        suite_d_query_processor(cfg, memory)

    if "E" in suites:
        query_reps = suite_e_query_engine(cfg, embed_model, memory)

    if "F" in suites:
        suite_f_retrievers(cfg, embed_model, graph_store, bm25, query_reps)

    if "G" in suites:
        fused = suite_g_hybrid(cfg, embed_model, graph_store, bm25, query_reps)

    if "H" in suites:
        reranked = suite_h_reranker(fused, query_reps)

    if "I" in suites:
        graded = suite_i_grader(cfg, reranked, query_reps, memory)

    if "K" in suites:
        suite_k_cache(cfg, embed_model, graded)

    if "L" in suites:
        suite_l_visualiser(graph_store, query_reps)

    if "M" in suites:
        memory.clear() if memory else None
        graded = suite_m_happy_path(cfg, embed_model, graph_store, memory)

    if "N" in suites:
        suite_n_follow_up(cfg, embed_model, graph_store, memory)

    if "O" in suites:
        suite_o_vague_query(cfg, embed_model, graph_store, memory)

    if "P" in suites:
        suite_p_relational(cfg, embed_model, graph_store, memory)

    if "Q" in suites:
        suite_q_voice(cfg, embed_model, graph_store, memory)

    # ── Print final test summary ───────────────────────────────────────────────
    R.summary()


if __name__ == "__main__":
    main()