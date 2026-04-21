# #!/usr/bin/env python3
# """
# test_flow.py — Real-document KG-RAG ingestion test with CombinedExtractor
# ==========================================================================
# Demonstrates a complete end-to-end test of the KG-RAG pipeline using:
#   • Extractors module (PDF, DOCX, PPTX, Audio, Video, URL) to extract text
#   • Chunking module (SemanticChunker, RecursiveChunker, etc.) to chunk text
#   • Embeddings module (HuggingFaceEmbedding) to vectorize chunks
#   • CombinedExtractor for simultaneous node & relationship extraction
#   • Neo4j integration (optional) with automatic database clearing

# No mock data — uses your own document and actual extraction modules.

# Run modes
# ---------
#     # Interactive (prompts for file path and Neo4j password)
#     python test_flow.py

#     # Fully specified via CLI
#     python test_flow.py \\
#         --file   /path/to/lecture.pdf \\
#         --source lecture_01 \\
#         --neo4j  \\
#         --password neo4j1234 \\
#         --model  llama-3.3-70b-versatile \\
#         -v

#     # Skip Neo4j — extraction + display only
#     python test_flow.py --file /path/to/notes.pdf

# CLI flags
# ---------
#     --file       Path to the document to ingest (PDF, TXT, DOCX, PPTX, …)
#     --source     Logical source name stored in the graph (default: file stem)
#     --neo4j      Enable Neo4j write tests
#     --uri        Neo4j URI           (default: neo4j://127.0.0.1:7687)
#     --user       Neo4j user          (default: neo4j)
#     --password   Neo4j password      (default: neo4j1234)
#     --model      Groq model          (default: llama-3.3-70b-versatile)
#     --max-chunks Limit chunks used   (default: all)
#     --batch      Chunks per combined LLM call (default: 5)
#     --mode       constrained | unconstrained (default: constrained)
#     -v / --verbose   Verbose output

# Environment variables
# ---------------------
#     GROQ_API_KEY   — Groq API key (required)
#     NEO4J_URI      — neo4j://127.0.0.1:7687
#     NEO4J_USER     — neo4j
#     NEO4J_PASSWORD — neo4j1234
# """
# from __future__ import annotations

# import argparse
# import os
# import sys
# import time
# import warnings
# from pathlib import Path
# from typing import List, Optional

# import numpy as np

# # ── .env loading ──────────────────────────────────────────────────────────────
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# # ── Path setup ────────────────────────────────────────────────────────────────
# _HERE = Path(__file__).resolve().parent
# _ROOT = _HERE.parent
# for _p in [str(_HERE), str(_ROOT)]:
#     if _p not in sys.path:
#         sys.path.insert(0, _p)


# # =============================================================================
# # Module imports — fail fast with clear messages
# # =============================================================================

# # ── Chunk class ───────────────────────────────────────────────────────────────
# try:
#     from chunking.chunk_base import Chunk
#     print("✓ Chunk class loaded  (chunking.chunk_base)")
# except ImportError as e:
#     print(f"ERROR: Could not import Chunk from chunking.chunk_base: {e}")
#     sys.exit(1)

# # ── Chunking module ───────────────────────────────────────────────────────────
# _HAS_CHUNKER = False
# _chunker_obj = None
# try:
#     from chunking import (
#         SemanticChunker,
#         RecursiveChunker,
#         SlidingWindowChunker,
#         SentenceChunker,
#         FixedSizeChunker,
#         ParagraphChunker,
#     )
#     _chunker_obj = RecursiveChunker()
#     _HAS_CHUNKER = True
#     print("✓ Chunking module loaded with RecursiveChunker (default)")
# except ImportError as e:
#     print(f"WARNING: Chunking module import failed ({e})")
#     print("         Will use built-in word-count fallback chunker.")

# # ── Embeddings module ─────────────────────────────────────────────────────────
# _HAS_EMBEDDING = False
# _embedding_fn  = None
# try:
#     from embeddings import HuggingFaceEmbedding
#     embedding_model = HuggingFaceEmbedding(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         device=None  # Auto-detect (cuda if available, else cpu)
#     )
#     _embedding_fn  = embedding_model.encode
#     _HAS_EMBEDDING = True
#     print("✓ HuggingFaceEmbedding loaded (embeddings.HuggingFaceEmbedding)")
#     print("   Model: sentence-transformers/all-MiniLM-L6-v2 (384 dim)")
# except ImportError as e:
#     print(f"WARNING: embeddings module not found ({e}) — using deterministic dummy (dim=384).")

# # ── Extractors module ─────────────────────────────────────────────────────────
# _HAS_EXTRACTORS = False
# _extractor_wrapper = None
# try:
#     from extractors.extractors_wrapper import SimpleExtractorWrapper
#     from extractors import get_type_for_extension
#     _extractor_wrapper = SimpleExtractorWrapper()
#     _HAS_EXTRACTORS = True
#     print("✓ Extractors module loaded  (extractors.*)")
#     print("   Supported: PDF, DOCX, PPTX, Audio, Video, URL")
# except ImportError as e:
#     print(f"WARNING: Extractors module not fully available ({e})")

# # ── Graph module ──────────────────────────────────────────────────────────────
# try:
#     from graph.llm_backend import LLMBackend
#     from graph.graph_store  import GraphStore
#     from graph.base         import ExtractedNode, ExtractedRelationship
#     print("✓ Graph module loaded  (graph.*)")
# except ImportError as e:
#     print(f"ERROR: Could not import graph module: {e}")
#     sys.exit(1)

# # ── CombinedExtractor ─────────────────────────────────────────────────────────
# try:
#     from graph.node_relation_extractor import CombinedExtractor, ALLOWED_RELATIONS
#     print("✓ CombinedExtractor loaded  (graph.node_relation_extractor)")
# except ImportError as e:
#     print(f"ERROR: Could not import CombinedExtractor: {e}")
#     print("       Make sure node_relation_extractor.py is inside your graph/ package.")
#     sys.exit(1)

# # ── Pipeline ──────────────────────────────────────────────────────────────────
# try:
#     from Pipeline import Pipeline, PipelineStats
#     print("✓ Pipeline loaded  (Pipeline)")
# except ImportError as e:
#     print(f"ERROR: Could not import Pipeline: {e}")
#     sys.exit(1)


# # =============================================================================
# # Test counters
# # =============================================================================
# _total  = 0
# _passed = 0


# def run_test(description: str, condition: bool, verbose: bool = False, detail: str = "") -> None:
#     global _total, _passed
#     _total += 1
#     icon = "✓ PASS" if condition else "✗ FAIL"
#     if condition:
#         _passed += 1
#     print(f"  {icon}  {description}")
#     if verbose and detail:
#         print(f"         → {detail}")


# def header(title: str) -> None:
#     print(f"\n{'─' * 70}")
#     print(f"  {title}")
#     print(f"{'─' * 70}")


# # =============================================================================
# # Embedding fallback
# # =============================================================================

# def _dummy_embedding_fn(texts: List[str]) -> np.ndarray:
#     """Deterministic dummy embedding (used when embedding module unavailable)."""
#     rng = np.random.default_rng(seed=42)
#     return rng.random((len(texts), 384)).astype(np.float32)


# # =============================================================================
# # File extraction and chunking
# # =============================================================================

# def extract_text_from_file(file_path: Path) -> str:
#     """
#     Extract text from various file formats using the extractors module.
#     Falls back to plain text reading for unsupported formats.
#     """
#     suffix = file_path.suffix.lower()

#     if suffix == ".txt":
#         return file_path.read_text(encoding="utf-8", errors="replace")

#     if not _HAS_EXTRACTORS:
#         print(f"WARNING: Extractors module not available. Reading as plain text.")
#         return file_path.read_text(encoding="utf-8", errors="replace")

#     try:
#         if suffix == ".pdf":
#             extracted = _extractor_wrapper.extract_pdf(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".docx", ".doc"]:
#             extracted = _extractor_wrapper.extract_docx(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".pptx", ".ppt"]:
#             extracted = _extractor_wrapper.extract_pptx(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
#             extracted = _extractor_wrapper.extract_audio(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]:
#             extracted = _extractor_wrapper.extract_video(str(file_path))
#             if extracted:
#                 return extracted
#     except Exception as e:
#         print(f"WARNING: Extraction failed for {suffix}: {e}")

#     # Fallback: read as plain text
#     print(f"WARNING: Could not extract {suffix}. Reading as plain text.")
#     return file_path.read_text(encoding="utf-8", errors="replace")


# # =============================================================================
# # Chunking helpers
# # =============================================================================

# def _fallback_chunker(text: str, source: str, chunk_size: int = 400) -> List[Chunk]:
#     """
#     Built-in word-count chunker (fallback when no chunker module is found).
#     Splits text into overlapping windows of ~chunk_size words.
#     """
#     words   = text.split()
#     overlap = chunk_size // 5
#     chunks  = []
#     idx     = 0
#     chunk_id= 0
#     while idx < len(words):
#         window   = words[idx: idx + chunk_size]
#         text_out = " ".join(window)
#         start    = len(" ".join(words[:idx]))
#         end      = start + len(text_out)
#         chunks.append(Chunk(
#             text     = text_out,
#             chunk_id = chunk_id,
#             start_char = start,
#             end_char   = end,
#             metadata   = {"source": source},
#         ))
#         chunk_id += 1
#         idx      += chunk_size - overlap
#     return chunks


# def chunk_text(text: str, source: str) -> List[Chunk]:
#     """
#     Chunk text using the chunking module (if available) or the built-in fallback.
#     """
#     if _HAS_CHUNKER and _chunker_obj is not None:
#         # Pattern 1: chunker.chunk(text) → List[Chunk]
#         if hasattr(_chunker_obj, "chunk"):
#             chunks = _chunker_obj.chunk(text)
#             for c in chunks:
#                 c.metadata.setdefault("source", source)
#             return chunks

#         # Pattern 2: chunker(text) → List[Chunk]   (callable)
#         if callable(_chunker_obj):
#             chunks = _chunker_obj(text)
#             for c in chunks:
#                 c.metadata.setdefault("source", source)
#             return chunks

#         print("WARNING: Chunker found but no known method (chunk / __call__).")

#     # Built-in fallback
#     return _fallback_chunker(text, source)


# # =============================================================================
# # Test functions
# # =============================================================================

# def test_file_extraction(file_path: Path, verbose: bool):
#     header(f"Step 1 — File extraction  [{file_path.name}]")

#     suffix = file_path.suffix.lower()
#     run_test(f"File exists and is readable", file_path.exists() and file_path.is_file(), verbose)

#     try:
#         text = extract_text_from_file(file_path)
#         run_test("Text extracted successfully", len(text) > 0, verbose,
#                  f"{len(text)} characters extracted")
#         if verbose:
#             preview = text[:300].replace("\n", " ")
#             print(f"\n  Text preview:")
#             print(f"  {preview}{'...' if len(text) > 300 else ''}\n")
#         return text
#     except Exception as e:
#         run_test("Text extraction successful", False, verbose, str(e))
#         return None


# def test_chunking(text: str, source: str, max_chunks: Optional[int], verbose: bool) -> List[Chunk]:
#     header("Step 2 — Document chunking")

#     try:
#         chunks = chunk_text(text, source)
#         run_test("chunk_text() returns a non-empty list", len(chunks) > 0, verbose,
#                  f"{len(chunks)} raw chunk(s) produced")

#         if chunks:
#             c0 = chunks[0]
#             run_test("Chunk has .text attribute",     hasattr(c0, "text"),     verbose)
#             run_test("Chunk has .chunk_id attribute", hasattr(c0, "chunk_id"), verbose)
#             run_test("Chunk has .metadata dict",      isinstance(getattr(c0, "metadata", None), dict), verbose)
#             run_test('metadata["source"] is set',     c0.metadata.get("source") == source, verbose,
#                      f"source = {c0.metadata.get('source')!r}")

#         if max_chunks:
#             chunks = chunks[:max_chunks]
#             print(f"  (limited to {max_chunks} chunks for this run)")

#         if verbose and chunks:
#             preview = chunks[0].text[:300].replace("\n", " ")
#             print(f"\n  Preview of chunk 0:")
#             print(f"  {preview}{'...' if len(chunks[0].text) > 300 else ''}\n")

#         return chunks
#     except Exception as e:
#         run_test("Chunking completed", False, verbose, str(e))
#         return []


# def test_llm_connection(llm: LLMBackend, verbose: bool):
#     header("Step 3 — LLM Backend connection")

#     run_test("LLMBackend instantiated", llm is not None, verbose)

#     print("  Sending test prompt to Groq...")
#     try:
#         resp = llm.generate("Reply with exactly: OK")
#         ok   = bool(resp and len(resp.strip()) > 0)
#         run_test("Groq API responds", ok, verbose, f"response: {resp.strip()[:80]!r}")
#     except Exception as e:
#         run_test("Groq API responds", False, verbose, str(e))


# def test_combined_extractor_init(llm: LLMBackend, embedding_fn, batch_chunks: int, mode: str, verbose: bool) -> CombinedExtractor:
#     header("Step 4 — CombinedExtractor initialisation")

#     ext = CombinedExtractor(
#         llm                  = llm,
#         embedding_fn         = embedding_fn,
#         mode                 = mode,
#         batch_chunks         = batch_chunks,
#         confidence_threshold = 0.6,
#         max_entity_pairs     = 20,
#     )
#     run_test("CombinedExtractor instantiated", ext is not None, verbose)
#     run_test("Correct batch_chunks",           ext.batch_chunks == batch_chunks, verbose,
#              f"batch_chunks = {ext.batch_chunks}")
#     run_test("Correct mode",                   ext.mode == mode, verbose)
#     run_test("Extended allowed types includes Theorem",
#              "Theorem" in ext.allowed_types, verbose)
#     run_test("Extended allowed types includes Operator",
#              "Operator" in ext.allowed_types, verbose)
#     run_test("ALLOWED_RELATIONS includes DEFINED_BY",
#              "DEFINED_BY" in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes IMPLIES",
#              "IMPLIES" in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes PROOF_OF",
#              "PROOF_OF" in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes CONVERGES_TO",
#              "CONVERGES_TO" in ALLOWED_RELATIONS, verbose)
#     if verbose:
#         print(f"\n  {ext}\n")
#     return ext


# def test_node_extraction(ext: CombinedExtractor, chunks: list, verbose: bool) -> tuple:
#     header(f"Step 5 — Node extraction  ({len(chunks)} chunks, batch_size={ext.batch_chunks})")

#     t0 = time.time()
#     try:
#         nodes = ext.extract_from_chunks(chunks, show_progress=True)
#     except Exception as e:
#         run_test("extract_from_chunks (nodes) completed", False, verbose, str(e))
#         print(f"  WARNING: Node extraction failed — pipeline will continue with empty nodes.")
#         return [], {}

#     elapsed = time.time() - t0

#     success = len(nodes) > 0
#     run_test("extract_from_chunks (nodes) completed",
#              success, verbose, f"{len(nodes)} node(s) extracted in {elapsed:.1f}s" if success else "Extraction produced no nodes (possible rate limit)")

#     if nodes:
#         n0 = nodes[0]
#         run_test("Each node has node_id",     bool(n0.node_id),      verbose)
#         run_test("Each node has name",        bool(n0.name),         verbose)
#         run_test("Each node has entity_type", bool(n0.entity_type),  verbose)
#         run_test("Each node has description", bool(n0.description),  verbose)
#         run_test("Each node has source",      bool(n0.source),       verbose)
#         run_test("Each node has embedding",
#                  n0.embedding is not None and isinstance(n0.embedding, np.ndarray),
#                  verbose, f"shape = {np.shape(n0.embedding)}")
#         run_test("No duplicate node_ids",
#                  len({n.node_id for n in nodes}) == len(nodes), verbose)

#     # Build node_id_map
#     node_id_map: dict = {}
#     for n in nodes:
#         node_id_map[n.name.lower()] = n.node_id
#         for alias in n.aliases:
#             node_id_map.setdefault(alias.lower(), n.node_id)

#     if verbose and nodes:
#         print(f"\n  Entity type distribution:")
#         from collections import Counter
#         counts = Counter(n.entity_type for n in nodes)
#         for etype, cnt in counts.most_common():
#             print(f"    {etype:20s}: {cnt}")
#         print(f"\n  Sample nodes:")
#         for n in nodes[:5]:
#             print(f"    [{n.entity_type}] {n.name}")
#             print(f"           {n.description[:100]}")
#         print()

#     return nodes, node_id_map


# def test_relationship_extraction(
#     ext         : CombinedExtractor,
#     chunks      : list,
#     nodes       : list,
#     node_id_map : dict,
#     verbose     : bool,
# ) -> list:
#     header(f"Step 6 — Relationship extraction  ({len(chunks)} chunks)")

#     # Build nodes_per_chunk
#     nodes_per_chunk: dict = {}
#     for idx, chunk in enumerate(chunks):
#         chunk_id = str(getattr(chunk, "chunk_id", idx))
#         nodes_per_chunk[idx] = [n for n in nodes if n.source_chunk == chunk_id]

#     t0 = time.time()
#     try:
#         rels = ext.extract_from_chunks(
#             chunks,
#             nodes_per_chunk = nodes_per_chunk,
#             node_id_map     = node_id_map,
#             show_progress   = True,
#         )
#     except Exception as e:
#         run_test("extract_from_chunks (rels) completed", False, verbose, str(e))
#         return []

#     elapsed = time.time() - t0
#     run_test("Returns a list of ExtractedRelationship",
#              isinstance(rels, list), verbose, f"{len(rels)} rel(s) in {elapsed:.1f}s")

#     if rels:
#         r0 = rels[0]
#         run_test("Each rel has source_id",    bool(r0.source_id),    verbose)
#         run_test("Each rel has target_id",    bool(r0.target_id),    verbose)
#         run_test("Each rel has relation_type",bool(r0.relation_type),verbose)
#         run_test("Each rel has description",  bool(r0.description),  verbose)
#         run_test("Each rel has confidence",   0 <= r0.confidence <= 1, verbose,
#                  f"confidence = {r0.confidence}")

#     if verbose and rels:
#         print(f"\n  Relation type distribution:")
#         from collections import Counter
#         rcounts = Counter(r.relation_type for r in rels)
#         for rtype, cnt in rcounts.most_common(10):
#             print(f"    {rtype:30s}: {cnt}")
#         print(f"\n  Sample relationships:")
#         for r in rels[:5]:
#             print(f"    {r.source_name} —[{r.relation_type}]→ {r.target_name}")
#             print(f"       conf={r.confidence:.2f}  {r.description[:80]}")
#         print()

#     return rels


# def test_graph_store(
#     store       : GraphStore,
#     nodes       : list,
#     rels        : list,
#     source      : str,
#     verbose     : bool,
# ):
#     header("Step 7 — GraphStore Neo4j write & verify")

#     try:
#         store.init_schema()
#         run_test("init_schema() runs without error", True, verbose)
#     except Exception as e:
#         run_test("init_schema() runs without error", False, verbose, str(e))
#         return

#     # Clear previous data for this source
#     try:
#         n_del, r_del = store.delete_by_source(source)
#         run_test(f"Deleted previous data for source '{source}'", True, verbose,
#                  f"Deleted {n_del} nodes, {r_del} relationships")
#     except Exception as e:
#         print(f"  WARNING: Could not clear previous data: {e}")

#     # Upsert nodes
#     try:
#         n_written = store.upsert_nodes(nodes)
#         run_test("upsert_nodes() succeeds",
#                  n_written >= 0, verbose, f"{n_written} node(s) written")
#     except Exception as e:
#         run_test("upsert_nodes() succeeds", False, verbose, str(e))
#         return

#     # Upsert relationships
#     try:
#         r_written = store.upsert_relationships(rels)
#         run_test("upsert_relationships() succeeds",
#                  r_written >= 0, verbose, f"{r_written} rel(s) written")
#     except Exception as e:
#         run_test("upsert_relationships() succeeds", False, verbose, str(e))

#     # Verify counts
#     try:
#         n_count = store.count_nodes()
#         r_count = store.count_relationships()
#         nodes_ok = n_count > 0 or len(nodes) == 0
#         rels_ok = r_count > 0 or len(rels) == 0
#         run_test("Graph has nodes", nodes_ok, verbose, f"{n_count} total nodes (expected {len(nodes)})")
#         run_test("Graph has rels",  rels_ok, verbose, f"{r_count} total rels (expected {len(rels)})")
#     except Exception as e:
#         run_test("count_nodes / count_relationships", False, verbose, str(e))


# def test_pipeline_end_to_end(
#     llm          : LLMBackend,
#     embedding_fn,
#     store        : GraphStore,
#     chunks       : list,
#     batch_chunks : int,
#     mode         : str,
#     source       : str,
#     verbose      : bool,
# ):
#     header("Step 8 — Full Pipeline.from_components() end-to-end run")

#     ext = CombinedExtractor(
#         llm                  = llm,
#         embedding_fn         = embedding_fn,
#         mode                 = mode,
#         batch_chunks         = batch_chunks,
#         confidence_threshold = 0.6,
#     )

#     try:
#         pipeline = Pipeline.from_components(
#             node_extractor         = ext,
#             relationship_extractor = ext,
#             graph_store            = store,
#             schema_already_exists  = True,
#             show_progress          = verbose,
#         )
#         run_test("Pipeline.from_components() succeeds", True, verbose)
#     except Exception as e:
#         run_test("Pipeline.from_components() succeeds", False, verbose, str(e))
#         return

#     # Clean previous data for this source
#     try:
#         store.delete_by_source(source)
#     except Exception:
#         pass

#     t0 = time.time()
#     try:
#         stats = pipeline.run(chunks)
#         elapsed = time.time() - t0

#         run_test("pipeline.run() returns PipelineStats", isinstance(stats, PipelineStats), verbose)
#         run_test("chunks_processed == len(chunks)",
#                  stats.chunks_processed == len(chunks), verbose,
#                  f"{stats.chunks_processed}")
#         run_test("nodes_extracted > 0",
#                  stats.nodes_extracted > 0, verbose,
#                  f"{stats.nodes_extracted}")
#         run_test("nodes_written > 0",
#                  stats.nodes_written > 0, verbose,
#                  f"{stats.nodes_written}")
#         run_test("relationships_extracted >= 0",
#                  stats.relationships_extracted >= 0, verbose,
#                  f"{stats.relationships_extracted}")

#         print(f"\n  Pipeline run completed in {elapsed:.1f}s")
#         print(stats)

#     except Exception as e:
#         run_test("pipeline.run() completes without fatal error", False, verbose, str(e))

#     with pipeline:
#         gstats = pipeline.graph_stats()
#         run_test("graph_stats() returns node count",
#                  isinstance(gstats.get("nodes"), int), verbose,
#                  f"{gstats}")


# # =============================================================================
# # Main
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="KG-RAG test flow using CombinedExtractor on a real document"
#     )
#     parser.add_argument("--file",       default=None,
#                         help="Path to the document to ingest")
#     parser.add_argument("--source",     default=None,
#                         help="Logical source name (defaults to file stem)")
#     parser.add_argument("--neo4j",      action="store_true",
#                         help="Run Neo4j write tests")
#     parser.add_argument("--uri",        default=os.environ.get("NEO4J_URI",      "neo4j://127.0.0.1:7687"))
#     parser.add_argument("--user",       default=os.environ.get("NEO4J_USER",     "neo4j"))
#     parser.add_argument("--password",   default=os.environ.get("NEO4J_PASSWORD", "neo4j1234"))
#     parser.add_argument("--model",      default="llama-3.3-70b-versatile")
#     parser.add_argument("--max-chunks", type=int, default=None,
#                         help="Limit the number of chunks used (useful for quick tests)")
#     parser.add_argument("--batch",      type=int, default=5,
#                         help="Chunks per combined LLM call (default 5)")
#     parser.add_argument("--mode",       default="constrained",
#                         choices=["constrained", "unconstrained"])
#     parser.add_argument("-v", "--verbose", action="store_true")
#     args = parser.parse_args()

#     # ── Prompt for file path if not given ────────────────────────────────────
#     file_path_str = args.file
#     if not file_path_str:
#         print()
#         file_path_str = input("Enter path to the document to ingest: ").strip()
#         if not file_path_str:
#             print("No file path provided. Exiting.")
#             sys.exit(1)

#     file_path = Path(file_path_str).expanduser().resolve()
#     if not file_path.exists():
#         print(f"ERROR: File not found: {file_path}")
#         sys.exit(1)

#     source = args.source or file_path.stem

#     # ── Print configuration banner ────────────────────────────────────────────
#     print(f"\n{'=' * 70}")
#     print(f"  KG-RAG Pipeline: Extraction → Chunking → Embedding → Graph")
#     print(f"{'=' * 70}")
#     print(f"  Document     : {file_path}")
#     print(f"  Source label : {source}")
#     print(f"  LLM model    : {args.model}")
#     print(f"  Batch chunks : {args.batch}")
#     print(f"  Mode         : {args.mode}")
#     print(f"  Max chunks   : {args.max_chunks or 'all'}")
#     print(f"  Neo4j        : {args.uri if args.neo4j else 'Skipped (add --neo4j)'}")
#     print(f"\n  MODULES:")
#     print(f"  Extraction   : {'Available (PDF, DOCX, PPTX, Audio, Video)' if _HAS_EXTRACTORS else 'Not available'}")
#     print(f"  Chunking     : {'SemanticChunker' if _HAS_CHUNKER else 'Fallback (word-count)'}")
#     print(f"  Embedding    : {'HuggingFaceEmbedding' if _HAS_EMBEDDING else 'Dummy (dim=384)'}")
#     print(f"{'=' * 70}")

#     v = args.verbose

#     # ── Build shared components ───────────────────────────────────────────────
#     print("\nConnecting to Groq...")
#     try:
#         llm = LLMBackend(model=args.model, max_tokens=3000, temperature=0.0)
#         print(f"  ✓ {llm}")
#     except Exception as e:
#         print(f"  ERROR: {e}")
#         print("  Set GROQ_API_KEY in your environment or .env file.")
#         sys.exit(1)

#     embedding_fn = _embedding_fn if _HAS_EMBEDDING else _dummy_embedding_fn

#     # ── Neo4j store ───────────────────────────────────────────────────────────
#     store = None
#     if args.neo4j:
#         # print(f"\nConnecting to Neo4j at {args.uri}...")
#         # try:
#         #     store = GraphStore(**args.neo4j, embedding_dim=384)


#         #     query = "MATCH (n) DETACH DELETE n"
#         #     with self._driver.session(database=self.database) as session:
#         #     session.run(query)

#         #     print("  ✓ Neo4j connected and fully cleaned")
#         # except Exception as e:
#         #     print(f"  ERROR: {e}\n  Neo4j tests will be skipped.")
#         #     store = None
#         print(f"\nConnecting to Neo4j at {args.uri}...")
#         try:
#             store = GraphStore(
#                 uri      = args.uri,
#                 user     = args.user,
#                 password = args.password,
#                 database = "neo4j",
#                 embedding_dim = 384,
#             )
#             store.init_schema()
#             # Clear database at the start of a new test
#             print(f"  Clearing database for fresh test...")
#             try:
#                 # Delete data for this source to ensure clean state
#                 query = "MATCH (n) DETACH DELETE n"
#                 with store._driver.session(database=store.database) as session:
#                     session.run(query)
#                 print(f"  ✓ Cleared entire database (all previous nodes and relationships deleted)")

#                 # n_del, r_del = store.delete_by_source(source)
#                 # print(f"  ✓ Cleared previous data: {n_del} nodes, {r_del} relationships")
#             except Exception as e:
#                 print(f"  WARNING: Could not clear previous data: {e}")
#             print("  ✓ Neo4j connected")
#         except Exception as e:
#             print(f"  ERROR: {e}")
#             print("  Neo4j tests will be skipped.")
#             store = None

#     # ═══════════════════════════════════════════════════════════════════════
#     # TEST SEQUENCE
#     # ═══════════════════════════════════════════════════════════════════════

#     # Step 1 — File extraction
#     text = test_file_extraction(file_path, v)
#     if not text:
#         print("ERROR: No text extracted. Cannot continue.")
#         sys.exit(1)

#     # Step 2 — Chunking
#     chunks = test_chunking(text, source, args.max_chunks, v)
#     if not chunks:
#         print("ERROR: No chunks produced. Cannot continue.")
#         sys.exit(1)

#     # Step 3 — LLM connection
#     test_llm_connection(llm, v)

#     # Step 4 — CombinedExtractor init
#     ext = test_combined_extractor_init(llm, embedding_fn, args.batch, args.mode, v)

#     # Step 5 — Node extraction
#     nodes, node_id_map = test_node_extraction(ext, chunks, v)

#     # Warn if node extraction failed
#     if len(nodes) == 0:
#         print(f"\n  ⚠ WARNING: No nodes extracted. This may be due to:")
#         print(f"    - Groq API rate limit (reduce --batch size)")
#         print(f"    - API connection issues")
#         print(f"    - LLM parsing errors")
#         print(f"\n  To avoid rate limits, try: --batch 1 --max-chunks 10")

#     # Step 6 — Relationship extraction
#     rels = test_relationship_extraction(ext, chunks, nodes, node_id_map, v)

#     # Step 7 — GraphStore (if Neo4j available)
#     if store is not None:
#         test_graph_store(store, nodes, rels, source, v)

#     # Step 8 — Full Pipeline end-to-end
#     if store is not None:
#         test_pipeline_end_to_end(
#             llm          = llm,
#             embedding_fn = embedding_fn,
#             store        = store,
#             chunks       = chunks,
#             batch_chunks = args.batch,
#             mode         = args.mode,
#             source       = source,
#             verbose      = v,
#         )
#     else:
#         header("Steps 7–8 — Neo4j tests SKIPPED")
#         print("  Run with: python test_flow.py --file <path> --neo4j --password <pwd>")

#     # ── Cleanup ───────────────────────────────────────────────────────────────
#     if store is not None:
#         store.close()

#     # ── Summary ───────────────────────────────────────────────────────────────
#     failed = _total - _passed
#     print(f"\n{'=' * 70}")
#     if failed == 0:
#         print(f"  ALL {_total} tests passed ✓")
#     else:
#         print(f"  {_passed} / {_total} passed   ({failed} FAILED ✗)")
#     print(f"{'=' * 70}\n")
#     sys.exit(0 if failed == 0 else 1)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# """
# test_pipeline.py — Full KG-RAG Pipeline test suite
# ====================================================
# Tests the consolidated Pipeline.py which uses CombinedExtractor internally.
# All components (node extraction + relationship extraction) are handled by the
# single Pipeline entry point — no direct use of NodeExtractor / RelationshipExtractor.

# Run modes
# ---------
#     # Interactive — prompts for file path and Neo4j password
#     python test_pipeline.py

#     # Fully specified
#     python test_pipeline.py \\
#         --file   /path/to/lecture.pdf \\
#         --source lecture_01 \\
#         --neo4j  \\
#         --password neo4j1234 \\
#         --model  llama-3.3-70b-versatile \\
#         -v

#     # Extraction + graph tests only (no Neo4j)
#     python test_pipeline.py --file /path/to/notes.pdf

# CLI flags
# ---------
#     --file        Path to document to ingest (PDF, TXT, DOCX, PPTX, …)
#     --source      Logical source name stored in the graph (default: file stem)
#     --neo4j       Enable Neo4j write & pipeline end-to-end tests
#     --uri         Neo4j URI           (default: neo4j://127.0.0.1:7687)
#     --user        Neo4j username      (default: neo4j)
#     --password    Neo4j password      (default: neo4j1234)
#     --model       Groq model          (default: llama-3.3-70b-versatile)
#     --max-chunks  Limit chunks used   (default: all)
#     --batch       Chunks per combined LLM call (default: 5)
#     --mode        constrained | unconstrained  (default: constrained)
#     -v / --verbose  Verbose output

# Environment variables
# ---------------------
#     GROQ_API_KEY   — Groq API key (required)
#     NEO4J_URI      — neo4j://127.0.0.1:7687
#     NEO4J_USER     — neo4j
#     NEO4J_PASSWORD — neo4j1234
# """
# from __future__ import annotations

# import argparse
# import os
# import sys
# import time
# import warnings
# from pathlib import Path
# from typing import List, Optional

# import numpy as np

# # ── .env loading ──────────────────────────────────────────────────────────────
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# # ── Path setup ────────────────────────────────────────────────────────────────
# _HERE = Path(__file__).resolve().parent
# _ROOT = _HERE.parent
# for _p in [str(_HERE), str(_ROOT)]:
#     if _p not in sys.path:
#         sys.path.insert(0, _p)


# # =============================================================================
# # Module imports — fail fast with clear messages
# # =============================================================================

# # ── Chunk class ───────────────────────────────────────────────────────────────
# try:
#     from chunking.chunk_base import Chunk
#     print("✓ Chunk class loaded  (chunking.chunk_base)")
# except ImportError as e:
#     print(f"ERROR: Could not import Chunk from chunking.chunk_base: {e}")
#     sys.exit(1)

# # ── Chunking module ───────────────────────────────────────────────────────────
# _HAS_CHUNKER = False
# _chunker_obj = None
# try:
#     from chunking import (
#         SemanticChunker,
#         RecursiveChunker,
#         SlidingWindowChunker,
#         SentenceChunker,
#         FixedSizeChunker,
#         ParagraphChunker,
#     )
#     _chunker_obj = SemanticChunker()
#     _HAS_CHUNKER = True
#     print("✓ Chunking module loaded with SemanticChunker (default)")
# except ImportError as e:
#     print(f"WARNING: Chunking module import failed ({e})")
#     print("         Will use built-in word-count fallback chunker.")

# # ── Embeddings module ─────────────────────────────────────────────────────────
# _HAS_EMBEDDING = False
# _embedding_fn  = None
# try:
#     from embeddings import HuggingFaceEmbedding
#     embedding_model = HuggingFaceEmbedding(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         device=None,
#     )
#     _embedding_fn  = embedding_model.encode
#     _HAS_EMBEDDING = True
#     print("✓ HuggingFaceEmbedding loaded (embeddings.HuggingFaceEmbedding)")
#     print("   Model: sentence-transformers/all-MiniLM-L6-v2 (384 dim)")
# except ImportError as e:
#     print(f"WARNING: embeddings module not found ({e}) — using deterministic dummy (dim=384).")

# # ── Extractors module ─────────────────────────────────────────────────────────
# _HAS_EXTRACTORS = False
# _extractor_wrapper = None
# try:
#     from extractors.extractors_wrapper import SimpleExtractorWrapper
#     from extractors import get_type_for_extension
#     _extractor_wrapper = SimpleExtractorWrapper()
#     _HAS_EXTRACTORS = True
#     print("✓ Extractors module loaded  (extractors.*)")
#     print("   Supported: PDF, DOCX, PPTX, Audio, Video, URL")
# except ImportError as e:
#     print(f"WARNING: Extractors module not fully available ({e})")

# # ── Graph module ──────────────────────────────────────────────────────────────
# try:
#     from graph.llm_backend import LLMBackend
#     from graph.graph_store  import GraphStore
#     from graph.base         import ExtractedNode, ExtractedRelationship
#     print("✓ Graph module loaded  (graph.*)")
# except ImportError as e:
#     print(f"ERROR: Could not import graph module: {e}")
#     sys.exit(1)

# # ── CombinedExtractor ─────────────────────────────────────────────────────────
# try:
#     from graph.node_relation_extractor import CombinedExtractor, ALLOWED_RELATIONS
#     print("✓ CombinedExtractor loaded  (graph.node_relation_extractor)")
# except ImportError as e:
#     print(f"ERROR: Could not import CombinedExtractor: {e}")
#     print("       Make sure node_relation_extractor.py is inside your graph/ package.")
#     sys.exit(1)

# # ── Pipeline ──────────────────────────────────────────────────────────────────
# try:
#     from Pipeline import Pipeline, PipelineStats
#     print("✓ Pipeline loaded  (Pipeline)")
# except ImportError as e:
#     print(f"ERROR: Could not import Pipeline: {e}")
#     sys.exit(1)


# # =============================================================================
# # Test counters
# # =============================================================================
# _total  = 0
# _passed = 0


# def run_test(description: str, condition: bool, verbose: bool = False, detail: str = "") -> None:
#     global _total, _passed
#     _total += 1
#     icon = "✓ PASS" if condition else "✗ FAIL"
#     if condition:
#         _passed += 1
#     print(f"  {icon}  {description}")
#     if verbose and detail:
#         print(f"         → {detail}")


# def header(title: str) -> None:
#     print(f"\n{'─' * 70}")
#     print(f"  {title}")
#     print(f"{'─' * 70}")


# # =============================================================================
# # Embedding fallback
# # =============================================================================

# def _dummy_embedding_fn(texts: List[str]) -> np.ndarray:
#     """Deterministic dummy embedding (used when embedding module unavailable)."""
#     rng = np.random.default_rng(seed=42)
#     return rng.random((len(texts), 384)).astype(np.float32)


# # =============================================================================
# # File extraction and chunking
# # =============================================================================

# def extract_text_from_file(file_path: Path) -> str:
#     """Extract text from various file formats; falls back to plain-text read."""
#     suffix = file_path.suffix.lower()

#     if suffix == ".txt":
#         return file_path.read_text(encoding="utf-8", errors="replace")

#     if not _HAS_EXTRACTORS:
#         print("WARNING: Extractors module not available. Reading as plain text.")
#         return file_path.read_text(encoding="utf-8", errors="replace")

#     try:
#         if suffix == ".pdf":
#             extracted = _extractor_wrapper.extract_pdf(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".docx", ".doc"]:
#             extracted = _extractor_wrapper.extract_docx(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".pptx", ".ppt"]:
#             extracted = _extractor_wrapper.extract_pptx(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
#             extracted = _extractor_wrapper.extract_audio(str(file_path))
#             if extracted:
#                 return extracted
#         elif suffix in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]:
#             extracted = _extractor_wrapper.extract_video(str(file_path))
#             if extracted:
#                 return extracted
#     except Exception as e:
#         print(f"WARNING: Extraction failed for {suffix}: {e}")

#     print(f"WARNING: Could not extract {suffix}. Reading as plain text.")
#     return file_path.read_text(encoding="utf-8", errors="replace")


# def _fallback_chunker(text: str, source: str, chunk_size: int = 400) -> List[Chunk]:
#     """Built-in word-count chunker — fallback when no chunker module is found."""
#     words   = text.split()
#     overlap = chunk_size // 5
#     chunks  = []
#     idx     = 0
#     chunk_id = 0
#     while idx < len(words):
#         window   = words[idx: idx + chunk_size]
#         text_out = " ".join(window)
#         start    = len(" ".join(words[:idx]))
#         end      = start + len(text_out)
#         chunks.append(Chunk(
#             text       = text_out,
#             chunk_id   = chunk_id,
#             start_char = start,
#             end_char   = end,
#             metadata   = {"source": source},
#         ))
#         chunk_id += 1
#         idx      += chunk_size - overlap
#     return chunks


# def chunk_text(text: str, source: str) -> List[Chunk]:
#     """Chunk using the chunking module if available; else use the built-in fallback."""
#     if _HAS_CHUNKER and _chunker_obj is not None:
#         if hasattr(_chunker_obj, "chunk"):
#             chunks = _chunker_obj.chunk(text)
#             for c in chunks:
#                 c.metadata.setdefault("source", source)
#             return chunks
#         if callable(_chunker_obj):
#             chunks = _chunker_obj(text)
#             for c in chunks:
#                 c.metadata.setdefault("source", source)
#             return chunks
#         print("WARNING: Chunker found but no known method (chunk / __call__).")
#     return _fallback_chunker(text, source)


# # =============================================================================
# # Test functions
# # =============================================================================

# # ── Step 1 — File extraction ──────────────────────────────────────────────────

# def test_file_extraction(file_path: Path, verbose: bool) -> Optional[str]:
#     header(f"Step 1 — File extraction  [{file_path.name}]")

#     run_test("File exists and is readable",
#              file_path.exists() and file_path.is_file(), verbose)

#     try:
#         text = extract_text_from_file(file_path)
#         run_test("Text extracted successfully",
#                  len(text) > 0, verbose, f"{len(text)} characters extracted")
#         if verbose:
#             preview = text[:300].replace("\n", " ")
#             print(f"\n  Text preview:")
#             print(f"  {preview}{'...' if len(text) > 300 else ''}\n")
#         return text
#     except Exception as e:
#         run_test("Text extraction successful", False, verbose, str(e))
#         return None


# # ── Step 2 — Chunking ─────────────────────────────────────────────────────────

# def test_chunking(
#     text: str,
#     source: str,
#     max_chunks: Optional[int],
#     verbose: bool,
# ) -> List[Chunk]:
#     header("Step 2 — Document chunking")

#     try:
#         chunks = chunk_text(text, source)
#         run_test("chunk_text() returns a non-empty list",
#                  len(chunks) > 0, verbose, f"{len(chunks)} raw chunk(s) produced")

#         if chunks:
#             c0 = chunks[0]
#             run_test("Chunk has .text attribute",     hasattr(c0, "text"),     verbose)
#             run_test("Chunk has .chunk_id attribute", hasattr(c0, "chunk_id"), verbose)
#             run_test("Chunk has .metadata dict",
#                      isinstance(getattr(c0, "metadata", None), dict), verbose)
#             run_test('metadata["source"] is set',
#                      c0.metadata.get("source") == source, verbose,
#                      f"source = {c0.metadata.get('source')!r}")

#         if max_chunks:
#             chunks = chunks[:max_chunks]
#             print(f"  (limited to {max_chunks} chunks for this run)")

#         if verbose and chunks:
#             preview = chunks[0].text[:300].replace("\n", " ")
#             print(f"\n  Preview of chunk 0:")
#             print(f"  {preview}{'...' if len(chunks[0].text) > 300 else ''}\n")

#         return chunks
#     except Exception as e:
#         run_test("Chunking completed", False, verbose, str(e))
#         return []


# # ── Step 3 — LLM backend ──────────────────────────────────────────────────────

# def test_llm_connection(llm: LLMBackend, verbose: bool) -> None:
#     header("Step 3 — LLM Backend connection")

#     run_test("LLMBackend instantiated", llm is not None, verbose)

#     print("  Sending test prompt to Groq...")
#     try:
#         resp = llm.generate("Reply with exactly: OK")
#         ok   = bool(resp and len(resp.strip()) > 0)
#         run_test("Groq API responds", ok, verbose,
#                  f"response: {resp.strip()[:80]!r}")
#     except Exception as e:
#         run_test("Groq API responds", False, verbose, str(e))


# # ── Step 4 — CombinedExtractor init ──────────────────────────────────────────

# def test_combined_extractor_init(
#     llm          : LLMBackend,
#     embedding_fn,
#     batch_chunks : int,
#     mode         : str,
#     verbose      : bool,
# ) -> CombinedExtractor:
#     header("Step 4 — CombinedExtractor initialisation")

#     ext = CombinedExtractor(
#         llm                  = llm,
#         embedding_fn         = embedding_fn,
#         mode                 = mode,
#         batch_chunks         = batch_chunks,
#         confidence_threshold = 0.6,
#         max_entity_pairs     = 20,
#     )

#     run_test("CombinedExtractor instantiated",    ext is not None,              verbose)
#     run_test("Correct batch_chunks",              ext.batch_chunks == batch_chunks, verbose,
#              f"batch_chunks = {ext.batch_chunks}")
#     run_test("Correct mode",                      ext.mode == mode,             verbose)
#     run_test("Allowed types includes Theorem",    "Theorem"  in ext.allowed_types, verbose)
#     run_test("Allowed types includes Operator",   "Operator" in ext.allowed_types, verbose)
#     run_test("Allowed types includes Distribution",
#              "Distribution" in ext.allowed_types, verbose)
#     run_test("Allowed types includes Space",      "Space"    in ext.allowed_types, verbose)
#     run_test("ALLOWED_RELATIONS includes DEFINED_BY",
#              "DEFINED_BY"   in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes IMPLIES",
#              "IMPLIES"      in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes PROOF_OF",
#              "PROOF_OF"     in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes CONVERGES_TO",
#              "CONVERGES_TO" in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes BOUNDED_BY",
#              "BOUNDED_BY"   in ALLOWED_RELATIONS, verbose)
#     run_test("ALLOWED_RELATIONS includes MAPS_TO",
#              "MAPS_TO"      in ALLOWED_RELATIONS, verbose)
#     run_test("_pending_rels starts empty",
#              len(ext._pending_rels) == 0, verbose)
#     if verbose:
#         print(f"\n  {ext}\n")
#     return ext


# # ── Step 5 — Node extraction (CombinedExtractor) ─────────────────────────────

# def test_node_extraction(
#     ext     : CombinedExtractor,
#     chunks  : list,
#     verbose : bool,
# ) -> tuple:
#     header(f"Step 5 — Node extraction  ({len(chunks)} chunks, batch_size={ext.batch_chunks})")

#     t0 = time.time()
#     try:
#         nodes = ext.extract_from_chunks(chunks, show_progress=True)
#         elapsed = time.time() - t0
#         success = isinstance(nodes, list)
#         run_test("extract_from_chunks() returns a list",
#                  success, verbose,
#                  f"{len(nodes)} node(s) extracted in {elapsed:.1f}s"
#                  if success else "extraction produced no nodes")
#     except Exception as e:
#         run_test("extract_from_chunks() (nodes) completed", False, verbose, str(e))
#         return [], {}

#     if nodes:
#         n0 = nodes[0]
#         run_test("Each node has node_id",
#                  bool(n0.node_id), verbose)
#         run_test("Each node has name",
#                  bool(n0.name), verbose)
#         run_test("Each node has entity_type",
#                  bool(n0.entity_type), verbose)
#         run_test("Each node has description",
#                  bool(n0.description), verbose)
#         run_test("Each node has source",
#                  bool(n0.source), verbose)
#         run_test("Each node has embedding",
#                  n0.embedding is not None and isinstance(n0.embedding, np.ndarray),
#                  verbose, f"shape = {np.shape(n0.embedding)}")
#         run_test("No duplicate node_ids",
#                  len({n.node_id for n in nodes}) == len(nodes), verbose)
#         run_test("_pending_rels populated after node extraction",
#                  len(ext._pending_rels) > 0, verbose,
#                  f"{len(ext._pending_rels)} chunk(s) with cached rels")
#     else:
#         print(f"\n  ⚠ WARNING: No nodes extracted. Possible causes:")
#         print(f"    - Groq API rate limit  (try --batch 1 --max-chunks 10)")
#         print(f"    - API connection issues")
#         print(f"    - LLM JSON parsing errors")

#     # Build node_id_map
#     node_id_map: dict = {}
#     for n in nodes:
#         node_id_map[n.name.lower()] = n.node_id
#         for alias in n.aliases:
#             node_id_map.setdefault(alias.lower(), n.node_id)

#     if verbose and nodes:
#         from collections import Counter
#         print(f"\n  Entity type distribution:")
#         counts = Counter(n.entity_type for n in nodes)
#         for etype, cnt in counts.most_common():
#             print(f"    {etype:20s}: {cnt}")
#         print(f"\n  Sample nodes:")
#         for n in nodes[:5]:
#             print(f"    [{n.entity_type}] {n.name}")
#             print(f"           {n.description[:100]}")
#         print()

#     return nodes, node_id_map


# # ── Step 6 — Relationship extraction (from cache) ────────────────────────────

# def test_relationship_extraction(
#     ext          : CombinedExtractor,
#     chunks       : list,
#     nodes        : list,
#     node_id_map  : dict,
#     verbose      : bool,
# ) -> list:
#     header(f"Step 6 — Relationship extraction  ({len(chunks)} chunks, reads cached rels)")

#     # Build nodes_per_chunk index
#     nodes_per_chunk: dict = {}
#     for idx, chunk in enumerate(chunks):
#         chunk_id = str(getattr(chunk, "chunk_id", idx))
#         nodes_per_chunk[idx] = [n for n in nodes if n.source_chunk == chunk_id]

#     t0 = time.time()
#     try:
#         rels = ext.extract_from_chunks(
#             chunks,
#             nodes_per_chunk = nodes_per_chunk,
#             node_id_map     = node_id_map,
#             show_progress   = True,
#         )
#     except Exception as e:
#         run_test("extract_from_chunks() (rels) completed", False, verbose, str(e))
#         return []

#     elapsed = time.time() - t0
#     run_test("Returns a list of ExtractedRelationship",
#              isinstance(rels, list), verbose, f"{len(rels)} rel(s) in {elapsed:.1f}s")

#     if rels:
#         r0 = rels[0]
#         run_test("Each rel has source_id",
#                  bool(r0.source_id),    verbose)
#         run_test("Each rel has target_id",
#                  bool(r0.target_id),    verbose)
#         run_test("Each rel has relation_type",
#                  bool(r0.relation_type), verbose)
#         run_test("Each rel has description",
#                  bool(r0.description),  verbose)
#         run_test("Each rel confidence in [0, 1]",
#                  0 <= r0.confidence <= 1, verbose,
#                  f"confidence = {r0.confidence}")
#         if ext.mode == "constrained":
#             all_allowed = all(r.relation_type in ALLOWED_RELATIONS for r in rels)
#             run_test("All relation_types in ALLOWED_RELATIONS (constrained mode)",
#                      all_allowed, verbose)
#         run_test("Relationships deduplicated (no exact duplicates)",
#                  len(rels) == len({(r.source_id, r.target_id, r.relation_type)
#                                    for r in rels}),
#                  verbose)

#     if verbose and rels:
#         from collections import Counter
#         print(f"\n  Relation type distribution:")
#         rcounts = Counter(r.relation_type for r in rels)
#         for rtype, cnt in rcounts.most_common(10):
#             print(f"    {rtype:30s}: {cnt}")
#         print(f"\n  Sample relationships:")
#         for r in rels[:5]:
#             print(f"    {r.source_name} —[{r.relation_type}]→ {r.target_name}")
#             print(f"       conf={r.confidence:.2f}  {r.description[:80]}")
#         print()

#     return rels


# # ── Step 7 — GraphStore direct writes ────────────────────────────────────────

# def test_graph_store(
#     store   : GraphStore,
#     nodes   : list,
#     rels    : list,
#     source  : str,
#     verbose : bool,
# ) -> None:
#     header("Step 7 — GraphStore Neo4j write & verify")

#     try:
#         store.init_schema()
#         run_test("init_schema() runs without error", True, verbose)
#     except Exception as e:
#         run_test("init_schema() runs without error", False, verbose, str(e))
#         return

#     # Clear previous data for this source
#     try:
#         n_del, r_del = store.delete_by_source(source)
#         run_test(f"Cleared previous data for source '{source}'", True, verbose,
#                  f"Deleted {n_del} nodes, {r_del} relationships")
#     except Exception as e:
#         print(f"  WARNING: Could not clear previous data: {e}")

#     # Upsert nodes
#     try:
#         n_written = store.upsert_nodes(nodes)
#         run_test("upsert_nodes() succeeds",
#                  n_written >= 0, verbose, f"{n_written} node(s) written")
#     except Exception as e:
#         run_test("upsert_nodes() succeeds", False, verbose, str(e))
#         return

#     # Upsert relationships
#     try:
#         r_written = store.upsert_relationships(rels)
#         run_test("upsert_relationships() succeeds",
#                  r_written >= 0, verbose, f"{r_written} rel(s) written")
#     except Exception as e:
#         run_test("upsert_relationships() succeeds", False, verbose, str(e))

#     # Verify counts
#     try:
#         n_count = store.count_nodes()
#         r_count = store.count_relationships()
#         run_test("Graph has nodes after write",
#                  n_count > 0 or len(nodes) == 0, verbose,
#                  f"{n_count} total nodes (expected {len(nodes)})")
#         run_test("Graph has rels after write",
#                  r_count > 0 or len(rels) == 0, verbose,
#                  f"{r_count} total rels (expected {len(rels)})")
#     except Exception as e:
#         run_test("count_nodes / count_relationships", False, verbose, str(e))


# # ── Step 8 — Pipeline default constructor ────────────────────────────────────

# def test_pipeline_default_constructor(
#     llm          : LLMBackend,
#     embedding_fn,
#     batch_chunks : int,
#     mode         : str,
#     verbose      : bool,
# ) -> None:
#     """
#     Tests that Pipeline() raises correctly when embedding_fn is omitted,
#     and constructs successfully when it is provided.
#     NOTE: This test does NOT call .run() — it only validates construction.
#           The full end-to-end run is in test_pipeline_end_to_end().
#     """
#     header("Step 8 — Pipeline default constructor validation")

#     # embedding_fn=None must raise ValueError
#     try:
#         _ = Pipeline(
#             neo4j_uri      = "neo4j://127.0.0.1:7687",
#             groq_api_key   = "dummy",
#             embedding_fn   = None,
#         )
#         run_test("Pipeline() raises ValueError when embedding_fn=None", False, verbose,
#                  "No exception raised — expected ValueError")
#     except ValueError:
#         run_test("Pipeline() raises ValueError when embedding_fn=None", True, verbose)
#     except Exception as e:
#         run_test("Pipeline() raises ValueError when embedding_fn=None", False, verbose,
#                  f"Unexpected exception: {e}")

#     # PipelineStats dataclass
#     stats = PipelineStats()
#     run_test("PipelineStats() default-constructs to zero counts",
#              stats.chunks_processed == 0 and stats.nodes_extracted == 0 and
#              stats.relationships_extracted == 0,
#              verbose)
#     run_test("PipelineStats.__str__() contains 'Pipeline run complete'",
#              "Pipeline run complete" in str(stats), verbose)

#     if verbose:
#         print(f"\n  Default PipelineStats:\n{stats}\n")


# # ── Step 9 — Pipeline.from_components() end-to-end ───────────────────────────

# def test_pipeline_end_to_end(
#     llm          : LLMBackend,
#     embedding_fn,
#     store        : GraphStore,
#     chunks       : list,
#     batch_chunks : int,
#     mode         : str,
#     source       : str,
#     verbose      : bool,
# ) -> None:
#     header("Step 9 — Pipeline.from_components() end-to-end run")

#     ext = CombinedExtractor(
#         llm                  = llm,
#         embedding_fn         = embedding_fn,
#         mode                 = mode,
#         batch_chunks         = batch_chunks,
#         confidence_threshold = 0.6,
#     )

#     # Build pipeline
#     try:
#         pipeline = Pipeline.from_components(
#             node_extractor         = ext,
#             relationship_extractor = ext,    # same instance — required
#             graph_store            = store,
#             schema_already_exists  = True,
#             show_progress          = verbose,
#         )
#         run_test("Pipeline.from_components() succeeds", True, verbose)
#     except Exception as e:
#         run_test("Pipeline.from_components() succeeds", False, verbose, str(e))
#         return

#     run_test("Pipeline uses the same CombinedExtractor for both roles",
#              pipeline._node_extractor is pipeline._relationship_extractor, verbose)

#     # Clear previous data
#     try:
#         store.delete_by_source(source)
#     except Exception:
#         pass

#     # run() on empty list must warn and return default stats
#     with warnings.catch_warnings(record=True) as caught:
#         warnings.simplefilter("always")
#         empty_stats = pipeline.run([])
#     run_test("pipeline.run([]) returns default PipelineStats",
#              isinstance(empty_stats, PipelineStats) and empty_stats.chunks_processed == 0,
#              verbose)
#     run_test("pipeline.run([]) issues a UserWarning",
#              any("empty" in str(w.message).lower() for w in caught), verbose)

#     # Full run
#     t0 = time.time()
#     try:
#         stats = pipeline.run(chunks)
#         elapsed = time.time() - t0

#         run_test("pipeline.run() returns PipelineStats",
#                  isinstance(stats, PipelineStats), verbose)
#         run_test("chunks_processed == len(chunks)",
#                  stats.chunks_processed == len(chunks), verbose,
#                  f"{stats.chunks_processed}")
#         run_test("nodes_extracted > 0",
#                  stats.nodes_extracted > 0, verbose,
#                  f"{stats.nodes_extracted}")
#         run_test("nodes_written > 0",
#                  stats.nodes_written > 0, verbose,
#                  f"{stats.nodes_written}")
#         run_test("relationships_extracted >= 0",
#                  stats.relationships_extracted >= 0, verbose,
#                  f"{stats.relationships_extracted}")
#         run_test("relationships_written >= 0",
#                  stats.relationships_written >= 0, verbose,
#                  f"{stats.relationships_written}")
#         run_test("elapsed_seconds > 0",
#                  stats.elapsed_seconds > 0, verbose,
#                  f"{stats.elapsed_seconds:.1f}s")
#         run_test("nodes_written <= nodes_extracted",
#                  stats.nodes_written <= stats.nodes_extracted + 1, verbose)

#         print(f"\n  Full pipeline run completed in {elapsed:.1f}s")
#         print(stats)
#     except Exception as e:
#         run_test("pipeline.run() completes without fatal error", False, verbose, str(e))
#         return

#     # graph_stats()
#     # NOTE: Do NOT use 'with pipeline:' here because it closes the driver.
#     # The store needs to stay open for the ingest_chunk test below.
#     gstats = pipeline.graph_stats()
#     run_test("graph_stats() returns dict with 'nodes' key",
#              isinstance(gstats.get("nodes"), int), verbose, f"{gstats}")
#     run_test("graph_stats() returns dict with 'relationships' key",
#              isinstance(gstats.get("relationships"), int), verbose)

#     # ingest_chunk() — incremental test
#     # header("Step 9b — ingest_chunk() incremental ingestion")
#     # single_ext = CombinedExtractor(
#     #     llm          = llm,
#     #     embedding_fn = embedding_fn,
#     #     mode         = mode,
#     #     batch_chunks = 1,
#     # )
#     # single_pipeline = Pipeline.from_components(
#     #     node_extractor         = single_ext,
#     #     relationship_extractor = single_ext,
#     #     graph_store            = store,
#     #     schema_already_exists  = True,
#     #     show_progress          = False,
#     # )
#     # try:
#     #     inc_stats = single_pipeline.ingest_chunk(chunks[0])
#     #     run_test("ingest_chunk() returns PipelineStats",
#     #              isinstance(inc_stats, PipelineStats), verbose)
#     #     run_test("ingest_chunk() processes exactly 1 chunk",
#     #              inc_stats.chunks_processed == 1, verbose,
#     #              f"chunks_processed = {inc_stats.chunks_processed}")
#     # except Exception as e:
#     #     run_test("ingest_chunk() completes", False, verbose, str(e))


# # ── Step 10 — Pipeline context manager & repr ────────────────────────────────

# def test_pipeline_context_manager(
#     llm          : LLMBackend,
#     embedding_fn,
#     store        : GraphStore,
#     batch_chunks : int,
#     mode         : str,
#     verbose      : bool,
# ) -> None:
#     header("Step 10 — Pipeline context-manager and __repr__")

#     ext = CombinedExtractor(
#         llm          = llm,
#         embedding_fn = embedding_fn,
#         mode         = mode,
#         batch_chunks = batch_chunks,
#     )
#     pipeline = Pipeline.from_components(
#         node_extractor         = ext,
#         relationship_extractor = ext,
#         graph_store            = store,
#         schema_already_exists  = True,
#         show_progress          = False,
#     )

#     run_test("Pipeline() usable as context manager (__enter__/__exit__)",
#              hasattr(pipeline, "__enter__") and hasattr(pipeline, "__exit__"), verbose)

#     with pipeline as p:
#         run_test("Context manager yields the Pipeline instance",
#                  p is pipeline, verbose)

#     repr_str = repr(pipeline)
#     run_test("Pipeline.__repr__() contains 'Pipeline('",
#              "Pipeline(" in repr_str, verbose, repr_str[:120])

#     if verbose:
#         print(f"\n  repr:\n  {repr_str}\n")


# # =============================================================================
# # Main
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="KG-RAG Pipeline test suite — tests Pipeline.py with CombinedExtractor"
#     )
#     parser.add_argument("--file",       default=None,
#                         help="Path to the document to ingest")
#     parser.add_argument("--source",     default=None,
#                         help="Logical source name (defaults to file stem)")
#     parser.add_argument("--neo4j",      action="store_true",
#                         help="Enable Neo4j write & end-to-end pipeline tests")
#     parser.add_argument("--uri",        default=os.environ.get("NEO4J_URI",      "neo4j://127.0.0.1:7687"))
#     parser.add_argument("--user",       default=os.environ.get("NEO4J_USER",     "neo4j"))
#     parser.add_argument("--password",   default=os.environ.get("NEO4J_PASSWORD", "neo4j1234"))
#     parser.add_argument("--model",      default="llama-3.3-70b-versatile")
#     parser.add_argument("--max-chunks", type=int, default=None,
#                         help="Limit number of chunks (useful for quick tests)")
#     parser.add_argument("--batch",      type=int, default=2,
#                         help="Chunks per combined LLM call (default 2)")
#     parser.add_argument("--mode",       default="constrained",
#                         choices=["constrained", "unconstrained"])
#     parser.add_argument("-v", "--verbose", action="store_true")
#     args = parser.parse_args()

#     # ── Prompt for file path ──────────────────────────────────────────────────
#     file_path_str = args.file
#     if not file_path_str:
#         print()
#         file_path_str = input("Enter path to the document to ingest: ").strip()
#         if not file_path_str:
#             print("No file path provided. Exiting.")
#             sys.exit(1)

#     file_path = Path(file_path_str).expanduser().resolve()
#     if not file_path.exists():
#         print(f"ERROR: File not found: {file_path}")
#         sys.exit(1)

#     source = args.source or file_path.stem

#     # ── Banner ────────────────────────────────────────────────────────────────
#     print(f"\n{'=' * 70}")
#     print(f"  KG-RAG Pipeline Test Suite")
#     print(f"  Pipeline.py  ×  CombinedExtractor  ×  GraphStore")
#     print(f"{'=' * 70}")
#     print(f"  Document     : {file_path}")
#     print(f"  Source label : {source}")
#     print(f"  LLM model    : {args.model}")
#     print(f"  Batch chunks : {args.batch}")
#     print(f"  Mode         : {args.mode}")
#     print(f"  Max chunks   : {args.max_chunks or 'all'}")
#     print(f"  Neo4j        : {args.uri if args.neo4j else 'Skipped (add --neo4j to enable)'}")
#     print(f"\n  MODULES:")
#     print(f"  Extraction : {'Available (PDF, DOCX, PPTX, Audio, Video)' if _HAS_EXTRACTORS else 'Not available (plain-text fallback)'}")
#     print(f"  Chunking   : {'SemanticChunker' if _HAS_CHUNKER else 'Fallback (word-count)'}")
#     print(f"  Embedding  : {'HuggingFaceEmbedding (384 dim)' if _HAS_EMBEDDING else 'Dummy (384 dim, seed=42)'}")
#     print(f"{'=' * 70}")

#     v = args.verbose

#     # ── LLM ──────────────────────────────────────────────────────────────────
#     print("\nConnecting to Groq...")
#     try:
#         llm = LLMBackend(model=args.model, max_tokens=3000, temperature=0.0)
#         print(f"  ✓ {llm}")
#     except Exception as e:
#         print(f"  ERROR: {e}")
#         print("  Set GROQ_API_KEY in your environment or .env file.")
#         sys.exit(1)

#     embedding_fn = _embedding_fn if _HAS_EMBEDDING else _dummy_embedding_fn

#     # ── Neo4j ─────────────────────────────────────────────────────────────────
#     store = None
#     if args.neo4j:
#         print(f"\nConnecting to Neo4j at {args.uri}...")
#         try:
#             store = GraphStore(
#                 uri           = args.uri,
#                 user          = args.user,
#                 password      = args.password,
#                 database      = "neo4j",
#                 embedding_dim = 384,
#             )
#             store.init_schema()
#             print("  Clearing database for a fresh test run...")
#             try:
#                 with store._driver.session(database=store.database) as session:
#                     session.run("MATCH (n) DETACH DELETE n")
#                 print("  ✓ Database cleared (all previous nodes and relationships deleted)")
#             except Exception as e:
#                 print(f"  WARNING: Could not clear database: {e}")
#             print("  ✓ Neo4j connected")
#         except Exception as e:
#             print(f"  ERROR: {e}")
#             print("  Neo4j tests will be skipped.")
#             store = None

#     # ═════════════════════════════════════════════════════════════════════════
#     # TEST SEQUENCE
#     # ═════════════════════════════════════════════════════════════════════════

#     # Step 1 — File extraction
#     text = test_file_extraction(file_path, v)
#     if not text:
#         print("ERROR: No text extracted. Cannot continue.")
#         sys.exit(1)

#     # Step 2 — Chunking
#     chunks = test_chunking(text, source, args.max_chunks, v)
#     if not chunks:
#         print("ERROR: No chunks produced. Cannot continue.")
#         sys.exit(1)

#     # Step 3 — LLM connection
#     test_llm_connection(llm, v)

#     # Step 4 — CombinedExtractor init
#     ext = test_combined_extractor_init(llm, embedding_fn, args.batch, args.mode, v)

#     # Step 5 — Node extraction
#     nodes, node_id_map = test_node_extraction(ext, chunks, v)

#     # Step 6 — Relationship extraction (reads _pending_rels cache)
#     rels = test_relationship_extraction(ext, chunks, nodes, node_id_map, v)

#     # Step 7 — GraphStore direct write (if Neo4j available)
#     if store is not None:
#         test_graph_store(store, nodes, rels, source, v)

#     # Step 8 — Pipeline default constructor validation
#     test_pipeline_default_constructor(llm, embedding_fn, args.batch, args.mode, v)

#     # Step 9 — Pipeline.from_components() end-to-end (needs Neo4j)
#     if store is not None:
#         test_pipeline_end_to_end(
#             llm          = llm,
#             embedding_fn = embedding_fn,
#             store        = store,
#             chunks       = chunks,
#             batch_chunks = args.batch,
#             mode         = args.mode,
#             source       = source,
#             verbose      = v,
#         )
#         # Step 10 — Context manager & repr
#         test_pipeline_context_manager(llm, embedding_fn, store, args.batch, args.mode, v)
#     else:
#         header("Steps 7 / 9 / 10 — Neo4j tests SKIPPED")
#         print("  Run with: python test_pipeline.py --file <path> --neo4j --password <pwd>")

#     # ── Cleanup ───────────────────────────────────────────────────────────────
#     if store is not None:
#         store.close()

#     # ── Summary ───────────────────────────────────────────────────────────────
#     failed = _total - _passed
#     print(f"\n{'=' * 70}")
#     if failed == 0:
#         print(f"  ALL {_total} tests passed ✓")
#     else:
#         print(f"  {_passed} / {_total} passed   ({failed} FAILED ✗)")
#     print(f"{'=' * 70}\n")
#     sys.exit(0 if failed == 0 else 1)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
test_pipeline.py — Full KG-RAG Pipeline test suite (OPTIMIZED)

✔ Removed:
    - Step 5: Node extraction
    - Step 6: Relationship extraction

✔ Now:
    - Only ONE LLM call (inside pipeline.run)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── .env loading ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Path setup ────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_HERE), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# =============================================================================
# Imports
# =============================================================================

from chunking.chunk_base import Chunk
from graph.llm_backend import LLMBackend
from graph.graph_store import GraphStore
from graph.node_relation_extractor import CombinedExtractor
from Pipeline import Pipeline, PipelineStats

# =============================================================================
# Helpers
# =============================================================================

def header(title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")

# =============================================================================
# Dummy embedding fallback
# =============================================================================

def _dummy_embedding_fn(texts: List[str]) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.random((len(texts), 384)).astype(np.float32)

# =============================================================================
# Chunking (fallback)
# =============================================================================

def _fallback_chunker(text: str, source: str, chunk_size: int = 400):
    words = text.split()
    overlap = chunk_size // 5
    chunks = []
    idx = 0
    cid = 0

    while idx < len(words):
        window = words[idx: idx + chunk_size]
        txt = " ".join(window)
        chunks.append(Chunk(
            text=txt,
            chunk_id=cid,
            start_char=0,
            end_char=len(txt),
            metadata={"source": source},
        ))
        cid += 1
        idx += chunk_size - overlap

    return chunks

# =============================================================================
# File read
# =============================================================================

def extract_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="replace")

# =============================================================================
# Step 1
# =============================================================================

def step1_extract(file_path: Path):
    header("Step 1 — File extraction")
    text = extract_text(file_path)
    print(f"✓ Extracted {len(text)} characters")
    return text

# =============================================================================
# Step 2
# =============================================================================

def step2_chunk(text: str, source: str, max_chunks=None):
    header("Step 2 — Chunking")
    chunks = _fallback_chunker(text, source)

    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"✓ {len(chunks)} chunks created")
    return chunks

# =============================================================================
# Step 3
# =============================================================================

def step3_llm(llm):
    header("Step 3 — LLM check")
    resp = llm.generate("Reply with OK")
    print(f"✓ LLM response: {resp.strip()}")

# =============================================================================
# Step 4
# =============================================================================

def step4_extractor(llm, embedding_fn, batch, mode):
    header("Step 4 — CombinedExtractor init")
    ext = CombinedExtractor(
        llm=llm,
        embedding_fn=embedding_fn,
        batch_chunks=batch,
        mode=mode
    )
    print("✓ CombinedExtractor ready")
    return ext

# =============================================================================
# Step 5 (Pipeline constructor validation)
# =============================================================================

def step5_pipeline_validation(llm, embedding_fn):
    header("Step 5 — Pipeline validation")

    try:
        Pipeline(
            neo4j_uri="neo4j://127.0.0.1:7687",
            groq_api_key="dummy",
            embedding_fn=None,
        )
        print("✗ Expected error not raised")
    except ValueError:
        print("✓ Proper validation (embedding_fn required)")

    stats = PipelineStats()
    print("✓ PipelineStats OK")

# =============================================================================
# Step 6 — FULL PIPELINE (ONLY LLM CALL)
# =============================================================================

def step6_pipeline_run(llm, embedding_fn, store, chunks, batch, mode, source):
    header("Step 6 — FULL Pipeline (single LLM call)")

    ext = CombinedExtractor(
        llm=llm,
        embedding_fn=embedding_fn,
        batch_chunks=batch,
        mode=mode,
    )

    pipeline = Pipeline.from_components(
        node_extractor=ext,
        relationship_extractor=ext,
        graph_store=store,
        schema_already_exists=True,
    )

    store.delete_by_source(source)

    t0 = time.time()
    stats = pipeline.run(chunks)
    print(stats)
    print(f"✓ Completed in {time.time()-t0:.1f}s")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KG-RAG Pipeline test suite — optimized end-to-end flow"
    )
    parser.add_argument("--file", required=True,
                        help="Path to the document to ingest")
    parser.add_argument("--source", default=None,
                        help="Logical source name (defaults to file stem)")
    parser.add_argument("--neo4j", action="store_true",
                        help="Enable Neo4j write tests")
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"),
                        help="Neo4j URI")
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"),
                        help="Neo4j username")
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "neo4j1234"),
                        help="Neo4j password")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model name")
    parser.add_argument("--batch", type=int, default=2,
                        help="Chunks per combined LLM call (default 2)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit number of chunks (default: all)")
    parser.add_argument("--mode", default="constrained",
                        choices=["constrained", "unconstrained"],
                        help="Entity type extraction mode")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    source = args.source or file_path.stem

    print("\n" + "="*70)
    print("  KG-RAG PIPELINE TEST SUITE (OPTIMIZED)")
    print("="*70)
    print(f"  File:         {file_path}")
    print(f"  Source:       {source}")
    print(f"  Model:        {args.model}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Mode:         {args.mode}")
    print(f"  Max chunks:   {args.max_chunks or 'all'}")
    print(f"  Neo4j:        {args.uri if args.neo4j else 'Disabled'}")
    print("="*70 + "\n")

    llm = LLMBackend(model=args.model, max_tokens=3000)
    embedding_fn = _dummy_embedding_fn

    store = None
    if args.neo4j:
        try:
            store = GraphStore(
                uri=args.uri,
                user=args.user,
                password=args.password,
                database="neo4j",
                embedding_dim=384,
            )
            store.init_schema()
            print(f"✓ Neo4j connected at {args.uri}")
        except Exception as e:
            print(f"✗ Neo4j connection failed: {e}")
            store = None

    # FLOW
    text   = step1_extract(file_path)
    chunks = step2_chunk(text, source, args.max_chunks)
    step3_llm(llm)
    step4_extractor(llm, embedding_fn, args.batch, args.mode)
    step5_pipeline_validation(llm, embedding_fn)

    if store:
        step6_pipeline_run(
            llm,
            embedding_fn,
            store,
            chunks,
            args.batch,
            args.mode,
            source
        )
    else:
        print("⚠ Skipping pipeline (no Neo4j)")

    # Cleanup
    if store:
        store.close()

if __name__ == "__main__":
    main()