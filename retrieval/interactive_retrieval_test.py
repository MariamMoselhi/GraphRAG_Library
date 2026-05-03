# """
# GraphRAG Interactive Query Session
# ===================================

# Runs the full setup pipeline ONCE (extraction → chunking → embedding → graph),
# then enters an interactive loop where you type queries (or audio file paths)
# and see retrieved, reranked, and graded results — until you type 'exit'.

# Usage
# -----
#     python retrieval_interactive.py

# You will be prompted for:
#   1. Input file path (PDF / DOCX / PPTX / TXT / MD / URL)
#   2. Working directory  (indices, cache, memory are stored here)
#   3. Neo4j connection details
#   4. Groq API keys

# Then the setup runs once, and you can ask as many questions as you like.
# Type  exit  or  quit  (or press Ctrl-C) to end the session.
# """
# from __future__ import annotations

# import os
# import sys
# from pathlib import Path
# from typing import Optional

# # ─────────────────────────────────────────────────────────────────────────────
# # Resolve project root (same logic as retrieval_test.py)
# # ─────────────────────────────────────────────────────────────────────────────
# PROJECT_ROOT = Path(__file__).parent.parent.absolute()
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# # ─────────────────────────────────────────────────────────────────────────────
# # Colour helpers  (copied verbatim from retrieval_test.py — no changes)
# # ─────────────────────────────────────────────────────────────────────────────
# _HAS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

# def _c(code: str, text: str) -> str:
#     return f"\033[{code}m{text}\033[0m" if _HAS_COLOR else text

# def header(title: str) -> None:
#     bar = "═" * 72
#     print(f"\n{_c('96;1', bar)}")
#     print(f"{_c('96;1', f'  {title}')}")
#     print(f"{_c('96', bar)}")

# def section(title: str) -> None:
#     print(f"\n{_c('94;1', f'  ▶  {title}')}")
#     print(f"{_c('94', '  ' + '─' * 60)}")

# def ok(msg: str) -> None:
#     print(f"  {_c('92', '✔')}  {msg}")

# def warn(msg: str) -> None:
#     print(f"  {_c('93', '⚠')}  {msg}")

# def fail(msg: str) -> None:
#     print(f"  {_c('91', '✗')}  {msg}")

# def info(msg: str) -> None:
#     print(f"  {_c('97', 'ℹ')}  {msg}")

# def show(label: str, value) -> None:
#     print(f"    {_c('2', label + ':')}  {value}")

# def separator() -> None:
#     print(f"  {_c('2', '─' * 60)}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 1 — Collect configuration from the user
# # ─────────────────────────────────────────────────────────────────────────────

# def collect_config() -> dict:
#     header("GraphRAG Interactive Session — Setup")
#     print("""
#   This session will:
#     1. Extract text from your file
#     2. Chunk it and build embedding + BM25 indices
#     3. Build a Neo4j knowledge graph
#     4. Then let you ask unlimited questions until you type 'exit'

#   Press ENTER to accept any default shown in [brackets].
# """)

#     # ── Input file ────────────────────────────────────────────────────────────
#     print(_c("97;1", "  ── Step 1: Input File ──"))
#     print("  Supported: PDF, DOCX, PPTX, TXT, MD, or a URL")
#     file_path = input("  File path or URL: ").strip()
#     if not file_path:
#         fail("No file path provided.  Exiting.")
#         sys.exit(1)

#     # ── Working directory ─────────────────────────────────────────────────────
#     print(f"\n{_c('97;1', '  ── Step 2: Working Directory ──')}")
#     print("  Indices, memory and cache will be stored here.")
#     work_dir = input("  Working directory [./rag_session]: ").strip() or "./rag_session"

#     # ── Neo4j ─────────────────────────────────────────────────────────────────
#     print(f"\n{_c('97;1', '  ── Step 3: Neo4j Connection ──')}")
#     neo4j_uri  = input("  Neo4j URI  [neo4j://127.0.0.1:7687]: ").strip() or "neo4j://127.0.0.1:7687"
#     neo4j_user = input("  Neo4j user [neo4j]: ").strip() or "neo4j"
#     neo4j_pass = input("  Neo4j password [neo4j1234]: ").strip() or "neo4j1234"

#     # ── Groq API keys ─────────────────────────────────────────────────────────
#     print(f"\n{_c('97;1', '  ── Step 4: Groq API Keys ──')}")
#     print("  One shared key for all roles, or separate keys per role.")
#     shared_key = input("  Shared Groq API key (leave blank to set individually): ").strip()

#     if shared_key:
#         whisper_key = query_key = grader_key = shared_key
#     else:
#         whisper_key = input("  GROQ_API_KEY_WHISPER (voice transcription): ").strip()
#         query_key   = input("  GROQ_API_KEY_QUERY   (intent / entity / Cypher): ").strip()
#         grader_key  = input("  GROQ_API_KEY_GRADER  (relevance grading): ").strip()

#     return {
#         "file_path"  : file_path,
#         "work_dir"   : work_dir,
#         "neo4j_uri"  : neo4j_uri,
#         "neo4j_user" : neo4j_user,
#         "neo4j_pass" : neo4j_pass,
#         "whisper_key": whisper_key,
#         "query_key"  : query_key,
#         "grader_key" : grader_key,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 2 — One-time setup  (extraction → chunking → indices → graph)
# # ─────────────────────────────────────────────────────────────────────────────

# def run_setup(cfg: dict) -> tuple:
#     """
#     Returns (embed_model, bm25, graph_store, memory_store).
#     Any component that fails to initialise is returned as None; the query loop
#     will still run with whatever is available.
#     """
#     os.makedirs(cfg["work_dir"], exist_ok=True)
#     work_dir  = cfg["work_dir"]
#     file_path = cfg["file_path"]

#     # ── Extract text ──────────────────────────────────────────────────────────
#     header("Setup Phase 1/4 — Text Extraction")
#     extracted_text = ""
#     try:
#         ext = Path(file_path).suffix.lower() if not file_path.startswith("http") else ".url"
#         info(f"Input : {file_path}  (type: {ext or 'url'})")

#         if file_path.startswith("http"):
#             from extractors.url_extractor import URLExtractor
#             result = URLExtractor().extract(file_path)
#         elif ext == ".pdf":
#             from extractors.pdf_extractor import PDFExtractor
#             result = PDFExtractor().extract(file_path)
#         elif ext in (".docx", ".doc"):
#             from extractors.docx_extractor import DOCXExtractor
#             result = DOCXExtractor().extract(file_path)
#         elif ext in (".pptx", ".ppt"):
#             from extractors.pptx_extractor import PPTXExtractor
#             result = PPTXExtractor().extract(file_path)
#         else:                              # .txt / .md / fallback
#             result = {"success": True, "extracted_text": Path(file_path).read_text(encoding="utf-8")}

#         if not result.get("success"):
#             fail(f"Extraction returned success=False: {result.get('error', 'unknown error')}")
#             sys.exit(1)

#         extracted_text = result.get("extracted_text", "")
#         if len(extracted_text) < 100:
#             fail(f"Extracted text is too short ({len(extracted_text)} chars).  Exiting.")
#             sys.exit(1)

#         ok(f"Extracted {len(extracted_text):,} characters  ({len(extracted_text.split()):,} words)")
#     except Exception as e:
#         fail(f"Extraction failed: {e}")
#         sys.exit(1)

#     # ── Chunk ─────────────────────────────────────────────────────────────────
#     header("Setup Phase 2/4 — Chunking")
#     chunks = []
#     try:
#         from chunking.chunking import get_chunker
#         try:
#             chunker = get_chunker("semantic", threshold=0.5, max_sentences_per_chunk=20)
#             chunks  = chunker.chunk(extracted_text)
#             info(f"Semantic chunker → {len(chunks)} chunks")
#         except Exception:
#             warn("Semantic chunker unavailable — falling back to paragraph chunker")
#             chunks = get_chunker("paragraph").chunk(extracted_text)
#             info(f"Paragraph chunker → {len(chunks)} chunks")

#         # Tag every chunk with its source
#         for c in chunks:
#             if not hasattr(c, "metadata") or c.metadata is None:
#                 c.metadata = {}
#             c.metadata["source"] = file_path

#         ok(f"{len(chunks)} chunks ready")
#     except Exception as e:
#         fail(f"Chunking failed: {e}")
#         sys.exit(1)

#     # ── Embedding + FAISS + BM25 ──────────────────────────────────────────────
#     header("Setup Phase 3/4 — Embedding & Index Building")
#     embed_model = None
#     bm25        = None
#     try:
#         from embeddings.huggingFace import HuggingFaceEmbedding
#         embed_model = HuggingFaceEmbedding(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             normalize=True,
#         )
#         ok(f"Embedding model loaded  (dim={embed_model.dimension})")

#         from retrieval.faiss_store import FAISSStore
#         faiss_store = FAISSStore(embed_model, verbose=False)
#         n = faiss_store.build_from_chunks(chunks)
#         faiss_store.save(work_dir)
#         ok(f"FAISS index built with {n} vectors and saved to {work_dir}")

#         from retrieval.bm25_retriever import BM25Retriever
#         bm25 = BM25Retriever(verbose=False)
#         bm25.build(chunks)
#         bm25.save(work_dir)
#         ok(f"BM25 index built with {n} documents and saved to {work_dir}")

#     except Exception as e:
#         fail(f"Embedding / index building failed: {e}")
#         warn("Continuing without vector search — results will degrade.")

#     # ── Neo4j knowledge graph ─────────────────────────────────────────────────
#     header("Setup Phase 4/4 — Knowledge Graph (Neo4j)")
#     graph_store = None
#     try:
#         from graph.graph_store import GraphStore
#         graph_store = GraphStore(
#             uri=cfg["neo4j_uri"],
#             user=cfg["neo4j_user"],
#             password=cfg["neo4j_pass"],
#         )
#         graph_store.init_schema()
#         ok(f"Neo4j connected  ({graph_store.count_nodes()} nodes already in graph)")

#         if embed_model:
#             from graph.node_relation_extractor import CombinedExtractor
#             from graph.llm_backend import LLMBackend

#             llm = LLMBackend(
#                 api_key    = cfg["query_key"] or cfg["grader_key"],
#                 model      = "llama-3.3-70b-versatile",
#                 max_tokens = 3000,
#             )
#             extractor = CombinedExtractor(
#                 llm          = llm,
#                 embedding_fn = embed_model.encode,
#                 batch_chunks = 2,
#                 mode         = "constrained",
#             )
#             info(f"Extracting entities from {len(chunks)} chunks (this may take a while)…")
#             nodes   = extractor.extract_from_chunks(chunks, show_progress=True)
#             written = graph_store.upsert_nodes(nodes)
#             ok(f"{written} nodes written to Neo4j  (total: {graph_store.count_nodes()})")
#         else:
#             warn("Skipping graph entity extraction — no embedding model available.")

#     except Exception as e:
#         fail(f"Neo4j / graph building failed: {e}")
#         warn("Continuing without graph retrieval.")
#         graph_store = None

#     # ── Memory store ──────────────────────────────────────────────────────────
#     from retrieval.memory_store import MemoryStore
#     memory = MemoryStore(
#         session_id  = "interactive_session",
#         storage_dir = work_dir,
#         max_turns   = 20,
#         verbose     = False,
#     )
#     memory.clear()           # fresh session each run
#     ok("Memory store initialised")

#     return embed_model, bm25, graph_store, memory


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 3 — Build the retrieval pipeline
# # ─────────────────────────────────────────────────────────────────────────────

# def build_pipeline(cfg: dict, embed_model, graph_store, memory):
#     from retrieval.retrieval_pipeline import RetrievalPipeline
#     work_dir = cfg["work_dir"]

#     return RetrievalPipeline(
#         embedding_model   = embed_model,
#         graph_store       = graph_store,
#         faiss_dir         = work_dir if Path(work_dir, "faiss.index").exists() else None,
#         bm25_dir          = work_dir if Path(work_dir, "bm25.pkl").exists() else None,
#         session_id        = memory.session_id if memory else "interactive",
#         memory_dir        = work_dir,
#         whisper_api_key   = cfg["whisper_key"],
#         query_llm_api_key = cfg["query_key"],
#         grader_api_key    = cfg["grader_key"],
#         enable_retry      = True,
#         show_graph_viz    = False,   # keep output clean in interactive mode
#         verbose           = False,   # suppress pipeline internals; results printed below
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 4 — Pretty-print a single graded result
# # ─────────────────────────────────────────────────────────────────────────────

# def print_result(query_input: str, result) -> None:
#     """Print the graded retrieval result in a readable format."""
#     print()
#     print(_c("96;1", "  ══  RETRIEVAL RESULT  " + "═" * 49))

#     # ── Query summary ─────────────────────────────────────────────────────────
#     section("Query")
#     show("Input",      query_input[:120])
#     show("Normalised", result.query.normalised_text[:120])
#     show("Intent",     result.query.intent.value)
#     show("Entities",   result.query.entities or "none detected")
#     show("Keywords",   (result.query.keywords or [])[:8])

#     # ── Pipeline stats ────────────────────────────────────────────────────────
#     section("Pipeline statistics")
#     t = result.trace
#     show("Input modality",   t.input_modality)
#     show("BM25 retrieved",   t.bm25_count)
#     show("Vector retrieved", t.vector_count)
#     show("Graph retrieved",  t.graph_count)
#     show("After fusion",     t.fused_count)
#     show("After reranking",  t.reranked_count)
#     show("Total time",       f"{t.total_elapsed_ms:.0f} ms")
#     if t.transcription_text:
#         show("Transcription", t.transcription_text[:100])

#     # ── Grader verdict ────────────────────────────────────────────────────────
#     section("Grader verdict")
#     verdict_color = "92;1" if result.verdict.value == "pass" else (
#                     "93;1" if result.verdict.value == "partial" else "91;1")
#     print(f"    {_c(verdict_color, result.verdict.value.upper())}  "
#           f"(confidence: {result.confidence:.3f})")
#     show("Passed chunks", len(result.passed_chunks))
#     show("Failed chunks", len(result.failed_chunks))
#     if result.reformulation:
#         show("Suggested reformulation", result.reformulation)

#     # ── Retrieved & graded chunks ─────────────────────────────────────────────
#     if result.passed_chunks:
#         section("Passed chunks (most relevant → least relevant)")
#         for i, gc in enumerate(result.passed_chunks, 1):
#             chunk_text = gc.fused.chunk.text if hasattr(gc, "fused") else gc.text
#             score      = gc.relevance_score if hasattr(gc, "relevance_score") else getattr(gc, "rerank_score", 0)
#             reason     = getattr(gc, "grader_reason", "")
#             print(f"\n    {_c('92', f'[{i}]')}  score={score:.3f}")
#             if reason:
#                 print(f"         {_c('2', reason[:100])}")
#             # Print chunk text, wrapped at 80 chars
#             text_preview = chunk_text.strip().replace("\n", " ")
#             # Break into 80-char lines for readability
#             while text_preview:
#                 print(f"         {text_preview[:80]}")
#                 text_preview = text_preview[80:]
#     else:
#         section("No chunks passed the relevance threshold")
#         if result.failed_chunks:
#             warn(f"{len(result.failed_chunks)} chunk(s) failed grading — "
#                  "try rephrasing your question.")

#     print()
#     print(_c("96", "  " + "═" * 70))
#     print()


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 5 — Interactive query loop
# # ─────────────────────────────────────────────────────────────────────────────

# _AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

# def interactive_loop(cfg: dict, pipeline, memory) -> None:
#     header("Interactive Query Session  —  type 'exit' to quit")
#     print(f"""
#   You can now ask questions about your document.
#   {_c('97', 'Text query')} : just type your question and press ENTER
#   {_c('97', 'Voice query')}: type the full path to an audio file (mp3/wav/m4a/…)
#   {_c('97', 'Exit')       }: type  exit  or  quit,  or press Ctrl-C
# """)

#     turn = 0
#     while True:
#         # ── Prompt ────────────────────────────────────────────────────────────
#         try:
#             raw = input(_c("96;1", f"  [{turn + 1}] Query > ")).strip()
#         except (EOFError, KeyboardInterrupt):
#             print()
#             info("Session ended by user.")
#             break

#         # ── Exit ──────────────────────────────────────────────────────────────
#         if raw.lower() in ("exit", "quit", "q", ""):
#             if raw == "":
#                 continue      # ignore blank lines
#             info("Goodbye!")
#             break

#         # ── Detect audio vs text ──────────────────────────────────────────────
#         query_input = raw
#         p = Path(raw)
#         is_audio = p.suffix.lower() in _AUDIO_EXTENSIONS and p.exists()

#         if is_audio:
#             info(f"Audio file detected: {raw}")
#         else:
#             info(f"Running retrieval for: '{raw[:80]}'")

#         # ── Run the pipeline ──────────────────────────────────────────────────
#         try:
#             result = pipeline.run(query_input)
#         except Exception as e:
#             fail(f"Pipeline error: {e}")
#             warn("You can try rephrasing and asking again.")
#             continue

#         # ── Display results ───────────────────────────────────────────────────
#         print_result(query_input, result)

#         # ── Record turn in memory for follow-up awareness ─────────────────────
#         try:
#             ai_response_summary = (
#                 result.context_text[:200] if result.context_text else "No relevant content found."
#             )
#             pipeline.record_turn(
#                 user_query    = result.query.normalised_text,
#                 ai_response   = ai_response_summary,
#                 graded_result = result,
#             )
#         except Exception:
#             pass   # memory recording is best-effort

#         turn += 1

#     # ── Session stats ─────────────────────────────────────────────────────────
#     header("Session Summary")
#     show("Total queries answered", turn)
#     show("Memory turns recorded",  memory.turn_count if memory else 0)
#     print()


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     cfg = collect_config()

#     embed_model, bm25, graph_store, memory = run_setup(cfg)

#     header("Setup Complete — Building Retrieval Pipeline")
#     try:
#         pipeline = build_pipeline(cfg, embed_model, graph_store, memory)
#         ok("Pipeline ready")
#     except Exception as e:
#         fail(f"Could not build pipeline: {e}")
#         sys.exit(1)

#     interactive_loop(cfg, pipeline, memory)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# retrieval_interactive.py
"""
GraphRAG Interactive Query Session
===================================

Runs the full setup pipeline ONCE (extraction → chunking → embedding → graph),
then enters an interactive loop where you type queries (or audio file paths)
and see retrieved, reranked, and graded results — until you type 'exit'.

Usage
-----
    python retrieval_interactive.py

You will be prompted for:
  1. Input file path (PDF / DOCX / PPTX / TXT / MD / URL)
  2. Working directory  (indices, cache, memory are stored here)
  3. Neo4j connection details
  4. Groq API keys

Then the setup runs once, and you can ask as many questions as you like.
Type  exit  or  quit  (or press Ctrl-C) to end the session.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Resolve project root (same logic as retrieval_test.py)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers  (copied verbatim from retrieval_test.py — no changes)
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
# Step 1 — Collect configuration from the user
# ─────────────────────────────────────────────────────────────────────────────

def collect_config() -> dict:
    header("GraphRAG Interactive Session — Setup")
    print("""
  This session will:
    1. Extract text from your file
    2. Chunk it and build embedding + BM25 indices
    3. Build a Neo4j knowledge graph
    4. Then let you ask unlimited questions until you type 'exit'

  Press ENTER to accept any default shown in [brackets].
""")

    # ── Input file ────────────────────────────────────────────────────────────
    print(_c("97;1", "  ── Step 1: Input File ──"))
    print("  Supported: PDF, DOCX, PPTX, TXT, MD, or a URL")
    file_path = input("  File path or URL: ").strip()
    if not file_path:
        fail("No file path provided.  Exiting.")
        sys.exit(1)

    # ── Working directory ─────────────────────────────────────────────────────
    print(f"\n{_c('97;1', '  ── Step 2: Working Directory ──')}")
    print("  Indices, memory and cache will be stored here.")
    work_dir = input("  Working directory [./rag_session]: ").strip() or "./rag_session"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    print(f"\n{_c('97;1', '  ── Step 3: Neo4j Connection ──')}")
    neo4j_uri  = input("  Neo4j URI  [neo4j://127.0.0.1:7687]: ").strip() or "neo4j://127.0.0.1:7687"
    neo4j_user = input("  Neo4j user [neo4j]: ").strip() or "neo4j"
    neo4j_pass = input("  Neo4j password [neo4j1234]: ").strip() or "neo4j1234"

    # ── Groq API keys ─────────────────────────────────────────────────────────
    print(f"\n{_c('97;1', '  ── Step 4: Groq API Keys ──')}")
    print("  One shared key for all roles, or separate keys per role.")
    shared_key = input("  Shared Groq API key (leave blank to set individually): ").strip()

    if shared_key:
        whisper_key = query_key = grader_key = shared_key
    else:
        whisper_key = input("  GROQ_API_KEY_WHISPER (voice transcription): ").strip()
        query_key   = input("  GROQ_API_KEY_QUERY   (intent / entity / Cypher): ").strip()
        grader_key  = input("  GROQ_API_KEY_GRADER  (relevance grading): ").strip()

    return {
        "file_path"  : file_path,
        "work_dir"   : work_dir,
        "neo4j_uri"  : neo4j_uri,
        "neo4j_user" : neo4j_user,
        "neo4j_pass" : neo4j_pass,
        "whisper_key": whisper_key,
        "query_key"  : query_key,
        "grader_key" : grader_key,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — One-time setup  (extraction → chunking → indices → graph)
# ─────────────────────────────────────────────────────────────────────────────

def run_setup(cfg: dict) -> tuple:
    """
    Returns (embed_model, bm25, graph_store, memory_store).
    Any component that fails to initialise is returned as None; the query loop
    will still run with whatever is available.
    """
    os.makedirs(cfg["work_dir"], exist_ok=True)
    work_dir  = cfg["work_dir"]
    file_path = cfg["file_path"]

    # ── Extract text ──────────────────────────────────────────────────────────
    header("Setup Phase 1/4 — Text Extraction")
    extracted_text = ""
    try:
        ext = Path(file_path).suffix.lower() if not file_path.startswith("http") else ".url"
        info(f"Input : {file_path}  (type: {ext or 'url'})")

        if file_path.startswith("http"):
            from extractors.url_extractor import URLExtractor
            result = URLExtractor().extract(file_path)
        elif ext == ".pdf":
            from extractors.pdf_extractor import PDFExtractor
            result = PDFExtractor().extract(file_path)
        elif ext in (".docx", ".doc"):
            from extractors.docx_extractor import DOCXExtractor
            result = DOCXExtractor().extract(file_path)
        elif ext in (".pptx", ".ppt"):
            from extractors.pptx_extractor import PPTXExtractor
            result = PPTXExtractor().extract(file_path)
        else:                              # .txt / .md / fallback
            result = {"success": True, "extracted_text": Path(file_path).read_text(encoding="utf-8")}

        if not result.get("success"):
            fail(f"Extraction returned success=False: {result.get('error', 'unknown error')}")
            sys.exit(1)

        extracted_text = result.get("extracted_text", "")
        if len(extracted_text) < 100:
            fail(f"Extracted text is too short ({len(extracted_text)} chars).  Exiting.")
            sys.exit(1)

        ok(f"Extracted {len(extracted_text):,} characters  ({len(extracted_text.split()):,} words)")
    except Exception as e:
        fail(f"Extraction failed: {e}")
        sys.exit(1)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    header("Setup Phase 2/4 — Chunking")
    chunks = []
    try:
        from chunking.chunking import get_chunker
        try:
            chunker = get_chunker("semantic", threshold=0.5, max_sentences_per_chunk=20)
            chunks  = chunker.chunk(extracted_text)
            info(f"Semantic chunker → {len(chunks)} chunks")
        except Exception:
            warn("Semantic chunker unavailable — falling back to paragraph chunker")
            chunks = get_chunker("paragraph").chunk(extracted_text)
            info(f"Paragraph chunker → {len(chunks)} chunks")

        # Tag every chunk with its source
        for c in chunks:
            if not hasattr(c, "metadata") or c.metadata is None:
                c.metadata = {}
            c.metadata["source"] = file_path

        ok(f"{len(chunks)} chunks ready")
    except Exception as e:
        fail(f"Chunking failed: {e}")
        sys.exit(1)

    # ── Embedding + FAISS + BM25 ──────────────────────────────────────────────
    header("Setup Phase 3/4 — Embedding & Index Building")
    embed_model = None
    bm25        = None
    try:
        from embeddings.huggingFace import HuggingFaceEmbedding
        # embed_model = HuggingFaceEmbedding(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     normalize=True,
        # )
        # ok(f"Embedding model loaded  (dim={embed_model.dimension})")

        # from vectordb.faiss_store import FaissVectorStore
        # faiss_store = FaissVectorStore(dim=embed_model.dimension, verbose=False)
        # n = faiss_store.add_chunks(chunks)
        # faiss_store.save(work_dir)
        # ok(f"FAISS index built with {n} vectors and saved to {work_dir}")
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True,
        )
        ok(f"Embedding model loaded  (dim={embed_model.dimension})")
 
        from vectordb.faiss_store import FaissVectorStore
 
        # FaissVectorStore follows the vectordb/base.py contract:
        # embeddings must be stored in chunk.metadata["embedding"] BEFORE
        # add_chunks() is called.  Embed all chunks now in one batched call.
        info(f"Embedding {len(chunks)} chunks...")
        texts   = [c.text for c in chunks]
        vectors = embed_model.encode(texts)          # shape (N, dim)
        for chunk, vec in zip(chunks, vectors):
            chunk.metadata["embedding"] = vec
 
        faiss_store = FaissVectorStore(dim=embed_model.dimension, verbose=False)
        faiss_store.add_chunks(chunks)               # add_chunks returns None
        faiss_store.save(work_dir)
        n = len(faiss_store)                         # correct count
        ok(f"FAISS index built with {n} vectors and saved to {work_dir}")
 
        from retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(verbose=False)
        bm25.build(chunks)
        bm25.save(work_dir)
        ok(f"BM25 index built with {n} documents and saved to {work_dir}")

    except Exception as e:
        fail(f"Embedding / index building failed: {e}")
        warn("Continuing without vector search — results will degrade.")

    # ── Neo4j knowledge graph ─────────────────────────────────────────────────
    header("Setup Phase 4/4 — Knowledge Graph (Neo4j)")
    graph_store = None
    try:
        from graph.graph_store import GraphStore
        graph_store = GraphStore(
            uri=cfg["neo4j_uri"],
            user=cfg["neo4j_user"],
            password=cfg["neo4j_pass"],
        )
        graph_store.init_schema()
        ok(f"Neo4j connected  ({graph_store.count_nodes()} nodes already in graph)")

        if embed_model:
            from graph.node_relation_extractor import CombinedExtractor
            from graph.llm_backend import LLMBackend
            from graph.Pipeline import Pipeline

            llm = LLMBackend(
                api_key    = cfg["query_key"] or cfg["grader_key"],
                model      = "llama-3.3-70b-versatile",
                max_tokens = 3000,
            )
            extractor = CombinedExtractor(
                llm          = llm,
                embedding_fn = embed_model.encode,
                batch_chunks = 2,
                mode         = "constrained",
            )
            
            # Use Pipeline.from_components — it handles node_id_map building internally
            pipeline = Pipeline.from_components(
                node_extractor         = extractor,
                relationship_extractor = extractor,
                graph_store            = graph_store,
                extract_cross_doc      = False,
                show_progress          = True,
            )
            
            info(f"Running extraction pipeline on {len(chunks)} chunks…")
            stats = pipeline.run(chunks)
            ok(f"Pipeline complete: {stats.nodes_written} nodes, {stats.relationships_written} relationships")
        else:
            warn("Skipping graph entity extraction — no embedding model available.")

    except Exception as e:
        fail(f"Neo4j / graph building failed: {e}")
        warn("Continuing without graph retrieval.")
        graph_store = None

    # ── Memory store ──────────────────────────────────────────────────────────
    from retrieval.memory_store import MemoryStore
    memory = MemoryStore(
        session_id  = "interactive_session",
        storage_dir = work_dir,
        max_turns   = 20,
        verbose     = False,
    )
    memory.clear()           # fresh session each run
    ok("Memory store initialised")

    return embed_model, bm25, graph_store, memory


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build the retrieval pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(cfg: dict, embed_model, graph_store, memory):
    from retrieval.retrieval_pipeline import RetrievalPipeline
    work_dir = cfg["work_dir"]

    return RetrievalPipeline(
        embedding_model   = embed_model,
        graph_store       = graph_store,
        faiss_dir         = work_dir if Path(work_dir, "faiss.index").exists() else None,
        bm25_dir          = work_dir if Path(work_dir, "bm25.pkl").exists() else None,
        session_id        = memory.session_id if memory else "interactive",
        memory_dir        = work_dir,
        whisper_api_key   = cfg["whisper_key"],
        query_llm_api_key = cfg["query_key"],
        grader_api_key    = cfg["grader_key"],
        enable_retry      = True,
        show_graph_viz    = False,   # keep output clean in interactive mode
        verbose           = False,   # suppress pipeline internals; results printed below
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Pretty-print a single graded result
# ─────────────────────────────────────────────────────────────────────────────

def _get_chunk_text(gc) -> str:
    """
    Robustly extract the text string from a GradedChunk (or any wrapper object).

    The object graph can vary depending on how the pipeline assembles results.
    We try every known attribute path before giving up.
    """
    # Path 1: GradedChunk → .fused (FusedResult) → .chunk (Chunk) → .text
    try:
        return gc.fused.chunk.text
    except AttributeError:
        pass
    # Path 2: GradedChunk → .fused → .text   (some FusedResult implementations
    #          expose text directly)
    try:
        return gc.fused.text
    except AttributeError:
        pass
    # Path 3: GradedChunk → .chunk → .text
    try:
        return gc.chunk.text
    except AttributeError:
        pass
    # Path 4: GradedChunk → .text   (flat layout)
    try:
        return gc.text
    except AttributeError:
        pass
    # Path 5: GradedChunk → .content
    try:
        return gc.content
    except AttributeError:
        pass
    # Fallback: stringify the object so we never crash
    return str(gc)


def _get_chunk_score(gc) -> float:
    """Return the best available score from a GradedChunk."""
    for attr in ("relevance_score", "rerank_score", "rrf_score", "score"):
        v = getattr(gc, attr, None)
        if v is not None:
            return float(v)
    # try gc.fused.*
    fused = getattr(gc, "fused", None)
    if fused is not None:
        for attr in ("rrf_score", "score"):
            v = getattr(fused, attr, None)
            if v is not None:
                return float(v)
    return 0.0


def print_result(query_input: str, result) -> None:
    """Print the graded retrieval result in a readable format."""
    print()
    print(_c("96;1", "  ══  RETRIEVAL RESULT  " + "═" * 49))

    # ── Query summary ─────────────────────────────────────────────────────────
    section("Query")
    show("Input",      query_input[:120])
    show("Normalised", result.query.normalised_text[:120])
    show("Intent",     result.query.intent.value)
    show("Entities",   result.query.entities or "none detected")
    show("Keywords",   (result.query.keywords or [])[:8])

    # ── Pipeline stats ────────────────────────────────────────────────────────
    section("Pipeline statistics")
    t = result.trace
    show("Input modality",   t.input_modality)
    show("BM25 retrieved",   t.bm25_count)
    show("Vector retrieved", t.vector_count)
    show("Graph retrieved",  t.graph_count)
    show("After fusion",     t.fused_count)
    show("After reranking",  t.reranked_count)
    show("Total time",       f"{t.total_elapsed_ms:.0f} ms")
    if t.transcription_text:
        show("Transcription", t.transcription_text[:100])

    # ── Grader verdict ────────────────────────────────────────────────────────
    section("Grader verdict")
    verdict_color = "92;1" if result.verdict.value == "pass" else (
                    "93;1" if result.verdict.value == "partial" else "91;1")
    print(f"    {_c(verdict_color, result.verdict.value.upper())}  "
          f"(confidence: {result.confidence:.3f})")
    show("Passed chunks", len(result.passed_chunks))
    show("Failed chunks", len(result.failed_chunks))
    if result.reformulation:
        show("Suggested reformulation", result.reformulation)

    # ── Retrieved & graded chunks ─────────────────────────────────────────────
    if result.passed_chunks:
        section("Passed chunks (most relevant → least relevant)")
        for i, gc in enumerate(result.passed_chunks, 1):
            chunk_text = _get_chunk_text(gc)
            score      = _get_chunk_score(gc)
            reason     = getattr(gc, "grader_reason", "")
            print(f"\n    {_c('92', f'[{i}]')}  score={score:.3f}")
            if reason:
                print(f"         {_c('2', reason[:100])}")
            # Print chunk text wrapped at 80 chars per line for readability
            text_preview = chunk_text.strip().replace("\n", " ")
            while text_preview:
                print(f"         {text_preview[:80]}")
                text_preview = text_preview[80:]
    else:
        section("No chunks passed the relevance threshold")
        if result.failed_chunks:
            warn(f"{len(result.failed_chunks)} chunk(s) failed grading — "
                 "try rephrasing your question.")

    print()
    print(_c("96", "  " + "═" * 70))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Interactive query loop
# ─────────────────────────────────────────────────────────────────────────────

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

def interactive_loop(cfg: dict, pipeline, memory) -> None:
    header("Interactive Query Session  —  type 'exit' to quit")
    print(f"""
  You can now ask questions about your document.
  {_c('97', 'Text query')} : just type your question and press ENTER
  {_c('97', 'Voice query')}: type the full path to an audio file (mp3/wav/m4a/…)
  {_c('97', 'Exit')       }: type  exit  or  quit,  or press Ctrl-C
""")

    turn = 0
    while True:
        # ── Prompt ────────────────────────────────────────────────────────────
        try:
            raw = input(_c("96;1", f"  [{turn + 1}] Query > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            info("Session ended by user.")
            break

        # ── Exit ──────────────────────────────────────────────────────────────
        if raw.lower() in ("exit", "quit", "q", ""):
            if raw == "":
                continue      # ignore blank lines
            info("Goodbye!")
            break

        # ── Detect audio vs text ──────────────────────────────────────────────
        query_input = raw
        p = Path(raw)
        is_audio = p.suffix.lower() in _AUDIO_EXTENSIONS and p.exists()

        if is_audio:
            info(f"Audio file detected: {raw}")
        else:
            info(f"Running retrieval for: '{raw[:80]}'")

        # ── Run the pipeline ──────────────────────────────────────────────────
        try:
            result = pipeline.run(query_input)
        except Exception as e:
            fail(f"Pipeline error: {e}")
            warn("You can try rephrasing and asking again.")
            continue

        # ── Display results ───────────────────────────────────────────────────
        print_result(query_input, result)

        # ── Record turn in memory for follow-up awareness ─────────────────────
        try:
            ai_response_summary = (
                result.context_text[:200] if result.context_text else "No relevant content found."
            )
            pipeline.record_turn(
                user_query    = result.query.normalised_text,
                ai_response   = ai_response_summary,
                graded_result = result,
            )
        except Exception:
            pass   # memory recording is best-effort

        turn += 1

    # ── Session stats ─────────────────────────────────────────────────────────
    header("Session Summary")
    show("Total queries answered", turn)
    show("Memory turns recorded",  memory.turn_count if memory else 0)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = collect_config()

    embed_model, bm25, graph_store, memory = run_setup(cfg)

    header("Setup Complete — Building Retrieval Pipeline")
    try:
        pipeline = build_pipeline(cfg, embed_model, graph_store, memory)
        ok("Pipeline ready")
    except Exception as e:
        fail(f"Could not build pipeline: {e}")
        sys.exit(1)

    interactive_loop(cfg, pipeline, memory)


if __name__ == "__main__":
    main()