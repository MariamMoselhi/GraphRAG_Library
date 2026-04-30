# GraphRAG Retrieval Phase — Complete Module Documentation

This document explains every file in the `retrieval/` package: what it does, why it was designed that way, what it receives, what it returns, and how it connects to the rest of the system.

---

## Table of Contents

1. [Package Overview](#1-package-overview)
2. [retrieval_context.py](#2-retrieval_contextpy)
3. [retrieval_logger.py](#3-retrieval_loggerpy)
4. [memory_store.py](#4-memory_storepy)
5. [query_cache.py](#5-query_cachepy)
6. [faiss_store.py](#6-faiss_storepy)
7. [query_processor.py](#7-query_processorpy)
8. [query_engine.py](#8-query_enginepy)
9. [bm25_retriever.py](#9-bm25_retrieverpy)
10. [vector_retriever.py](#10-vector_retrieverpy)
11. [graph_retriever.py](#11-graph_retrieverpy)
12. [graph_visualizer.py](#12-graph_visualizerpy)
13. [hybrid_retriever.py](#13-hybrid_retrieverpy)
14. [reranker.py](#14-rerankerpy)
15. [grader.py](#15-graderpy)
16. [retrieval_pipeline.py](#16-retrieval_pipelinepy)
17. [__init__.py](#17-initpy)
18. [Data Flow Diagram](#18-data-flow-diagram)
19. [Environment Variables Reference](#19-environment-variables-reference)
20. [Dependency Installation](#20-dependency-installation)

---

## 1. Package Overview

The `retrieval/` package implements the full retrieval phase of the GraphRAG system. It sits between the ingestion pipeline (extractors → chunkers → embedders → Neo4j) and the answer-generation phase.

Its job is to take a raw user input — which may be voice audio or text — and return a `GradedResult`: a ranked, filtered, relevance-verified list of text chunks ready to be passed as context to an answer-generation LLM.

The pipeline is strictly sequential with one branch point (the cache check) and one feedback loop (the grader-triggered retry on failure):

```
User input
  → QueryProcessor     (voice→text, memory injection)
  → [QueryCache]       (skip pipeline if semantically identical query was cached)
  → QueryEngine        (intent, entities, Cypher, embedding vector)
  ├─ BM25Retriever     (keyword search)
  ├─ VectorRetriever   (FAISS ANN semantic search)
  └─ GraphRetriever    (Neo4j traversal + path finding)
       → HybridRetriever   (Reciprocal Rank Fusion)
            → Reranker         (cross-encoder precision boost)
                 → Grader          (LLM quality gate)
                      → GradedResult (→ answer generation)
                           ↑ retry with reformulation on FAIL
```

Three separate Groq API keys are used to prevent rate-limit contention between pipeline stages:

| Key | Used by | Purpose |
|-----|---------|---------|
| `GROQ_API_KEY_WHISPER` | QueryProcessor | Voice transcription via Whisper |
| `GROQ_API_KEY_QUERY` | QueryEngine | Intent detection, entity extraction, Cypher generation |
| `GROQ_API_KEY_GRADER` | Grader | Per-chunk relevance scoring, reformulation suggestion |

---

## 2. `retrieval_context.py`

### Purpose
Defines all shared data types used across the entire retrieval pipeline. Every module communicates through these typed dataclasses — no raw dictionaries are passed between stages.

### Why this design
Typed dataclasses give you IDE autocomplete, catch field-name errors at definition time, and make inter-module contracts explicit. When a bug occurs, you can inspect any object in the pipeline and know exactly which fields it has and what they mean.

### Key classes

#### `QueryIntent` (Enum)
The six possible intents detected from a user query:
- `FACTUAL` — "What is X?" Definition or description lookups.
- `RELATIONAL` — "How does X relate to Y?" Multi-hop graph questions.
- `PROCEDURAL` — "How do I do X?" Step-by-step questions.
- `COMPARATIVE` — "Compare X and Y." Contrast questions.
- `FOLLOW_UP` — References prior conversation ("elaborate on that").
- `UNKNOWN` — Could not determine intent reliably.

Intent matters because `hybrid_retriever.py` uses it to apply adaptive weights: relational queries get a higher graph retrieval weight, procedural queries get higher BM25 weight.

#### `GraderVerdict` (Enum)
- `PASS` — Enough relevant chunks found; proceed to generation.
- `PARTIAL` — Some chunks are relevant; generation proceeds with caveats.
- `FAIL` — No relevant chunks; trigger query reformulation and retry.
- `REFORMULATE` — Explicit reformulation signal.

#### `QueryRepresentation`
Produced by `query_processor.py` and enriched by `query_engine.py`. Contains:
- `raw_text` — Original text after transcription.
- `normalised_text` — Cleaned version used downstream.
- `intent` — Detected `QueryIntent`.
- `entities` — Named entities extracted (e.g. `["Gradient Descent", "Adam Optimizer"]`).
- `keywords` — BM25 search terms (stop-word filtered).
- `cypher_query` — Neo4j Cypher string for graph retrieval (or `None`).
- `embedding` — Dense numpy vector for FAISS search.
- `memory_context` — Injected prior conversation context.

#### `RetrievedChunk`
The atomic unit returned by each retriever. Contains `chunk_id`, `text`, `source`, `score`, `retriever` (which of the three produced it), and `metadata`.

#### `FusedResult`
A `RetrievedChunk` after RRF fusion. Adds `rrf_score`, `contributing` (which retrievers contributed), and `individual_ranks` (each retriever's rank for this chunk).

#### `RerankedResult`
A `FusedResult` after cross-encoder reranking. Adds `rerank_score`, `original_rank`, `final_rank`, and `delta` (rank change, used in logging).

#### `GradedChunk`
A `RerankedResult` after the LLM grader evaluated it. Adds `relevance_score`, `passed` (bool), and `grader_reason` (LLM explanation string).

#### `GradedResult`
The pipeline's final output. Contains `verdict`, `passed_chunks`, `failed_chunks`, `query`, `confidence`, `reformulation` (if FAIL), and `trace`. Has a `context_text` property that concatenates passed chunks into a ready-to-use generation context string.

#### `GraphTraversalResult`
Stores raw Neo4j output: `nodes`, `relationships`, `paths`, `cypher_used`, `traversal_depth`. Used by `graph_visualizer.py`.

#### `ConversationTurn`
One turn in the conversation memory: `turn_id`, `user_query`, `ai_response`, `retrieved_chunks`, `timestamp`, `query_intent`, `entities`.

#### `RetrievalTrace`
Full audit log accumulated across all pipeline phases. Each phase appends a `PhaseStats` entry. The logger reads this to print the final structured summary.

---

## 3. `retrieval_logger.py`

### Purpose
Centralised, structured console logger for the entire retrieval pipeline. Uses ANSI escape codes for color-coded output so you can instantly see which phase is running, what it returned, and whether it succeeded or failed.

### Why this design
Routing all output through a single logger (instead of scattered `print()` calls) means:
- Verbose mode can be toggled once and affects all phases simultaneously.
- Color coding is consistent: green = success/passed, yellow = warnings/partial, red = failures/rejected, cyan = phase headers.
- The `RetrievalTrace` is the single truth; the logger only reads from it, never writes to it.

### ANSI color scheme

| Color | Meaning |
|-------|---------|
| Cyan | Phase headers, graph traversal data |
| Green | Successes, passed chunks, high-confidence edges |
| Yellow | Warnings, BM25 scores, medium-confidence edges, cache hits |
| Red | Errors, failed chunks, low-confidence edges |
| Blue | Vector cosine similarities, info messages |
| Purple | Graph relation types, intent labels |
| White | Neutral labels |
| Dim gray | Previews, secondary info |

### Key methods
- `phase_start(name)` — Prints a cyan banner for the phase.
- `phase_end(name, count, elapsed_ms)` — Prints a one-line completion line.
- `print_query(...)` — Prints the full parsed query representation.
- `print_bm25_results(chunks)` — Top-5 BM25 results with scores and previews.
- `print_vector_results(chunks)` — Top-5 vector results with cosine similarities.
- `print_graph_results(...)` — Graph results with node/rel counts and hops.
- `print_fusion_results(results)` — RRF fused results with contributing retrievers.
- `print_reranker_results(results)` — Reranked results with delta arrows (↑ moved up, ↓ moved down).
- `print_grader_results(...)` — Passed/failed chunks with LLM reasons.
- `print_trace(trace)` — Full pipeline summary with timing bars, counts, and verdict.
- `print_memory_summary(...)` — Memory store state.

---

## 4. `memory_store.py`

### Purpose
Stores the conversation history as a sliding window of `ConversationTurn` objects, persisted to disk as JSON. Provides context injection for coreference resolution and entity bias in follow-up queries.

### Why memory lives in retrieval (not generation)
Memory is a retrieval signal. When a user asks "elaborate on that", the retrievers need to know what "that" refers to before they can query anything. If memory only lived in the generation phase, retrievers would see a decontextualised query and retrieve irrelevant chunks — the context collapse problem.

Specifically, memory influences:
1. `query_processor.py` — detects follow-up queries and injects prior turn context for coreference resolution.
2. `query_engine.py` — biases entity extraction toward entities already established in conversation.
3. `grader.py` — checks retrieved chunks for consistency with prior answers.

### Sliding window design
- Keeps the last `max_turns` turns (default 10) in the active window.
- When `summarise_at` turns accumulate (default 15), older turns are compressed into a summary string using the LLM, keeping the injected context bounded in token count.
- If no LLM is provided, old turns are simply dropped (their queries are logged as a comma-separated list).
- All state is written atomically to `memory_{session_id}.json` via a temp-file-then-rename pattern to prevent corruption on crash.

### Key methods
- `add_turn(user_query, ai_response, retrieved_chunks, intent, entities)` — Call this after generation completes.
- `build_context_string(n_turns)` — Returns a compact multi-line string of recent turns for injection into the query.
- `get_recent_entities(n_turns)` — Returns deduplicated entity names from recent turns for entity bias.
- `is_follow_up(query)` — Heuristic check: does the query contain pronouns or referential phrases ("it", "that method", "as you said")?
- `clear()` — Wipes all memory for this session.

### Thread safety
All state mutations are protected by `threading.Lock()` so the store is safe for concurrent use.

---

## 5. `query_cache.py`

### Purpose
LRU cache that stores `GradedResult` objects, keyed by query embedding similarity rather than exact string match. A cache hit returns the full pipeline output instantly — zero retriever calls.

### Why cosine similarity, not exact string matching
"What is gradient descent?" and "Explain gradient descent to me" are semantically identical. Exact string matching would miss this and re-run all three retrievers. Cosine similarity between query embeddings correctly identifies near-duplicate queries.

### Design details
- Fixed-capacity `OrderedDict` maintains LRU order.
- Linear scan over cached embeddings is acceptable for ≤ 200 entries; for larger caches a FAISS flat index would be substituted.
- Hit condition: `cosine_similarity(query_embedding, cached_embedding) ≥ threshold` (default 0.95).
- Threshold guidance: 0.98 = very tight (near-verbatim only), 0.95 = recommended (catches most paraphrases), 0.90 = loose (may over-generalise).
- Optional disk persistence via `pickle` so the cache survives process restarts.
- Thread-safe via `threading.Lock`.

### Key methods
- `get(query_embedding)` — Returns `GradedResult` if hit, `None` if miss.
- `put(query_embedding, result)` — Stores a result, evicting LRU entry if at capacity.
- `invalidate()` — Clears all entries.
- `stats()` — Returns `{size, capacity, hits, misses, hit_rate, threshold}`.

---

## 6. `faiss_store.py`

### Purpose
Wraps Facebook AI Similarity Search (FAISS) to build, persist, load, and query a dense vector index of chunk embeddings.

### Why FAISS Flat (IndexFlatIP)
For corpora up to ~100K chunks, the exact flat inner-product index gives perfect recall with sub-50ms search time for top-10. `IndexFlatIP` uses inner product, which equals cosine similarity when vectors are L2-normalised (which `HuggingFaceEmbedding` does by default). For larger corpora, `IndexHNSWFlat` would be substituted.

### Two files on disk
- `faiss.index` — FAISS binary index (all embedding vectors).
- `faiss_meta.pkl` — Parallel metadata list: `chunk_id`, `text`, `source`, `metadata` for each vector.

### Key methods
- `build_from_chunks(chunks)` — Embeds all chunk texts in one batched call, then adds them to the index.
- `add(chunk, embedding)` — Incremental add of a single chunk.
- `search(query_vector, top_k)` — Returns `List[RetrievedChunk]` sorted by cosine similarity.
- `save(directory)` — Saves both files.
- `FAISSStore.load(directory, embedding_model)` — Class method restoring from disk.

---

## 7. `query_processor.py`

### Purpose
First pipeline stage. Detects whether user input is audio or text, transcribes audio using Groq Whisper (API key #1), normalises the text, checks the memory store for prior context, and detects follow-up queries.

### Audio detection
If the input string is a path to a file with an audio extension (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.webm`) and the file exists, it is treated as audio. Otherwise it is treated as text. This is transparent to the caller — the same `run()` method handles both.

### Groq Whisper (API key #1)
Uses `GROQ_API_KEY_WHISPER` (separate from the LLM key) so transcription rate limits never block LLM extraction calls. The `whisper-large-v3` model is used by default. Maximum file size is 25 MB (Groq limit). Language can be set explicitly or left as `None` for auto-detection.

### Text normalisation
1. Strip control characters (null bytes, etc.).
2. Replace smart quotes with straight quotes.
3. Collapse multiple spaces and newlines.
4. Remove trailing weak punctuation (period, comma, semicolon) that would confuse keyword extraction.

### Memory injection
If `MemoryStore.is_follow_up(query)` returns `True`, `build_context_string(n_turns=3)` is called and stored in `QueryRepresentation.memory_context` for use by `query_engine.py`.

### Returns
`QueryRepresentation` with `raw_text`, `normalised_text`, `intent` (set to `FOLLOW_UP` if detected, else `UNKNOWN`), and `memory_context`.

---

## 8. `query_engine.py`

### Purpose
Second pipeline stage. Takes the `QueryRepresentation` from `query_processor.py` and fans it out into three retrieval representations: a keyword list for BM25, a Cypher query for Neo4j, and a dense embedding vector for FAISS.

### Single LLM call (Groq API key #2)
One call to `GROQ_API_KEY_QUERY` extracts all three outputs simultaneously:
- **Intent** — One of the six `QueryIntent` values.
- **Entities** — Canonical full names of technical concepts (e.g. "Gradient Descent", not "GD").
- **Cypher** — A Neo4j query string with `$entity`, `$entity1`, `$entity2` placeholders that are substituted with the extracted entity names.

The LLM is given a strict JSON output format and the response is parsed by `_safe_json_parse` (reused from your existing `node_relation_extractor.py`).

### Memory-biased entity extraction
If `MemoryStore` provides recent entities, they are appended to the prompt as "prefer these entity names from prior context if relevant." This prevents the same concept from being treated as two different entities across turns (e.g. "GD" in turn 1 vs "Gradient Descent" in turn 3).

### Keyword extraction (no LLM)
Keywords are extracted locally by tokenising on word boundaries, filtering stop words, and keeping tokens ≥ 3 characters. Entity tokens are prepended as highest-priority BM25 terms. This is intentionally simple — BM25 does not need high-precision keywords, and avoiding an LLM call here keeps latency low.

### Embedding (local model)
`HuggingFaceEmbedding.encode([text])` is called once. If `memory_context` is set, the first 200 characters are prepended to the query text before embedding so the vector is biased toward prior conversation topics.

### Returns
The same `QueryRepresentation`, mutated in-place with `intent`, `entities`, `keywords`, `cypher_query`, and `embedding` populated.

---

## 9. `bm25_retriever.py`

### Purpose
Keyword retriever using the BM25Okapi scoring model. Handles both index building (at ingestion time) and search (at query time).

### Scientific basis — BM25 formula
```
BM25(q, D) = Σ_t  IDF(t) · (tf(t,D) · (k1+1)) / (tf(t,D) + k1·(1 - b + b·|D|/avgdl))
```
- `IDF(t)` — Inverse document frequency: penalises terms that appear in many documents.
- `tf(t,D)` — Term frequency in document D.
- `k1=1.5` — Term saturation: prevents one very frequent term from dominating.
- `b=0.75` — Length normalisation: penalises long documents slightly.
- `avgdl` — Average document length across the corpus.

BM25 outperforms TF-IDF for exact-match queries and is the backbone of Elasticsearch and Lucene.

### Implementation
Uses `rank_bm25.BM25Okapi` (pip package) when available. Falls back to a pure-Python implementation (`_FallbackBM25`) that implements the exact same formula — slower for large corpora but has zero extra dependencies.

### Index persistence
The `rank_bm25.BM25Okapi` object and metadata list are pickled to `bm25.pkl`. Built once at ingestion time; loaded at retrieval time.

### Tokenisation
`_tokenise(text)` lowercases, splits on `\W+`, and filters stop words and tokens shorter than 2 characters. The same stop word set is used in `query_engine.py` for keyword extraction, ensuring query and document tokens match.

### Returns from `search()`
`List[RetrievedChunk]` with `retriever="bm25"`, sorted by descending BM25 score. Zero-score results are excluded.

---

## 10. `vector_retriever.py`

### Purpose
Thin wrapper around `FAISSStore` that applies a minimum-score threshold, updates `RetrievalTrace`, and returns standardised `RetrievedChunk` objects.

### Why a separate wrapper module
`FAISSStore` is a storage layer; `VectorRetriever` is a retrieval policy. Keeping them separate allows the retriever to apply a minimum-score threshold (default 0.3 cosine similarity — below this the match is too weak), log `PhaseStats`, and be swapped for a different ANN backend without touching `FAISSStore`.

### Returns from `search()`
`List[RetrievedChunk]` with `retriever="vector"`, filtered to `score ≥ min_score`, sorted by descending cosine similarity.

---

## 11. `graph_retriever.py`

### Purpose
Graph traversal retriever. Wraps your existing `GraphStore` and orchestrates four complementary strategies to maximise recall from the knowledge graph.

### Four retrieval strategies (executed in order)

#### Strategy 1 — Direct Cypher execution
If `query_engine.py` produced a Cypher query, it is executed directly against Neo4j. This gives maximum precision for well-formed entity queries.

#### Strategy 2 — Entity vector search + neighbourhood traversal
For each query entity, `GraphStore.similarity_search(query_embedding, top_k=5)` finds the best-matching graph nodes using the Neo4j vector index on node embeddings. For each matched node, `GraphStore.get_neighbourhood(node_id, hops=1)` fetches the surrounding subgraph (nodes + relationships). Node descriptions and relationship descriptions are converted to `RetrievedChunk` objects.

**Adaptive depth expansion:** If fewer than `min_graph_results` (default 3) chunks are returned at `hops=1`, the retriever automatically retries with `hops=2` to widen the neighbourhood.

#### Strategy 3 — Exact name lookup
For each entity name extracted by the query engine, `GraphStore.get_node_by_name(name)` performs a case-insensitive exact match. This catches entities the vector search may have missed due to embedding distance.

#### Strategy 4 — Path finding (relational/comparative queries only)
When `QueryIntent` is `RELATIONAL` or `COMPARATIVE` and ≥2 entities are present, `GraphStore.get_paths_between(src_id, tgt_id, max_hops=3)` is called for every entity pair. The path (alternating node → relation → node) is converted to a readable text string and wrapped in a `RetrievedChunk`. This is graph retrieval's killer feature for multi-hop reasoning.

### Returns
A tuple `(List[RetrievedChunk], GraphTraversalResult)`. The `GraphTraversalResult` contains the raw nodes, relationships, and paths for the visualiser.

---

## 12. `graph_visualizer.py`

### Purpose
Generates two representations of the retrieved subgraph for development-time inspection.

### Console output (`print_graph`)
ANSI color-coded display showing:
- All nodes with entity type and description preview. Query-matched entities are highlighted with a green star (★) and bold name.
- All relationships as `SOURCE --[RELATION_TYPE]--> TARGET  conf=0.85`. Edge color indicates confidence: green ≥ 0.8, yellow 0.6–0.8, red < 0.6.
- Paths found (for relational queries) as readable chain strings.

### Mermaid diagram output (`to_mermaid` / `print_mermaid`)
Generates a valid Mermaid flowchart (`graph LR`) string. Relationship line style encodes confidence:
- Solid arrow `--"RELATION"-->` for confidence ≥ 0.8.
- Dashed arrow `--"RELATION"-.->` for 0.6–0.8.
- Double arrow `=="RELATION"==>` for < 0.6.
- Query-matched nodes are rendered as rounded rectangles with green styling.

The Mermaid output can be pasted into https://mermaid.live or any `mermaid` code block renderer.

---

## 13. `hybrid_retriever.py`

### Purpose
Combines the three retriever outputs into a single ranked list using Reciprocal Rank Fusion (RRF).

### Scientific basis — RRF formula
```
RRF(d) = Σ_r  weight_r / (k + rank_r(d))
```
- `rank_r(d)` — Rank of document d in retriever r's list (1-based).
- `k = 60` — Smoothing constant. Empirically optimal per Cormack et al. (2009).
- `weight_r` — Per-retriever weight (see adaptive weighting below).

**Why k=60?** A document ranked 1st contributes 1/61 ≈ 0.016. A document ranked 60th contributes 1/120 ≈ 0.008 — half the value. This smooth decay does not over-penalise mid-ranked results from a single retriever.

**Why RRF instead of score normalisation?** BM25 scores and cosine similarities live on incompatible scales. Min-max normalisation is sensitive to outliers and changes with corpus size. RRF bypasses score normalisation entirely by working only on rank order — robust and essentially parameter-free.

### Adaptive intent-based weights
When `use_adaptive=True` (default), the per-retriever weights are adjusted based on detected query intent:

| Intent | BM25 | Vector | Graph |
|--------|------|--------|-------|
| FACTUAL | 1.0 | 1.2 | 1.0 |
| RELATIONAL | 0.7 | 1.0 | **1.5** |
| PROCEDURAL | **1.2** | 1.0 | 0.9 |
| COMPARATIVE | 0.8 | 1.1 | **1.4** |
| FOLLOW_UP | 0.9 | **1.3** | 1.0 |
| UNKNOWN | 1.0 | 1.0 | 1.0 |

Relational and comparative queries get a higher graph weight because the graph excels at multi-hop reasoning. Procedural queries get higher BM25 weight because they benefit from exact keyword matches (steps, commands, parameters).

### Returns
`List[FusedResult]` with up to `top_k` (default 15) entries, sorted by descending `rrf_score`. Each entry records which retrievers contributed and their individual ranks.

---

## 14. `reranker.py`

### Purpose
Applies cross-encoder reranking to the RRF-fused shortlist to boost precision before the grader.

### Scientific distinction — bi-encoder vs cross-encoder
The three retrievers are effectively bi-encoders: query and documents are encoded independently. This is fast but misses fine-grained query-document interaction.

A **cross-encoder** encodes `(query, document)` as a single input, allowing full attention across both sequences. This produces much more accurate relevance scores at the cost of being 10–50× slower per document. The cross-encoder is applied to a shortlist (≤ 30 candidates) rather than the full corpus — making the trade-off worthwhile.

### Default model
`cross-encoder/ms-marco-MiniLM-L-6-v2`
- Trained on the MS MARCO passage ranking benchmark (530K query-passage pairs).
- 6 transformer layers, ~22M parameters — fast enough for ≤ 50 candidates.
- Raw logits are sigmoid-normalised to `[0, 1]` so scores are interpretable.

Alternative: `cross-encoder/ms-marco-MiniLM-L-12-v2` for higher accuracy at ~2× latency.

### Fallback
If `sentence-transformers` is not installed, falls back to cosine similarity between query and document embeddings using the embedding model. Much weaker but prevents a hard failure.

### Delta tracking
Each `RerankedResult` records `original_rank` (from RRF) and `final_rank` (after reranking). The `delta = final_rank - original_rank` is logged with directional arrows: ↑ = moved up (better), ↓ = moved down (worse). Large deltas indicate the reranker significantly changed the ordering — worth inspecting during development.

### Returns
`List[RerankedResult]` sorted by descending `rerank_score`. All input items are preserved (nothing is dropped here — that is the grader's job).

---

## 15. `grader.py`

### Purpose
LLM-powered quality gate that evaluates each reranked chunk for relevance to the query before allowing it to proceed to answer generation.

### Why an LLM grader (not just a threshold on rerank score)
The reranker scores relevance on a general ranking benchmark (MS MARCO). It does not know your specific document domain, the user's conversational context, or whether a chunk contradicts what was already said in prior turns. The LLM grader uses Groq API key #3 to apply nuanced, domain-aware judgment that the cross-encoder cannot.

### Single batched LLM call
All chunks are graded in one call to minimise latency. The prompt includes:
- The normalised query and detected intent.
- Up to 2 prior conversation turns (from memory) for consistency checking.
- All chunks as a JSON array with 300-character text previews.
- A scoring rubric: 1.0 = directly answers, 0.8 = highly relevant, 0.6 = partially relevant, 0.4 = tangential, 0.2 = adjacent, 0.0 = irrelevant or contradictory.

The LLM returns a JSON array of `{index, score, reason}` entries, one per chunk.

### Verdict logic

| Condition | Verdict |
|-----------|---------|
| `n_passed ≥ min_passed` (default 1) | `PASS` |
| `n_passed ≥ partial_threshold` (default 1) | `PARTIAL` |
| `n_passed < partial_threshold` | `FAIL` |

### Reformulation on FAIL
When verdict is `FAIL`, the grader fires a second (short) LLM call asking the model to suggest a clearer reformulation of the original query. This suggestion is stored in `GradedResult.reformulation` and used by `retrieval_pipeline.py` to trigger one automatic retry with an expanded `top_k`.

### Memory consistency check
If `MemoryStore` has prior turns, the grader prompt includes a block of recent conversation context. The LLM is instructed to flag chunks that contradict established facts with a score near 0.0 and a reason like "contradicts prior answer about X."

---

## 16. `retrieval_pipeline.py`

### Purpose
The single entry point that wires all 15 other modules into a complete, production-ready retrieval pipeline.

### Initialisation
The constructor takes all configuration parameters and builds every component. BM25 and FAISS indices are loaded from disk if paths are provided; if not, those retrievers are disabled gracefully with a warning (the pipeline continues with the remaining retrievers).

### `run(user_input)` — The main method
Calls `_run_once()` internally. If the grader returns `FAIL` and a reformulation is suggested, automatically retries once with the reformulated query and increased `top_k` values (+5 each). The final `RetrievalTrace` is printed via `retrieval_logger.print_trace()`.

### `_run_once()` — Single pipeline pass
Executes all 8 stages in order:
1. Query processing (voice → text, memory injection)
2. Query engine (intent, entities, Cypher, embedding)
3. Cache check (return cached result if hit)
4. Parallel retrieval (BM25 + Vector + Graph)
5. Graph visualisation
6. RRF fusion
7. Cross-encoder reranking
8. LLM grading

### `record_turn(user_query, ai_response, graded_result)` — Memory management
Call this after the answer-generation phase completes. It passes the full response, chunk IDs used, detected intent, and extracted entities to `MemoryStore.add_turn()`.

### `build_indices(chunks, save_dir)` — Index building helper
Convenience method that builds both the BM25 and FAISS indices from a list of chunks and saves them to disk. Call once after ingestion.

### `cache_stats()` and `memory_summary()`
Utility methods for monitoring the pipeline's state.

---

## 17. `__init__.py`

### Purpose
Exposes the complete public API of the `retrieval` package. All 16 classes/functions from all submodules are re-exported so callers only need `from retrieval import RetrievalPipeline, GradedResult, ...`.

---

## 18. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  INGESTION (done once)                                              │
│  Extractors → Chunkers → CombinedExtractor → Neo4j (GraphStore)    │
│                       → HuggingFaceEmbedding → FAISSStore → disk   │
│                       → BM25Retriever → disk                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ (indices loaded at startup)
┌─────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (once per query)                                         │
│                                                                     │
│  user_input (str or audio path)                                     │
│      │                                                              │
│      ▼                                                              │
│  QueryProcessor  ←── MemoryStore (prior turns, coreference)        │
│      │ QueryRepresentation (raw_text, normalised, memory_context)   │
│      ▼                                                              │
│  QueryCache ─── HIT ──────────────────────────────────► GradedResult│
│      │ MISS                                                         │
│      ▼                                                              │
│  QueryEngine (Groq API #2)                                          │
│      │ QueryRepresentation + intent + entities + cypher + embedding │
│      ▼                                                              │
│  ┌──────────────────────────────────────────┐                      │
│  │  Three retrievers (parallel)             │                      │
│  │  BM25Retriever   → List[RetrievedChunk]  │                      │
│  │  VectorRetriever → List[RetrievedChunk]  │                      │
│  │  GraphRetriever  → List[RetrievedChunk]  │                      │
│  │                  + GraphTraversalResult  │                      │
│  └──────────────────────────────────────────┘                      │
│      │                          │                                   │
│      │                   GraphVisualizer (console + Mermaid)        │
│      ▼                                                              │
│  HybridRetriever (RRF)  → List[FusedResult]                        │
│      ▼                                                              │
│  Reranker (cross-encoder) → List[RerankedResult]                   │
│      ▼                                                              │
│  Grader (Groq API #3) → GradedResult                               │
│      │                                                              │
│      ├── PASS/PARTIAL ──────────────────────► context_text          │
│      │                                        → Answer Generation   │
│      └── FAIL ──► reformulation ──► _run_once() retry (once)       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ (after generation)
┌─────────────────────────────────────────────────────────────────────┐
│  pipeline.record_turn(query, response, graded_result)               │
│  → MemoryStore.add_turn() → JSON persistence                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 19. Environment Variables Reference

| Variable | Used in | Description |
|----------|---------|-------------|
| `GROQ_API_KEY_WHISPER` | `query_processor.py` | Groq key for Whisper voice transcription |
| `GROQ_API_KEY_QUERY` | `query_engine.py` | Groq key for intent/entity/Cypher LLM calls |
| `GROQ_API_KEY_GRADER` | `grader.py` | Groq key for relevance grading LLM calls |
| `GROQ_API_KEY` | All three (fallback) | Shared fallback if specific keys not set |
| `NEO4J_URI` | `GraphStore` | Neo4j connection URI (default `neo4j://127.0.0.1:7687`) |
| `NEO4J_USER` | `GraphStore` | Neo4j username (default `neo4j`) |
| `NEO4J_PASSWORD` | `GraphStore` | Neo4j password (default `neo4j1234`) |

All three Groq keys can be set to the same value if you only have one key — the separation exists to prevent rate-limit contention, not because different accounts are required.

---

## 20. Dependency Installation

```bash
# Core retrieval dependencies
pip install faiss-cpu          # FAISS vector index
pip install rank-bm25          # BM25 keyword search
pip install sentence-transformers  # Cross-encoder reranker + embeddings
pip install groq               # Groq API client (Whisper + LLM)

# Already required by ingestion pipeline
pip install neo4j              # Neo4j driver
pip install numpy              # Numerical arrays
pip install spacy              # NLP (used by CombinedExtractor)
python -m spacy download en_core_web_sm

# Extractor dependencies (for the test pipeline)
pip install PyMuPDF            # PDF extraction (fitz)
pip install python-docx        # DOCX extraction
pip install python-pptx        # PPTX extraction
pip install requests beautifulsoup4  # URL extraction
```

All packages are standard PyPI packages with no LangChain dependency.