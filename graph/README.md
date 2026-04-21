# KG-RAG Graph Module

A knowledge-graph ingestion pipeline for academic documents. Extracts entity nodes and semantic relationships from text chunks using a local spaCy NER pass and a Groq-hosted LLM, then persists everything to Neo4j with vector-indexed embeddings for downstream retrieval.

---

## 1. Architecture Overview

```
Chunks
  └─ CombinedExtractor ──── one LLM call per batch ────► List[ExtractedNode]
       │                     (spaCy pre-pass + Groq)      + _pending_rels cache
       │
       └─ (same instance) ── reads from _pending_rels ──► List[ExtractedRelationship]
            │                 (NO second LLM call)
            │
            └─ GraphStore ──────────────────────────────► Neo4j
                             (MERGE upsert + vector index)
```

The critical design insight is that `CombinedExtractor` does both jobs — node extraction and relationship resolution — in a single class. During the node-extraction phase it fires one LLM call per batch that returns both entities **and** raw relationships simultaneously. Those raw relationships are held in an in-memory cache (`_pending_rels`). When the pipeline later calls the relationship-resolution phase, no second LLM call is made — the extractor simply reads from the cache, resolves entity names to node IDs using the now-finalized `node_id_map`, and filters by confidence.

---

## 2. File Reference

### 2.1 `base.py`

Defines the shared data models and the abstract contracts that all extractors must satisfy. Nothing in this file imports from any other file in the module — it is the foundation everything else builds on.

#### `ExtractedNode`

A dataclass representing a single entity ready for Neo4j storage.

```python
@dataclass
class ExtractedNode:
    node_id     : str            # MD5 hash of (name.lower() + entity_type.lower())
    name        : str            # Canonical entity name
    entity_type : str            # One of the 22 allowed types
    description : str            # One-sentence explanation (LaTeX preserved)
    source_chunk: str            # chunk.chunk_id — which chunk this came from
    source      : str            # chunk.metadata["source"] — document identifier
    embedding   : np.ndarray     # Set after extraction; used by Neo4j vector index
    aliases     : List[str]      # Lowercase alternative names (e.g. "bert", "gpt")
```

**`node_id` stability:** The ID is computed as `MD5(name.lower() + ":" + entity_type.lower())[:16]`. This means the same entity — say "Gradient Descent" as a "Method" — always produces the same `node_id` regardless of which chunk or document it was found in. This is what makes deduplication work: two chunks mentioning Gradient Descent produce two `ExtractedNode` objects with identical `node_id`s, and the deduplicator collapses them into one, keeping the longer description and merging aliases.

**`to_neo4j_properties()`:** Serializes to a flat dict for Neo4j writes. Deliberately excludes `embedding` because the Neo4j vector index API expects embeddings to be set via a separate mechanism, not as part of the property map in the `MERGE` query.

#### `ExtractedRelationship`

A dataclass representing a directed semantic edge between two entity nodes.

```python
@dataclass
class ExtractedRelationship:
    source_id    : str    # node_id of the source entity
    target_id    : str    # node_id of the target entity
    source_name  : str    # Human-readable source entity name
    target_name  : str    # Human-readable target entity name
    relation_type: str    # e.g. "USES", "DEFINED_BY", "PROOF_OF"
    description  : str    # Technical explanation of the relationship
    source_chunk : str    # chunk.chunk_id — provenance
    source       : str    # chunk.metadata["source"] — provenance
    confidence   : float  # LLM-assigned 0.0–1.0; edges below threshold are dropped
    mode         : str    # "constrained" or "unconstrained"
```

**Direction is meaningful.** `BERT -[USES]-> Attention Mechanism` is semantically different from `Attention Mechanism -[USED_BY]-> BERT`. The extractor always follows the logical direction as stated in the source text.

**`to_neo4j_properties()`:** Returns the properties stored on the Neo4j relationship. Notably it does not include `source_id` or `target_id` — those are used to `MATCH` the two endpoint nodes in the Cypher `MERGE` query, but they are not stored as relationship properties.

#### `BaseNodeExtractor`

Abstract base class with two required methods.

**`extract_from_chunks(chunks, nodes_per_chunk=None, node_id_map=None, show_progress=True)`**

The signature has optional `nodes_per_chunk` and `node_id_map` parameters to accommodate `CombinedExtractor`'s unified dispatch: when those parameters are `None` the method runs node extraction; when they are provided it runs relationship resolution. Simple implementations that only do node extraction can ignore the extra parameters. The return type is `List[ExtractedNode]` in the node-extraction path.

**`extract_from_single_chunk_public(chunk)`**

Single-chunk extraction used during incremental ingestion. Must return nodes with embeddings already set — unlike the batch path where embedding happens once after full deduplication.

#### `BaseRelationshipExtractor`

Abstract base class with four required methods.

**`extract_from_chunks(chunks, nodes_per_chunk, node_id_map, show_progress=True)`**

Added in the current version. The `Pipeline` calls this directly on `self._relationship_extractor` to resolve relationships after the node step is complete. Without this method on the base, there would be no contract guaranteeing the method exists — any subclass missing it would pass `isinstance` checks and then crash at runtime. In `CombinedExtractor` this is the same method as the one on `BaseNodeExtractor`, dispatched by the presence of the non-`None` arguments.

**`extract_from_chunk(chunk, nodes_in_chunk, node_id_map)`**

Single-chunk relationship extraction for incremental ingestion. Fires a standalone LLM prompt for one chunk.

**`extract_cross_document_references(chunk, node_id_map, all_document_nodes)`**

Detects when a chunk explicitly references concepts introduced in other documents — phrases like "as shown in", "building on", "recall that". Returns `ExtractedRelationship` objects typed with cross-document relation types such as `BUILDS_ON`, `CITES`, or `DEPENDS_ON`.

**`deduplicate_relationships(relationships)`**

Removes duplicate edges. The deduplication key is `(source_id, target_id, relation_type)`. When two edges share the same triple, the one with higher confidence is kept.

---

### 2.2 `llm_backend.py`

A thin, self-contained HTTP wrapper around the Groq chat completions API. It uses only Python's standard library (`urllib`, `json`, `time`) — no `requests`, no `openai` SDK, no local model weights.

#### Construction

```python
llm = LLMBackend(
    api_key     = "gsk_...",          # or set GROQ_API_KEY env var
    model       = "llama-3.3-70b-versatile",
    max_tokens  = 3000,
    temperature = 0.0,                # greedy by default — deterministic output
    batch_size  = 1,                  # how many prompts to combine per API call
)
```

The `api_key` falls back to the `GROQ_API_KEY` environment variable. `python-dotenv` is used automatically if installed.

#### `generate(prompt)` — single call

Sends one user message and returns the assistant reply as a plain string. Retries up to 3 times with a 15-second sleep on HTTP 429 (rate limit) and 5xx (server errors).

#### `generate_batch(prompts)` — batched calls

Takes a list of prompts and reduces API round-trips by combining up to `batch_size` prompts into one request. Each prompt is wrapped with a `### CHUNK N ###` header. The combined response is split back on a `---CHUNK_SEPARATOR---` token. If the model doesn't split cleanly, the raw response is returned for that batch rather than silently losing data.

Note: `CombinedExtractor` does **not** use `generate_batch`. It builds its own combined prompt directly and calls `_call_api` with a per-batch `max_tokens` computed as `max(3000, llm.max_tokens × len(chunks))`.

#### Retry logic

Three error categories are handled:
- `urllib.error.HTTPError` with a retryable status code (429, 500, 502, 503, 504) — sleep and retry.
- `urllib.error.URLError` (network failures, DNS, timeouts) — sleep and retry.
- HTTP 403 with Cloudflare error code 1010 — immediately raises a descriptive error pointing to IP/region restrictions rather than retrying uselessly.

---

### 2.3 `node_relation_extractor.py`

The core of the pipeline. `CombinedExtractor` replaces the old separate `NodeExtractor` and `RelationshipExtractor` classes with a single class that handles both roles in one LLM call per batch.

#### Relation taxonomy

42 allowed relationship types organized into eight groups:

| Group | Types |
|---|---|
| Methodological | `USES`, `EXTENDS`, `IMPLEMENTS`, `IMPROVES_UPON`, `REPLACES`, `COMBINES` |
| Structural | `DEPENDS_ON`, `PART_OF`, `BASED_ON`, `DERIVED_FROM` |
| Evaluation | `EVALUATED_ON`, `TRAINED_ON`, `BENCHMARKED_AGAINST`, `OUTPERFORMS` |
| Attribution | `INTRODUCES`, `PROPOSED_BY`, `DEVELOPED_BY`, `PUBLISHED_IN` |
| Comparison | `COMPARES_WITH`, `CONTRASTS_WITH`, `EQUIVALENT_TO` |
| Knowledge flow | `CITES`, `BUILDS_ON`, `MOTIVATED_BY`, `ADDRESSES`, `SOLVES` |
| Domain | `APPLIED_TO`, `GENERALIZES`, `SPECIALIZES` |
| Mathematical | `DEFINED_BY`, `EXPRESSED_AS`, `PARAMETERIZED_BY`, `CHARACTERIZED_BY`, `IMPLIES`, `SATISFIES`, `ASSUMES`, `PROOF_OF`, `TRANSFORMS_TO`, `MAPS_TO`, `APPROXIMATES`, `SPECIAL_CASE_OF`, `OPTIMIZES`, `CONVERGES_TO`, `BOUNDED_BY` |

The mathematical group is the new addition — the original design only covered the first seven groups. This extension allows the extractor to represent formal mathematical structure: a `Theorem` linked to its `Proof` via `PROOF_OF`, an `Algorithm` linked to a convergence bound via `CONVERGES_TO`, an `Operator` linked to a `Space` via `MAPS_TO`.

#### Entity types

22 types in two groups:

**Conceptual (inherited):** `Concept`, `Algorithm`, `Method`, `Model`, `System`, `Component`, `Signal`, `Formula`, `Theory`, `Metric`, `Task`, `Dataset`, `Framework`, `Author`, `Institution`

**Mathematical (new):** `Theorem`, `Lemma`, `Proof`, `Operator`, `Property`, `Distribution`, `Space`

`Concept` is deliberately the fallback of last resort — the LLM prompt instructs the model to use it only when no more specific type fits.

#### Construction

```python
ext = CombinedExtractor(
    llm                  = llm,            # LLMBackend instance
    embedding_fn         = my_embed_fn,    # Callable[[List[str]], np.ndarray]
    allowed_types        = None,           # defaults to DEFAULT_ALLOWED_TYPES
    mode                 = "constrained",  # or "unconstrained"
    confidence_threshold = 0.6,            # edges below this are dropped
    max_entity_pairs     = 20,             # cap on pairs evaluated per chunk
    batch_chunks         = 2,             # chunks combined per LLM call
)
```

**`mode`** controls relationship type enforcement:
- `"constrained"` — the LLM must use only the 42 types in `ALLOWED_RELATIONS`. Any relationship with a type outside this list is silently dropped during post-processing.
- `"unconstrained"` — the LLM can choose any descriptive label. The label is normalized to `UPPER_SNAKE_CASE` but not filtered.

#### Internal state

`CombinedExtractor` carries two thread-safe caches:

**`_cache`:** A nested dict `text → cache_signature → {"entities": [...]}`. Maps raw chunk text to the list of entities extracted from it. The cache signature includes the model name and allowed types, so changing either parameter invalidates all cached entries automatically. Relationships are deliberately **not** cached here because they carry provenance (`source_chunk`, `source`) that would be wrong when a chunk reappears in a different context.

**`_pending_rels`:** A dict `chunk_id → [raw_rel_dicts]`. Populated as a side-effect of `_extract_nodes`. Consumed — and cleared — by `_extract_rels` when the relationship-resolution phase runs. Raw relationship dicts at this stage contain entity name strings, not node IDs; resolution against `node_id_map` happens in `_resolve_and_filter`.

#### Node extraction flow (`_extract_nodes`)

1. Clear `_pending_rels` from any previous run.
2. Group chunks into batches of `batch_chunks`.
3. For each batch, call `_process_batch`:
   - Run spaCy NER on each chunk (free, local, no API call). spaCy recognizes `PERSON → Author`, `ORG → Institution`, `WORK_OF_ART → Model`, `PRODUCT → Framework`. These become hint annotations passed into the LLM prompt.
   - Check the text-keyed node cache for each chunk. Chunks with cached entities skip the LLM entirely.
   - For uncached chunks, fire one combined LLM call with `_call_combined_llm`. The prompt includes all chunks in the batch, the spaCy hints for each, the full entity type guide, the full relationship taxonomy, and explicit LaTeX preservation instructions. The model returns a single JSON object with a `"chunks"` array, each element containing `"entities"` and `"relationships"` for that chunk.
   - Merge spaCy entities as gap-fill: LLM results are authoritative. spaCy-only entities are appended only for names the LLM missed entirely.
   - Write entities to cache. Write raw relationships to `_pending_rels`.
4. After all batches: deduplicate nodes across the full list (same `node_id` → keep longest description, merge aliases), then embed all deduplicated nodes in a single batched call to `embedding_fn`.

#### Relationship resolution flow (`_extract_rels`)

1. Iterate over chunks.
2. For each chunk, look up `chunk_id` in `_pending_rels`.
   - **Cache hit:** use the raw relationship dicts stored during node extraction.
   - **Cache miss** (e.g. `_extract_rels` called standalone without a prior `_extract_nodes`): fire a standalone single-chunk LLM relationship prompt as fallback.
3. Pass raw dicts through `_resolve_and_filter`:
   - Drop any relationship below `confidence_threshold`.
   - Look up `source_name.lower()` and `target_name.lower()` in `node_id_map`. Drop any relationship where either endpoint cannot be resolved to a known `node_id`.
   - In `constrained` mode: drop any relationship whose type is not in `ALLOWED_RELATIONS`.
   - In `unconstrained` mode: normalize the type to `UPPER_SNAKE_CASE`.
4. Deduplicate the full list (highest confidence wins per triple).

#### Single-chunk incremental path

`extract_from_single_chunk_public(chunk)` and `extract_from_chunk(chunk, nodes_in_chunk, node_id_map)` handle the case where `Pipeline.ingest_chunk` is called for a single new document arriving after the initial bulk ingestion. The node path checks the text cache first, then fires a simpler entity-only prompt (`_build_entity_only_prompt`) if the chunk is new. The relationship path fires a standalone relationship prompt directly.

#### Cross-document reference detection

`extract_cross_document_references` scans chunk text for reference markers (`"recall"`, `"as shown in"`, `"building on"`, `"cite"`, etc.). If any marker is found it fires a targeted LLM prompt asking specifically for cross-document references and returns relationships typed from `CROSS_DOC_RELATIONS = [DEPENDS_ON, EXTENDS, BUILDS_ON, CITES, MOTIVATED_BY, BASED_ON]`.

#### Entity pair prioritization

When a chunk contains more than `max_entity_pairs` candidate pairs, `_select_prompt_nodes` uses `_TYPE_PRIORITY_PAIRS` to rank pairs by expected relationship richness. Mathematical pairs rank highest: `(Theorem, Proof)` and `(Proof, Theorem)` have priority 1. Standard pairs like `(Model, Method)` rank at 3. Pairs not in the map get a default priority of 99 and are dropped when the cap is reached.

#### `_safe_json_parse`

A module-level helper used by all LLM response parsers. Handles:
- Markdown code fences (` ```json ... ``` `) — strips them before parsing.
- Unescaped backslashes (common in Windows paths or LaTeX) — attempts a regex fix.
- JSON embedded in prose — extracts the outermost `{...}` block.
- Top-level JSON arrays — wraps them as `{"items": [...]}`.
Returns an empty dict on total failure rather than raising, so callers can degrade gracefully.

---

### 2.4 `graph_store.py`

The Neo4j persistence layer. Handles all reads and writes to the database. Has no awareness of how extraction works — it only sees `ExtractedNode` and `ExtractedRelationship` objects.

#### Construction

```python
store = GraphStore(
    uri          = "neo4j://127.0.0.1:7687",
    user         = "neo4j",
    password     = "neo4j1234",
    database     = "neo4j",
    embedding_dim = 384,   # must match your embedding model's output dimension
    batch_size   = 256,    # nodes or relationships written per transaction
)
```

The driver calls `verify_connectivity()` immediately on construction — a `ConnectionError` is raised right away if Neo4j is unreachable rather than at the first write.

#### Schema initialization — `init_schema()`

Creates three database objects, all idempotent (`IF NOT EXISTS`):

- **Uniqueness constraint on `Entity.node_id`** — prevents duplicate nodes and creates a backing B-tree index automatically.
- **B-tree index on `Entity.name`** — speeds up `get_node_by_name` lookups.
- **Vector index on `Entity.embedding`** — requires Neo4j 5.11+ with built-in vector support. If creation fails (older Neo4j or missing GDS plugin) a warning is issued and `similarity_search` falls back to brute-force cosine automatically.

Always safe to call on every application start because all three operations are idempotent.

#### `upsert_nodes(nodes)` — write logic

Uses a Cypher `MERGE ... ON CREATE SET ... ON MATCH SET` pattern so ingestion is fully idempotent — running the same document twice produces the same graph state.

Before writing, nodes in the same batch are pre-merged in Python: if two `ExtractedNode` objects share a `node_id`, their aliases are unioned and the longer description wins. This avoids unnecessary round-trips.

`ON MATCH SET` updates only `embedding` (unconditionally, to keep vectors fresh) and `description` (only if the incoming description is longer than what is already stored — mirrors `CombinedExtractor`'s in-memory deduplication policy). `name`, `entity_type`, and `source_chunk` are never overwritten on an existing node.

#### `upsert_relationships(relationships)` — write logic

Neo4j does not allow parameterized relationship type names (`:$type` is not valid Cypher). The workaround is to group relationships by `relation_type` and issue one Cypher statement per type with the type name interpolated directly into the query string. Before interpolation, `_sanitise_rel_type` validates that the type is pure `UPPER_SNAKE_CASE` (`[A-Z][A-Z0-9_]*`) to prevent Cypher injection. Any relationship with a type that fails this check is skipped with a warning.

`ON MATCH SET` keeps the higher-confidence description and confidence score when a relationship already exists — consistent with `deduplicate_relationships` in the extractor.

#### `delete_by_source(source)`

Removes all nodes and relationships whose `source` property matches the given string. Returns `(nodes_deleted, relationships_deleted)`. Designed for re-ingestion: delete the old data for a document, then run the pipeline on the updated version.

#### Retrieval methods

| Method | What it returns |
|---|---|
| `get_node_by_id(node_id)` | Single node dict or `None` |
| `get_node_by_name(name)` | Case-insensitive first match or `None` |
| `similarity_search(query_vector, top_k)` | Top-k nearest nodes by cosine similarity; tries Neo4j vector index, falls back to brute-force |
| `get_neighbourhood(node_id, hops)` | All nodes and relationships within N hops |
| `get_subgraph(node_ids)` | All nodes in a set plus every relationship between them |
| `get_paths_between(source_id, target_id, max_hops)` | All shortest paths as ordered lists of alternating node/relationship dicts |
| `count_nodes()` | Total `Entity` node count |
| `count_relationships()` | Total relationship count |

**`similarity_search`:** First tries the Neo4j built-in vector index (fast ANN). On `ClientError` (index not available) falls back to pulling all embeddings into Python and computing cosine similarity in NumPy. The fallback is slow for large graphs — the vector index should always be preferred in production.

**`get_neighbourhood`:** Uses a variable-length Cypher path match (`[*1..hops]`). Returns both nodes and relationships with full property dicts plus `source_id`, `target_id`, and `relation_type` added to each relationship dict for easy consumption by the retriever.

---

### 2.5 `Pipeline.py`

The orchestration layer. Wires `CombinedExtractor` and `GraphStore` together and exposes a single `.run(chunks)` entry point.

#### Construction — default

```python
pipeline = Pipeline(
    neo4j_uri      = "neo4j://127.0.0.1:7687",   # or NEO4J_URI env var
    neo4j_user     = "neo4j",                      # or NEO4J_USER env var
    neo4j_password = "neo4j1234",                  # or NEO4J_PASSWORD env var
    neo4j_database = "neo4j",
    groq_api_key   = "gsk_...",                    # or GROQ_API_KEY env var
    groq_model     = "llama-3.3-70b-versatile",
    embedding_fn   = my_embed_fn,                  # required — no default
    embedding_dim  = 384,
    batch_chunks   = 2,
    llm_max_tokens = 3000,
    rel_mode       = "constrained",
    rel_confidence = 0.6,
    max_entity_pairs = 20,
    graph_batch_size = 256,
    extract_cross_doc = True,
    show_progress  = True,
)
```

`embedding_fn` is the only required parameter with no default. It must accept `List[str]` and return a 2D `np.ndarray` of shape `(len(texts), embedding_dim)`.

Internally this constructor creates one `LLMBackend`, one `CombinedExtractor`, and one `GraphStore`, then assigns the same `CombinedExtractor` instance to both `self._node_extractor` and `self._relationship_extractor`. This shared reference is essential: the cache written by the node-extraction phase must be readable by the relationship-resolution phase, and they are the same object.

#### Construction — `from_components`

```python
pipeline = Pipeline.from_components(
    node_extractor         = ext,    # CombinedExtractor instance
    relationship_extractor = ext,    # THE SAME instance — not a copy
    graph_store            = store,
    extract_cross_doc      = True,
    schema_already_exists  = False,  # set True to skip init_schema()
)
```

Pass the **same** `CombinedExtractor` instance for both parameters. If you pass two different instances the relationship phase will read from an empty `_pending_rels` cache and fall back to a fresh LLM call for every chunk, negating the whole point of the combined design.

#### `run(chunks)` — the seven steps

1. **Node extraction.** Calls `self._node_extractor.extract_from_chunks(chunks, show_progress=...)`. `CombinedExtractor` runs the full spaCy + LLM + dedup + embed flow and populates `_pending_rels` as a side-effect. Returns deduplicated `List[ExtractedNode]`.

2. **Build `node_id_map`.** Builds a `name.lower() → node_id` mapping from the deduplicated node list. A priority policy handles name collisions: if two nodes have the same surface name but different types, the more specific type wins (`Model` beats `Concept`). Aliases are added with `setdefault` so they never overwrite a canonical name entry. The mathematical entity types (`Theorem`, `Lemma`, etc.) currently fall through to priority 99 — they are in the map but do not have explicit priority tiers yet.

3. **Write nodes to Neo4j.** Calls `self._store.upsert_nodes(nodes)`. Raises and aborts on failure — a partial node write would leave relationships referencing non-existent nodes.

4. **Build `nodes_per_chunk` index.** Iterates over chunks and builds `Dict[int, List[ExtractedNode]]` mapping each chunk's positional index to the nodes that came from it. Used by `CombinedExtractor._extract_rels` to build fallback prompts for any chunk whose pending rels are missing.

5. **Relationship resolution.** Calls `self._relationship_extractor.extract_from_chunks(chunks, nodes_per_chunk, node_id_map, show_progress=...)`. Because `_relationship_extractor` is the same `CombinedExtractor` instance, this dispatches to `_extract_rels`, which reads `_pending_rels` without making any LLM calls.

6. **Cross-document relationships (optional).** If `extract_cross_doc=True` and there is more than one chunk, iterates over chunks calling `extract_cross_document_references`. Results are appended to the main relationship list. Failures per-chunk are caught and recorded in `stats.errors` rather than aborting.

7. **Deduplicate and write relationships.** Calls `deduplicate_relationships` on the combined list, then `self._store.upsert_relationships`. Raises and aborts on failure — at this point all nodes are already committed, so a partial relationship write leaves the graph inconsistent.

#### `PipelineStats`

Returned by `run()`. Carries `chunks_processed`, `nodes_extracted`, `nodes_written`, `relationships_extracted`, `relationships_written`, `cross_doc_relationships`, `elapsed_seconds`, and `errors`. `__str__` formats a readable summary table suitable for printing.

#### `ingest_chunk(chunk)`

Convenience wrapper that calls `self.run([chunk])`. Uses the incremental paths in `CombinedExtractor` rather than the batch paths because `batch_chunks > 1` with a single-element list still works — the batch just has one item.

#### Error philosophy

Node write failures are fatal and re-raise immediately — the pipeline aborts. Relationship extraction failures per-chunk are non-fatal — a warning is recorded in `stats.errors` and extraction continues. Relationship write failures are fatal — all nodes are already committed so re-raising lets the caller retry `upsert_relationships` with the extracted data.

---

### 2.6 `__init__.py`

The public API surface of the `graph` package. Controls what is importable with `from graph import ...`.

```python
from graph import (
    ExtractedNode,
    ExtractedRelationship,
    BaseNodeExtractor,
    BaseRelationshipExtractor,
    LLMBackend,
    CombinedExtractor,
    GraphStore,            # lazy — neo4j not imported until this is accessed
)
```

#### Lazy `GraphStore`

`GraphStore` is not imported at module load time. `graph_store.py` imports `neo4j` at its top level, which means an eager import would make `neo4j` a hard dependency for **every** `from graph import ...` statement — including environments that only do extraction and have no Neo4j. The `__getattr__` hook defers the import until the caller actually accesses the name:

```python
def __getattr__(name: str):
    if name == "GraphStore":
        from .graph_store import GraphStore
        return GraphStore
    raise AttributeError(f"module 'graph' has no attribute {name!r}")
```

`CombinedExtractor`, `LLMBackend`, and the base classes are eagerly imported because they do not carry heavy optional dependencies.

#### What is NOT exposed

The internal implementation details — `_safe_json_parse`, `ALLOWED_RELATIONS`, `_TYPE_PRIORITY_PAIRS`, `_sanitise_rel_type` — are all accessible via their submodules directly but are not part of the public `__all__`. This keeps the package interface narrow and stable.

---

## 3. Data Flow: End to End

```
chunks (List[Chunk])
  │
  │  each chunk has: .text, .chunk_id, .metadata["source"]
  │
  ▼
CombinedExtractor._extract_nodes(chunks)
  │
  ├─ [per batch of batch_chunks]
  │    ├─ spaCy NER → hint dict per chunk
  │    ├─ text cache lookup → skip LLM for cached chunks
  │    └─ LLM call (one per batch) → {"chunks": [{entities, relationships}]}
  │         ├─ entities → merged with spaCy hints
  │         └─ relationships → stored in _pending_rels[chunk_id]
  │
  ├─ deduplication  (same node_id → merge description + aliases)
  └─ batch embedding call → sets node.embedding on each node
  │
  ▼
List[ExtractedNode]  (deduplicated, with embeddings)
  │
  ├─ build node_id_map  (name.lower() → node_id)
  │
  ▼
GraphStore.upsert_nodes(nodes)   → Neo4j :Entity nodes (MERGE, idempotent)
  │
  ▼
CombinedExtractor._extract_rels(chunks, nodes_per_chunk, node_id_map)
  │
  ├─ [per chunk]
  │    ├─ read _pending_rels[chunk_id]  (no LLM call)
  │    └─ _resolve_and_filter
  │         ├─ drop confidence < threshold
  │         ├─ resolve names → node IDs via node_id_map
  │         └─ constrained: drop unknown types
  │
  └─ deduplicate_relationships (highest confidence per triple)
  │
  ▼
[optional] extract_cross_document_references  → additional relationships
  │
  ▼
GraphStore.upsert_relationships(rels)  → Neo4j typed edges (MERGE, idempotent)
  │
  ▼
PipelineStats
```

---

## 4. Key Design Decisions

**One LLM call per batch instead of two.** The previous design made two separate LLM passes — one for nodes, one for relationships. `CombinedExtractor` collapses these into one call per batch. The tradeoff is a more complex prompt and response parsing, but it halves the number of API calls and total latency.

**`_pending_rels` cache over returning relationships from the node step.** The node-extraction method is contractually required to return `List[ExtractedNode]`. Returning relationships as a side channel through a shared cache keeps the interface clean while still avoiding a second LLM call. The cache is cleared at the start of each `_extract_nodes` call to prevent stale data from a previous run leaking into the current one.

**Stable `node_id` hash.** Because every mention of "Gradient Descent" as a "Method" produces the same 16-character MD5 hex string, deduplication is purely set-based — no fuzzy string matching required. The vector index on embeddings handles the fuzzy case at query time.

**MERGE semantics in Neo4j.** All writes use `MERGE` rather than `CREATE`. This makes the pipeline idempotent: ingesting the same document twice leaves the graph identical to ingesting it once. It also means partial failures are recoverable — re-running the pipeline after a crash will write the missing nodes/relationships without duplicating what was already committed.

**Lazy neo4j import.** Keeping `neo4j` out of the eager import path means the extraction code can run in lightweight environments (e.g., a CPU-only Lambda function that only processes text) without requiring the Neo4j driver to be installed.

**`constrained` vs `unconstrained` mode.** Constrained mode gives predictable, consistent relationship types across all documents — essential when the downstream retriever queries by relationship type. Unconstrained mode is better for exploratory ingestion of documents with unusual domain vocabulary where the fixed taxonomy would produce too many dropped relationships.

---

## 5. Configuration Reference

| Parameter | Where set | Default | Effect |
|---|---|---|---|
| `batch_chunks` | `Pipeline`, `CombinedExtractor` | `2` | Chunks per LLM call. Higher = fewer API calls, larger prompts. |
| `llm_max_tokens` | `Pipeline` → `LLMBackend` | `3000` | Max tokens per LLM response. Scaled up automatically for multi-chunk batches. |
| `rel_mode` | `Pipeline`, `CombinedExtractor` | `"constrained"` | Whether to enforce the 42-type taxonomy. |
| `rel_confidence` | `Pipeline`, `CombinedExtractor` | `0.6` | Edges below this score are dropped. |
| `max_entity_pairs` | `Pipeline`, `CombinedExtractor` | `20` | Cap on entity pairs per chunk. Limits relationship prompt size. |
| `embedding_dim` | `Pipeline`, `GraphStore` | `384` | Must match your `embedding_fn` output dimension. |
| `graph_batch_size` | `Pipeline`, `GraphStore` | `256` | Nodes or relationships per Neo4j transaction. |
| `extract_cross_doc` | `Pipeline` | `True` | Whether to run the cross-document reference pass. |
| `temperature` | `LLMBackend` | `0.0` | Greedy sampling — deterministic output. Raise slightly for more variety. |

---

## 6. Quick Start

**Minimal setup:**

```python
from Pipeline import Pipeline

def my_embed_fn(texts):
    # Replace with your actual embedding model
    # Must return np.ndarray of shape (len(texts), embedding_dim)
    ...

pipeline = Pipeline(
    groq_api_key   = "gsk_...",
    neo4j_uri      = "neo4j://127.0.0.1:7687",
    neo4j_password = "neo4j1234",
    embedding_fn   = my_embed_fn,
)

stats = pipeline.run(chunks)
print(stats)
```

**Advanced setup with custom components:**

```python
from graph import LLMBackend, GraphStore
from graph.node_relation_extractor import CombinedExtractor
from Pipeline import Pipeline

llm = LLMBackend(
    api_key    = "gsk_...",
    model      = "llama-3.3-70b-versatile",
    max_tokens = 3000,
)

ext = CombinedExtractor(
    llm                  = llm,
    embedding_fn         = my_embed_fn,
    mode                 = "constrained",
    confidence_threshold = 0.7,
    batch_chunks         = 3,
)

store = GraphStore(
    uri          = "neo4j://127.0.0.1:7687",
    password     = "neo4j1234",
    embedding_dim = 384,
)
store.init_schema()

# Pass the SAME ext instance for both roles
pipeline = Pipeline.from_components(
    node_extractor         = ext,
    relationship_extractor = ext,
    graph_store            = store,
)

stats = pipeline.run(chunks)
```

**Incremental ingestion:**

```python
# After initial bulk ingestion, ingest one new chunk at a time
stats = pipeline.ingest_chunk(new_chunk)
```

**Querying the graph directly:**

```python
# Vector similarity search
results = store.similarity_search(query_embedding, top_k=5)

# Neighbourhood traversal
subgraph = store.get_neighbourhood(node_id, hops=2)

# All paths between two concepts
paths = store.get_paths_between(source_id, target_id, max_hops=3)

# Re-ingest an updated document
store.delete_by_source("lecture_03.pdf")
pipeline.run(new_chunks_for_lecture_03)
```