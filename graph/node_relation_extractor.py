# graph/node_relation_extractor.py
"""
Combined node AND relationship extractor for academic documents.

Key design goals
----------------
1. ONE LLM call per batch of up to BATCH_CHUNKS (default 2) chunks — extracts
   both entities and relationships simultaneously, halving round-trips.
2. First-class support for two academic document modes:
      • Conceptual  — ideas, methods, models, frameworks, authors
      • Mathematical — theorems, operators, proofs, formulae, spaces
   LaTeX expressions are preserved verbatim in entity descriptions and
   relationship descriptions so the downstream graph keeps full math context.
3. Extended relationship taxonomy that covers standard academic relations
   (USES, EXTENDS, BASED_ON, …) PLUS mathematical/structural relations
   (DEFINED_BY, IMPLIES, PROOF_OF, CONVERGES_TO, TRANSFORMS_TO, …).
4. All original NodeExtractor + RelationshipExtractor logic is preserved:
      • spaCy NER pre-pass (free, fast gap-fill)
      • LLM is authoritative — spaCy fills what LLM misses
      • Text-keyed cache with version signature
      • Global node deduplication + alias merging
      • Batched embedding call after deduplication
      • Confidence threshold filtering
      • UPPER_SNAKE_CASE normalisation for unconstrained mode
      • Cross-document reference detection (markers scan + LLM)
      • Relationship deduplication — highest confidence wins per triple
5. Pipeline-compatible duck-typing:
      extract_from_chunks(chunks, show_progress)
          → List[ExtractedNode]          [node-extractor interface]
      extract_from_chunks(chunks, nodes_per_chunk, node_id_map, show_progress)
          → List[ExtractedRelationship]  [relationship-extractor interface]

Usage
-----
    from graph.node_relation_extractor import CombinedExtractor
    from graph import LLMBackend, GraphStore
    from Pipeline import Pipeline

    llm  = LLMBackend(api_key="gsk_...", model="llama-3.3-70b-versatile")
    ext  = CombinedExtractor(llm=llm, embedding_fn=my_embed_fn)

    pipeline = Pipeline.from_components(
        node_extractor         = ext,   # CombinedExtractor satisfies both roles
        relationship_extractor = ext,
        graph_store            = store,
    )
    stats = pipeline.run(chunks)
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import warnings
from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import spacy
    from spacy.language import Language as SpacyModel
except ImportError as e:
    raise ImportError(
        "spacy is required.\n"
        "pip install spacy && python -m spacy download en_core_web_sm"
    ) from e

from .base import (
    BaseNodeExtractor,
    BaseRelationshipExtractor,
    ExtractedNode,
    ExtractedRelationship,
)
from .llm_backend import LLMBackend


# ── Extended relation taxonomy ────────────────────────────────────────────────

# ── Methodological (inherited) ────────────────────────────────────────────────
_METHODOLOGICAL = [
    "USES",           # A uses B as a component or sub-method
    "EXTENDS",        # A builds on / is a variant of B
    "IMPLEMENTS",     # A is a concrete implementation of B
    "IMPROVES_UPON",  # A outperforms or advances beyond B
    "REPLACES",       # A supersedes B
    "COMBINES",       # A merges B and C (multi-target)
]

# ── Dependency & Structure (inherited) ───────────────────────────────────────
_STRUCTURAL = [
    "DEPENDS_ON",     # Understanding / running A requires B
    "PART_OF",        # A is a sub-component of B
    "BASED_ON",       # A is grounded in / derived from B
    "DERIVED_FROM",   # A is a direct derivation of B
]

# ── Evaluation & Training (inherited) ────────────────────────────────────────
_EVALUATION = [
    "EVALUATED_ON",
    "TRAINED_ON",
    "BENCHMARKED_AGAINST",
    "OUTPERFORMS",
]

# ── Attribution (inherited) ──────────────────────────────────────────────────
_ATTRIBUTION = [
    "INTRODUCES",
    "PROPOSED_BY",
    "DEVELOPED_BY",
    "PUBLISHED_IN",
]

# ── Comparison & Contrast (inherited) ────────────────────────────────────────
_COMPARISON = [
    "COMPARES_WITH",
    "CONTRASTS_WITH",
    "EQUIVALENT_TO",  # also used for mathematical equivalence
]

# ── Knowledge Flow (inherited) ───────────────────────────────────────────────
_KNOWLEDGE_FLOW = [
    "CITES",
    "BUILDS_ON",
    "MOTIVATED_BY",
    "ADDRESSES",
    "SOLVES",
]

# ── Domain Application (inherited) ───────────────────────────────────────────
_DOMAIN = [
    "APPLIED_TO",
    "GENERALIZES",
    "SPECIALIZES",
]

# ── Mathematical Relations (NEW) ─────────────────────────────────────────────
_MATHEMATICAL = [
    # Definition & representation
    "DEFINED_BY",       # A is formally defined using/as B
    "EXPRESSED_AS",     # A can be written/expressed as mathematical form B
    "PARAMETERIZED_BY", # A is parameterized or controlled by B
    "CHARACTERIZED_BY", # A is characterized / uniquely determined by property B

    # Logical structure
    "IMPLIES",          # A logically / mathematically implies B
    "SATISFIES",        # A satisfies condition, constraint, or property B
    "ASSUMES",          # A requires assumption / hypothesis B
    "PROOF_OF",         # A is a proof or formal derivation of theorem B

    # Algebraic / analytic operations
    "TRANSFORMS_TO",    # A is transformed into B (via an operator / change of basis)
    "MAPS_TO",          # Function or operator A maps its domain to B
    "APPROXIMATES",     # A is an approximation or estimate of B
    "SPECIAL_CASE_OF",  # A is a special / degenerate case of B

    # Optimization & convergence
    "OPTIMIZES",        # A minimizes or maximizes objective B
    "CONVERGES_TO",     # Sequence or iterative method A converges to value / solution B
    "BOUNDED_BY",       # A is bounded above / below by expression B
]

ALLOWED_RELATIONS: List[str] = (
    _METHODOLOGICAL +
    _STRUCTURAL     +
    _EVALUATION     +
    _ATTRIBUTION    +
    _COMPARISON     +
    _KNOWLEDGE_FLOW +
    _DOMAIN         +
    _MATHEMATICAL
)

# Cross-document relations (subset used by cross-document pass)
CROSS_DOC_RELATIONS: List[str] = [
    "DEPENDS_ON", "EXTENDS", "BUILDS_ON",
    "CITES", "MOTIVATED_BY", "BASED_ON",
]

# ── Entity-type pair priority for relation candidate selection ────────────────
_TYPE_PRIORITY_PAIRS: Dict[Tuple[str, str], int] = {
    # Math-specific high-priority pairs
    ("Theorem",    "Proof")       : 1,
    ("Proof",      "Theorem")     : 1,
    ("Formula",    "Theorem")     : 1,
    ("Theorem",    "Formula")     : 1,
    ("Operator",   "Space")       : 2,
    ("Property",   "Theorem")     : 2,
    ("Theory",     "Theorem")     : 2,
    # Standard high-priority pairs (from original)
    ("Model",      "Method")      : 3,
    ("Method",     "Model")       : 3,
    ("Method",     "Concept")     : 4,
    ("Concept",    "Method")      : 4,
    ("Model",      "Dataset")     : 5,
    ("Method",     "Task")        : 5,
    ("Method",     "Theory")      : 5,
    ("Theory",     "Method")      : 5,
    ("Algorithm",  "Theory")      : 5,
    ("Theory",     "Algorithm")   : 5,
    ("Model",      "Metric")      : 6,
    ("Author",     "Model")       : 7,
    ("Author",     "Method")      : 7,
    ("Author",     "Theorem")     : 7,
}


# ── JSON Parsing Helpers ──────────────────────────────────────────────────────

def _safe_json_parse(raw_text: str) -> dict:
    """
    Robustly parse JSON from LLM output, handling common formatting issues.
    
    Handles:
      - Markdown code fences (```json ... ```)
      - Unescaped backslashes (Windows paths, regex patterns)
      - Partial JSON (extracts innermost {...})
    
    Returns empty dict if parsing fails after all recovery attempts.
    """
    if not raw_text:
        return {}
    
    # Step 1: Remove markdown code fences
    clean = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
    clean = re.sub(r'\s*```$', '', clean).strip()
    
    # Step 2: Attempt direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    
    # Step 3: Try to fix unescaped backslashes
    # This is a common issue when LLM generates Windows paths or regex patterns
    try:
        # Replace single backslashes with escaped backslashes,
        # but avoid double-escaping already-escaped ones
        fixed = re.sub(r'(?<!\\)\\(?!["/\\bfnrtu])', r'\\\\', clean)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Step 4: Extract JSON-like object from text
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    # Step 5: Try extracting list if no object found
    start = clean.find("[")
    end   = clean.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(clean[start:end + 1])
            if isinstance(data, list):
                return {"items": data}
        except json.JSONDecodeError:
            pass
    
    return {}


# ── CombinedExtractor ─────────────────────────────────────────────────────────

class CombinedExtractor(BaseNodeExtractor, BaseRelationshipExtractor):
    """
    Extracts entity nodes AND semantic relationships from academic text chunks
    using a single LLM call per batch of up to ``batch_chunks`` (default 2) chunks.

    Dual interface
    --------------
    This class satisfies both the NodeExtractor and RelationshipExtractor
    interfaces used by Pipeline.from_components().  Pass the same instance
    for both parameters:

        pipeline = Pipeline.from_components(
            node_extractor         = combined,
            relationship_extractor = combined,
            graph_store            = store,
        )

    The Pipeline first calls extract_from_chunks(chunks, show_progress=...)
    which runs the combined extraction and caches raw relationships keyed by
    chunk_id.  It then calls extract_from_chunks(chunks, nodes_per_chunk,
    node_id_map, show_progress=...) which resolves those cached relationships
    against the finalized node_id_map and returns filtered ExtractedRelationship
    objects.

    Entity types
    ------------
    Conceptual (from NodeExtractor):
        Concept, Algorithm, Method, Model, System, Component, Signal,
        Formula, Theory, Metric, Task, Dataset, Framework, Author, Institution

    Mathematical (NEW):
        Theorem, Lemma, Proof, Operator, Property, Distribution, Space

    Relationship types
    ------------------
    All 27 original types PLUS 15 mathematical types:
        DEFINED_BY, EXPRESSED_AS, PARAMETERIZED_BY, CHARACTERIZED_BY,
        IMPLIES, SATISFIES, ASSUMES, PROOF_OF,
        TRANSFORMS_TO, MAPS_TO, APPROXIMATES, SPECIAL_CASE_OF,
        OPTIMIZES, CONVERGES_TO, BOUNDED_BY

    LaTeX preservation
    ------------------
    Descriptions returned by the LLM are allowed to contain LaTeX
    (e.g. ``$\\nabla_\\theta \\mathcal{L}$``).  No stripping is applied
    so the downstream graph retains full mathematical context.

    Args
    ----
    llm                  : LLMBackend instance.
    embedding_fn         : Callable[[List[str]], np.ndarray].
    max_workers          : (Unused — batches run sequentially to preserve
                           pending_rels consistency; kept for API compat.)
    allowed_types        : Entity types to keep.  Defaults to DEFAULT_ALLOWED_TYPES.
    mode                 : "constrained" | "unconstrained".
    confidence_threshold : Minimum confidence to keep a relationship (default 0.6).
    max_entity_pairs     : Cap on entity pairs evaluated per chunk (default 20).
    batch_chunks         : Number of chunks combined into a single LLM call (default 2).
    """

    # ── Entity type registry ─────────────────────────────────────────────────
    DEFAULT_ALLOWED_TYPES: List[str] = [
        # ── Inherited from NodeExtractor ────────────────────────────────────
        "Concept",        # General ideas (Entropy, Stability, Voltage)
        "Algorithm",      # Step-by-step procedures (Gradient Descent, Binary Search)
        "Method",         # Techniques (Backpropagation, Fourier Analysis)
        "Model",          # Systems/models (BERT, State Space Model, Linear Regression)
        "System",         # Physical or abstract systems (Control System)
        "Component",      # Physical/electronic parts (Resistor, Capacitor, Sensor)
        "Signal",         # Signal types (Analog Signal, Step Signal)
        "Formula",        # Equations (Ohm's Law, Transfer Function)
        "Theory",         # Formal theories (Control Theory, Probability Theory)
        "Metric",         # Evaluation measures (Accuracy, Error Rate)
        "Task",           # Problems (Classification, Filtering)
        "Dataset",        # Data collections (MNIST, SQuAD)
        "Framework",      # Tools (PyTorch, MATLAB)
        "Author",         # Researchers / people
        "Institution",    # Organisations / labs
        # ── Mathematical extensions (NEW) ───────────────────────────────────
        "Theorem",        # Named mathematical theorems (Bayes' Theorem, CLT)
        "Lemma",          # Intermediate mathematical results / sub-theorems
        "Proof",          # Formal mathematical proof or derivation technique
        "Operator",       # Mathematical operators / transforms (Fourier, Laplacian)
        "Property",       # Mathematical properties / conditions (Linearity, Convexity)
        "Distribution",   # Probability distributions (Gaussian, Bernoulli, Poisson)
        "Space",          # Mathematical / function spaces (Hilbert, L2, Banach)
    ]

    _CACHE_SCHEMA_VERSION = "v1"

    def __init__(
        self,
        llm                  : LLMBackend,
        embedding_fn,
        max_workers          : int                                     = 2,
        allowed_types        : Optional[List[str]]                     = None,
        mode                 : Literal["constrained", "unconstrained"] = "constrained",
        confidence_threshold : float                                   = 0.6,
        max_entity_pairs     : int                                     = 20,
        batch_chunks         : int                                     = 2,
    ):
        if mode not in ("constrained", "unconstrained"):
            raise ValueError(f"mode must be 'constrained' or 'unconstrained', got '{mode}'")

        self.llm                  = llm
        self.embedding_fn         = embedding_fn
        self.max_workers          = max_workers          # kept for API compatibility
        self.allowed_types        = allowed_types or self.DEFAULT_ALLOWED_TYPES
        self.mode                 = mode
        self.confidence_threshold = confidence_threshold
        self.max_entity_pairs     = max_entity_pairs
        self.batch_chunks         = max(1, batch_chunks)

        # ── Node cache (text → sig → {"entities": [...]}) ───────────────────
        # Stores only stable extraction payloads (no provenance).
        # Thread-safe via _cache_lock.
        self._cache: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}
        self._cache_lock = threading.Lock()

        # ── Pending relationship cache (chunk_id_str → [raw_rel_dicts]) ─────
        # Populated during the node-extraction phase.
        # Consumed (and cleared) during the relationship-extraction phase.
        self._pending_rels: Dict[str, List[dict]] = {}
        self._pending_rels_lock = threading.Lock()

        self._nlp = self._load_spacy()

    # ═════════════════════════════════════════════════════════════════════════
    # BaseNodeExtractor  — public contract
    # ═════════════════════════════════════════════════════════════════════════

    def extract_from_chunks(
        self,
        chunks,
        nodes_per_chunk = None,
        node_id_map     = None,
        show_progress   : bool = True,
    ):
        """
        Unified entry point — dispatches on argument signature.

        Called with only (chunks, show_progress)
            → node-extractor role: runs combined LLM extraction, returns nodes.
        Called with (chunks, nodes_per_chunk, node_id_map, show_progress)
            → relationship-extractor role: resolves cached raw rels, returns rels.
        """
        if nodes_per_chunk is None and node_id_map is None:
            return self._extract_nodes(chunks, show_progress)
        else:
            return self._extract_rels(chunks, nodes_per_chunk, node_id_map, show_progress)

    def extract_from_single_chunk_public(self, chunk) -> List[ExtractedNode]:
        """Single-chunk extraction for incremental ingestion."""
        nodes = self._process_single_chunk_nodes(chunk)
        return self._embed_nodes(nodes)

    # ═════════════════════════════════════════════════════════════════════════
    # BaseRelationshipExtractor  — public contract
    # ═════════════════════════════════════════════════════════════════════════

    def extract_from_chunk(
        self,
        chunk,
        nodes_in_chunk : List[ExtractedNode],
        node_id_map    : dict,
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships for a single chunk (incremental ingestion).
        Falls back to building a fresh combined prompt for that one chunk.
        """
        if len(nodes_in_chunk) < 2:
            return []

        prompt_nodes = self._select_prompt_nodes(nodes_in_chunk)
        if len(prompt_nodes) < 2:
            return []

        raw_rels = self._llm_extract_rels_for_prompt_nodes(chunk.text, prompt_nodes)
        return self._resolve_and_filter(raw_rels, node_id_map, chunk)

    def extract_cross_document_references(
        self,
        chunk,
        node_id_map        : dict,
        all_document_nodes : dict,
    ) -> List[ExtractedRelationship]:
        """
        Detect references to concepts introduced in other documents.
        Identical logic to RelationshipExtractor — preserved verbatim.
        """
        text = chunk.text

        markers = [
            "recall", "remember", "as we saw", "as shown in",
            "previously", "introduced in", "building on",
            "as discussed", "following", "based on prior work",
            "cite", "cited", "reference", "as proposed by",
        ]
        if not any(m in text.lower() for m in markers):
            return []

        cross_types = ", ".join(CROSS_DOC_RELATIONS)
        prompt = f"""Analyse this academic text for explicit references to work or concepts
introduced in other documents, earlier sections, or prior literature.

Text:
{text}

Look for language like:
"recall", "as shown in [X]", "building on [X]", "as proposed by", "cite", etc.

For each cross-reference found:
- source: concept / method discussed NOW
- target: concept / work being REFERENCED
- relation_type: one of [{cross_types}]
- description: quote the reference phrase exactly
- confidence: 0.9 if explicit, 0.6 if implied

Respond ONLY with JSON — no preamble, no markdown:
{{
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "relation_type": "relation type",
      "description": "quoted phrase",
      "confidence": 0.9
    }}
  ]
}}"""

        try:
            raw    = self.llm.generate(prompt)
            parsed = _safe_json_parse(raw)
            items  = parsed.get("relationships", [])
        except Exception as e:
            warnings.warn(f"Cross-document extraction failed: {e}")
            return []

        if self.mode == "constrained":
            items = [
                i for i in items
                if i.get("relation_type") in CROSS_DOC_RELATIONS
            ]

        return self._resolve_and_filter(items, node_id_map, chunk)

    def deduplicate_relationships(
        self,
        relationships: List[ExtractedRelationship],
    ) -> List[ExtractedRelationship]:
        """
        Keep highest-confidence edge per (source_id, target_id, relation_type).
        Preserved verbatim from RelationshipExtractor.
        """
        seen: Dict[Tuple, ExtractedRelationship] = {}
        for rel in relationships:
            key = (rel.source_id, rel.target_id, rel.relation_type)
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        return list(seen.values())

    # ═════════════════════════════════════════════════════════════════════════
    # Internal — node extraction flow
    # ═════════════════════════════════════════════════════════════════════════

    def _extract_nodes(self, chunks: list, show_progress: bool) -> List[ExtractedNode]:
        """
        Main node-extraction driver.

        1. Clear pending-rel cache for this run.
        2. Process chunks in batches of ``self.batch_chunks``.
        3. Deduplicate and embed.
        """
        # Clear stale pending rels from any previous run
        with self._pending_rels_lock:
            self._pending_rels.clear()

        all_nodes: List[ExtractedNode] = []
        total   = len(chunks)
        batches = [chunks[i:i + self.batch_chunks]
                   for i in range(0, total, self.batch_chunks)]

        for b_idx, batch in enumerate(batches):
            if show_progress:
                done = b_idx * self.batch_chunks
                print(f"  Combined extract: batch {b_idx + 1}/{len(batches)} "
                      f"({done}–{min(done + self.batch_chunks, total)}/{total} chunks)",
                      end="\r")
            try:
                batch_result = self._process_batch(batch)
            except RuntimeError:
                raise  # propagate LLM-fatal errors
            except Exception as e:
                warnings.warn(f"Batch {b_idx} extraction failed: {e}")
                continue

            for chunk in batch:
                chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))
                source   = chunk.metadata.get("source", "unknown")
                payload  = batch_result.get(chunk_id, {})
                entities = payload.get("entities", [])
                raw_rels = payload.get("relationships", [])

                nodes = self._build_nodes_from_entities(entities, source, chunk_id)
                all_nodes.extend(nodes)

                with self._pending_rels_lock:
                    self._pending_rels[chunk_id] = raw_rels

        if show_progress:
            print(f"  Combined extract: {total}/{total} chunks ✓" + " " * 20)

        deduplicated = self._deduplicate_nodes(all_nodes)
        deduplicated = self._embed_nodes(deduplicated)
        print(f"  ✓ {len(deduplicated)} unique nodes from {total} chunks")
        return deduplicated

    def _process_batch(self, batch: list) -> Dict[str, dict]:
        """
        Run spaCy + one LLM call for a batch of chunks.

        Returns
        -------
        dict mapping chunk_id (str) → {"entities": [...], "relationships": [...]}
        """
        cache_sig = self._cache_signature()
        results: Dict[str, dict] = {}

        # ── Phase 1: per-chunk spaCy NER (free, local) ───────────────────────
        spacy_per_chunk: Dict[str, List[dict]] = {}
        for chunk in batch:
            chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))
            spacy_per_chunk[chunk_id] = self._extract_with_spacy(chunk.text)

        # ── Phase 2: check node cache per chunk ──────────────────────────────
        uncached_chunks = []
        for chunk in batch:
            chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))
            with self._cache_lock:
                cached = self._cache.get(chunk.text, {}).get(cache_sig)
            if cached is not None:
                # Rebuild from cache — relationships are never cached (provenance-free)
                entities = cached.get("entities", [])
                results[chunk_id] = {"entities": entities, "relationships": []}
            else:
                uncached_chunks.append(chunk)

        if not uncached_chunks:
            return results

        # ── Phase 3: single combined LLM call for all uncached chunks ─────────
        llm_result = self._call_combined_llm(uncached_chunks, spacy_per_chunk)

        # ── Phase 4: merge + write to cache ──────────────────────────────────
        for chunk in uncached_chunks:
            chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))
            llm_payload = llm_result.get(chunk_id, {"entities": [], "relationships": []})

            # Merge spaCy entities as gap-fill (LLM is authoritative)
            llm_entities   = llm_payload.get("entities", [])
            spacy_entities = spacy_per_chunk.get(chunk_id, [])
            merged_entities = self._merge_entity_sets(spacy_entities, llm_entities)
            normalised      = self._normalize_cache_entities(merged_entities)

            # Write to node cache (entities only — rels are not cached)
            with self._cache_lock:
                bucket = self._cache.setdefault(chunk.text, {})
                bucket.setdefault(cache_sig, {"entities": normalised})
                final_entities = bucket[cache_sig]["entities"]

            results[chunk_id] = {
                "entities"     : final_entities,
                "relationships": llm_payload.get("relationships", []),
            }

        return results

    def _call_combined_llm(
        self,
        chunks         : list,
        spacy_per_chunk: Dict[str, List[dict]],
    ) -> Dict[str, dict]:
        """
        Build and fire the combined prompt; parse the JSON response.

        Returns
        -------
        dict mapping chunk_id (str) → {"entities": [...], "relationships": [...]}
        """
        prompt = self._build_combined_prompt(chunks, spacy_per_chunk)

        # Allow generous tokens: entities + rels per chunk × number of chunks
        effective_max = max(3000, self.llm.max_tokens * len(chunks))
        try:
            raw = self.llm._call_api(prompt, max_tokens=effective_max)
        except Exception as e:
            warnings.warn(f"Combined LLM call failed: {e}")
            raise RuntimeError("LLM extraction failed — pipeline degraded") from e

        if not raw:
            raise RuntimeError("LLM extraction failed — empty model output")

        parsed = _safe_json_parse(raw)
        if not parsed:
            raise RuntimeError("LLM extraction failed — could not parse JSON from model output")

        chunk_results = parsed.get("chunks", [])
        if not isinstance(chunk_results, list):
            raise RuntimeError("LLM extraction failed — 'chunks' is not a list")

        # Map chunk_index → chunk_id
        output: Dict[str, dict] = {}
        for item in chunk_results:
            if not isinstance(item, dict):
                continue
            idx = item.get("chunk_index")
            if idx is None or not (0 <= idx < len(chunks)):
                continue
            chunk     = chunks[idx]
            chunk_id  = str(getattr(chunk, "chunk_id", id(chunk)))
            entities  = self._validate_entities(item.get("entities", []))
            rels      = self._validate_rels(item.get("relationships", []))
            output[chunk_id] = {"entities": entities, "relationships": rels}

        return output

    # ═════════════════════════════════════════════════════════════════════════
    # Internal — relationship extraction flow
    # ═════════════════════════════════════════════════════════════════════════

    def _extract_rels(
        self,
        chunks          : list,
        nodes_per_chunk,
        node_id_map     : dict,
        show_progress   : bool,
    ) -> List[ExtractedRelationship]:
        """
        Resolve cached raw relationships from the node-extraction phase.
        Falls back to a fresh single-chunk LLM call for any chunk whose
        pending rels are missing (e.g. if called standalone without prior
        _extract_nodes).
        """
        all_rels: List[ExtractedRelationship] = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if show_progress:
                print(f"  Rels: resolving {i + 1}/{total} chunks", end="\r")

            chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))

            with self._pending_rels_lock:
                raw_rels = self._pending_rels.get(chunk_id)

            if raw_rels is None:
                # No cached rels — build a fresh call for this chunk
                nodes_here = (nodes_per_chunk or {}).get(i, [])
                if len(nodes_here) >= 2:
                    prompt_nodes = self._select_prompt_nodes(nodes_here)
                    if len(prompt_nodes) >= 2:
                        raw_rels = self._llm_extract_rels_for_prompt_nodes(
                            chunk.text, prompt_nodes
                        )
                    else:
                        raw_rels = []
                else:
                    raw_rels = []

            resolved = self._resolve_and_filter(raw_rels, node_id_map, chunk)
            all_rels.extend(resolved)

        if show_progress:
            print(f"  Rels: {total}/{total} chunks resolved ✓" + " " * 20)

        deduped = self.deduplicate_relationships(all_rels)
        print(f"  ✓ {len(deduped)} unique relationships from {total} chunks")
        return deduped

    def _llm_extract_rels_for_prompt_nodes(
        self,
        text        : str,
        prompt_nodes: List[ExtractedNode],
    ) -> List[dict]:
        """
        Fire a standalone relationship-extraction prompt (single chunk fallback).
        Mirrors the logic from RelationshipExtractor._build_extraction_prompt.
        """
        entity_list = self._format_entity_list(prompt_nodes)
        constrained = (self.mode == "constrained")

        if constrained:
            relation_section = (
                "Allowed relationship types (use ONLY these):\n"
                + "\n".join(f"  {r}" for r in ALLOWED_RELATIONS)
            )
            rule_line = "- relation_type MUST be one of the allowed types above"
        else:
            relation_section = (
                "Relationship types:\n"
                "  Choose any descriptive relation type in UPPER_SNAKE_CASE."
            )
            rule_line = "- relation_type should be UPPER_SNAKE_CASE, descriptive, concise"

        prompt = f"""You are an expert in technical and mathematical education.

Extract HIGH-QUALITY semantic relationships between the following entities
found in the given academic text.

Entities:
{entity_list}

{relation_section}

Text:
{text}

Rules:
- Only extract relationships EXPLICITLY stated or strongly implied
- {rule_line}
- Confidence 0.9+ only when explicitly stated; 0.6–0.8 for strong implications
- Maximum 10 relationships
- Preserve any LaTeX in descriptions verbatim

Output (STRICT JSON ONLY — no markdown):
{{
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "relation_type": "RELATION_TYPE",
      "description": "clear technical explanation",
      "confidence": 0.9
    }}
  ]
}}"""

        try:
            raw   = self.llm.generate(prompt)
            parsed = _safe_json_parse(raw)
            return parsed.get("relationships", [])
        except Exception as e:
            warnings.warn(f"Standalone rel extraction failed: {e}")
            return []

    # ═════════════════════════════════════════════════════════════════════════
    # Internal — single-chunk node helper (for incremental ingestion)
    # ═════════════════════════════════════════════════════════════════════════

    def _process_single_chunk_nodes(self, chunk) -> List[ExtractedNode]:
        """spaCy + LLM for exactly one chunk (used by extract_from_single_chunk_public)."""
        text     = chunk.text
        source   = chunk.metadata.get("source", "unknown")
        chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))
        cache_sig = self._cache_signature()

        with self._cache_lock:
            cached = self._cache.get(text, {}).get(cache_sig)

        if cached is not None:
            return self._build_nodes_from_entities(
                cached.get("entities", []), source, chunk_id
            )

        spacy_items = self._extract_with_spacy(text)
        llm_items   = self._extract_single_chunk_llm(text)
        merged      = self._merge_entity_sets(spacy_items, llm_items)
        normalised  = self._normalize_cache_entities(merged)

        with self._cache_lock:
            bucket = self._cache.setdefault(text, {})
            bucket.setdefault(cache_sig, {"entities": normalised})
            entities = bucket[cache_sig]["entities"]

        return self._build_nodes_from_entities(entities, source, chunk_id)

    def _extract_single_chunk_llm(self, text: str) -> List[dict]:
        """
        Fires the entity-only extraction prompt for a single chunk.
        Mirrors NodeExtractor._extract_with_llm — preserved verbatim.
        """
        allowed = ", ".join(self.allowed_types)
        prompt  = self._build_entity_only_prompt(text, allowed)
        try:
            raw   = self.llm.generate(prompt)
            parsed = _safe_json_parse(raw)
            if not parsed:
                raise RuntimeError("Could not parse JSON from model output")
            return self._validate_entities(parsed.get("entities", []))
        except Exception as e:
            warnings.warn(f"Single-chunk LLM entity extraction failed: {e}")
            raise RuntimeError("LLM extraction failed — pipeline degraded") from e

    # ═════════════════════════════════════════════════════════════════════════
    # Prompt construction
    # ═════════════════════════════════════════════════════════════════════════

    def _build_combined_prompt(
        self,
        chunks         : list,
        spacy_per_chunk: Dict[str, List[dict]],
    ) -> str:
        """
        Build the combined extraction prompt for up to 5 chunks.

        The prompt instructs the LLM to:
        1. Extract entities (nodes) for each chunk.
        2. Extract relationships using ONLY the entities found in that chunk.
        3. Preserve LaTeX verbatim in descriptions.
        4. Return a single JSON object with a "chunks" array.
        """
        allowed_types  = ", ".join(self.allowed_types)
        constrained    = (self.mode == "constrained")

        if constrained:
            relation_section = (
                "Allowed relationship types — use ONLY these:\n"
                + "\n".join(f"  • {r}" for r in ALLOWED_RELATIONS)
            )
            rule_line = "relation_type MUST be one of the allowed types above"
        else:
            relation_section = (
                "Relationship types: choose any descriptive UPPER_SNAKE_CASE label\n"
                "  (e.g. IMPROVES_UPON, TESTED_WITH, CHARACTERIZED_BY)."
            )
            rule_line = "relation_type should be UPPER_SNAKE_CASE, descriptive, concise"

        # ── Chunk blocks ─────────────────────────────────────────────────────
        chunk_blocks = []
        for idx, chunk in enumerate(chunks):
            chunk_id      = str(getattr(chunk, "chunk_id", id(chunk)))
            spacy_hints   = spacy_per_chunk.get(chunk_id, [])
            hint_str      = ""
            if spacy_hints:
                names    = [h["name"] for h in spacy_hints[:8]]
                hint_str = (
                    f"\n  [spaCy NER hints — use only if relevant: "
                    f"{', '.join(names)}]"
                )
            block = (
                f"### CHUNK {idx} ###"
                f"{hint_str}\n"
                f"{chunk.text.strip()}"
            )
            chunk_blocks.append(block)

        chunks_section = "\n\n".join(chunk_blocks)
        num_chunks     = len(chunks)

        prompt = f"""You are an expert knowledge-graph builder specialising in academic documents.
You will receive {num_chunks} text chunk(s) from an academic lecture or paper.
The content may be CONCEPTUAL (ideas, methods, models) or MATHEMATICAL (theorems,
proofs, operators, formulae).  Both modes require precise, high-quality extraction.

══════════════════════════════════════════════
ENTITY EXTRACTION RULES
══════════════════════════════════════════════

Allowed entity types: {allowed_types}

Type-selection guide:
  Algorithm   → clearly defined step-by-step procedure (e.g. Dijkstra's Algorithm)
  Method      → general technique (e.g. Dropout, Regularisation)
  Model       → mathematical or learned representation (e.g. BERT, Hidden Markov Model)
  System      → physical or abstract system (e.g. Control System, Embedded System)
  Component   → physical/electronic part (e.g. Resistor, Op-Amp)
  Signal      → time-varying input/output (e.g. Step Signal, Impulse Response)
  Formula     → explicit equation or named law (e.g. Euler's Formula, Ohm's Law)
  Theory      → formal framework (e.g. Probability Theory, Control Theory)
  Concept     → general idea — USE ONLY if no more specific type fits
  Metric      → evaluation measure (e.g. Accuracy, BLEU, MSE)
  Task        → problem definition (e.g. Classification, Filtering)
  Dataset     → named data collection (e.g. MNIST, SQuAD)
  Framework   → software / tool (e.g. PyTorch, MATLAB)
  Author      → researcher or person
  Institution → organisation / lab
  Theorem     → named mathematical theorem (e.g. Bayes' Theorem, Central Limit Theorem)
  Lemma       → intermediate mathematical result
  Proof       → formal mathematical proof technique or argument
  Operator    → mathematical operator or transform (e.g. Fourier Transform, Gradient)
  Property    → mathematical property or condition (e.g. Linearity, Convexity, Stability)
  Distribution→ probability distribution (e.g. Gaussian, Bernoulli, Poisson)
  Space       → mathematical / function space (e.g. Hilbert Space, L2 Space)

General rules:
  • Focus on entities a student MUST understand — avoid over-extraction
  • Ignore teaching filler: "example", "case", "step", "note", "basically"
  • Canonicalise to full widely-accepted names:
      "normal dist." → "Gaussian Distribution"
      "GD" → "Gradient Descent"
  • LaTeX MUST be preserved verbatim inside descriptions
      e.g. description: "Minimises $\\mathcal{{L}}(\\theta) = \\sum_i (y_i - \\hat{{y}}_i)^2$"
  • Return 3–15 entities per chunk depending on density
  • Provide a clear, self-contained 1-sentence description per entity

══════════════════════════════════════════════
RELATIONSHIP EXTRACTION RULES
══════════════════════════════════════════════

{relation_section}

Mathematical guidance:
  Theorem/Lemma proven via Proof      → PROOF_OF
  Formula expressing a Concept        → EXPRESSED_AS  or  DEFINED_BY
  Method/Algorithm satisfying Property→ SATISFIES
  Operator applied to a Space         → MAPS_TO  or  TRANSFORMS_TO
  Method converging to a solution     → CONVERGES_TO
  Concept being a special case        → SPECIAL_CASE_OF
  Hypothesis/Condition enabling proof → ASSUMES
  Result following from theorem       → IMPLIES

Conceptual guidance:
  Component → System                  → PART_OF
  Method solving Task                 → SOLVES  or  APPLIED_TO
  Method built on Theory              → BASED_ON
  Model evaluated on Dataset          → EVALUATED_ON
  Author proposing concept            → PROPOSED_BY

Rules:
  • {rule_line}
  • Use ONLY entity names you listed above — do not invent new ones
  • Confidence ≥ 0.9 if explicitly stated; 0.6–0.8 if strongly implied
  • Skip any relationship below confidence 0.6
  • Maximum 10 relationships per chunk
  • Preserve LaTeX verbatim in relationship descriptions

══════════════════════════════════════════════
CHUNKS
══════════════════════════════════════════════

{chunks_section}

══════════════════════════════════════════════
OUTPUT FORMAT  (STRICT JSON ONLY — no markdown, no preamble, no trailing text)
══════════════════════════════════════════════

{{
  "chunks": [
    {{
      "chunk_index": 0,
      "entities": [
        {{
          "name": "Canonical Entity Name",
          "entity_type": "One allowed type",
          "description": "Clear one-sentence explanation (LaTeX preserved)",
          "aliases": ["alias1", "alias2"]
        }}
      ],
      "relationships": [
        {{
          "source": "source entity name (must match an entity above)",
          "target": "target entity name (must match an entity above)",
          "relation_type": "RELATION_TYPE",
          "description": "Technical explanation (LaTeX preserved)",
          "confidence": 0.9
        }}
      ]
    }}
  ]
}}

Produce exactly {num_chunks} chunk object(s), indexed 0 to {num_chunks - 1}.
"""
        return prompt

    def _build_entity_only_prompt(self, text: str, allowed: str) -> str:
        """Entity-only prompt for the single-chunk incremental path."""
        return f"""Extract high-quality technical entities and key learning concepts from the
following academic text. The text may be conceptual OR mathematical.

Text:
{text}

General Rules:
- Focus ONLY on the most important technical concepts
- Limit to 3–15 non-redundant entities
- Ignore filler words: "example", "case", "thing", "step", "value", "note"
- Ignore overly generic terms UNLESS part of a named concept
- Canonicalise: "stack" → "Stack Data Structure", "normal dist." → "Gaussian Distribution"
- LaTeX MUST be preserved verbatim in descriptions

Allowed entity types: {allowed}

Output (STRICT JSON ONLY — no markdown):
{{
  "entities": [
    {{
      "name": "Canonical Entity Name",
      "entity_type": "One allowed type",
      "description": "Clear one-sentence explanation (LaTeX preserved)",
      "aliases": ["alias1", "alias2"]
    }}
  ]
}}"""

    # ═════════════════════════════════════════════════════════════════════════
    # Internal — shared post-processing
    # ═════════════════════════════════════════════════════════════════════════

    def _resolve_and_filter(
        self,
        items       : List[dict],
        node_id_map : dict,
        chunk,
    ) -> List[ExtractedRelationship]:
        """
        1. Filter by confidence threshold.
        2. Resolve entity names → node IDs.
        3. Constrained mode: drop unauthorised relation types.
        4. Unconstrained mode: normalise to UPPER_SNAKE_CASE.
        Preserved verbatim from RelationshipExtractor.
        """
        source   = chunk.metadata.get("source", "unknown")
        chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))
        results  = []

        for item in items:
            conf = float(item.get("confidence", 1.0))
            if conf < self.confidence_threshold:
                continue

            src_name = str(item.get("source", "")).strip()
            tgt_name = str(item.get("target", "")).strip()
            rel_type = str(item.get("relation_type", "")).strip()

            source_id = node_id_map.get(src_name.lower())
            target_id = node_id_map.get(tgt_name.lower())
            if not source_id or not target_id:
                continue

            rel_type = self._normalize_relation(rel_type)
            if not rel_type:
                continue

            if self.mode == "constrained":
                if rel_type not in ALLOWED_RELATIONS:
                    continue
            # unconstrained: keep all

            results.append(ExtractedRelationship(
                source_id    = source_id,
                target_id    = target_id,
                source_name  = src_name,
                target_name  = tgt_name,
                relation_type= rel_type,
                description  = str(item.get("description", "")).strip(),
                source_chunk = chunk_id,
                source       = source,
                confidence   = conf,
                mode         = self.mode,
            ))

        return results

    def _deduplicate_nodes(self, nodes: List[ExtractedNode]) -> List[ExtractedNode]:
        """One canonical node per node_id; keeps longest description + merged aliases."""
        seen: Dict[str, ExtractedNode] = {}
        for node in nodes:
            if node.node_id not in seen:
                seen[node.node_id] = node
            else:
                existing = seen[node.node_id]
                if len(node.description) > len(existing.description):
                    existing.description = node.description
                existing.aliases = list(set(existing.aliases + node.aliases))
        return list(seen.values())

    def _embed_nodes(self, nodes: List[ExtractedNode]) -> List[ExtractedNode]:
        """One batched embedding call after deduplication."""
        if not nodes:
            return nodes
        texts = [f"{n.name}: {n.description}" for n in nodes]
        try:
            vectors = self.embedding_fn(texts)
            for node, vec in zip(nodes, vectors):
                node.embedding = vec
        except Exception as e:
            warnings.warn(f"Node embedding failed: {e}")
        return nodes

    def _merge_entity_sets(
        self,
        spacy_items: List[dict],
        llm_items  : List[dict],
    ) -> List[dict]:
        """LLM is authoritative; spaCy fills gaps the LLM missed."""
        llm_names = {item["name"].lower() for item in llm_items}
        merged    = list(llm_items)
        for item in spacy_items:
            if item["name"].lower() not in llm_names:
                merged.append(item)
        return merged

    def _build_nodes_from_entities(
        self,
        entities: List[dict],
        source  : str,
        chunk_id: str,
    ) -> List[ExtractedNode]:
        nodes = []
        for item in entities:
            if item.get("entity_type") not in self.allowed_types:
                continue
            nodes.append(ExtractedNode(
                node_id      = self._make_node_id(item["name"], item["entity_type"]),
                name         = item["name"],
                entity_type  = item["entity_type"],
                description  = item["description"],
                source_chunk = chunk_id,
                source       = source,
                aliases      = list(item.get("aliases", [])),
            ))
        return nodes

    def _validate_entities(self, raw: list) -> List[dict]:
        """Normalise and validate LLM-returned entity list."""
        result = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name        = str(item.get("name", "")).strip()
            entity_type = str(item.get("entity_type", "")).strip()
            description = str(item.get("description", "")).strip()
            aliases     = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            if not name or not entity_type or not description:
                continue
            result.append({
                "name"       : name,
                "entity_type": entity_type,
                "description": description,
                "aliases"    : [str(a).strip().lower() for a in aliases if str(a).strip()],
            })
        return result

    def _validate_rels(self, raw: list) -> List[dict]:
        """Normalise and validate LLM-returned relationship list."""
        result = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            src  = str(item.get("source", "")).strip()
            tgt  = str(item.get("target", "")).strip()
            rtype= str(item.get("relation_type", "")).strip()
            desc = str(item.get("description", "")).strip()
            try:
                conf = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                conf = 1.0
            if not src or not tgt or not rtype:
                continue
            result.append({
                "source"       : src,
                "target"       : tgt,
                "relation_type": rtype,
                "description"  : desc,
                "confidence"   : conf,
            })
        return result

    def _normalize_cache_entities(self, items: List[dict]) -> List[dict]:
        """Keep only stable extraction fields; drop provenance-bearing data."""
        normalised = []
        for item in items:
            name        = str(item.get("name", "")).strip()
            entity_type = str(item.get("entity_type", "")).strip()
            description = str(item.get("description", "")).strip()
            aliases     = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            if not name or not entity_type or not description:
                continue
            normalised.append({
                "name"       : name,
                "entity_type": entity_type,
                "description": description,
                "aliases"    : [str(a).strip().lower() for a in aliases if str(a).strip()],
            })
        return normalised

    def _cache_signature(self) -> str:
        """Versioned cache key: model + allowed types."""
        model_name  = getattr(self.llm, "model", "unknown-model")
        allowed_sig = ",".join(sorted(self.allowed_types))
        return f"{self._CACHE_SCHEMA_VERSION}|model={model_name}|types={allowed_sig}"

    # ═════════════════════════════════════════════════════════════════════════
    # Internal — spaCy NER
    # ═════════════════════════════════════════════════════════════════════════

    def _extract_with_spacy(self, text: str) -> List[dict]:
        """
        Fast local NER pass — same label map as NodeExtractor.
        Results used as hints / gap-fill for the LLM.
        """
        label_map = {
            "PERSON"     : "Author",
            "ORG"        : "Institution",
            "WORK_OF_ART": "Model",
            "PRODUCT"    : "Framework",
        }
        items = []
        for ent in self._nlp(text).ents:
            if ent.label_ not in label_map:
                continue
            items.append({
                "name"       : ent.text.strip().title(),
                "entity_type": label_map[ent.label_],
                "description": f"{ent.text} — mentioned in source.",
                "aliases"    : [ent.text.lower()],
            })
        return items

    # ═════════════════════════════════════════════════════════════════════════
    # Internal — entity pair selection (relationship extraction)
    # ═════════════════════════════════════════════════════════════════════════

    def _select_prompt_nodes(self, nodes: List[ExtractedNode]) -> List[ExtractedNode]:
        """
        Select nodes for prompt construction while enforcing max_entity_pairs.
        Preserved from RelationshipExtractor — extended priority map included.
        """
        if len(nodes) < 2:
            return nodes

        pairs = list(combinations(nodes, 2))
        if len(pairs) <= self.max_entity_pairs:
            return nodes

        selected_pairs = self._prioritise_pairs(pairs, self.max_entity_pairs)
        selected_ids   = set()
        for left, right in selected_pairs:
            selected_ids.add(id(left))
            selected_ids.add(id(right))

        return [n for n in nodes if id(n) in selected_ids]

    def _prioritise_pairs(
        self,
        pairs    : List[Tuple],
        max_pairs: int,
    ) -> List[Tuple]:
        """Sort entity pairs by expected relationship richness (extended for math)."""
        return sorted(
            pairs,
            key=lambda p: _TYPE_PRIORITY_PAIRS.get(
                (p[0].entity_type, p[1].entity_type), 99
            )
        )[:max_pairs]

    def _format_entity_list(self, nodes: List[ExtractedNode]) -> str:
        return "\n".join(
            f"- {n.name} ({n.entity_type}): {n.description}"
            for n in nodes
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Static helpers
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_node_id(name: str, entity_type: str) -> str:
        """MD5 hash of lowercased name + type — stable across documents."""
        raw = f"{name.lower().strip()}:{entity_type.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _normalize_relation(rel_type: str) -> str:
        """Normalise free-form labels to UPPER_SNAKE_CASE."""
        return rel_type.strip().upper().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _load_spacy() -> SpacyModel:
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model en_core_web_sm...")
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    # ═════════════════════════════════════════════════════════════════════════
    # Dunder
    # ═════════════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"CombinedExtractor("
            f"model={getattr(self.llm, 'model', '?')!r}, "
            f"mode={self.mode!r}, "
            f"batch_chunks={self.batch_chunks}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"allowed_types={len(self.allowed_types)} types)"
        )