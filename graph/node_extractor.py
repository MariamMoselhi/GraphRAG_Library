# graph/node_extractor.py
from __future__ import annotations

import hashlib
import json
import re
import threading
import warnings
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    import spacy
    from spacy.language import Language as SpacyModel
except ImportError as e:
    raise ImportError(
        "spacy is required.\n"
        "pip install spacy && python -m spacy download en_core_web_sm"
    ) from e

from .base import ExtractedNode, BaseNodeExtractor
from .llm_backend import LLMBackend


class NodeExtractor(BaseNodeExtractor):
    """
    Extracts entity nodes from text chunks using spaCy + HuggingFace LLM.

    Strategy
    --------
    1. spaCy NER   — fast, local, zero API cost. Catches proper nouns,
                     persons, organisations, and named works.
    2. LLM         — catches domain-specific concepts spaCy misses
                     (methods, metrics, datasets, tasks, abstract concepts).
    3. Merge       — LLM is authoritative; spaCy fills any remaining gaps.
    4. Deduplicate — one canonical node per (name, entity_type) pair.
    5. Embed       — one batched embedding call after deduplication.

    Entity types (academia-focused)
    --------------------------------
    Concept      Abstract ideas: "overfitting", "attention", "entropy"
    Method       Algorithms / techniques: "backpropagation", "dropout"
    Model        Named architectures: "BERT", "GPT-4", "ResNet"
    Dataset      Named corpora: "ImageNet", "SQuAD", "GLUE"
    Metric       Evaluation measures: "BLEU", "F1", "perplexity"
    Author       Researchers: "Vaswani", "LeCun", "Hinton"
    Institution  Organisations: "Google Brain", "OpenAI", "DeepMind"
    Task         ML/NLP tasks: "image classification", "NER", "MT"
    Theory       Theoretical frameworks: "PAC learning", "VC dimension"
    Framework    Software / libraries: "PyTorch", "TensorFlow", "JAX"

    Source convention
    -----------------
    Set chunk.metadata["source"] before calling any extract method.
    Defaults to "unknown" if not set.

    Args
    ----
    llm           : LLMBackend instance (caller constructs)
    embedding_fn  : Callable[[List[str]], np.ndarray]
                    Pass HuggingFaceEmbedding.encode directly
    max_workers   : Thread pool size for parallel chunk processing
    allowed_types : Entity types to keep (default = DEFAULT_ALLOWED_TYPES)
    """

    DEFAULT_ALLOWED_TYPES = [
        "Concept",        # General ideas (Entropy, Stability, Voltage)
        "Algorithm",      # Step-by-step procedures (Gradient Descent, Binary Search)
        "Method",         # Techniques (Backpropagation, Fourier Analysis)
        "Model",          # Systems/models (BERT, State Space Model, Linear Regression)
        "System",         # Physical or abstract systems (Control System, Embedded System)
        "Component",      # Physical/electronic parts (Resistor, Capacitor, Sensor)
        "Signal",         # Signal types (Analog Signal, Step Signal)
        "Formula",        # Equations (Ohm's Law, Transfer Function)
        "Theory",         # Formal theories (Control Theory, Probability Theory)
        "Metric",         # Evaluation measures (Accuracy, Error Rate)
        "Task",           # Problems (Classification, Filtering)
        "Dataset",        # Data collections (MNIST)
        "Framework",      # Tools (PyTorch, MATLAB)
        "Author",         # Researchers / people
        "Institution",    # Organisations / labs
]

    _CACHE_SCHEMA_VERSION = "v1"

    def __init__(
        self,
        llm          : LLMBackend,
        embedding_fn,
        max_workers  : int            = 2,
        allowed_types: Optional[List[str]] = None,
    ):
        self.llm           = llm
        self.embedding_fn  = embedding_fn
        self.max_workers   = max_workers
        self.allowed_types = allowed_types or self.DEFAULT_ALLOWED_TYPES

        # chunk_text -> cache_signature -> {"entities": List[dict]}
        # Cache stores intermediate extraction payload only (no provenance-bearing nodes).
        self._cache: dict[str, dict[str, dict[str, List[dict]]]] = {}
        self._cache_lock = threading.Lock()
        self._nlp = self._load_spacy()

    # ── BaseNodeExtractor contract ────────────────────────────────────────────

    def extract_from_chunks(
        self,
        chunks: list,
        show_progress: bool = True,
    ) -> List[ExtractedNode]:
        """
        Extract and deduplicate nodes from all chunks in parallel.
        Implements BaseNodeExtractor.extract_from_chunks.
        """
        all_nodes: List[ExtractedNode] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._extract_from_single_chunk, chunk): chunk
                for chunk in chunks
            }
            completed = 0
            for future in as_completed(future_map):
                completed += 1
                if show_progress:
                    print(f"  Nodes: {completed}/{len(chunks)} chunks", end="\r")
                try:
                    all_nodes.extend(future.result())
                except Exception as e:
                    if str(e) == "LLM extraction failed — pipeline degraded":
                        raise RuntimeError("LLM extraction failed — pipeline degraded") from e
                    chunk = future_map[future]
                    warnings.warn(
                        f"Node extraction failed for chunk "
                        f"{getattr(chunk, 'chunk_id', '?')}: {e}"
                    )

        if show_progress:
            print(f"  Nodes: {len(chunks)}/{len(chunks)} chunks ✓")

        deduplicated = self._deduplicate_nodes(all_nodes)
        deduplicated = self._embed_nodes(deduplicated)
        print(f"  ✓ {len(deduplicated)} unique nodes from {len(chunks)} chunks")
        return deduplicated

    def extract_from_single_chunk_public(self, chunk) -> List[ExtractedNode]:
        """
        Single-chunk extraction for incremental ingestion.
        Implements BaseNodeExtractor.extract_from_single_chunk_public.
        """
        nodes = self._extract_from_single_chunk(chunk)
        return self._embed_nodes(nodes)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_from_single_chunk(self, chunk) -> List[ExtractedNode]:
        """spaCy + LLM combined extraction for one chunk."""
        text   = chunk.text
        source = chunk.metadata.get("source", "unknown")
        cache_sig = self._cache_signature()

        # Fast path: lock-free lookup first.
        cached_entry = self._cache.get(text, {}).get(cache_sig)
        if cached_entry is not None:
            return self._build_nodes_from_entities(
                entities=cached_entry.get("entities", []),
                source=source,
                chunk_id=chunk.chunk_id,
            )

        # Compute outside lock to avoid blocking all workers during API/network calls.
        spacy_items = self._extract_with_spacy(text)
        llm_items   = self._extract_with_llm(text)
        merged      = self._merge_entity_sets(spacy_items, llm_items)
        entities_payload = self._normalize_cache_entities(merged)

        # Double-check locking to avoid race overwrites.
        with self._cache_lock:
            cache_bucket = self._cache.setdefault(text, {})
            cache_bucket.setdefault(cache_sig, {"entities": entities_payload})
            entities = cache_bucket[cache_sig]["entities"]

        return self._build_nodes_from_entities(
            entities=entities,
            source=source,
            chunk_id=chunk.chunk_id,
        )

    def _cache_signature(self) -> str:
        """Versioned cache signature so cache entries are not keyed by raw text only."""
        model_name = getattr(self.llm, "model", "unknown-model")
        allowed_sig = ",".join(sorted(self.allowed_types))
        return f"{self._CACHE_SCHEMA_VERSION}|model={model_name}|types={allowed_sig}"

    def _normalize_cache_entities(self, items: List[dict]) -> List[dict]:
        """Keep only stable extraction fields in cache payload."""
        normalized: List[dict] = []
        for item in items:
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("entity_type", "")).strip()
            description = str(item.get("description", "")).strip()
            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            if not name or not entity_type or not description:
                continue

            normalized.append({
                "name": name,
                "entity_type": entity_type,
                "description": description,
                "aliases": [str(a).strip().lower() for a in aliases if str(a).strip()],
            })
        return normalized

    def _build_nodes_from_entities(
        self,
        entities: List[dict],
        source: str,
        chunk_id: int,
    ) -> List[ExtractedNode]:
        """Build fresh ExtractedNode objects with current chunk provenance."""
        nodes: List[ExtractedNode] = []
        for item in entities:
            if item["entity_type"] not in self.allowed_types:
                continue
            nodes.append(ExtractedNode(
                node_id      = self._make_node_id(item["name"], item["entity_type"]),
                name         = item["name"],
                entity_type  = item["entity_type"],
                description  = item["description"],
                source_chunk = str(chunk_id),
                source       = source,
                aliases      = list(item.get("aliases", [])),
            ))
        return nodes

    def _extract_with_spacy(self, text: str) -> List[dict]:
        """
        Fast local NER pass.

        spaCy label   → entity_type
        PERSON        → Author
        ORG           → Institution
        WORK_OF_ART   → Model
        PRODUCT       → Framework
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

    def _extract_with_llm(self, text: str) -> List[dict]:
        """
        LLM pass for domain concepts spaCy cannot detect.
        Raises RuntimeError on failure so callers can fail fast.
        """
        allowed = ", ".join(self.allowed_types)
        prompt = f"""Extract high-quality technical entities and key learning concepts from the following lecture or academic text.

Text:
{text}

General Rules:
- Focus ONLY on the most important technical concepts that a student needs to understand or revise
- Limit extraction to the most relevant and non-redundant entities (avoid over-extraction)
- Ignore filler or teaching words such as: "example", "case", "thing", "step", "value", "note", "basically"
- Ignore overly generic terms like: "model", "data", "result", "method" UNLESS part of a specific named concept
- Do NOT extract sentence fragments or vague phrases

Domain Awareness:
- The text may belong to any technical field (Computer Science, Machine Learning, NLP, Mathematics, Engineering, Control, Electronics, etc.)
- Adapt entity extraction accordingly

Canonicalization:
- Always return the most standard, widely accepted full name
  - "stack" → "Stack Data Structure"
  - "last in first out" → "LIFO (Last In First Out)"
  - "normal distribution" → "Gaussian Distribution"
- If a concept is implied but not explicitly named, infer the correct canonical term

Entity Types:
- Allowed types: {allowed}

Type Selection Rules (STRICT):
- Algorithm → clearly defined step-by-step procedure
- Method → general technique (less strict than algorithm)
- Model → mathematical or learned representation
- System → physical or abstract system (e.g., control system)
- Component → physical/electronic part (resistor, sensor)
- Signal → time-varying input/output
- Formula → explicit equation or law
- Theory → formal framework or law
- Concept → general idea (fallback if nothing else fits)
- Metric → evaluation measure
- Task → problem definition
- Dataset → named dataset
- Framework → software/tool

- Prefer the MOST specific type possible
- Use "Concept" ONLY if no better type applies

Descriptions:
- Provide a clear, concise, student-friendly definition (1 sentence)
- Make it self-contained
- Do NOT copy directly from the text

Aliases:
- Include common abbreviations, synonyms, or variations
- Use lowercase for aliases
- Avoid duplicates

Output Constraints:
- Return between 3 to 15 entities depending on text richness
- Avoid duplicate or overlapping entities

Output Format (STRICT JSON ONLY — no markdown, no explanation):
{{
  "entities": [
    {{
      "name": "Canonical Entity Name",
      "entity_type": "One allowed type",
      "description": "Clear one-sentence explanation",
      "aliases": ["alias1", "alias2"]
    }}
  ]
}}
"""

        try:
            raw = self.llm.generate(prompt)
            clean = re.sub(r'^```(?:json)?\s*', '', (raw or '').strip())
            clean = re.sub(r'\s*```$', '', clean).strip()

            if not clean:
                raise RuntimeError("LLM extraction failed — empty model output")

            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError:
                # Some models prepend commentary/safety text. Extract the first JSON object if present.
                start = clean.find("{")
                end = clean.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise RuntimeError("LLM extraction failed — no JSON object found in response")
                parsed = json.loads(clean[start:end + 1])

            entities = parsed.get("entities", [])
            if not isinstance(entities, list):
                raise RuntimeError("LLM extraction failed — 'entities' is not a list")

            normalized: List[dict] = []
            for item in entities:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                entity_type = str(item.get("entity_type", "")).strip()
                description = str(item.get("description", "")).strip()
                aliases = item.get("aliases", [])
                if not isinstance(aliases, list):
                    aliases = []

                if not name or not entity_type or not description:
                    continue

                normalized.append({
                    "name": name,
                    "entity_type": entity_type,
                    "description": description,
                    "aliases": [str(a).strip().lower() for a in aliases if str(a).strip()],
                })

            if not normalized:
                raise RuntimeError("LLM extraction failed — no valid entities returned")

            return normalized
        except Exception as e:
            warnings.warn(f"LLM entity extraction failed: {e}")
            raise RuntimeError("LLM extraction failed — pipeline degraded") from e

    def _merge_entity_sets(
        self,
        spacy_items: List[dict],
        llm_items  : List[dict],
    ) -> List[dict]:
        """LLM is authoritative. spaCy fills gaps LLM missed."""
        llm_names = {item["name"].lower() for item in llm_items}
        merged    = list(llm_items)
        for item in spacy_items:
            if item["name"].lower() not in llm_names:
                merged.append(item)
        return merged

    def _deduplicate_nodes(self, nodes: List[ExtractedNode]) -> List[ExtractedNode]:
        """
        One canonical node per node_id.
        Keeps the longest description and merges aliases.
        """
        seen: dict[str, ExtractedNode] = {}
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
        """
        One batched embedding call for all nodes.
        Runs after deduplication — no wasted calls.
        """
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

    @staticmethod
    def _make_node_id(name: str, entity_type: str) -> str:
        """MD5 hash of lowercased name + type — stable across all documents."""
        raw = f"{name.lower().strip()}:{entity_type.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _load_spacy() -> SpacyModel:
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model en_core_web_sm...")
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            return spacy.load("en_core_web_sm")