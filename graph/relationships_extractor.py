# graph/relationships_extractor.py
from __future__ import annotations

import json
import re
import warnings
from itertools import combinations
from typing import List, Literal, Tuple

from .base import ExtractedNode, ExtractedRelationship, BaseRelationshipExtractor
from .llm_backend import LLMBackend


# ── Relation taxonomy ─────────────────────────────────────────────────────────

# Methodological
_METHODOLOGICAL = [
    "USES",           # A uses B as a component or sub-method
    "EXTENDS",        # A builds on / is a variant of B
    "IMPLEMENTS",     # A is a concrete implementation of B
    "IMPROVES_UPON",  # A outperforms or advances beyond B
    "REPLACES",       # A supersedes B
    "COMBINES",       # A merges B and C (multi-target)
]

# Dependency & Structure
_STRUCTURAL = [
    "DEPENDS_ON",     # Understanding / running A requires B
    "PART_OF",        # A is a sub-component of B
    "BASED_ON",       # A is grounded in / derived from B
    "DERIVED_FROM",   # A is a direct derivation of B
]

# Evaluation & Training
_EVALUATION = [
    "EVALUATED_ON",       # A is tested/benchmarked on B
    "TRAINED_ON",         # A was trained using dataset B
    "BENCHMARKED_AGAINST",# A is compared against B on a benchmark
    "OUTPERFORMS",        # A achieves better results than B
]

# Introduction & Attribution
_ATTRIBUTION = [
    "INTRODUCES",     # A introduces / proposes B (paper → concept)
    "PROPOSED_BY",    # A was proposed by author/group B
    "DEVELOPED_BY",   # A was built/released by institution B
    "PUBLISHED_IN",   # A appeared in venue B
]

# Comparison & Contrast
_COMPARISON = [
    "COMPARES_WITH",  # A is explicitly compared to B
    "CONTRASTS_WITH", # A and B are presented as alternatives
    "EQUIVALENT_TO",  # A and B are shown to be the same concept
]

# Knowledge Flow (cross-document / citation)
_KNOWLEDGE_FLOW = [
    "CITES",          # A cites / references B
    "BUILDS_ON",      # A advances work started in B
    "MOTIVATED_BY",   # A is motivated by problem B
    "ADDRESSES",      # A attempts to solve problem B
    "SOLVES",         # A fully solves problem B
]

# Domain Application
_DOMAIN = [
    "APPLIED_TO",     # A is applied to domain / task B
    "GENERALIZES",    # A is a more general form of B
    "SPECIALIZES",    # A is a specialisation of B
]

ALLOWED_RELATIONS: List[str] = (
    _METHODOLOGICAL +
    _STRUCTURAL     +
    _EVALUATION     +
    _ATTRIBUTION    +
    _COMPARISON     +
    _KNOWLEDGE_FLOW +
    _DOMAIN
)

# Cross-document relations (subset used for reference detection)
CROSS_DOC_RELATIONS: List[str] = [
    "DEPENDS_ON", "EXTENDS", "BUILDS_ON",
    "CITES", "MOTIVATED_BY", "BASED_ON",
]


# ── Concrete implementation ───────────────────────────────────────────────────

class RelationshipExtractor(BaseRelationshipExtractor):
    """
    Extracts directed semantic relationships between entity nodes.

    Mode (set at construction time)
    --------------------------------
    "constrained"
        LLM must choose from ALLOWED_RELATIONS (27 types).
        Graph schema stays clean and predictable.
        Invalid relation types returned by the LLM are silently dropped.
        Best for: production pipelines, structured knowledge bases.

    "unconstrained"
        LLM freely names any relation type it sees fit.
        More expressive — captures nuanced academic language.
        relation_type is normalised to UPPER_SNAKE_CASE automatically.
        Best for: exploratory analysis, discovering new relation patterns.

    In both modes the same confidence threshold and pair-priority logic apply.

    Args
    ----
    llm                  : LLMBackend instance (caller constructs)
    mode                 : "constrained" or "unconstrained" (default "constrained")
    confidence_threshold : Edges below this score are discarded (default 0.6)
    max_entity_pairs     : Cap on entity pairs evaluated per chunk (default 20)
    """

    def __init__(
        self,
        llm                 : LLMBackend,
        mode                : Literal["constrained", "unconstrained"] = "constrained",
        confidence_threshold: float = 0.6,
        max_entity_pairs    : int   = 20,
    ):
        if mode not in ("constrained", "unconstrained"):
            raise ValueError(f"mode must be 'constrained' or 'unconstrained', got '{mode}'")

        self.llm                  = llm
        self.mode                 = mode
        self.confidence_threshold = confidence_threshold
        self.max_entity_pairs     = max_entity_pairs

    # ── BaseRelationshipExtractor contract ────────────────────────────────────

    def extract_from_chunk(
        self,
        chunk,
        nodes_in_chunk: List[ExtractedNode],
        node_id_map   : dict,
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships between entity pairs in one chunk.
        Implements BaseRelationshipExtractor.extract_from_chunk.
        """
        if len(nodes_in_chunk) < 2:
            return []

        prompt_nodes = self._select_prompt_nodes(nodes_in_chunk)
        if len(prompt_nodes) < 2:
            return []

        prompt = self._build_extraction_prompt(
            text               = chunk.text,
            nodes              = prompt_nodes,
            constrained        = (self.mode == "constrained"),
        )

        try:
            raw    = self.llm.generate(prompt)
            clean  = re.sub(r'^```(?:json)?\s*', '', raw.strip())
            clean  = re.sub(r'\s*```$', '', clean).strip()
            parsed = json.loads(clean)
            items  = parsed.get("relationships", [])
        except Exception as e:
            warnings.warn(f"Relationship extraction failed for chunk {chunk.chunk_id}: {e}")
            return []

        return self._resolve_and_filter(items, node_id_map, chunk)

    def extract_from_chunks(
        self,
        chunks        : list,
        nodes_per_chunk: "dict[int, List[ExtractedNode]]",
        node_id_map   : dict,
        show_progress : bool = True,
    ) -> "List[ExtractedRelationship]":
        """
        Extract relationships from multiple chunks with batched LLM calls.

        Instead of one API call per chunk, this method groups up to
        ``self.llm.batch_size`` chunks into a single request, then splits
        the response and processes each part independently.

        Args
        ----
        chunks          : list of Chunk objects
        nodes_per_chunk : mapping of chunk index → list of ExtractedNode
                          for that chunk (from NodeExtractor)
        node_id_map     : global name.lower() → node_id mapping
        show_progress   : print progress to stdout

        Returns
        -------
        Deduplicated list of ExtractedRelationship across all chunks.
        """
        eligible = [
            (i, chunk) for i, chunk in enumerate(chunks)
            if len(nodes_per_chunk.get(i, [])) >= 2
        ]

        if not eligible:
            return []

        batch_size = max(1, getattr(self.llm, "batch_size", 1))

        # Build one prompt per eligible chunk
        prompts = []
        eligible_with_nodes = []
        for i, chunk in eligible:
            prompt_nodes = self._select_prompt_nodes(nodes_per_chunk[i])
            if len(prompt_nodes) < 2:
                continue
            prompts.append(
                self._build_extraction_prompt(
                    text        = chunk.text,
                    nodes       = prompt_nodes,
                    constrained = (self.mode == "constrained"),
                )
            )
            eligible_with_nodes.append((i, chunk))

        if not prompts:
            return []

        # Fire batched or individual API calls
        if batch_size > 1 and hasattr(self.llm, "generate_batch"):
            n_calls = (len(prompts) + batch_size - 1) // batch_size
            if show_progress:
                print(f"  Rels: {len(prompts)} chunks → {n_calls} batched API calls "
                      f"(batch_size={batch_size})")
            responses = self.llm.generate_batch(prompts)
        else:
            responses = []
            for pos, prompt in enumerate(prompts):
                if show_progress:
                    print(f"  Rels: {pos + 1}/{len(prompts)} chunks", end="\r")
                try:
                    responses.append(self.llm.generate(prompt))
                except Exception as e:
                    warnings.warn(f"Relationship extraction failed: {e}")
                    responses.append("")
            if show_progress:
                print()

        # Parse each response and collect relationships
        all_rels: List[ExtractedRelationship] = []
        for pos, (chunk_idx, chunk) in enumerate(eligible_with_nodes):
            raw = responses[pos] if pos < len(responses) else ""
            try:
                clean  = re.sub(r'^```(?:json)?\s*', '', raw.strip())
                clean  = re.sub(r'\s*```$', '', clean).strip()
                parsed = json.loads(clean)
                items  = parsed.get("relationships", [])
            except Exception as e:
                warnings.warn(
                    f"Relationship parse failed for chunk {chunk.chunk_id}: {e}"
                )
                items = []

            all_rels.extend(self._resolve_and_filter(items, node_id_map, chunk))

        deduped = self.deduplicate_relationships(all_rels)
        if show_progress:
            print(f"  ✓ {len(deduped)} unique relationships from {len(chunks)} chunks")
        return deduped

    def extract_cross_document_references(
        self,
        chunk             : object,
        node_id_map       : dict,
        all_document_nodes: dict,
    ) -> List[ExtractedRelationship]:
        """
        Detect references to concepts from other documents.
        Implements BaseRelationshipExtractor.extract_cross_document_references.
        """
        text = chunk.text

        # Pre-screen: skip LLM call if no reference markers present
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
            clean  = re.sub(r'^```(?:json)?\s*', '', raw.strip())
            clean  = re.sub(r'\s*```$', '', clean).strip()
            parsed = json.loads(clean)
            items  = parsed.get("relationships", [])
        except Exception as e:
            warnings.warn(f"Cross-document extraction failed: {e}")
            return []

        # In constrained mode, only keep cross-doc relation types
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
        Implements BaseRelationshipExtractor.deduplicate_relationships.
        """
        seen: dict[Tuple, ExtractedRelationship] = {}
        for rel in relationships:
            key = (rel.source_id, rel.target_id, rel.relation_type)
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        return list(seen.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_extraction_prompt(
        self,
        text      : str,
        nodes     : List[ExtractedNode],
        constrained: bool,
    ) -> str:
        entity_list = self._format_entity_list(nodes)

        if constrained:
            relation_section = f"""Allowed relationship types (use ONLY these):
{chr(10).join(f"  {r}" for r in ALLOWED_RELATIONS)}"""
            rule_line = "- relation_type MUST be one of the allowed types above"
        else:
            relation_section = """Relationship types:
  Choose any descriptive relation type in UPPER_SNAKE_CASE that best describes
  the academic relationship (e.g. IMPROVES_UPON, TESTED_WITH, INSPIRED_BY)."""
            rule_line = "- relation_type should be UPPER_SNAKE_CASE, descriptive, and concise"

        return f"""You are an expert in engineering and technical education.

Your task is to extract HIGH-QUALITY semantic relationships between entities
from an engineering lecture or academic text.

Entities present in this text:
{entity_list}

{relation_section}

Text:
{text}

Engineering Context Awareness:
The text may belong to fields such as:
- Control Systems
- Electronics / Circuits
- Signal Processing
- Embedded Systems
- Machine Learning / AI
- Mathematics

You MUST interpret relationships in a technical and functional way.

---

Relationship Guidelines (VERY IMPORTANT):

1. STRUCTURAL RELATIONSHIPS:
- Component → System → use PART_OF
- Subsystem → System → use PART_OF
- System → Component → use DEPENDS_ON

2. FUNCTIONAL RELATIONSHIPS:
- Method/Algorithm solving a Task → use APPLIED_TO or SOLVES
- Method improving another → use IMPROVES_UPON
- Method replacing another → use REPLACES
- Method combining techniques → use COMBINES

3. THEORETICAL RELATIONSHIPS:
- Concept derived from Theory → use BASED_ON or DERIVED_FROM
- Formula representing a Concept → use BASED_ON
- Method based on Theory → use BASED_ON

4. SIGNAL & SYSTEM RELATIONSHIPS:
- Signal used inside System → use PART_OF
- System processing Signal → use APPLIED_TO
- Input/Output relations → use DEPENDS_ON

5. MODEL & DATA RELATIONSHIPS:
- Model trained using Dataset → use TRAINED_ON
- Model evaluated using Dataset → use EVALUATED_ON
- Model compared to another → use COMPARES_WITH or OUTPERFORMS

6. AUTHOR / SOURCE RELATIONSHIPS:
- Author → Method/Model → use PROPOSED_BY
- Institution → System/Framework → use DEVELOPED_BY

---

Strict Rules:
- Only extract relationships EXPLICITLY stated or strongly implied in the text
- Only use the entity names listed above — do not invent new entities
- {rule_line}
- Each relationship is directional: source → target
- Confidence 0.9+ only when explicitly stated; 0.6–0.8 for strong implications
- Do not extract speculative or assumed relationships
- Prefer fewer high-confidence relationships over many weak ones
- Maximum 10 relationships per chunk unless the text is very dense

Confidence:
- 0.9–1.0 → explicitly stated
- 0.7–0.8 → strongly implied
- < 0.6 → DO NOT include

---

Output format (STRICT JSON ONLY):
{{
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "relation_type": "RELATION_TYPE",
      "description": "clear technical explanation of the relationship",
      "confidence": 0.9
    }}
  ]
}}
"""

    def _resolve_and_filter(
        self,
        items      : List[dict],
        node_id_map: dict,
        chunk,
    ) -> List[ExtractedRelationship]:
        """
        Shared post-processing:
        1. Filter by confidence threshold
        2. Resolve entity names → node IDs
        3. In constrained mode: drop unauthorised relation types
        4. In unconstrained mode: normalise to UPPER_SNAKE_CASE
        """
        source   = chunk.metadata.get("source", "unknown")
        chunk_id = str(chunk.chunk_id)
        results  = []

        for item in items:
            conf = float(item.get("confidence", 1.0))
            if conf < self.confidence_threshold:
                continue

            src_name = item.get("source", "")
            tgt_name = item.get("target", "")
            rel_type = item.get("relation_type", "")

            source_id = node_id_map.get(src_name.lower())
            target_id = node_id_map.get(tgt_name.lower())
            if not source_id or not target_id:
                continue  # LLM hallucinated an entity

            rel_type = self._normalize_relation(rel_type)

            if not rel_type:
                continue

            if self.mode == "constrained":
                if rel_type not in ALLOWED_RELATIONS:
                    continue  # drop if not allowed

            elif self.mode == "unconstrained":
                pass  # ✅ keep everything

            else:
                continue

            results.append(ExtractedRelationship(
                source_id    = source_id,
                target_id    = target_id,
                source_name  = src_name,
                target_name  = tgt_name,
                relation_type= rel_type,
                description  = item.get("description", ""),
                source_chunk = chunk_id,
                source       = source,
                confidence   = conf,
                mode         = self.mode,
            ))

        return results

    @staticmethod
    def _normalize_relation(rel_type: str) -> str:
        """Normalize free-form relation labels into UPPER_SNAKE_CASE."""
        return rel_type.strip().upper().replace(" ", "_").replace("-", "_")

    def _format_entity_list(self, nodes: List[ExtractedNode]) -> str:
        return "\n".join(
            f"- {n.name} ({n.entity_type}): {n.description}"
            for n in nodes
        )

    def _prioritise_pairs(
        self,
        pairs    : List[Tuple],
        max_pairs: int,
    ) -> List[Tuple]:
        """
        Sort entity pairs by expected relationship richness.
        High-value pairs (Model↔Method, Method↔Concept) come first.
        """
        priority_map = {
            ("Model",   "Method")     : 1,
            ("Method",  "Model")      : 1,
            ("Method",  "Concept")    : 2,
            ("Concept", "Method")     : 2,
            ("Model",   "Dataset")    : 3,
            ("Dataset", "Model")      : 3,
            ("Model",   "Metric")     : 3,
            ("Method",  "Task")       : 4,
            ("Task",    "Method")     : 4,
            ("Model",   "Framework")  : 5,
            ("Method",  "Theory")     : 5,
            ("Theory",  "Method")     : 5,
            ("Author",  "Model")      : 6,
            ("Author",  "Method")     : 6,
        }
        return sorted(
            pairs,
            key=lambda p: priority_map.get(
                (p[0].entity_type, p[1].entity_type), 99
            )
        )[:max_pairs]

    def _select_prompt_nodes(self, nodes: List[ExtractedNode]) -> List[ExtractedNode]:
        """
        Select nodes used for prompt construction while enforcing max_entity_pairs.

        If combinations exceed max_entity_pairs, keep only nodes appearing in
        prioritised top-K pairs.
        """
        if len(nodes) < 2:
            return nodes

        pairs = list(combinations(nodes, 2))
        if len(pairs) <= self.max_entity_pairs:
            return nodes

        selected_pairs = self._prioritise_pairs(pairs, self.max_entity_pairs)
        selected_ids = set()
        for left, right in selected_pairs:
            selected_ids.add(id(left))
            selected_ids.add(id(right))

        # Preserve original order for stable prompting.
        selected_nodes = [n for n in nodes if id(n) in selected_ids]
        return selected_nodes