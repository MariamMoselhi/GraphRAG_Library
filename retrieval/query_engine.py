"""
Query Engine — second stage of the GraphRAG retrieval pipeline.

Takes the QueryRepresentation from query_processor.py and fans it out into
three parallel retrieval representations:

  1. keyword list    → BM25Retriever
  2. Cypher query    → GraphRetriever
  3. embedding vector→ VectorRetriever

Uses Groq LLM (API key #2 — GROQ_API_KEY_QUERY) to:
  - Detect query intent (FACTUAL / RELATIONAL / PROCEDURAL / COMPARATIVE / FOLLOW_UP)
  - Extract named entities (e.g. ["Gradient Descent", "Adam Optimizer"])
  - Generate a Cypher query for Neo4j graph traversal

Keyword extraction is done locally (no LLM) via stop-word filtering + stemming
to keep latency low — BM25 does not need high-precision keywords.

Embedding is done via the HuggingFaceEmbedding model (local, no API call).

Design note on memory-biased entity extraction
----------------------------------------------
When MemoryStore returns recent entities, they are appended to the LLM prompt
as "context entities" — the LLM is instructed to prefer these names over
generic alternatives.  This prevents the graph retriever from treating the
same concept as two different entities across turns
(e.g. "GD" in turn 1 vs "Gradient Descent" in turn 3).
"""
from __future__ import annotations

import os
import re
import time
from typing import List, Optional

import numpy as np

from .retrieval_context import (
    PhaseStats,
    QueryIntent,
    QueryRepresentation,
    RetrievalTrace,
)
from .memory_store import MemoryStore
from .retrieval_logger import RetrievalLogger


# Stop word set for keyword extraction
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its",
    "how", "why", "when", "where", "me", "my", "you", "your", "we", "our",
    "they", "their", "i", "am", "not", "no", "so", "as", "about", "just",
    "more", "also", "than", "then", "into", "through", "between", "explain",
    "describe", "tell", "give", "show", "list", "define", "compare", "use",
    "used", "using", "works", "work", "mean", "means",
}

# Cypher template catalogue — filled in by the LLM with entity names
_CYPHER_TEMPLATES = {
    QueryIntent.FACTUAL: """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($entity)
   OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($entity))
RETURN e.node_id, e.name, e.entity_type, e.description, e.source
LIMIT 5
""".strip(),

    QueryIntent.RELATIONAL: """
MATCH (a:Entity)-[r]-(b:Entity)
WHERE toLower(a.name) CONTAINS toLower($entity1)
   OR toLower(b.name) CONTAINS toLower($entity2)
RETURN a.name, type(r) AS relation, r.description,
       b.name, r.confidence
ORDER BY r.confidence DESC
LIMIT 10
""".strip(),

    QueryIntent.PROCEDURAL: """
MATCH (e:Entity)-[r:USES|IMPLEMENTS|DEPENDS_ON|BASED_ON|PART_OF]-(related:Entity)
WHERE toLower(e.name) CONTAINS toLower($entity)
RETURN e.name, type(r) AS step_type, related.name,
       related.description, related.entity_type
ORDER BY related.entity_type
LIMIT 10
""".strip(),

    QueryIntent.COMPARATIVE: """
MATCH (a:Entity), (b:Entity)
WHERE toLower(a.name) CONTAINS toLower($entity1)
  AND toLower(b.name) CONTAINS toLower($entity2)
OPTIONAL MATCH (a)-[r]-(b)
RETURN a.name, a.description, b.name, b.description,
       collect(type(r)) AS shared_relations
LIMIT 5
""".strip(),
}


class QueryEngine:
    """
    Fan-out query engine: text → {keywords, Cypher, embedding}.

    Args
    ----
    llm_api_key     : Groq API key for intent/entity/Cypher extraction.
                      Falls back to GROQ_API_KEY_QUERY then GROQ_API_KEY.
    llm_model       : Groq model to use (default "llama-3.3-70b-versatile").
    embedding_model : Any object with encode(texts) → np.ndarray method.
    memory_store    : MemoryStore for entity bias injection.
    logger          : RetrievalLogger instance.
    verbose         : Print phase details.
    """

    def __init__(
        self,
        embedding_model,
        llm_api_key     : Optional[str]     = None,
        llm_model       : str               = "llama-3.3-70b-versatile",
        memory_store    : Optional[MemoryStore] = None,
        logger          : Optional[RetrievalLogger] = None,
        verbose         : bool              = True,
    ):
        from graph.llm_backend import LLMBackend

        api_key = (
             os.environ.get("GROQ_API_KEY_QUERY")
        )
        self.llm             = LLMBackend(api_key=api_key, model=llm_model, max_tokens=1024)
        self.embedding_model = embedding_model
        self.memory_store    = memory_store
        self.logger          = logger or RetrievalLogger(verbose=verbose)
        self.verbose         = verbose

    # ── Main entry point 

    def process(
        self,
        rep   : QueryRepresentation,
        trace : Optional[RetrievalTrace] = None,
    ) -> QueryRepresentation:
        """
        Enrich a QueryRepresentation with intent, entities, keywords,
        Cypher, and embedding.

        Mutates rep in place and returns it.
        """
        t0 = time.time()
        self.logger.phase_start("Query Engine")

        # Step 1: LLM — intent + entities + Cypher
        prior_entities = []
        if self.memory_store:
            prior_entities = self.memory_store.get_recent_entities(n_turns=5)

        llm_output = self._call_llm(rep.normalised_text, rep.memory_context, prior_entities)

        rep.intent      = llm_output["intent"]
        rep.entities    = llm_output["entities"]
        rep.cypher_query= llm_output["cypher"]

        # Step 2: Keywords (local, no LLM)
        rep.keywords = self._extract_keywords(rep.normalised_text, rep.entities)

        # Step 3: Embedding (local model)
        embed_text  = rep.normalised_text
        if rep.memory_context:
            # Prepend compressed memory context to bias the embedding
            embed_text = rep.memory_context[:200] + " " + embed_text

        rep.embedding = self._embed(embed_text)

        elapsed_ms = (time.time() - t0) * 1000

        self.logger.print_query(
            raw_text    = rep.raw_text,
            intent      = rep.intent.value,
            entities    = rep.entities,
            keywords    = rep.keywords,
            cypher      = rep.cypher_query,
            from_memory = bool(rep.memory_context),
        )

        if trace:
            trace.add_phase(PhaseStats(
                phase_name   = "Query Engine",
                elapsed_ms   = elapsed_ms,
                input_count  = 1,
                output_count = 1,
                notes        = f"intent={rep.intent.value}  entities={len(rep.entities)}",
            ))

        self.logger.phase_end("Query Engine", count=1, elapsed_ms=elapsed_ms,
                              notes=f"{len(rep.entities)} entities · {len(rep.keywords)} keywords")
        return rep

    # LLM call

    def _call_llm(
        self,
        query          : str,
        memory_context : Optional[str],
        prior_entities : List[str],
    ) -> dict:
        """
        Single LLM call that returns intent, entities, and Cypher in JSON.
        """
        from graph.node_relation_extractor import _safe_json_parse  # reuse existing parser

        memory_block = ""
        if memory_context:
            memory_block = f"""
Prior conversation context (use for coreference resolution):
{memory_context}
"""

        entity_bias = ""
        if prior_entities:
            entity_bias = (
                f"\nPrefer these entity names from prior context if relevant: "
                f"{', '.join(prior_entities[:10])}\n"
            )

        allowed_intents = ", ".join(i.value for i in QueryIntent)

        prompt = f"""You are an expert query analyser for a knowledge-graph RAG system.
Analyse the user query and return ONLY a JSON object (no preamble, no markdown).
{memory_block}
User query: "{query}"
{entity_bias}
Tasks:
1. Detect the query intent. Allowed values: {allowed_intents}
2. Extract named technical entities (concept names, algorithm names,
   model names, theorem names, etc.). Return canonical full names.
   Return an empty list if no clear entities exist.
3. Generate a Neo4j Cypher query to retrieve relevant graph nodes.
   Use parameters $entity, $entity1, $entity2 where needed.
   If no meaningful Cypher can be written (e.g. very vague query), return null.

Response format (STRICT JSON ONLY):
{{
  "intent": "<intent value>",
  "entities": ["Entity Name 1", "Entity Name 2"],
  "cypher": "<Cypher query string or null>"
}}"""

        try:
            raw    = self.llm.generate(prompt)
            parsed = _safe_json_parse(raw)

            intent_str = parsed.get("intent", QueryIntent.UNKNOWN.value)
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.UNKNOWN

            entities = [
                str(e).strip() for e in parsed.get("entities", [])
                if str(e).strip()
            ]
            cypher = parsed.get("cypher") or None
            if cypher:
                cypher = cypher.strip()
                # Inject entities into template if LLM returned a placeholder
                if "$entity" in cypher and entities:
                    cypher = cypher.replace("$entity", f'"{entities[0]}"')
                if "$entity1" in cypher and len(entities) >= 1:
                    cypher = cypher.replace("$entity1", f'"{entities[0]}"')
                if "$entity2" in cypher and len(entities) >= 2:
                    cypher = cypher.replace("$entity2", f'"{entities[1]}"')

            return {"intent": intent, "entities": entities, "cypher": cypher}

        except Exception as e:
            self.logger.warn(f"Query engine LLM call failed ({e}) — using defaults")
            return {
                "intent"  : QueryIntent.UNKNOWN,
                "entities": [],
                "cypher"  : None,
            }

    # Keyword extraction

    def _extract_keywords(self, text: str, entities: List[str]) -> List[str]:
        """
        Extract BM25 keywords locally (no LLM needed).

        Algorithm
        ---------
        1. Tokenise on word boundaries.
        2. Lowercase and filter stop words.
        3. Keep tokens ≥ 3 characters.
        4. Prepend entity names (split into tokens) — entities are the
           highest-priority BM25 terms.
        5. Deduplicate while preserving order.
        """
        seen : set  = set()
        kws  : list = []

        # Entity tokens first (highest priority for BM25)
        for ent in entities:
            for tok in re.split(r"[\s_\-]+", ent):
                tok_l = tok.lower()
                if len(tok_l) >= 3 and tok_l not in _STOP_WORDS and tok_l not in seen:
                    seen.add(tok_l)
                    kws.append(tok)

        # General query tokens
        for tok in re.split(r"\W+", text):
            tok_l = tok.lower()
            if len(tok_l) >= 3 and tok_l not in _STOP_WORDS and tok_l not in seen:
                seen.add(tok_l)
                kws.append(tok)

        return kws

    # Embedding

    def _embed(self, text: str) -> np.ndarray:
        """Embed text using the local embedding model."""
        try:
            return self.embedding_model.encode([text])[0]
        except Exception as e:
            self.logger.warn(f"Embedding failed ({e}) — returning zero vector")
            dim = getattr(self.embedding_model, "dimension", 384) or 384
            return np.zeros(dim, dtype=np.float32)