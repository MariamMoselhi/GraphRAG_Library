"""
Conversation memory store for the GraphRAG retrieval pipeline.

Architecture decision (why memory lives in retrieval, not generation)
----------------------------------------------------------------------
Memory is a *retrieval signal*, not just a display aid.  When a user asks
"elaborate on that" or "how does it compare to what we discussed?", the
retrievers need to know what "that" refers to before they can query anything.

Placing memory here means:
  1. query_processor.py can resolve coreferences ("it", "that method")
     by injecting prior entities into the query before retrieval.
  2. query_engine.py can bias entity extraction toward already-seen concepts.
  3. The grader can check whether retrieved chunks are *consistent* with what
     was already told to the user in prior turns (no contradictions).

Design
------
- Sliding window of ConversationTurn objects, configurable depth (default 10).
- Persisted to disk as JSON so memory survives process restarts.
- When the window exceeds ``max_turns``, older turns are summarised using
  the Groq API and stored as a compressed "summary turn" — this keeps the
  injected context bounded in token count.
- Thread-safe: uses a threading.Lock around all state mutations.
- No external dependencies beyond stdlib + the project's LLMBackend.

Storage format (JSON)
---------------------
{
  "session_id": "...",
  "turns": [ { ConversationTurn fields } ],
  "summary": "optional compressed summary of older turns"
}
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .retrieval_context import ConversationTurn, QueryIntent


class MemoryStore:
    """
    Sliding-window conversation memory with JSON persistence.

    Args
    ----
    session_id      : Unique identifier for this conversation session.
                      Used as the filename: ``memory_{session_id}.json``.
    storage_dir     : Directory where JSON files are saved.
    max_turns       : Maximum turns kept in the active window (default 10).
    summarise_at    : Trigger summarisation when turns exceed this count.
                      Must be ≥ max_turns (default 15).
    llm_backend     : Optional LLMBackend for summarisation.
                      If None, old turns are simply dropped (no summarisation).
    verbose         : Print memory operations to console.
    """

    def __init__(
        self,
        session_id    : str                 = "default",
        storage_dir   : str                 = "memory",
        max_turns     : int                 = 10,
        summarise_at  : int                 = 15,
        llm_backend                         = None,
        verbose       : bool                = True,
    ):
        self.session_id   = session_id
        self.storage_dir  = Path(storage_dir)
        self.max_turns    = max_turns
        self.summarise_at = max(summarise_at, max_turns)
        self.llm          = llm_backend
        self.verbose      = verbose

        self._lock        : threading.Lock       = threading.Lock()
        self._turns       : List[ConversationTurn] = []
        self._summary     : str                  = ""
        self._next_id     : int                  = 0

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # Public API 

    def add_turn(
        self,
        user_query      : str,
        ai_response     : str,
        retrieved_chunks : Optional[List[str]] = None,
        intent          : str = QueryIntent.UNKNOWN.value,
        entities        : Optional[List[str]] = None,
    ) -> ConversationTurn:
        """
        Record a completed conversation turn.

        Call this AFTER the answer generation phase completes, passing
        the chunk_ids that were used to produce the answer.

        Returns the new ConversationTurn.
        """
        turn = ConversationTurn(
            turn_id          = self._next_id,
            user_query       = user_query,
            ai_response      = ai_response,
            retrieved_chunks = retrieved_chunks or [],
            timestamp        = datetime.now().isoformat(),
            query_intent     = intent,
            entities         = entities or [],
        )

        with self._lock:
            self._turns.append(turn)
            self._next_id += 1

            # Trigger summarisation if we've accumulated enough turns
            if len(self._turns) >= self.summarise_at:
                self._maybe_summarise()

            # Enforce window cap after summarisation
            if len(self._turns) > self.max_turns:
                self._turns = self._turns[-self.max_turns:]

        self._save()

        if self.verbose:
            print(
                f" Memory: turn {turn.turn_id} saved "
                f"(window={len(self._turns)}/{self.max_turns})"
            )
        return turn

    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Return the last n turns (default: all active window turns)."""
        with self._lock:
            turns = list(self._turns)
        return turns[-n:] if n else turns

    def build_context_string(self, n_turns: int = 3) -> str:
        """
        Build a compact context string from the last n_turns for injection
        into the query processor.

        Format
        ------
        [Summary of earlier conversation if available]
        Turn N-2: User asked "..." → Answer covered: entity1, entity2
        Turn N-1: User asked "..." → Answer covered: entity3
        Turn N  : User asked "..." → Answer covered: entity4

        This is intentionally brief — it feeds into the query, not the
        generation context.  Full turn text is in passed_chunks for generation.
        """
        parts = []

        if self._summary:
            parts.append(f"Earlier conversation summary:\n{self._summary}")

        recent = self.get_recent_turns(n_turns)
        for turn in recent:
            entity_hint = ""
            if turn.entities:
                entity_hint = f" Entities: {', '.join(turn.entities[:5])}."
            parts.append(
                f"Turn {turn.turn_id}: "
                f"User asked \"{turn.user_query[:120]}\".{entity_hint}"
            )

        return "\n".join(parts) if parts else ""

    def get_recent_entities(self, n_turns: int = 5) -> List[str]:
        """
        Return deduplicated entities from the last n_turns.

        Used by query_engine.py to bias entity extraction toward concepts
        already established in the conversation.
        """
        seen   = set()
        result = []
        for turn in self.get_recent_turns(n_turns):
            for e in turn.entities:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    result.append(e)
        return result

    def is_follow_up(self, query: str) -> bool:
        """
        Heuristic check: does the query reference prior conversation context?

        Looks for pronouns and referential phrases that signal coreference.
        """
        follow_up_markers = [
            "that", "this", "it ", "they", "them", "those", "these",
            "the same", "aforementioned", "mentioned", "above", "earlier",
            "previous", "before", "last time", "you said", "what about",
            "elaborate", "more on", "explain further", "expand on",
            "can you clarify", "how does it", "how do they",
        ]
        q_lower = query.lower()
        return any(marker in q_lower for marker in follow_up_markers)

    @property
    def turn_count(self) -> int:
        with self._lock:
            return len(self._turns)

    def clear(self) -> None:
        """Clear all memory for this session and delete the JSON file."""
        with self._lock:
            self._turns   = []
            self._summary = ""
            self._next_id = 0
        path = self._json_path()
        if path.exists():
            path.unlink()
        if self.verbose:
            print(f" Memory cleared for session '{self.session_id}'")

    # Summarisation
    
    def _maybe_summarise(self) -> None:
        """
        Compress the oldest half of the active window into self._summary.

        Called inside the write lock — do not acquire the lock again.
        Uses the LLM to summarise if available; otherwise simply drops turns.
        """
        n_to_summarise = len(self._turns) - self.max_turns
        if n_to_summarise <= 0:
            return

        old_turns  = self._turns[:n_to_summarise]
        self._turns = self._turns[n_to_summarise:]

        if self.llm is None:
            # No LLM — just note which queries were covered
            topics = [t.user_query[:60] for t in old_turns]
            self._summary = (
                (self._summary + "\n" if self._summary else "") +
                "Earlier queries: " + " | ".join(topics)
            )
            if self.verbose:
                print(
                    f" Memory: {n_to_summarise} turns dropped "
                    "(no LLM for summarisation)"
                )
            return

        # Build the summarisation prompt
        dialogue = "\n".join(
            f"User: {t.user_query}\nAI: {t.ai_response[:300]}"
            for t in old_turns
        )
        prior = (
            f"Existing summary:\n{self._summary}\n\n"
            if self._summary else ""
        )
        prompt = (
            f"{prior}"
            f"Summarise the following conversation turns in 3-5 sentences, "
            f"focusing on the key concepts, entities, and questions covered. "
            f"This summary will be used to resolve coreferences in future queries.\n\n"
            f"{dialogue}\n\nSummary:"
        )

        try:
            new_summary = self.llm.generate(prompt).strip()
            self._summary = new_summary
            if self.verbose:
                print(
                    f" Memory: {n_to_summarise} turns summarised "
                    f"({len(new_summary)} chars)"
                )
        except Exception as e:
            # Summarisation failure is non-fatal — just keep prior summary
            if self.verbose:
                print(f" Memory: summarisation failed ({e}), using prior summary")

    # Persistence

    def _json_path(self) -> Path:
        return self.storage_dir / f"memory_{self.session_id}.json"

    def _save(self) -> None:
        """Persist current state to JSON (called after every add_turn)."""
        with self._lock:
            payload = {
                "session_id" : self.session_id,
                "next_id"    : self._next_id,
                "summary"    : self._summary,
                "turns"      : [
                    {
                        "turn_id"         : t.turn_id,
                        "user_query"      : t.user_query,
                        "ai_response"     : t.ai_response,
                        "retrieved_chunks": t.retrieved_chunks,
                        "timestamp"       : t.timestamp,
                        "query_intent"    : t.query_intent,
                        "entities"        : t.entities,
                    }
                    for t in self._turns
                ],
            }
        try:
            tmp = self._json_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._json_path())
        except Exception as e:
            if self.verbose:
                print(f" Memory: save failed ({e})")

    def _load(self) -> None:
        """Load persisted state from JSON (called at __init__)."""
        path = self._json_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._next_id = payload.get("next_id", 0)
            self._summary = payload.get("summary", "")
            raw_turns     = payload.get("turns", [])
            self._turns   = [
                ConversationTurn(
                    turn_id          = t["turn_id"],
                    user_query       = t["user_query"],
                    ai_response      = t["ai_response"],
                    retrieved_chunks = t.get("retrieved_chunks", []),
                    timestamp        = t.get("timestamp", ""),
                    query_intent     = t.get("query_intent", QueryIntent.UNKNOWN.value),
                    entities         = t.get("entities", []),
                )
                for t in raw_turns
            ]
            # Enforce window cap on load (in case max_turns changed)
            if len(self._turns) > self.max_turns:
                self._turns = self._turns[-self.max_turns:]

            if self.verbose and self._turns:
                print(
                    f" Memory: loaded {len(self._turns)} turn(s) "
                    f"for session '{self.session_id}'"
                )
        except Exception as e:
            if self.verbose:
                print(f" Memory: load failed ({e}) — starting fresh")
            self._turns   = []
            self._summary = ""
            self._next_id = 0

    def __repr__(self) -> str:
        return (
            f"MemoryStore(session='{self.session_id}', "
            f"turns={self.turn_count}/{self.max_turns})"
        )