"""
GraphRAG Answer Generator
=========================

Takes a GradedResult from the RetrievalPipeline and generates a final,
coherent natural-language answer using a Groq LLM.

Features
--------
- Loads GROQ_API_KEY_GENERATE from a .env file automatically (via python-dotenv),
  mirroring the pattern used by retrieval_pipeline.py for its own keys.
- Accepts an explicit llm_api_key argument as an override (highest priority).
- Builds a structured prompt from passed_chunks + query intent/entities.
- Handles PASS / PARTIAL / FAIL verdicts with appropriate tone/caveats.
- Saves the answer text to a UTF-8 .txt file for TTS consumption.
- Updates the pipeline's QueryCache with the generated answer.
- Records the turn in MemoryStore with the real AI response.
- Returns a rich AnswerResult dataclass.

Environment variable resolution order
--------------------------------------
  1. Explicit ``llm_api_key`` constructor argument (if non-empty)
  2. GROQ_API_KEY_GENERATE in the .env file (loaded at import time)
  3. GROQ_API_KEY_GENERATE already present in the OS environment

Flow
----
    GradedResult  (from RetrievalPipeline.run())
         └─ AnswerGenerator.generate()
               ├─ _build_prompt()         build system + user prompt
               ├─ _call_llm()             Groq chat completion
               ├─ _save_to_file()         write answer to TTS file
               ├─ pipeline.record_turn()  update memory
               └─ AnswerResult            returned to caller

Usage
-----
    # .env file (never commit this file):
    #   GROQ_API_KEY_GENERATE=gsk_...

    from retrieval.answer_generator import AnswerGenerator

    gen = AnswerGenerator(
        llm_model  = "llama-3.3-70b-versatile",
        output_dir = "answers",
        verbose    = True,
    )

    answer = gen.generate(graded_result, pipeline=pipeline)
    print(answer.text)      # the answer string
    print(answer.tts_file)  # path to saved .txt file
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import urllib.request
import urllib.error

# ──────────────────────────────────────────────────────────────────────────────
# .env loading  —  must happen before any os.environ.get() call below
# ──────────────────────────────────────────────────────────────────────────────

def _load_dotenv(env_path: Optional[str] = None) -> None:
    """
    Load a .env file into os.environ using python-dotenv.

    Search order (first found wins):
      1. Explicit ``env_path`` argument.
      2. .env in the current working directory.
      3. .env two levels up from this file (project root convention).

    Falls back silently if python-dotenv is not installed, or if no .env
    file exists — the caller is responsible for ensuring the key is set
    through another means (e.g., shell export, CI secrets).
    """
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
    except ImportError:
        # python-dotenv not installed; rely on environment already being set.
        return

    if env_path:
        # Explicit path: load it, warn if it doesn't exist.
        target = Path(env_path)
        if target.is_file():
            load_dotenv(dotenv_path=target, override=False)
        else:
            import warnings
            warnings.warn(
                f"AnswerGenerator: specified .env path not found: {env_path}",
                stacklevel=3,
            )
        return

    # Auto-discovery: walk up from CWD, then fall back to project root.
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(dotenv_path=discovered, override=False)
        return

    # Last-resort: look two directories above this source file.
    project_root_env = Path(__file__).resolve().parents[2] / ".env"
    if project_root_env.is_file():
        load_dotenv(dotenv_path=project_root_env, override=False)


# Load .env immediately so os.environ is populated before class instantiation.
_load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Name of the environment variable that holds the Groq generation API key.
_ENV_KEY_NAME: str = "GROQ_API_KEY_GENERATE"

#: Groq chat-completion endpoint.
_GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"

#: HTTP timeout (seconds) for the LLM call.
_LLM_TIMEOUT_SECONDS: int = 60


# ──────────────────────────────────────────────────────────────────────────────
# AnswerResult  —  returned by AnswerGenerator.generate()
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnswerResult:
    """
    The final output of the answer generation phase.

    Attributes
    ----------
    text          : The generated answer string (ready to display / TTS).
    verdict       : The grader verdict that triggered this answer
                    ("pass" | "partial" | "fail").
    confidence    : Grader confidence score (0.0 – 1.0).
    sources       : List of source file names that contributed to the answer.
    chunk_ids     : List of chunk IDs used (for audit / traceability).
    tts_file      : Absolute path to the saved TTS .txt file (or None if save
                    was disabled / failed).
    elapsed_ms    : Time taken for LLM call in milliseconds.
    model         : Groq model used for generation.
    session_id    : Session ID from the pipeline's memory store.
    turn_recorded : True if pipeline.record_turn() succeeded.
    cache_updated : True if the answer was written to the query cache.
    """
    text          : str
    verdict       : str
    confidence    : float
    sources       : List[str]     = field(default_factory=list)
    chunk_ids     : List[str]     = field(default_factory=list)
    tts_file      : Optional[str] = None
    elapsed_ms    : float         = 0.0
    model         : str           = ""
    session_id    : str           = ""
    turn_recorded : bool          = False
    cache_updated : bool          = False

    def __str__(self) -> str:
        return self.text


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  —  robust attribute extraction (mirrors interactive_retrieval_test.py)
# ──────────────────────────────────────────────────────────────────────────────

def _get_chunk_text(gc) -> str:
    """Robustly extract text from a GradedChunk regardless of nesting."""
    for path in (
        lambda g: g.reranked.fused.chunk.text,
        lambda g: g.fused.chunk.text,
        lambda g: g.fused.text,
        lambda g: g.chunk.text,
        lambda g: g.text,
        lambda g: g.content,
    ):
        try:
            val = path(gc)
            if isinstance(val, str) and val.strip():
                return val
        except AttributeError:
            pass
    return str(gc)


def _get_chunk_id(gc) -> str:
    """Robustly extract chunk_id from a GradedChunk."""
    for path in (
        lambda g: g.reranked.fused.chunk.chunk_id,
        lambda g: g.fused.chunk.chunk_id,
        lambda g: g.chunk.chunk_id,
        lambda g: g.chunk_id,
    ):
        try:
            return str(path(gc))
        except AttributeError:
            pass
    return "unknown"


def _get_chunk_source(gc) -> str:
    """Extract the source/file name from a GradedChunk's metadata."""
    for path in (
        lambda g: g.reranked.fused.chunk.metadata.get("source", ""),
        lambda g: g.fused.chunk.metadata.get("source", ""),
        lambda g: g.chunk.metadata.get("source", ""),
    ):
        try:
            val = path(gc)
            if val:
                return str(val)
        except (AttributeError, TypeError):
            pass
    return ""


def _get_chunk_score(gc) -> float:
    """Extract the best available relevance score from a GradedChunk."""
    for attr in ("relevance_score", "rerank_score", "rrf_score", "score"):
        v = getattr(gc, attr, None)
        if v is not None:
            return float(v)
    fused = getattr(gc, "fused", None)
    if fused:
        for attr in ("rrf_score", "score"):
            v = getattr(fused, attr, None)
            if v is not None:
                return float(v)
    return 0.0


def _sanitize_filename(text: str, max_len: int = 60) -> str:
    """Turn a query string into a safe filename fragment."""
    cleaned = re.sub(r"[^\w\s-]", "", text.lower())
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip("_")
    return cleaned[:max_len] or "answer"


def _resolve_api_key(explicit_key: Optional[str]) -> str:
    """
    Resolve the Groq API key following a clear priority order:

      1. Explicit argument passed to the constructor.
      2. Environment variable GROQ_API_KEY_GENERATE (populated from .env
         by _load_dotenv() at module import time, or set in the shell).

    Raises
    ------
    ValueError
        If no key can be resolved from any source.
    """
    key = explicit_key or os.environ.get(_ENV_KEY_NAME, "")
    if not key:
        raise ValueError(
            f"AnswerGenerator: no API key found.\n"
            f"  • Add '{_ENV_KEY_NAME}=gsk_...' to your .env file, or\n"
            f"  • Export it as a shell variable:  "
            f"export {_ENV_KEY_NAME}=gsk_..., or\n"
            f"  • Pass it explicitly:  "
            f"AnswerGenerator(llm_api_key='gsk_...')"
        )
    return key


# ──────────────────────────────────────────────────────────────────────────────
# AnswerGenerator
# ──────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates natural-language answers from GradedResult objects.

    Args
    ----
    llm_api_key        : Groq API key.  When omitted the key is read from
                         GROQ_API_KEY_GENERATE in the .env file (or the
                         shell environment).
    env_file           : Path to a custom .env file.  Defaults to
                         auto-discovery (CWD → project root).
    llm_model          : Groq model for generation
                         (default llama-3.3-70b-versatile).
    output_dir         : Directory where TTS .txt files are saved.
                         Pass None to disable file saving.
    max_context_chunks : Max passed_chunks included in the prompt (default 6).
    max_tokens         : Max tokens for the LLM response (default 1024).
    temperature        : LLM temperature (default 0.3 — factual but natural).
    verbose            : Print progress to stdout (default True).
    """

    # ── Prompt templates ──────────────────────────────────────────────────────

    _SYSTEM_PROMPT = (
        "You are a precise, helpful AI assistant integrated into a GraphRAG "
        "system.  Your job is to answer the user's question based ONLY on the "
        "retrieved context passages provided.\n\n"
        "Rules:\n"
        "1. Answer in clear, fluent prose.  Do NOT use bullet points or "
        "numbered lists unless the question explicitly asks for a list.\n"
        "2. Stay strictly grounded in the provided context.  Do NOT invent facts.\n"
        "3. If the context only partially covers the question, answer what you "
        "can and acknowledge the gap naturally — do not fabricate missing details.\n"
        "4. Be concise: prefer 2–4 paragraphs unless a longer answer is clearly "
        "needed.\n"
        "5. Do NOT mention \"chunks\", \"retrieval\", \"grading\", or any internal "
        "pipeline terminology.  Write as if you simply know the answer.\n"
        "6. End with a single sentence that summarises the key takeaway.\n"
    )

    _PARTIAL_CAVEAT = (
        "\n[Note: The available information is incomplete.  Answer as best you "
        "can and indicate what is uncertain.]\n"
    )

    _FAIL_ANSWER = (
        "I wasn't able to find enough relevant information in the document to "
        "answer your question confidently.  Please try rephrasing, or check "
        "whether the topic is covered in the source material."
    )

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        llm_api_key          : Optional[str] = None,
        env_file             : Optional[str] = None,
        llm_model            : str           = "llama-3.3-70b-versatile",
        output_dir           : Optional[str] = "answers",
        max_context_chunks   : int           = 6,
        max_tokens           : int           = 1024,
        temperature          : float         = 0.3,
        verbose              : bool          = True,
    ) -> None:
        # If the caller specifies a custom .env path, reload dotenv now so
        # it takes effect before _resolve_api_key reads os.environ.
        if env_file:
            _load_dotenv(env_path=env_file)

        # Resolve the API key — raises ValueError early if missing.
        self._api_key = _resolve_api_key(llm_api_key)

        self.llm_model          = llm_model
        self.output_dir         = output_dir
        self.max_context_chunks = max_context_chunks
        self.max_tokens         = max_tokens
        self.temperature        = temperature
        self.verbose            = verbose

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._log(f"AnswerGenerator ready  (model={llm_model})")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        graded_result,
        pipeline=None,
        original_query: Optional[str] = None,
    ) -> AnswerResult:
        """
        Generate an answer from a GradedResult.

        Args
        ----
        graded_result  : GradedResult returned by RetrievalPipeline.run().
        pipeline       : RetrievalPipeline instance.  When provided:
                           • pipeline.record_turn() is called with the real answer
                           • pipeline.cache is updated with the answer text
        original_query : The raw user query string.  Falls back to
                         graded_result.query.normalised_text if omitted.

        Returns
        -------
        AnswerResult
        """
        t0 = time.time()

        # ── Extract query info ────────────────────────────────────────────────
        query_rep   = graded_result.query
        query_text  = original_query or query_rep.normalised_text
        verdict_val = graded_result.verdict.value   # "pass" | "partial" | "fail"
        confidence  = graded_result.confidence
        passed      = graded_result.passed_chunks

        session_id = (
            pipeline.memory.session_id
            if pipeline and hasattr(pipeline, "memory")
            else "unknown"
        )

        self._log(
            f"Generating answer  |  verdict={verdict_val.upper()}  "
            f"confidence={confidence:.3f}  chunks={len(passed)}  "
            f"session={session_id}"
        )

        # ── Gather chunk data ─────────────────────────────────────────────────
        top_chunks = passed[: self.max_context_chunks]

        chunk_texts  : List[str]   = [_get_chunk_text(gc)   for gc in top_chunks]
        chunk_ids    : List[str]   = [_get_chunk_id(gc)     for gc in top_chunks]
        chunk_scores : List[float] = [_get_chunk_score(gc)  for gc in top_chunks]

        raw_sources = [_get_chunk_source(gc) for gc in top_chunks]
        sources = sorted({Path(s).name for s in raw_sources if s})

        # ── Generate answer ───────────────────────────────────────────────────
        if verdict_val == "fail" or not chunk_texts:
            answer_text = self._FAIL_ANSWER
            self._log("Verdict FAIL — returning fallback answer.")
        else:
            prompt = self._build_prompt(
                query_text   = query_text,
                intent       = getattr(query_rep, "intent", None),
                entities     = getattr(query_rep, "entities", []),
                chunk_texts  = chunk_texts,
                chunk_scores = chunk_scores,
                verdict      = verdict_val,
            )
            answer_text = self._call_llm(prompt)

        elapsed_ms = (time.time() - t0) * 1000

        # ── Save to TTS file ──────────────────────────────────────────────────
        tts_file = self._save_to_file(
            answer_text = answer_text,
            query_text  = query_text,
            session_id  = session_id,
            verdict     = verdict_val,
            sources     = sources,
        )

        # ── Update memory (real answer, not stub) ─────────────────────────────
        turn_recorded = self._record_turn(
            pipeline      = pipeline,
            query_text    = query_text,
            answer_text   = answer_text,
            graded_result = graded_result,
            session_id    = session_id,
        )

        # ── Update query cache ────────────────────────────────────────────────
        cache_updated = self._update_cache(
            pipeline      = pipeline,
            query_rep     = query_rep,
            graded_result = graded_result,
            answer_text   = answer_text,
        )

        self._log(f"Answer generation complete  ({elapsed_ms:.0f} ms)")

        return AnswerResult(
            text          = answer_text,
            verdict       = verdict_val,
            confidence    = confidence,
            sources       = sources,
            chunk_ids     = chunk_ids,
            tts_file      = tts_file,
            elapsed_ms    = elapsed_ms,
            model         = self.llm_model,
            session_id    = session_id,
            turn_recorded = turn_recorded,
            cache_updated = cache_updated,
        )

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        query_text   : str,
        intent,
        entities     : List[str],
        chunk_texts  : List[str],
        chunk_scores : List[float],
        verdict      : str,
    ) -> str:
        """Assemble the full user-turn prompt sent to the LLM."""
        lines: List[str] = []

        # Intent + entity hint (helps the LLM calibrate response style)
        intent_str = intent.value if hasattr(intent, "value") else str(intent or "")
        if intent_str or entities:
            lines.append("## Query metadata")
            if intent_str:
                lines.append(f"- Intent: {intent_str}")
            if entities:
                lines.append(f"- Key entities: {', '.join(entities[:8])}")
            lines.append("")

        # Partial caveat
        if verdict == "partial":
            lines.append(self._PARTIAL_CAVEAT)

        # Context passages
        lines.append("## Retrieved context passages")
        lines.append(
            "(Listed from most to least relevant.  Use ALL passages that help "
            "answer the question.)"
        )
        lines.append("")

        for i, (text, score) in enumerate(zip(chunk_texts, chunk_scores), 1):
            lines.append(f"### Passage {i}  (relevance: {score:.3f})")
            lines.append(text.strip())
            lines.append("")

        # The actual question
        lines.append("## Question")
        lines.append(query_text)
        lines.append("")
        lines.append("## Answer")

        return "\n".join(lines)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> str:
        """
        Call the Groq chat-completion API and return the answer string.

        Uses ``requests`` instead of ``urllib`` so that a proper User-Agent
        header is sent automatically, avoiding Cloudflare's bot-protection
        HTTP 403 / error-code 1010 that ``urllib`` triggers with its default
        Python/<version> agent string.

        The API key is never logged or embedded in error messages.
        A structured error hierarchy is handled:
          - requests.HTTPError      → log status + reason, return fallback
          - requests.ConnectionError / Timeout → network failure, return fallback
          - json.JSONDecodeError    → malformed response, return fallback
          - KeyError / IndexError   → unexpected payload shape, return fallback
        """
        try:
            import requests as _requests  # type: ignore
        except ImportError:
            self._warn(
                "'requests' package not found — falling back to urllib.  "
                "Install it with: pip install requests"
            )
            return self._call_llm_urllib(user_prompt)

        headers = {
            "Content-Type"  : "application/json",
            # Key is injected at runtime from the resolved secret — never
            # logged or stored in any other attribute.
            "Authorization" : f"Bearer {self._api_key}",
            # Explicit User-Agent prevents Cloudflare WAF 403/1010 blocks
            # that occur when urllib sends its bare "Python-urllib/<ver>" agent.
            "User-Agent"    : "GraphRAG-AnswerGenerator/1.0",
        }
        payload = {
            "model"       : self.llm_model,
            "max_tokens"  : self.max_tokens,
            "temperature" : self.temperature,
            "messages"    : [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        }

        try:
            resp = _requests.post(
                _GROQ_API_URL,
                json    = payload,
                headers = headers,
                timeout = _LLM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except _requests.exceptions.HTTPError as exc:
            self._warn(f"HTTP {exc.response.status_code}: {exc.response.text}")
            return self._FAIL_ANSWER
        except (_requests.exceptions.ConnectionError,
                _requests.exceptions.Timeout) as exc:
            self._warn(f"Network error calling LLM: {exc}")
            return self._FAIL_ANSWER
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            self._warn(
                f"Unexpected LLM response shape ({exc!r}) — "
                "returning fallback answer."
            )
            return self._FAIL_ANSWER

    def _call_llm_urllib(self, user_prompt: str) -> str:
        """
        Fallback LLM caller using only the stdlib ``urllib``.
        Includes an explicit User-Agent header to avoid Cloudflare 403/1010.
        Only used when the ``requests`` package is unavailable.
        """
        payload = json.dumps({
            "model"       : self.llm_model,
            "max_tokens"  : self.max_tokens,
            "temperature" : self.temperature,
            "messages"    : [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            url     = _GROQ_API_URL,
            data    = payload,
            method  = "POST",
            headers = {
                "Content-Type"  : "application/json",
                "Authorization" : f"Bearer {self._api_key}",
                # Explicit User-Agent prevents Cloudflare WAF 403/1010 blocks.
                "User-Agent"    : "GraphRAG-AnswerGenerator/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT_SECONDS) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            self._warn(f"HTTP {exc.code}: {body}")
            return self._FAIL_ANSWER
        except urllib.error.URLError as exc:
            self._warn(f"Network error calling LLM: {exc}")
            return self._FAIL_ANSWER
        try:
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"].strip()
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            self._warn(
                f"Unexpected LLM response shape ({exc!r}) — "
                "returning fallback answer."
            )
            return self._FAIL_ANSWER

    # ── File saver ────────────────────────────────────────────────────────────

    def _save_to_file(
        self,
        answer_text : str,
        query_text  : str,
        session_id  : str,
        verdict     : str,
        sources     : List[str],
    ) -> Optional[str]:
        """
        Write the answer to a UTF-8 .txt file and return the absolute path.

        File name format:
            {session_slug}__{timestamp}__{query_slug}.txt

        The file includes a small header block (query, verdict, sources) so the
        TTS consumer can optionally skip it and read only the answer body.
        The answer body is separated by a line of dashes for easy splitting.

        Returns None if ``output_dir`` is falsy, or if the write fails.
        """
        if not self.output_dir:
            return None

        try:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_slug = _sanitize_filename(query_text)
            safe_sid   = _sanitize_filename(session_id, max_len=20) or "session"
            filename   = f"{safe_sid}__{timestamp}__{query_slug}.txt"
            filepath   = Path(self.output_dir) / filename

            header_lines = [
                f"SESSION   : {session_id}",
                f"TIMESTAMP : {datetime.now().isoformat(timespec='seconds')}",
                f"QUERY     : {query_text}",
                f"VERDICT   : {verdict.upper()}",
                f"SOURCES   : {', '.join(sources) if sources else 'n/a'}",
                "-" * 72,
                "",     # blank line before answer body
            ]

            content = "\n".join(header_lines) + answer_text + "\n"
            filepath.write_text(content, encoding="utf-8")
            self._log(f"TTS file saved → {filepath}")
            return str(filepath.resolve())

        except OSError as exc:
            self._warn(f"Could not save TTS file: {exc}")
            return None

    # ── Pipeline integration helpers ──────────────────────────────────────────

    def _record_turn(
        self,
        pipeline,
        query_text    : str,
        answer_text   : str,
        graded_result,
        session_id    : str,
    ) -> bool:
        """Call pipeline.record_turn() and return True on success."""
        if not pipeline:
            return False
        try:
            pipeline.record_turn(
                user_query    = query_text,
                ai_response   = answer_text,
                graded_result = graded_result,
            )
            self._log(
                f"Memory updated  "
                f"(session={session_id}  "
                f"turns={pipeline.memory.turn_count})"
            )
            return True
        except Exception as exc:   # noqa: BLE001
            self._warn(f"Memory record_turn failed: {exc}")
            return False

    def _update_cache(
        self,
        pipeline,
        query_rep,
        graded_result,
        answer_text: str,
    ) -> bool:
        """
        Attach the generated answer to the GradedResult in the query cache.
        Returns True on success.
        """
        if not (pipeline and hasattr(pipeline, "cache")):
            return False
        try:
            embedding = getattr(query_rep, "embedding", None)
            if embedding is None:
                return False
            # Attach the generated answer so future cache hits include it.
            graded_result.generated_answer = answer_text
            pipeline.cache.put(embedding, graded_result)
            stats = pipeline.cache.stats()
            self._log(
                f"Cache updated  "
                f"(hits={stats.get('hits', '?')}  "
                f"misses={stats.get('misses', '?')}  "
                f"size={stats.get('size', '?')})"
            )
            return True
        except Exception as exc:   # noqa: BLE001
            self._warn(f"Cache update failed: {exc}")
            return False

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  \033[94m[AnswerGenerator]\033[0m  {msg}")

    def _warn(self, msg: str) -> None:
        print(f"  \033[93m[AnswerGenerator ⚠]\033[0m  {msg}")