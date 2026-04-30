"""
Query Processor — first stage of the GraphRAG retrieval pipeline.

Responsibilities
----------------
1. Detect input modality (audio vs text).
2. If audio: transcribe using Groq Whisper (API key #1 — GROQ_API_KEY_WHISPER).
3. Normalise the text query (encoding, whitespace, punctuation).
4. Check MemoryStore for prior conversation context.
5. Detect if this is a follow-up query and inject memory context for
   coreference resolution (so "it" and "that method" resolve correctly).
6. Return a QueryRepresentation with raw_text, normalised_text, and
   memory_context ready for query_engine.py.

API key separation
------------------
Whisper transcription uses its own Groq API key (GROQ_API_KEY_WHISPER)
so audio transcription never competes with LLM extraction calls for
rate limits.  The key is read from environment or passed directly.

Audio format support
--------------------
Supported: mp3, wav, m4a, flac, ogg, aac, webm (Groq Whisper limits).
Max file size: 25 MB (Groq limit).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

from .retrieval_context import (
    InputModality,
    PhaseStats,
    QueryIntent,
    QueryRepresentation,
    RetrievalTrace,
)
from .memory_store import MemoryStore
from .retrieval_logger import RetrievalLogger


# Supported audio extensions for auto-detection
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


class QueryProcessor:
    """
    First pipeline stage: turn raw user input into a QueryRepresentation.

    Args
    ----
    whisper_api_key     : Groq API key for Whisper transcription.
                          Falls back to GROQ_API_KEY_WHISPER env var.
    whisper_model       : Groq Whisper model (default "whisper-large-v3-turbo").
    whisper_language    : ISO language code ("en", "ar", None = auto-detect).
    whisper_temperature : Sampling temperature for transcription (default 0.0).
    memory_store        : MemoryStore instance (shared across all pipeline stages).
    memory_window       : Number of prior turns to inject as context (default 3).
    logger              : RetrievalLogger instance.
    verbose             : Whether to print phase details.
    """

    def __init__(
        self,
        whisper_api_key     : Optional[str]     = None,
        whisper_model       : str               = "whisper-large-v3-turbo",
        whisper_language    : Optional[str]     = None,
        whisper_temperature : float             = 0.0,
        memory_store        : Optional[MemoryStore] = None,
        memory_window       : int               = 3,
        logger              : Optional[RetrievalLogger] = None,
        verbose             : bool              = True,
    ):
        self.whisper_model       = whisper_model
        self.whisper_language    = whisper_language
        self.whisper_temperature = whisper_temperature
        self.memory_store        = memory_store
        self.memory_window       = memory_window
        self.logger              = logger or RetrievalLogger(verbose=verbose)
        self.verbose             = verbose

        # Initialise Groq client for Whisper
        api_key = (
            whisper_api_key
            or os.environ.get("GROQ_API_KEY_WHISPER")
        )
        if _GROQ_AVAILABLE and api_key:
            self._groq = Groq(api_key=api_key)
        else:
            self._groq = None
            if verbose:
                print(
                    "  ⚠  QueryProcessor: Groq client not initialised — "
                    "audio transcription unavailable.  "
                    "Set GROQ_API_KEY_WHISPER in your environment."
                )

    # entry point

    def process(
        self,
        user_input  : str,
        trace       : Optional[RetrievalTrace] = None,
    ) -> QueryRepresentation:
        """
        Process raw user input into a QueryRepresentation.

        Args
        ----
        user_input  : Either a text query string or a file path to an audio file.
        trace       : RetrievalTrace to update (optional).

        Returns
        -------
        QueryRepresentation with raw_text, normalised_text, and memory_context set.
        """
        t0 = time.time()

        self.logger.phase_start("Query Processor")

        # Detect modality
        modality, raw_text = self._detect_modality(user_input)

        if trace:
            trace.input_modality = modality.value
            trace.query_raw      = user_input

        # Transcribe audio
        transcription_text: Optional[str] = None

        if modality == InputModality.AUDIO:
            self.logger.info(f"Audio input detected: {Path(user_input).name}")
            transcription_text = self._transcribe(user_input)

            if not transcription_text:
                self.logger.error("Transcription failed — using empty string")
                transcription_text = ""

            raw_text = transcription_text

            if trace:
                trace.transcription_text = transcription_text
            self.logger.info(f"Transcribed: {transcription_text[:120]}")

        # Normalise
        normalised = self._normalise(raw_text)

        # Memory context
        memory_context : Optional[str] = None
        memory_turns   : int           = 0
        is_follow_up   : bool          = False

        if self.memory_store and self.memory_store.turn_count > 0:
            is_follow_up = self.memory_store.is_follow_up(normalised)
            if is_follow_up:
                memory_context = self.memory_store.build_context_string(
                    n_turns=self.memory_window
                )
                memory_turns   = min(self.memory_window, self.memory_store.turn_count)
                self.logger.info(
                    f"Follow-up detected — injecting {memory_turns} memory turn(s)"
                )

        if trace:
            trace.memory_turns_used = memory_turns

        # Build representation
        rep = QueryRepresentation(
            raw_text        = raw_text,
            normalised_text = normalised,
            intent          = QueryIntent.FOLLOW_UP if is_follow_up else QueryIntent.UNKNOWN,
            memory_context  = memory_context,
        )

        elapsed_ms = (time.time() - t0) * 1000

        self.logger.print_query(
            raw_text    = raw_text,
            intent      = rep.intent.value,
            entities    = [],
            keywords    = [],
            cypher      = None,
            from_memory = is_follow_up,
        )

        if trace:
            trace.add_phase(PhaseStats(
                phase_name  = "Query Processor",
                elapsed_ms  = elapsed_ms,
                input_count = 1,
                output_count= 1,
                notes       = modality.value + (" (follow-up)" if is_follow_up else ""),
            ))

        self.logger.phase_end("Query Processor", count=1, elapsed_ms=elapsed_ms)
        return rep

    # Transcription

    def _transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe an audio file using Groq Whisper API.

        Returns the transcribed text, or None on failure.
        """
        if self._groq is None:
            self.logger.error("Groq client not available — cannot transcribe audio")
            return None

        audio_path = Path(audio_path)
        if not audio_path.exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return None

        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            self.logger.error(
                f"Audio file too large: {file_size_mb:.1f} MB (Groq limit: 25 MB)"
            )
            return None

        try:
            with open(audio_path, "rb") as f:
                transcription = self._groq.audio.transcriptions.create(
                    file         = (audio_path.name, f.read()),
                    model        = self.whisper_model,
                    language     = self.whisper_language,
                    temperature  = self.whisper_temperature,
                    response_format = "verbose_json",
                )
            text = getattr(transcription, "text", "").strip()
            lang = getattr(transcription, "language", "unknown")
            dur  = getattr(transcription, "duration", 0)
            if self.verbose:
                self.logger.info(
                    f"Whisper: language={lang}  duration={dur:.1f}s  "
                    f"chars={len(text)}"
                )
            return text

        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            return None

    # Helpers

    @staticmethod
    def _detect_modality(user_input: str) -> tuple:
        """
        Determine whether user_input is an audio file path or a text query.

        Returns (InputModality, str) — the modality and the raw string.
        For audio, the string is the file path.
        For text, the string is the query text itself.
        """
        stripped = user_input.strip()
        if Path(stripped).suffix.lower() in _AUDIO_EXTENSIONS and Path(stripped).exists():
            return InputModality.AUDIO, stripped
        return InputModality.TEXT, stripped

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Lightweight text normalisation.

        Steps
        -----
        1. Strip leading/trailing whitespace.
        2. Collapse internal whitespace (multiple spaces → one space).
        3. Replace smart quotes with straight quotes.
        4. Remove null bytes and other control characters.
        5. Ensure the query ends without trailing punctuation that would
           confuse keyword extraction (remove trailing period/comma/semicolon).
        """
        # Control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

        # Smart quotes → straight
        text = (
            text
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
        )

        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", " ", text)
        text = text.strip()

        # Remove trailing weak punctuation (not ?, !)
        text = re.sub(r"[.,;]+$", "", text).strip()

        return text