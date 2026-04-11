from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from text_cleaner import TextCleaner


# CHUNK DATACLASS

@dataclass
class Chunk:
    """
    A single piece of text produced by any chunker.

    Attributes
    ----------
    text        : The actual chunk content (cleaned, trimmed).
    chunk_id    : Zero-based index of this chunk within its parent document.
    start_char  : Character offset where this chunk begins in the cleaned source.
    end_char    : Character offset where this chunk ends in the cleaned source.
    metadata    : Free-form dict — chunker type, overlap info, etc.
                  Downstream phases (embedding, graph, retrieval) may add their
                  own keys here without breaking the dataclass contract.
    """
    text       : str
    chunk_id   : int
    start_char : int
    end_char   : int
    metadata   : dict = field(default_factory=dict)

    # Convenience property used by embedding phase
    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id}, chars={self.char_count}, "
            f"words={self.word_count}, preview='{preview}...')"
        )


# BASE CHUNKER

class BaseChunker:
    """
    Abstract base class for all chunkers.

    Responsibilities
    ----------------
    • Holds a shared TextCleaner instance.
    • Exposes a clean(text) helper used by every subclass before splitting.
    • Declares the .chunk(text) interface that subclasses must implement.
    • Provides _make_chunk() factory to create Chunk objects consistently.
    • Provides _find_start() to compute real character offsets inside the
      cleaned text — required by embedding and retrieval phases.
    """

    chunker_type: str = "base"  # overridden by each subclass

    def __init__(self):
        self._cleaner = TextCleaner()

    # public interface

    def clean(self, text: str) -> str:
        """Clean raw extracted text. Call this before chunking."""
        return self._cleaner.clean(text)

    def chunk(self, text: str) -> List[Chunk]:
        """
        Split *text* into Chunk objects.
        Subclasses must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement .chunk(text)"
        )

    # helpers used by subclasses

    def _make_chunk(
        self,
        text: str,
        chunk_id: int,
        source_text: str,
        search_start: int = 0,
        extra_meta: Optional[dict] = None,
    ) -> Chunk:
        """
        Build a Chunk, computing char offsets inside *source_text*.

        Parameters
        ----------
        text         : The chunk's text content.
        chunk_id     : Zero-based position index.
        source_text  : The full cleaned document (used for offset search).
        search_start : Hint — start searching for *text* from this offset to
                       avoid matching an earlier identical substring.
        extra_meta   : Additional metadata dict merged into Chunk.metadata.
        """
        start = source_text.find(text, search_start)
        if start == -1:
            # Fallback: linear scan with stripped comparison
            start = search_start
        end = start + len(text)

        metadata = {"chunker": self.chunker_type}
        if extra_meta:
            metadata.update(extra_meta)

        return Chunk(
            text=text,
            chunk_id=chunk_id,
            start_char=start,
            end_char=end,
            metadata=metadata,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text into sentences using punctuation heuristics.

        Strategy
        --------
        Use a regex that splits AFTER sentence-ending punctuation
        (. ! ?) followed by whitespace, but avoids splitting on
        common abbreviations like "Dr.", "Fig.", "e.g.", "i.e.".

        This is intentionally simple and fast — a full NLP tokenizer
        would add heavyweight dependencies not needed here.
        """

        # Protect common abbreviations by temporarily replacing their dots
        abbreviations = [
            "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr",
            "Fig", "Eq", "No", "vs", "etc", "e.g", "i.e", "al",
            "approx", "ref", "ch", "vol", "pp",
        ]
        placeholder = "ABBR_DOT"
        for abbr in abbreviations:
            # Match "Abbr." followed by a space or digit
            text = re.sub(
                rf"\b{re.escape(abbr)}\.",
                f"{abbr}{placeholder}",
                text,
            )

        # Split after . ! ? followed by whitespace + uppercase or digit
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\(])", text)

        # Restore protected dots
        sentences = [p.replace(placeholder, ".").strip() for p in parts]

        # Drop empty strings
        return [s for s in sentences if s]