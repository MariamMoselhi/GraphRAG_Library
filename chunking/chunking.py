"""
chunking.py — GraphRAG Chunking Module  (Public API)

Internal layout
---------------
text_cleaner.py          — TextCleaner (cleaning & preprocessing)
chunk_base.py            — Chunk dataclass + BaseChunker abstract class
chunk_implementations.py — all six concrete chunker classes
chunking.py              — this file: registry, factory, public API
"""

from __future__ import annotations

from typing import List

# ── re-export the public dataclass so users can do: from chunking import Chunk
from chunk_base import Chunk, BaseChunker

# ── import all concrete chunkers from the internal implementations module
from chunk_implementations import (
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    RecursiveChunker,
    SlidingWindowChunker,
    SemanticChunker,
)

__all__ = [
    "Chunk",
    "get_chunker",
    "chunk_text",
    "DEFAULT_CHUNKER",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "SlidingWindowChunker",
    "SemanticChunker",
]


# Registry maps string name → chunker class
_CHUNKER_REGISTRY = {
    "fixed_size"    : FixedSizeChunker,
    "sentence"      : SentenceChunker,
    "paragraph"     : ParagraphChunker,
    "recursive"     : RecursiveChunker,
    "sliding_window": SlidingWindowChunker,
    "semantic"      : SemanticChunker,       # default
}

DEFAULT_CHUNKER = "semantic"


def get_chunker(chunker_type: str = DEFAULT_CHUNKER, **kwargs) -> BaseChunker:
    """
    Instantiate a chunker by name.

    Parameters
    ----------
    chunker_type : str — one of the keys in _CHUNKER_REGISTRY.
                         Defaults to "semantic".
    **kwargs     : forwarded to the chunker's __init__.

    Returns
    -------
    An instance of the requested BaseChunker subclass.

    Example
    -------
    >>> chunker = get_chunker("sentence", sentences_per_chunk=6)
    >>> chunks  = chunker.chunk(extracted_text)
    """
    key = chunker_type.lower().strip()
    cls = _CHUNKER_REGISTRY.get(key)
    if cls is None:
        valid = list(_CHUNKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown chunker type '{chunker_type}'. Valid options: {valid}"
        )
    return cls(**kwargs)


def chunk_text(
    text: str,
    chunker_type: str = DEFAULT_CHUNKER,
    **kwargs,
) -> List[Chunk]:
    """
    One-liner convenience function: clean + chunk in a single call.

    Parameters
    ----------
    text         : Raw text from an extractor (PDF, DOCX, PPTX, audio …).
    chunker_type : Which strategy to use (default "semantic").
    **kwargs     : Passed to the chunker constructor.

    Returns
    -------
    List[Chunk]

    Example
    -------
    >>> from chunking import chunk_text
    >>> chunks = chunk_text(raw_pdf_text, chunker_type="semantic", threshold=0.45)
    """
    chunker = get_chunker(chunker_type, **kwargs)
    return chunker.chunk(text)