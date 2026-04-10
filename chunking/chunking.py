import re
import math
import hashlib
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple


def _make_chunk_id(source_id: str, index: int) -> str:
    """Return a short, stable identifier: '<source_id>_<index>'."""
    return f"{source_id}_{index}"


def _approx_tokens(text: str) -> int:
    """
    Estimate token count as whitespace-split word count.
    Close enough to BPE token counts for typical English/technical text
    without requiring a tokeniser library.
    """
    return len(text.split())


def _build_chunk(
    text: str,
    source_id: str,
    index: int,
    char_start: int,
    char_end: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the standardised chunk dictionary returned by every chunker.

    Parameters
    ----------
    text        : The chunk's content (may be stripped).
    source_id   : Caller-supplied document identifier.
    index       : Position of this chunk in the output list (0-based).
    char_start  : Character offset where this chunk begins in the *original* text.
    char_end    : Character offset where this chunk ends (exclusive).
    metadata    : Arbitrary extra fields added by the calling chunker.
    """
    return {
        "chunk_id":    _make_chunk_id(source_id, index),
        "text":        text,
        "char_start":  char_start,
        "char_end":    char_end,
        "chunk_index": index,
        "token_count": _approx_tokens(text),
        "metadata":    metadata or {},
    }


# Sentence splitter (regex, no NLTK)
# ─────────────────────────────────────────────────────────────────────────────

# Sentence splitter: a two-step approach because Python's `re` module
# requires fixed-width look-behinds, making it impossible to exclude variable-
# length abbreviations in a single pattern.

# Step 1: Match any [.!?] followed by whitespace + uppercase / end-of-string.
# Step 2: In _split_sentences() we manually skip matches that are preceded by
#        a known abbreviation or a digit (to avoid splitting on "3.14", "Mr.", etc.).
_SENT_END = re.compile(
    r'[.!?]'
    r'(?:["\'])?'               # optional closing quote
    r'(?=\s+[A-Z\d"\'\(]|$)',  # followed by whitespace+uppercase/digit or end
    re.UNICODE
)

# Abbreviations whose trailing period must NOT trigger a sentence split.
# Stored as a frozenset of lowercase strings (without the dot).
_ABBREVS = frozenset([
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "eg", "ie", "fig", "st", "no", "vol", "pp", "ph", "ca",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "oct", "nov", "dec", "approx", "dept", "est", "govt",
])


def _split_sentences(text: str) -> List[str]:
    """
    Split *text* into sentences using a two-step heuristic (no NLTK/spaCy).

    Step 1 — Regex: find every [.!?] followed by whitespace + uppercase / end.
    Step 2 — Filter: skip matches preceded by a known abbreviation or a digit.
    """
    if not text.strip():
        return []

    boundaries: List[int] = []
    for match in _SENT_END.finditer(text):
        pos = match.start()                         # position of [.!?]

        # Walk backwards to get the word immediately before the punctuation
        i = pos - 1
        while i >= 0 and text[i].isalpha():
            i -= 1
        preceding_word = text[i + 1: pos].lower()

        if preceding_word in _ABBREVS:
            continue                                # abbreviation — skip

        if pos > 0 and text[pos - 1].isdigit():
            continue                                # decimal number — skip

        boundaries.append(match.end())             # record split position

    sentences: List[str] = []
    prev = 0
    for boundary in boundaries:
        fragment = text[prev:boundary].strip()
        if fragment:
            sentences.append(fragment)
        prev = boundary

    tail = text[prev:].strip()
    if tail:
        sentences.append(tail)

    if not sentences:
        sentences = [text.strip()]

    return sentences


def _split_paragraphs(text: str) -> List[str]:
    """
    Split *text* on one-or-more blank lines.

    Structural markers injected by the extractors (e.g. '--- Page 3 ---',
    '=== SLIDE 7 ===') also act as paragraph boundaries because they are
    always surrounded by blank lines.
    """
    # Normalise Windows line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Split on two-or-more consecutive newlines
    raw_paragraphs = re.split(r'\n{2,}', text)
    return [p.strip() for p in raw_paragraphs if p.strip()]


# Structural-marker detector (for extracted documents)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns produced by the project's extractors:
#   PDFExtractor   → "--- Page 3 ---"
#   PPTXExtractor  → "======  SLIDE 7  ======"  or "SLIDE 7"
#   DOCXExtractor  → "[HEADING 1] Introduction"
#   AudioExtractor / VideoExtractor → no special markers (plain text)
_STRUCTURAL_MARKER = re.compile(
    r'(?:'
    r'---+\s*(?:Page|Slide|Table|Figure)\s*\d+\s*---+'  # PDF / generic
    r'|={3,}\s*(?:SLIDE|SECTION|CHAPTER|HEADING)\s*\d*\s*={3,}'  # PPTX headings
    r'|\[HEADING\s+\d+\]\s*.+'                          # DOCX headings
    r'|\[SPEAKER\s+NOTES\]'                             # PPTX speaker notes
    r')',
    re.IGNORECASE
)


def _is_structural_marker(line: str) -> bool:
    """Return True if *line* is a document-structure marker (not body text)."""
    return bool(_STRUCTURAL_MARKER.match(line.strip()))


# Overlap helper
# ─────────────────────────────────────────────────────────────────────────────

def _add_overlap(chunks: List[str], overlap_sentences: int) -> List[str]:
    """
    Return a new list where each chunk (except the first) is prepended with
    the last *overlap_sentences* sentences of the preceding chunk.

    This preserves cross-boundary context without duplicating full chunks.
    """
    if overlap_sentences <= 0 or len(chunks) < 2:
        return chunks

    result: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sents = _split_sentences(chunks[i - 1])
        tail = prev_sents[-overlap_sentences:] if len(prev_sents) >= overlap_sentences else prev_sents
        overlap_text = " ".join(tail).strip()
        result.append((overlap_text + "\n\n" + chunks[i]).strip())

    return result


#  ABSTRACT BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AbstractChunker(ABC):
    """
    All chunkers inherit from this class and must implement `chunk()`.

    Parameters shared by every subclass
    ------------------------------------
    max_chunk_size : int
        Maximum number of *characters* per chunk (hard ceiling).
        Chunks are never split mid-sentence if avoidable.
    overlap_sentences : int
        How many sentences from the end of chunk N are prepended to chunk N+1
        to preserve cross-boundary context.  0 = no overlap.
    min_chunk_size : int
        Chunks shorter than this many characters are merged with the next one
        (prevents tiny orphan chunks).
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,
        overlap_sentences: int = 1,
        min_chunk_size: int = 100,
    ) -> None:
        self.max_chunk_size    = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.min_chunk_size    = min_chunk_size


    @abstractmethod
    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split *text* into chunks and return a list of chunk dicts.

        Parameters
        ----------
        text      : The full text to be chunked (pre-cleaned by TextCleaner).
        source_id : Human-readable document identifier (used in chunk IDs).
        metadata  : Key-value pairs merged into every chunk's ``metadata``
                    field (e.g. ``{"filename": "lecture.pdf", "page": 3}``).
        """


    def _enforce_size_ceiling(self, text: str) -> List[str]:
        """
        If *text* exceeds ``max_chunk_size`` characters, break it on sentence
        boundaries first, then on whitespace, ensuring no sub-chunk exceeds the
        ceiling.  Returns a list of strings, each ≤ max_chunk_size chars.
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        # Try to split on sentences first
        sentences = _split_sentences(text)
        sub_chunks: List[str] = []
        current = ""

        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent

            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                # Flush current accumulator
                if current:
                    sub_chunks.append(current)
                # If the single sentence itself is too long, split on whitespace
                if len(sent) > self.max_chunk_size:
                    words = sent.split()
                    buf = ""
                    for word in words:
                        trial = (buf + " " + word).strip() if buf else word
                        if len(trial) <= self.max_chunk_size:
                            buf = trial
                        else:
                            if buf:
                                sub_chunks.append(buf)
                            buf = word
                    if buf:
                        sub_chunks.append(buf)
                    current = ""
                else:
                    current = sent

        if current:
            sub_chunks.append(current)

        return sub_chunks if sub_chunks else [text[: self.max_chunk_size]]

    def _merge_tiny_chunks(self, texts: List[str]) -> List[str]:
        """
        Merge consecutive chunks that are shorter than ``min_chunk_size``
        characters into the following chunk, preventing orphan micro-chunks.
        """
        if not texts:
            return texts

        merged: List[str] = []
        pending = ""

        for t in texts:
            combined = (pending + "\n\n" + t).strip() if pending else t
            if len(combined) < self.min_chunk_size:
                pending = combined            # keep accumulating
            else:
                merged.append(combined)
                pending = ""

        if pending:                           # tail leftover
            if merged:
                merged[-1] = (merged[-1] + "\n\n" + pending).strip()
            else:
                merged.append(pending)

        return merged

    def _finalise(
        self,
        raw_chunks: List[str],
        original_text: str,
        source_id: str,
        base_metadata: Dict[str, Any],
        chunker_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of raw text strings into fully-formed chunk dicts.

        Steps
        -----
        1. Apply overlap (prepend tail of previous chunk).
        2. Apply size ceiling (break oversized chunks).
        3. Merge tiny chunks.
        4. Build chunk dicts with correct char_start / char_end offsets.
        """
        # Step 1 — overlap
        if self.overlap_sentences > 0:
            raw_chunks = _add_overlap(raw_chunks, self.overlap_sentences)

        # Step 2 — size ceiling (flatten any chunks that are still too large)
        sized: List[str] = []
        for chunk in raw_chunks:
            sized.extend(self._enforce_size_ceiling(chunk))

        # Step 3 — merge tiny chunks
        sized = self._merge_tiny_chunks(sized)

        # Step 4 — build dicts, computing char offsets by searching the original
        result: List[Dict[str, Any]] = []
        search_from = 0                      # advance cursor to avoid false matches
        for idx, text in enumerate(sized):
            # Find the first occurrence of the chunk text in the original,
            # starting from where the previous chunk ended.
            # We use only the first 60 chars as a search key to handle cases
            # where overlap changed the beginning of the chunk.
            key = text[:60].strip()
            pos = original_text.find(key, search_from)
            if pos == -1:
                # Overlap text prepended — search without the prepended portion
                # by finding the first non-overlap sentence
                sents = _split_sentences(text)
                for s in sents:
                    p = original_text.find(s[:40].strip(), search_from)
                    if p != -1:
                        pos = p
                        break
            if pos == -1:
                pos = search_from            # fallback: use cursor position

            char_start = pos
            char_end   = min(pos + len(text), len(original_text))
            search_from = max(search_from, char_end - 50)   # slight backtrack for safety

            meta = {**base_metadata, "chunker": chunker_name}
            result.append(
                _build_chunk(text, source_id, idx, char_start, char_end, meta)
            )

        return result


#  CHUNKER IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

# FixedSizeChunker

class FixedSizeChunker(AbstractChunker):
    """
    Strategy: Split text into fixed-character-count windows with optional
    character-level overlap.

    Why it exists
    -------------
    The simplest possible baseline.  Fast and deterministic, but produces
    chunks that may cut mid-sentence.  Useful for quick prototyping or as a
    fallback when text has no structural markers.

    Parameters
    ----------
    chunk_size    : Target size in characters for each chunk window.
    char_overlap  : Number of characters from the end of chunk N to prepend
                    to chunk N+1  (character-level, not sentence-level).
    """

    def __init__(
        self,
        chunk_size: int = 800,
        char_overlap: int = 100,
        min_chunk_size: int = 100,
    ) -> None:
        # We pass overlap_sentences=0 because overlap is handled at char level here
        super().__init__(
            max_chunk_size=chunk_size,
            overlap_sentences=0,
            min_chunk_size=min_chunk_size,
        )
        self.chunk_size   = chunk_size
        self.char_overlap = max(0, char_overlap)

    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Slide a window of `chunk_size` characters across *text* with a step
        of (chunk_size - char_overlap).
        """
        if not text.strip():
            return []

        base_meta = metadata or {}
        step       = max(1, self.chunk_size - self.char_overlap)
        raw_chunks: List[str] = []
        positions:  List[Tuple[int, int]] = []     # (char_start, char_end)

        # Slide window across text
        start = 0
        while start < len(text):
            end  = min(start + self.chunk_size, len(text))
            fragment = text[start:end].strip()
            if fragment:
                raw_chunks.append(fragment)
                positions.append((start, end))
            start += step

        # Build dicts (bypass _finalise because we have exact char positions)
        result: List[Dict[str, Any]] = []
        for idx, (frag, (cs, ce)) in enumerate(zip(raw_chunks, positions)):
            meta = {**base_meta, "chunker": "FixedSizeChunker"}
            result.append(_build_chunk(frag, source_id, idx, cs, ce, meta))

        return result


# SentenceChunker

class SentenceChunker(AbstractChunker):
    """
    Strategy: Group a fixed number of sentences per chunk, with sentence-level
    overlap between adjacent chunks.

    Why it exists
    -------------
    Guarantees that no chunk cuts a sentence in half.  Better than FixedSize
    for tasks that reason at the sentence level (NER, relation extraction).

    Parameters
    ----------
    sentences_per_chunk : Target number of sentences to include in each chunk.
    overlap_sentences   : Sentences shared between consecutive chunks.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1,
        max_chunk_size: int = 1200,
        min_chunk_size: int = 80,
    ) -> None:
        super().__init__(
            max_chunk_size=max_chunk_size,
            overlap_sentences=overlap_sentences,
            min_chunk_size=min_chunk_size,
        )
        self.sentences_per_chunk = max(1, sentences_per_chunk)

    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        1. Split *text* into sentences with _split_sentences().
        2. Group every `sentences_per_chunk` sentences.
        3. Apply overlap, size ceiling, and merge-tiny via _finalise().
        """
        if not text.strip():
            return []

        base_meta = metadata or {}
        sentences  = _split_sentences(text)

        # Group sentences into windows
        raw_chunks: List[str] = []
        step = max(1, self.sentences_per_chunk - self.overlap_sentences)

        i = 0
        while i < len(sentences):
            window = sentences[i: i + self.sentences_per_chunk]
            chunk_text = " ".join(window).strip()
            if chunk_text:
                raw_chunks.append(chunk_text)
            # Do not advance by overlap_sentences here; _finalise handles overlap
            i += self.sentences_per_chunk   # move by full window (overlap added later)

        return self._finalise(raw_chunks, text, source_id, base_meta, "SentenceChunker")


# ParagraphChunker

class ParagraphChunker(AbstractChunker):
    """
    Strategy: Use blank-line boundaries (paragraphs) as natural chunk delimiters,
    then merge small paragraphs and split oversized ones.

    Why it exists
    -------------
    Human authors write paragraphs as self-contained ideas.  Paragraph
    boundaries are the most reliable semantic unit without any ML inference.
    Works extremely well for PDFs and DOCX files.

    Parameters
    ----------
    max_paragraphs_per_chunk : Merge up to this many consecutive paragraphs
                               before forcing a new chunk.
    """

    def __init__(
        self,
        max_paragraphs_per_chunk: int = 3,
        max_chunk_size: int = 1200,
        overlap_sentences: int = 1,
        min_chunk_size: int = 100,
    ) -> None:
        super().__init__(
            max_chunk_size=max_chunk_size,
            overlap_sentences=overlap_sentences,
            min_chunk_size=min_chunk_size,
        )
        self.max_paragraphs_per_chunk = max(1, max_paragraphs_per_chunk)

    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        1. Split *text* into paragraphs.
        2. Accumulate up to `max_paragraphs_per_chunk` paragraphs while the
           combined length stays below `max_chunk_size`.
        3. Pass to _finalise() for overlap, size ceiling, merge-tiny.
        """
        if not text.strip():
            return []

        base_meta  = metadata or {}
        paragraphs = _split_paragraphs(text)

        raw_chunks: List[str] = []
        current_paras: List[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            will_overflow = (
                len(current_paras) >= self.max_paragraphs_per_chunk
                or (current_len + para_len) > self.max_chunk_size
            )

            if will_overflow and current_paras:
                raw_chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_len   = 0

            current_paras.append(para)
            current_len += para_len

        # Flush remaining paragraphs
        if current_paras:
            raw_chunks.append("\n\n".join(current_paras))

        return self._finalise(raw_chunks, text, source_id, base_meta, "ParagraphChunker")


# RecursiveChunker

class RecursiveChunker(AbstractChunker):
    """
    Strategy: Try successively finer-grained separators until each piece fits
    within `max_chunk_size`.  Mimics LangChain's RecursiveCharacterTextSplitter
    but built from scratch.

    Separator hierarchy (tried in order)
    -------------------------------------
    1.  Double newline (paragraph)
    2.  Single newline (line break)
    3.  Period + space (sentence end)
    4.  Comma + space (clause boundary)
    5.  Space (word boundary)
    6.  Empty string (character-level last resort)

    Why it exists
    -------------
    Works on ANY text format.  It will always produce chunks ≤ max_chunk_size
    while trying to keep the most meaningful text unit intact.
    """

    # Hierarchy of separators to try, from coarsest to finest
    _SEPARATORS: List[str] = ["\n\n", "\n", ". ", ", ", " ", ""]

    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Entry point: recursively split *text* using the separator hierarchy."""
        if not text.strip():
            return []

        base_meta  = metadata or {}
        raw_chunks = self._recursive_split(text, self._SEPARATORS)

        return self._finalise(raw_chunks, text, source_id, base_meta, "RecursiveChunker")

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        1. If *text* fits in max_chunk_size → return it as-is.
        2. Try the first separator in the list:
           a. Split the text on it.
           b. Merge consecutive pieces that fit together.
           c. Any piece that still exceeds max_chunk_size is recursively split
              with the next separator in the list.
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        if not separators:
            # Last resort: hard character split
            return [text[i: i + self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]

        sep = separators[0]
        remaining = separators[1:]

        # Split on the current separator
        if sep:
            pieces = text.split(sep)
            # Re-attach the separator to each piece (except the last)
            pieces_joined: List[str] = []
            for p in pieces:
                pieces_joined.append(p)
        else:
            # Empty separator = character split
            pieces_joined = list(text)

        # Greedily merge pieces until max_chunk_size is reached
        merged_chunks: List[str] = []
        current = ""

        for piece in pieces_joined:
            if not piece.strip():
                continue
            connector = sep if sep else ""
            candidate = (current + connector + piece).strip() if current else piece.strip()

            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    merged_chunks.append(current)
                # This single piece might still be too large → recurse
                if len(piece) > self.max_chunk_size:
                    merged_chunks.extend(self._recursive_split(piece, remaining))
                    current = ""
                else:
                    current = piece.strip()

        if current:
            merged_chunks.append(current)

        return merged_chunks if merged_chunks else [text[: self.max_chunk_size]]


# SlidingWindowChunker

class SlidingWindowChunker(AbstractChunker):
    """
    Strategy: Sentence-level sliding window — at every step of `step_sentences`
    sentences, emit a chunk of `window_sentences` sentences.

    Why it exists
    -------------
    Maximises context coverage by ensuring every sentence appears in multiple
    chunks.  Useful when recall matters more than efficiency (e.g. dense
    question-answering).  Note: produces more chunks than other methods.

    Parameters
    ----------
    window_sentences : Number of sentences in each chunk window.
    step_sentences   : How many sentences to advance at each step.
                       step < window → overlapping windows.
    """

    def __init__(
        self,
        window_sentences: int = 6,
        step_sentences: int = 3,
        max_chunk_size: int = 1400,
        min_chunk_size: int = 80,
    ) -> None:
        super().__init__(
            max_chunk_size=max_chunk_size,
            overlap_sentences=0,          # overlap handled via step
            min_chunk_size=min_chunk_size,
        )
        self.window_sentences = max(1, window_sentences)
        self.step_sentences   = max(1, step_sentences)

    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        1. Split *text* into sentences.
        2. Slide a window of `window_sentences` across them with a step of
           `step_sentences`.
        3. Pass to _finalise() for size ceiling and merge-tiny.
        """
        if not text.strip():
            return []

        base_meta = metadata or {}
        sentences  = _split_sentences(text)
        n          = len(sentences)

        raw_chunks: List[str] = []
        i = 0
        while i < n:
            window = sentences[i: i + self.window_sentences]
            chunk_text = " ".join(window).strip()
            if chunk_text:
                raw_chunks.append(chunk_text)
            i += self.step_sentences
            # Stop when next step would start past the last sentence
            if i >= n:
                break

        # Ensure final sentences are captured if step skips over them
        if n > 0 and raw_chunks:
            last_start = ((n - 1) // self.step_sentences) * self.step_sentences
            if last_start < n:
                tail = " ".join(sentences[last_start:]).strip()
                if tail and tail != raw_chunks[-1]:
                    raw_chunks.append(tail)

        return self._finalise(raw_chunks, text, source_id, base_meta, "SlidingWindowChunker")


# SemanticChunker  (TF-IDF cosine similarity, zero external deps)

class SemanticChunker(AbstractChunker):
    """
    Strategy: Detect topic shifts by measuring the cosine similarity between
    consecutive sentence-window vectors (TF-IDF, built from scratch).  A new
    chunk starts wherever the similarity drops below `similarity_threshold`.

    How the TF-IDF is built (no sklearn/numpy)
    -------------------------------------------
    1. Tokenise each sentence: lowercase, strip punctuation, split on whitespace.
    2. Build a corpus-level vocabulary.
    3. For each sentence compute TF (term frequency) and IDF (inverse document
       frequency across the sentence corpus).
    4. Represent each sentence as a sparse dict {term: tf*idf}.
    5. Compute cosine similarity between adjacent windows of `buffer_size`
       sentences using pure Python dot products and square-root norms.

    Breakpoint detection
    --------------------
    1. Compute pairwise similarities between consecutive windows.
    2. Compute a dynamic threshold: percentile_threshold-th percentile of all
       similarities (so boundaries adapt to the document's topic density).
    3. Mark a boundary wherever similarity falls below that threshold.

    Parameters
    ----------
    buffer_size           : Number of sentences on each side of a point to
                            include in its representative vector.  Larger →
                            smoother, coarser chunking.
    similarity_threshold  : Fixed fallback threshold (0–1).  When percentile
                            detection is active, this acts as a minimum floor.
    percentile_threshold  : Dynamic: find the Nth percentile of all similarities
                            and use that as the breakpoint threshold.
                            Set to None to use `similarity_threshold` only.
    """

    def __init__(
        self,
        max_chunk_size: int = 1200,
        overlap_sentences: int = 2,
        min_chunk_size: int = 120,
        buffer_size: int = 2,
        similarity_threshold: float = 0.20,
        percentile_threshold: Optional[float] = 25.0,
    ) -> None:
        super().__init__(
            max_chunk_size=max_chunk_size,
            overlap_sentences=overlap_sentences,
            min_chunk_size=min_chunk_size,
        )
        self.buffer_size           = max(1, buffer_size)
        self.similarity_threshold  = similarity_threshold
        self.percentile_threshold  = percentile_threshold

    # Public entry point
    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline:
        sentences → TF-IDF vectors → windowed cosine similarities →
        breakpoint detection → group into chunks → _finalise().
        """
        if not text.strip():
            return []

        base_meta = metadata or {}
        sentences  = _split_sentences(text)

        # With very few sentences, fall back to a single chunk
        if len(sentences) <= self.buffer_size * 2 + 1:
            return self._finalise(
                [text.strip()], text, source_id, base_meta, "SemanticChunker"
            )

        # Build TF-IDF vectors for every sentence
        tfidf_vecs = self._build_tfidf_vectors(sentences)

        # Compute similarity between adjacent buffered windows
        similarities = self._compute_window_similarities(tfidf_vecs)

        # Find breakpoint positions
        breakpoints = self._find_breakpoints(similarities)

        # Group sentences into chunks
        raw_chunks = self._group_sentences(sentences, breakpoints)

        return self._finalise(raw_chunks, text, source_id, base_meta, "SemanticChunker")

    # TF-IDF machinery (pure Python)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Lowercase, remove punctuation, split on whitespace.
        Returns a list of tokens (words).
        """
        text = text.lower()
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens (1-char noise) and pure numbers
        return [t for t in tokens if len(t) > 1 and not t.isdigit()]

    def _build_tfidf_vectors(
        self, sentences: List[str]
    ) -> List[Dict[str, float]]:
        """
        Build a TF-IDF vector (as a sparse dict) for each sentence.

        Steps
        -----
        1.  Tokenise every sentence.
        2.  Compute TF = count(term, sentence) / len(sentence_tokens).
        3.  Compute IDF = log(N / (1 + df(term))) where N = num sentences,
            df = number of sentences containing the term.
        4.  Vector[term] = TF * IDF.
        """
        N = len(sentences)
        tokenised: List[List[str]] = [self._tokenise(s) for s in sentences]

        # Document frequency: how many sentences contain each term
        df: Dict[str, int] = defaultdict(int)
        for tokens in tokenised:
            for term in set(tokens):            # set → count each term once per sentence
                df[term] += 1

        # Pre-compute IDF for every term
        idf: Dict[str, float] = {
            term: math.log(N / (1.0 + count))
            for term, count in df.items()
        }

        # Build TF-IDF vector per sentence
        vectors: List[Dict[str, float]] = []
        for tokens in tokenised:
            if not tokens:
                vectors.append({})
                continue
            tf: Dict[str, float] = {}
            total = len(tokens)
            for term in tokens:
                tf[term] = tf.get(term, 0.0) + 1.0 / total   # normalised TF

            vec: Dict[str, float] = {
                term: tf_val * idf.get(term, 0.0)
                for term, tf_val in tf.items()
            }
            vectors.append(vec)

        return vectors

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        """
        Cosine similarity between two sparse TF-IDF vectors (dicts).

        Formula: (a · b) / (||a|| * ||b||)

        Returns 0.0 if either vector is the zero vector (avoids division-by-zero).
        """
        if not a or not b:
            return 0.0

        # Dot product: iterate over the smaller dict for efficiency
        if len(a) > len(b):
            a, b = b, a

        dot   = sum(a_val * b.get(term, 0.0) for term, a_val in a.items())
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _window_vector(
        self, vectors: List[Dict[str, float]], center: int
    ) -> Dict[str, float]:
        """
        Aggregate the TF-IDF vectors of sentences in the window
        [center - buffer_size, center + buffer_size] into a single vector
        by summing term weights (then re-normalising to unit length).

        This smooths out noise in individual sentence vectors.
        """
        lo  = max(0, center - self.buffer_size)
        hi  = min(len(vectors), center + self.buffer_size + 1)

        combined: Dict[str, float] = defaultdict(float)
        for vec in vectors[lo:hi]:
            for term, weight in vec.items():
                combined[term] += weight

        # L2-normalise so length of window doesn't bias similarity
        norm = math.sqrt(sum(v * v for v in combined.values()))
        if norm > 0:
            combined = {t: v / norm for t, v in combined.items()}

        return dict(combined)

    def _compute_window_similarities(
        self, vectors: List[Dict[str, float]]
    ) -> List[float]:
        """
        For each pair of adjacent sentences (i, i+1), compute the cosine
        similarity between their buffered window vectors.

        Returns a list of length len(vectors) - 1.
        """
        similarities: List[float] = []
        for i in range(len(vectors) - 1):
            win_i   = self._window_vector(vectors, i)
            win_i1  = self._window_vector(vectors, i + 1)
            sim     = self._cosine_similarity(win_i, win_i1)
            similarities.append(sim)
        return similarities

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        Return the list of sentence indices after which a new chunk starts.

        Algorithm
        ---------
        1. If `percentile_threshold` is set, compute that percentile of all
           similarity values.  Use it as the dynamic threshold (but never
           below `similarity_threshold`).
        2. Mark a breakpoint at every index i where similarities[i] <  threshold.

        The percentile approach adapts to the document: in a highly homogeneous
        text the threshold will be higher (making more cuts); in a varied text
        it will find natural topic boundaries.
        """
        if not similarities:
            return []

        # Compute dynamic threshold via percentile
        threshold = self.similarity_threshold
        if self.percentile_threshold is not None:
            sorted_sims = sorted(similarities)
            n = len(sorted_sims)
            # Linear interpolation percentile
            rank  = (self.percentile_threshold / 100.0) * (n - 1)
            lo    = int(rank)
            hi    = min(lo + 1, n - 1)
            frac  = rank - lo
            dynamic = sorted_sims[lo] + frac * (sorted_sims[hi] - sorted_sims[lo])
            # Use the more conservative (lower) of the two so we don't over-cut
            threshold = min(threshold, dynamic)

        # Collect breakpoints
        breakpoints: List[int] = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # +1: breakpoint is *before* sentence i+1

        return breakpoints

    @staticmethod
    def _group_sentences(
        sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        """
        Use the detected breakpoints to group sentences into chunks.

        breakpoints is a sorted list of sentence indices where a new chunk
        starts.  E.g. [0, 4, 9] means:
          chunk 0 → sentences 0-3
          chunk 1 → sentences 4-8
          chunk 2 → sentences 9-end
        """
        if not sentences:
            return []

        # Ensure 0 is the first boundary and len(sentences) is the last
        boundaries = sorted(set([0] + breakpoints + [len(sentences)]))

        chunks: List[str] = []
        for start, end in zip(boundaries, boundaries[1:]):
            group = sentences[start:end]
            text  = " ".join(group).strip()
            if text:
                chunks.append(text)

        return chunks


# HybridChunker "DEFAULT" 

class HybridChunker(AbstractChunker):
    """
    Strategy: Structural segmentation → paragraph grouping → sentence-aware
    size enforcement → sentence-level overlap.

    Why it is the best default
    --------------------------
    1. **Structure-aware**: Recognises the markers injected by every extractor
       in this project (PDF pages, PPTX slides, DOCX headings, speaker notes).
       Each structural section becomes a natural graph node candidate.

    2. **No ML inference**: Runs in milliseconds on any hardware.

    3. **Semantic coherence**: Paragraphs (written by humans) already represent
       one idea.  Grouping 1-3 of them produces chunks that are ideal for
       entity- and relation-extraction LLMs.

    4. **Size-safe**: Never produces a chunk that exceeds max_chunk_size,
       which is critical for LLM context windows.

    5. **Format-agnostic**: Works on PDFs, DOCX, PPTX, audio transcripts (which
       have no structure, so it falls back to sentence grouping), and URLs.

    Processing pipeline
    -------------------
    Phase 1 — Structural split
        Split the document on structural markers (page/slide/heading lines).
        Each section between markers becomes a "section block".

    Phase 2 — Paragraph grouping within each section
        Further split each section block into paragraphs.
        Greedily merge paragraphs until max_chunk_size would be exceeded or
        max_paragraphs_per_chunk is reached.

    Phase 3 — Size enforcement
        Any chunk still > max_chunk_size is split at sentence boundaries by
        _enforce_size_ceiling() (inherited from AbstractChunker).

    Phase 4 — Merge tiny chunks
        Chunks shorter than min_chunk_size are merged with the next one.

    Phase 5 — Overlap
        Each chunk (except the first) gets the last `overlap_sentences`
        sentences of the preceding chunk prepended.

    Parameters
    ----------
    max_chunk_size          : Hard character ceiling per chunk.
    overlap_sentences       : Sentences of overlap between consecutive chunks.
    min_chunk_size          : Minimum characters; shorter chunks get merged.
    max_paragraphs_per_chunk: Maximum paragraphs to group before forcing a cut.
    keep_structural_markers : If True, include the marker line (e.g. "SLIDE 3")
                              as the first line of its chunk — useful for
                              provenance tracking in Graph RAG.
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,
        overlap_sentences: int = 2,
        min_chunk_size: int = 100,
        max_paragraphs_per_chunk: int = 3,
        keep_structural_markers: bool = True,
    ) -> None:
        super().__init__(
            max_chunk_size=max_chunk_size,
            overlap_sentences=overlap_sentences,
            min_chunk_size=min_chunk_size,
        )
        self.max_paragraphs_per_chunk = max(1, max_paragraphs_per_chunk)
        self.keep_structural_markers  = keep_structural_markers

    def chunk(
        self,
        text: str,
        source_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full HybridChunker pipeline — see class docstring.
        """
        if not text.strip():
            return []

        base_meta = metadata or {}

        # Phase 1: Structural split
        section_blocks = self._split_on_structural_markers(text)

        # Phase 2 + 3 + 4: Paragraph grouping + size + merge 
        raw_chunks: List[str] = []
        for marker, body in section_blocks:
            # Prepend marker to body if requested
            prefix = (marker + "\n") if (marker and self.keep_structural_markers) else ""
            paragraphs = _split_paragraphs(body) if body.strip() else []

            if not paragraphs:
                if prefix.strip():
                    raw_chunks.append(prefix.strip())
                continue

            current_paras: List[str] = []
            current_len = len(prefix)

            for para in paragraphs:
                para_len = len(para)
                will_exceed = (
                    len(current_paras) >= self.max_paragraphs_per_chunk
                    or (current_len + para_len) > self.max_chunk_size
                )

                if will_exceed and current_paras:
                    # Flush — prepend marker only to the first chunk of this section
                    chunk_text = prefix + "\n\n".join(current_paras)
                    raw_chunks.append(chunk_text.strip())
                    prefix = ""            # marker only appears in first chunk
                    current_paras = []
                    current_len   = 0

                current_paras.append(para)
                current_len += para_len

            # Flush remaining paragraphs
            if current_paras:
                chunk_text = prefix + "\n\n".join(current_paras)
                raw_chunks.append(chunk_text.strip())

        return self._finalise(raw_chunks, text, source_id, base_meta, "HybridChunker")

    def _split_on_structural_markers(
        self, text: str
    ) -> List[Tuple[str, str]]:
        """
        Walk through *text* line by line.  When a structural marker is found,
        close the current section and start a new one.

        Returns a list of (marker_line, body_text) tuples.
        The very first segment has an empty marker.
        """
        lines  = text.splitlines(keepends=True)
        sections: List[Tuple[str, str]] = []

        current_marker = ""
        current_body_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if _is_structural_marker(stripped):
                # Save current section
                body = "".join(current_body_lines).strip()
                if body or current_marker:
                    sections.append((current_marker, body))
                current_marker     = stripped
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Final section
        body = "".join(current_body_lines).strip()
        if body or current_marker:
            sections.append((current_marker, body))

        # If no structural markers were found, return the whole text as one section
        if not sections:
            sections = [("", text)]

        return sections


# CONVENIENCE FUNCTION  (default = HybridChunker)

def chunk_text(
    text: str,
    source_id: str = "doc",
    method: str = "hybrid",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    One-line entry point.  Selects the chunker by name and runs it.

    Parameters
    ----------
    text      : The text to chunk.
    source_id : Identifier used in chunk IDs (e.g. the resource_name from
                an extractor result dict).
    method    : One of:
                  "hybrid"    ← default, recommended for Graph RAG
                  "semantic"
                  "recursive"
                  "paragraph"
                  "sentence"
                  "sliding"
                  "fixed"
    metadata  : Arbitrary key-value pairs attached to every chunk's metadata.
    **kwargs  : Passed directly to the chosen chunker's constructor.

    Returns
    -------
    List of chunk dicts (see module docstring for keys).
    """
    method = method.lower().strip()

    CHUNKER_MAP: Dict[str, type] = {
        "hybrid":    HybridChunker,
        "semantic":  SemanticChunker,
        "recursive": RecursiveChunker,
        "paragraph": ParagraphChunker,
        "sentence":  SentenceChunker,
        "sliding":   SlidingWindowChunker,
        "fixed":     FixedSizeChunker,
    }

    chunker_cls = CHUNKER_MAP.get(method)
    if chunker_cls is None:
        raise ValueError(
            f"Unknown chunking method '{method}'. "
            f"Choose from: {list(CHUNKER_MAP.keys())}"
        )

    chunker = chunker_cls(**kwargs)
    return chunker.chunk(text, source_id=source_id, metadata=metadata or {})


# CHUNK ANALYSER  (optional diagnostic helper)

class ChunkAnalyser:
    """
    Compute descriptive statistics over a list of chunks returned by any chunker.
    Useful during development or benchmarking.

    Usage
    -----
        stats = ChunkAnalyser.analyse(chunks)
        ChunkAnalyser.print_report(stats)
    """

    @staticmethod
    def analyse(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Return a dict of statistics about the chunk list.

        Keys
        ----
        count           : Total number of chunks.
        total_chars     : Sum of character lengths across all chunks.
        total_tokens    : Sum of estimated token counts.
        avg_chars       : Mean character length.
        min_chars       : Minimum character length.
        max_chars       : Maximum character length.
        avg_tokens      : Mean token count.
        empty_chunks    : Number of chunks with no text.
        chunker         : Name of the chunker (from metadata if available).
        """
        if not chunks:
            return {"count": 0}

        lengths = [len(c["text"]) for c in chunks]
        tokens  = [c["token_count"] for c in chunks]

        return {
            "count":        len(chunks),
            "total_chars":  sum(lengths),
            "total_tokens": sum(tokens),
            "avg_chars":    round(sum(lengths) / len(lengths), 1),
            "min_chars":    min(lengths),
            "max_chars":    max(lengths),
            "avg_tokens":   round(sum(tokens) / len(tokens), 1),
            "empty_chunks": sum(1 for t in lengths if t == 0),
            "chunker":      chunks[0]["metadata"].get("chunker", "unknown"),
        }

    @staticmethod
    def print_report(stats: Dict[str, Any]) -> None:
        """Pretty-print the statistics dict to stdout."""
        print("\n" + "═" * 60)
        print(f"  Chunk Analysis Report — {stats.get('chunker', '?')}")
        print("═" * 60)
        for key, val in stats.items():
            print(f"  {key:<20s}: {val}")
        print("═" * 60 + "\n")