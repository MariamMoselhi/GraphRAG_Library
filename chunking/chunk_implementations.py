from __future__ import annotations

import re
from typing import List, Optional

from chunk_base import BaseChunker, Chunk



class FixedSizeChunker(BaseChunker):
    """
    Splits text into chunks of exactly *chunk_size* characters with an
    optional *overlap* of characters shared between consecutive chunks.

    Why useful in GraphRAG
    ----------------------
    Guarantees uniform token budgets when feeding chunks to an LLM.
    Overlap ensures that context spanning a chunk boundary is not lost.

    Parameters
    ----------
    chunk_size : int  — target character count per chunk (default 512).
    overlap    : int  — characters repeated at the start of the next chunk
                        (default 50). Must be < chunk_size.
    """

    chunker_type = "fixed_size"

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        super().__init__()
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        """
        Slice the cleaned text with a sliding character window.

        Algorithm
        ---------
        • Advance pointer by (chunk_size - overlap) each step.
        • Each window is [pos : pos + chunk_size].
        • Last window may be shorter than chunk_size — that is fine.
        """
        text = self.clean(text)
        if not text:
            return []

        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap  # how far we move each step
        pos = 0                                # current position in text
        chunk_id = 0

        while pos < len(text):
            # Slice out a window of chunk_size characters
            window = text[pos : pos + self.chunk_size]

            chunk = Chunk(
                text=window,
                chunk_id=chunk_id,
                start_char=pos,
                end_char=pos + len(window),
                metadata={
                    "chunker"   : self.chunker_type,
                    "chunk_size": self.chunk_size,
                    "overlap"   : self.overlap,
                },
            )
            chunks.append(chunk)

            pos += step        # slide forward
            chunk_id += 1

        return chunks


class SentenceChunker(BaseChunker):
    """
    Groups *sentences_per_chunk* sentences into one chunk, with an overlap of
    *overlap_sentences* sentences shared between consecutive chunks.

    Why useful in GraphRAG
    ----------------------
    Sentence-aligned chunks preserve grammatical units, which produces
    better embeddings and cleaner graph-node text.

    Parameters
    ----------
    sentences_per_chunk : int — how many sentences per chunk (default 5).
    overlap_sentences   : int — how many trailing sentences from the previous
                               chunk are prepended to the next (default 1).
    """

    chunker_type = "sentence"

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        super().__init__()
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError("overlap_sentences must be < sentences_per_chunk")
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into sentences, then group them into chunks.

        Algorithm
        ---------
        • Tokenise into sentence list using BaseChunker._split_sentences().
        • Slide a window of size *sentences_per_chunk* by
          (sentences_per_chunk - overlap_sentences) each step.
        • Join sentences within each window with a single space.
        """
        text = self.clean(text)
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        step = self.sentences_per_chunk - self.overlap_sentences
        search_start = 0   # character offset hint for _make_chunk
        chunk_id = 0
        pos = 0            # sentence index

        while pos < len(sentences):
            # Take a window of sentences
            window_sents = sentences[pos : pos + self.sentences_per_chunk]
            window_text  = " ".join(window_sents)

            chunk = self._make_chunk(
                text=window_text,
                chunk_id=chunk_id,
                source_text=text,
                search_start=search_start,
                extra_meta={
                    "sentences_per_chunk": self.sentences_per_chunk,
                    "overlap_sentences"  : self.overlap_sentences,
                    "sentence_count"     : len(window_sents),
                },
            )
            chunks.append(chunk)

            search_start = chunk.start_char   # next search starts here
            pos += step
            chunk_id += 1

        return chunks


class ParagraphChunker(BaseChunker):
    """
    Splits text on blank lines (\\n\\n), treating each block as one chunk.
    Short paragraphs below *min_chars* are merged with the next paragraph.

    Why useful in GraphRAG
    ----------------------
    Paragraphs are the natural semantic units in most documents (lecture
    notes, papers, articles). Graph edges drawn between paragraphs tend to
    capture coherent topic shifts.

    Parameters
    ----------
    min_chars : int — paragraphs shorter than this are merged with the next
                      one (default 100). Prevents tiny orphan chunks.
    """

    chunker_type = "paragraph"

    def __init__(self, min_chars: int = 100):
        super().__init__()
        self.min_chars = min_chars

    def chunk(self, text: str) -> List[Chunk]:
        """
        Split on double newlines, then merge short paragraphs.

        Algorithm
        ---------
        1. Split cleaned text on \\n\\n.
        2. Strip each block; skip empty blocks.
        3. If a block is shorter than min_chars, append it to an accumulator.
        4. When the accumulator reaches min_chars, flush it as a chunk.
        5. Any remaining accumulator content forms the final chunk.
        """
        text = self.clean(text)
        if not text:
            return []

        # Split on one or more blank lines
        raw_paragraphs = re.split(r"\n\s*\n", text)

        # Strip and filter empty blocks
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        # Merge short paragraphs
        merged: List[str] = []
        accumulator = ""

        for para in paragraphs:
            if accumulator:
                accumulator += "\n\n" + para
            else:
                accumulator = para

            if len(accumulator) >= self.min_chars:
                merged.append(accumulator)
                accumulator = ""

        # Flush any remaining text
        if accumulator:
            if merged:
                # Attach orphan to the last chunk rather than creating a
                # tiny chunk at the end
                merged[-1] += "\n\n" + accumulator
            else:
                merged.append(accumulator)

        # Build Chunk objects
        chunks: List[Chunk] = []
        search_start = 0

        for chunk_id, para_text in enumerate(merged):
            chunk = self._make_chunk(
                text=para_text,
                chunk_id=chunk_id,
                source_text=text,
                search_start=search_start,
                extra_meta={"min_chars": self.min_chars},
            )
            chunks.append(chunk)
            search_start = chunk.end_char

        return chunks



class RecursiveChunker(BaseChunker):
    """
    Tries a hierarchy of delimiters to split text.  If the resulting pieces
    are still too large, it recurses with the next delimiter until every
    chunk fits within *max_chunk_size* characters.

    Default delimiter hierarchy: ["\\n\\n", "\\n", ". ", " "]

    Why useful in GraphRAG
    ----------------------
    Handles heterogeneous documents (slides mixed with prose, code blocks,
    tables) where no single delimiter works uniformly.  The recursion ensures
    that every chunk is bounded in size — critical for LLM context windows.

    Parameters
    ----------
    max_chunk_size : int        — hard upper bound on chunk size in chars
                                  (default 512).
    delimiters     : List[str]  — ordered list of split tokens tried in
                                  sequence (outermost → innermost).
    """

    chunker_type = "recursive"

    # Default delimiter order: paragraph → line → sentence → word boundary
    DEFAULT_DELIMITERS = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        max_chunk_size: int = 512,
        delimiters: Optional[List[str]] = None,
    ):
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.delimiters = delimiters if delimiters is not None else self.DEFAULT_DELIMITERS

    def chunk(self, text: str) -> List[Chunk]:
        """
        Public entry point.  Cleans text then recursively splits it.
        """
        text = self.clean(text)
        if not text:
            return []

        # Recursively gather raw text pieces
        pieces = self._split(text, delimiter_index=0)

        # Build Chunk objects
        chunks: List[Chunk] = []
        search_start = 0

        for chunk_id, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunk = self._make_chunk(
                text=piece,
                chunk_id=chunk_id,
                source_text=text,
                search_start=search_start,
                extra_meta={
                    "max_chunk_size": self.max_chunk_size,
                    "delimiters"    : self.delimiters,
                },
            )
            chunks.append(chunk)
            search_start = chunk.end_char

        # Re-number after filtering empties
        for i, c in enumerate(chunks):
            c.chunk_id = i

        return chunks

    def _split(self, text: str, delimiter_index: int) -> List[str]:
        """
        Recursively split *text* using delimiters[delimiter_index].

        Base cases
        ----------
        • text is short enough  → return [text].
        • we have run out of delimiters → return [text] (last resort).

        Recursive case
        --------------
        • Split on current delimiter.
        • For each piece that is still too large, recurse with next delimiter.
        • Re-join pieces that are too small into the previous piece (greedy
          packing) to avoid tiny orphan chunks.
        """
        # Base case 1: already fits
        if len(text) <= self.max_chunk_size:
            return [text]

        # Base case 2: no more delimiters
        if delimiter_index >= len(self.delimiters):
            return [text]

        delimiter = self.delimiters[delimiter_index]

        # Split on current delimiter, keep the delimiter at the end of each
        # piece so the text can be reconstructed (except the last piece)
        raw_parts = text.split(delimiter)

        # Re-attach the delimiter to all parts except the last
        parts = []
        for i, part in enumerate(raw_parts):
            if i < len(raw_parts) - 1:
                parts.append(part + delimiter)
            else:
                parts.append(part)

        # Greedy packing: merge small parts into a running buffer
        packed: List[str] = []
        buffer = ""

        for part in parts:
            candidate = buffer + part
            if len(candidate) <= self.max_chunk_size:
                # Still fits — accumulate
                buffer = candidate
            else:
                # Current candidate is too big
                if buffer:
                    # Flush the buffer as a good-sized chunk
                    packed.append(buffer)
                if len(part) > self.max_chunk_size:
                    # This single part is already oversized — recurse deeper
                    sub_pieces = self._split(part, delimiter_index + 1)
                    packed.extend(sub_pieces)
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            packed.append(buffer)

        return packed



class SlidingWindowChunker(BaseChunker):
    """
    Creates overlapping chunks using a sliding window of *window_size*
    sentences that advances *step_size* sentences at a time.

    Why useful in GraphRAG
    ----------------------
    The heavy overlap means neighbouring chunks share substantial context,
    which helps the graph-builder create dense edges between related nodes
    and improves retrieval recall for questions that span chunk boundaries.

    Parameters
    ----------
    window_size : int — number of sentences in each chunk (default 8).
    step_size   : int — how many sentences the window advances per step
                        (default 4).  step_size < window_size creates overlap.
    """

    chunker_type = "sliding_window"

    def __init__(self, window_size: int = 8, step_size: int = 4):
        super().__init__()
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if step_size > window_size:
            raise ValueError("step_size should be <= window_size to get overlap")
        self.window_size = window_size
        self.step_size = step_size

    def chunk(self, text: str) -> List[Chunk]:
        """
        Tokenise into sentences then slide a window over them.

        Algorithm
        ---------
        • sentences[i : i + window_size] → one chunk.
        • Advance i by step_size.
        • Continue until i >= len(sentences).
        """
        text = self.clean(text)
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        search_start = 0
        chunk_id = 0
        i = 0

        while i < len(sentences):
            window = sentences[i : i + self.window_size]
            window_text = " ".join(window)

            chunk = self._make_chunk(
                text=window_text,
                chunk_id=chunk_id,
                source_text=text,
                search_start=search_start,
                extra_meta={
                    "window_size"      : self.window_size,
                    "step_size"        : self.step_size,
                    "overlap_sentences": self.window_size - self.step_size,
                    "sentence_count"   : len(window),
                },
            )
            chunks.append(chunk)
            search_start = chunk.start_char
            i += self.step_size
            chunk_id += 1

        return chunks


class SemanticChunker(BaseChunker):
    """
    Groups consecutive sentences into chunks by measuring the cosine
    similarity of adjacent sentence embeddings.  A new chunk boundary is
    inserted wherever similarity drops below *threshold*.

    This is the DEFAULT chunker for the GraphRAG pipeline because it
    produces topically coherent chunks — essential for building a
    meaningful knowledge graph where nodes should represent single concepts.

    Model
    sentence-transformers/all-MiniLM-L6-v2

    Parameters
    model_name  : str   — HuggingFace model identifier.
    threshold   : float — cosine-similarity cutoff [0, 1].
                          Similarities BELOW this value trigger a new chunk.
                          Lower  → larger chunks (topics merged).
                          Higher → smaller chunks (topics separated).
                          Default 0.5 works well for lecture / article text.
    max_sentences_per_chunk : int — hard cap so chunks don't grow unbounded
                                    (default 20).
    min_sentences_per_chunk : int — merge chunks that are too small
                                    (default 2).
    batch_size  : int   — embedding batch size; tune for your GPU/CPU RAM.
    """

    chunker_type = "semantic"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.5,
        max_sentences_per_chunk: int = 20,
        min_sentences_per_chunk: int = 2,
        batch_size: int = 64,
    ):
        super().__init__()
        self.model_name = model_name
        self.threshold = threshold
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.batch_size = batch_size

        # Lazy-load the model — import only when this class is first used
        # so that environments without sentence-transformers can still use
        # all other chunkers without any import error.
        self._model = None

    def _load_model(self):
        """Load the SentenceTransformer model (only once, then cached)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for SemanticChunker.\n"
                    "Install it with:  pip install sentence-transformers"
                ) from exc

    # embedding helpers

    def _embed(self, sentences: List[str]):
        """
        Encode a list of sentences into L2-normalised embeddings.

        Returns
        -------
        numpy.ndarray of shape (N, 384) with float32 dtype.
        Each row is a unit vector — cosine similarity between two rows equals
        their dot product (since ||v|| = 1 after normalisation).
        """
        import numpy as np
        self._load_model()

        # encode() with normalize_embeddings=True returns unit vectors
        embeddings = self._model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise → dot = cosine sim
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """
        Cosine similarity between two 1-D numpy vectors.

        Because _embed() returns L2-normalised vectors, this simplifies to
        a plain dot product: sim = a · b   (both are unit vectors).
        """
        import numpy as np
        # Dot product of two unit vectors == cosine similarity
        sim = float(np.dot(a, b))
        # Clip to [-1, 1] to guard against floating-point rounding
        return max(-1.0, min(1.0, sim))

    # main algorithm

    def chunk(self, text: str) -> List[Chunk]:
        """
        Segment text into semantically coherent chunks.

        Algorithm (step by step)
        1. Clean the text.
        2. Split into sentences (using BaseChunker._split_sentences).
        3. Embed all sentences in one batched forward pass.
        4. Walk through adjacent sentence pairs.
           — Compute cosine similarity between sentence[i] and sentence[i+1].
           — If similarity < threshold  OR  current chunk hit max size:
               → close the current chunk, start a new one.
        5. Merge chunks that are smaller than min_sentences_per_chunk into
           their neighbour (whichever has higher average similarity).
        6. Build and return Chunk objects with full metadata.
        """
        import numpy as np

        # Step 1: clean the text
        text = self.clean(text)
        if not text:
            return []

        # Step 2: split into sentences
        sentences = self._split_sentences(text)
        n = len(sentences)

        if n == 0:
            return []

        # If there is only one sentence, return it as a single chunk
        if n == 1:
            return [
                self._make_chunk(
                    text=sentences[0],
                    chunk_id=0,
                    source_text=text,
                    extra_meta={"threshold": self.threshold, "model": self.model_name},
                )
            ]

        # Step 3: embed all sentences
        embeddings = self._embed(sentences)   # shape (n, 384)

        # Step 4: detect boundaries
        # Each group is a list of sentence indices [i, j, k, ...]
        groups: List[List[int]] = [[0]]       # start with first sentence

        for i in range(1, n):
            # Similarity between sentence i-1 and sentence i
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            current_group_size = len(groups[-1])
            topic_shift = sim < self.threshold
            too_long = current_group_size >= self.max_sentences_per_chunk

            if topic_shift or too_long:
                # Open a new group
                groups.append([i])
            else:
                # Continue current group
                groups[-1].append(i)

        # Step 5: merge tiny groups
        # Any group with fewer than min_sentences is merged with its best
        # neighbour (left or right).
        groups = self._merge_small_groups(groups, embeddings, sentences)

        # Step 6: build Chunk objects
        chunks: List[Chunk] = []
        search_start = 0

        for chunk_id, group in enumerate(groups):
            chunk_sentences = [sentences[idx] for idx in group]
            chunk_text = " ".join(chunk_sentences)

            # Average pairwise similarity inside the chunk (quality signal
            # for downstream graph-builder to weight edges)
            if len(group) > 1:
                sims = [
                    self._cosine_similarity(embeddings[group[j]], embeddings[group[j + 1]])
                    for j in range(len(group) - 1)
                ]
                avg_sim = float(np.mean(sims))
            else:
                avg_sim = 1.0   # single sentence is perfectly self-coherent

            chunk = self._make_chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_text=text,
                search_start=search_start,
                extra_meta={
                    "model"                 : self.model_name,
                    "threshold"             : self.threshold,
                    "sentence_count"        : len(group),
                    "avg_internal_similarity": round(avg_sim, 4),
                },
            )
            chunks.append(chunk)
            search_start = chunk.end_char

        return chunks

    def _merge_small_groups(
        self,
        groups: List[List[int]],
        embeddings,
        sentences: List[str],
    ) -> List[List[int]]:
        """
        Merge groups that have fewer than *min_sentences_per_chunk* sentences
        into whichever neighbour (left or right) has the highest mean
        embedding similarity to the small group's centroid.

        This avoids single-sentence or very short chunks that would create
        low-quality graph nodes.
        """
        import numpy as np

        if self.min_sentences_per_chunk <= 1:
            return groups

        changed = True
        while changed:
            changed = False
            new_groups: List[List[int]] = []
            skip = False

            for g_idx in range(len(groups)):
                if skip:
                    skip = False
                    continue

                group = groups[g_idx]

                if len(group) >= self.min_sentences_per_chunk:
                    new_groups.append(group)
                    continue

                # Group is too small — decide merge direction
                has_left  = len(new_groups) > 0
                has_right = g_idx + 1 < len(groups)

                if not has_left and not has_right:
                    # Only one group total
                    new_groups.append(group)
                elif not has_left:
                    # Merge right
                    groups[g_idx + 1] = group + groups[g_idx + 1]
                    skip = False     # the merged group will be processed next
                    changed = True
                elif not has_right:
                    # Merge into left
                    new_groups[-1] = new_groups[-1] + group
                    changed = True
                else:
                    # Compare similarity to left and right neighbours
                    centroid = embeddings[group].mean(axis=0)
                    left_centroid  = embeddings[new_groups[-1]].mean(axis=0)
                    right_centroid = embeddings[groups[g_idx + 1]].mean(axis=0)

                    sim_left  = float(np.dot(centroid, left_centroid))
                    sim_right = float(np.dot(centroid, right_centroid))

                    if sim_left >= sim_right:
                        new_groups[-1] = new_groups[-1] + group
                    else:
                        groups[g_idx + 1] = group + groups[g_idx + 1]

                    changed = True

            groups = new_groups

        return groups