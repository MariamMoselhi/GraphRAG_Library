# GraphRAG Chunking Module — Structure & Logic Reference

> This document describes the four files that make up the chunking module,
> what each one is responsible for, how they connect to each other, and the
> logic behind every significant design choice.

---

## Overview — The Four-File Layout

The original single `chunking.py` (977 lines) was split into four files with
clear, non-overlapping responsibilities. The dependency arrow points downward:
nothing in a lower layer imports from a higher one.

```
chunking.py                ← the ONLY file users import from
    │
    ├── chunk_implementations.py   ← all six chunker classes
    │       │
    │       └── chunk_base.py      ← Chunk dataclass + BaseChunker
    │               │
    │               └── text_cleaner.py   ← TextCleaner (lowest layer)
```

| File | Lines | Exposed to users? | One-line role |
|---|---|---|---|
| `text_cleaner.py` | 65 | No (internal) | Cleans raw extractor output |
| `chunk_base.py` | 173 | `Chunk` only | Data contract + abstract base |
| `chunk_implementations.py` | 766 | No (internal) | All six chunker algorithms |
| `chunking.py` | 123 | Yes (everything) | Public API, registry, factory |

---

## 1. `text_cleaner.py`

### Role
Lowest layer of the stack. Contains exactly one class: `TextCleaner`.
No other module in the package imports it except `chunk_base.py`.

### Why a separate file?
Cleaning logic is completely independent of chunking logic. Isolating it
here means it can be updated, tested, or swapped out without touching any
chunker. It also mirrors `utils/text_cleaner.py` in the repo so the
chunking module can stand alone when the utils package is unavailable.

### Imports
```python
import re          # standard library only — no internal dependencies
```

### Class: `TextCleaner`

Four class-level compiled regex patterns (compiled once at class load time,
reused on every call — O(1) lookup cost):

| Attribute | Pattern | What it removes |
|---|---|---|
| `_PAGE_MARKER` | `-{3,}\s*Page\s+\d+\s*-{3,}` | PDF page dividers inserted by `PDFExtractor` e.g. `--- Page 3 ---` |
| `_MULTI_NEWLINE` | `\n{3,}` | Three or more consecutive newlines — collapsed to `\n\n` |
| `_MULTI_SPACE` | `[ \t]+` | Runs of spaces or tabs — collapsed to one space |
| `_CONTROL_CHARS` | `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]` | Null bytes and non-printable control characters (keeps `\n` and `\t`) |

#### `clean(text: str) -> str` — six sequential steps

```
Step 1  _CONTROL_CHARS  → replace with space
Step 2  _PAGE_MARKER    → replace with \n
Step 3  _MULTI_NEWLINE  → collapse to \n\n
Step 4  _MULTI_SPACE    → collapse to single space
Step 5  per-line strip  → strip each line individually (catches leading/trailing spaces left by step 4)
Step 6  full strip      → strip the whole document
```

The steps are ordered so that earlier steps don't reintroduce noise for
later steps. For example, removing page markers (step 2) can create
triple newlines, which step 3 then collapses.

---

## 2. `chunk_base.py`

### Role
Defines the data contract (`Chunk`) and the abstract interface
(`BaseChunker`) that every chunker must satisfy. It imports `TextCleaner`
from `text_cleaner.py` and is imported by both `chunk_implementations.py`
and `chunking.py`.

### Imports
```python
import re
from dataclasses import dataclass, field
from typing import List, Optional
from text_cleaner import TextCleaner   # only internal import
```

---

### Dataclass: `Chunk`

The universal return type of every chunker's `.chunk()` method. Every
downstream phase (embedding, graph-building, retrieval, answer generation)
receives a `List[Chunk]` and reads from this dataclass.

```python
@dataclass
class Chunk:
    text       : str   # the actual chunk content
    chunk_id   : int   # zero-based position within the document
    start_char : int   # character offset in the cleaned source where this chunk begins
    end_char   : int   # character offset where it ends
    metadata   : dict  # extensible key-value store (default: empty dict)
```

#### Why `start_char` / `end_char`?
These offsets let the retrieval phase highlight the exact passage in the
source document and let the graph-builder detect when chunks from different
strategies overlap — a signal for adding edges.

#### Why `metadata` is a plain `dict`
Each downstream phase adds its own keys (`"embedding"`, `"node_id"`,
`"retrieval_score"`, …) without modifying the dataclass. Open/closed
principle: the contract never changes, but its content grows.

#### Computed properties (read-only)

| Property | Logic | Used by |
|---|---|---|
| `word_count` | `len(self.text.split())` | Tests, downstream filtering |
| `char_count` | `len(self.text)` | Convenience alias; used in `__repr__` |

#### `__repr__`
Returns a compact string like:
```
Chunk(id=2, chars=312, words=52, preview='Graph Neural Networks are a class...')
```
Useful for debugging without printing the full chunk text.

---

### Abstract class: `BaseChunker`

All six concrete chunkers inherit from this class. It:
- Owns one `TextCleaner` instance (created in `__init__`, reused across calls)
- Exposes `clean()` as a thin delegation to `TextCleaner.clean()`
- Declares `chunk()` as the interface subclasses must implement
- Provides two shared helpers: `_make_chunk()` and `_split_sentences()`

#### `__init__`
```python
self._cleaner = TextCleaner()
# One instance per chunker — shared across all .chunk() calls on that instance.
```

#### `clean(text) -> str`
```python
return self._cleaner.clean(text)
# Pure delegation. Every subclass calls this as the first line of .chunk().
```

#### `chunk(text) -> List[Chunk]`
```python
raise NotImplementedError(...)
# Forces every subclass to provide its own implementation.
```

#### `_make_chunk(text, chunk_id, source_text, search_start, extra_meta) -> Chunk`

Shared factory used by five of the six chunkers (all except `FixedSizeChunker`,
which knows exact offsets from its loop counter and constructs `Chunk` directly).

```python
start = source_text.find(text, search_start)
# str.find() with a search_start hint avoids re-matching an earlier occurrence
# of the same substring (critical for overlapping chunkers like SentenceChunker
# and SlidingWindowChunker where the same sentence appears in two chunks).

if start == -1:
    start = search_start   # fallback: use the hint as-is

end = start + len(text)

metadata = {"chunker": self.chunker_type}
if extra_meta:
    metadata.update(extra_meta)   # merge chunker-specific keys

return Chunk(text=text, chunk_id=chunk_id, start_char=start, end_char=end, metadata=metadata)
```

#### `_split_sentences(text) -> List[str]` (static method)

Shared sentence tokeniser used by `SentenceChunker`, `SlidingWindowChunker`,
and `SemanticChunker`. No NLP library — purely regex.

**Step 1 — protect abbreviations:**
```python
abbreviations = ["Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr",
                 "Fig", "Eq", "No", "vs", "etc", "e.g", "i.e", "al",
                 "approx", "ref", "ch", "vol", "pp"]
placeholder = "ABBR_DOT"
# "Dr." → "DrABBR_DOT"  so the dot is invisible to the splitter
```

**Step 2 — split on sentence boundaries:**
```python
re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'(])", text)
# lookbehind: must follow . ! ?
# \s+: one or more whitespace (the actual split point)
# lookahead: next char must start a sentence (uppercase, digit, or quote)
```

**Step 3 — restore protected dots and filter empties.**

---

## 3. `chunk_implementations.py`

### Role
Contains all six concrete chunker classes. Users never import from this
file — `chunking.py` re-exports everything. The only internal import is:

```python
from chunk_base import BaseChunker, Chunk
```

All six classes follow the same structure: `__init__` validates parameters
and stores them, `chunk()` cleans the text first then applies the algorithm.

---

### Class 1: `FixedSizeChunker`

**`chunker_type = "fixed_size"`**

Splits text by character count with optional overlap.

```
Parameters:  chunk_size (default 512)
             overlap    (default 50, must be < chunk_size)
```

**Algorithm in `chunk()`:**
```
step = chunk_size - overlap      ← how far to advance per iteration

pos = 0
while pos < len(text):
    window = text[pos : pos + chunk_size]   ← slice (O(chunk_size), not O(N²))
    build Chunk directly (pos is the exact start_char — no str.find() needed)
    pos += step
```

**Why `Chunk(...)` directly instead of `_make_chunk()`?**
`pos` is already the correct `start_char`. Calling `str.find()` would be
redundant work. This makes `FixedSizeChunker` O(N) with no search overhead.

**Constructor guard:**
```python
if overlap >= chunk_size:
    raise ValueError(...)
# Prevents step ≤ 0, which would cause an infinite loop.
```

---

### Class 2: `SentenceChunker`

**`chunker_type = "sentence"`**

Groups `sentences_per_chunk` consecutive sentences per chunk, overlapping
by `overlap_sentences` sentences.

```
Parameters:  sentences_per_chunk (default 5)
             overlap_sentences   (default 1, must be < sentences_per_chunk)
```

**Algorithm in `chunk()`:**
```
sentences = _split_sentences(text)
step = sentences_per_chunk - overlap_sentences

pos = 0  (sentence index)
while pos < len(sentences):
    window_sents = sentences[pos : pos + sentences_per_chunk]
    window_text  = " ".join(window_sents)
    chunk = _make_chunk(window_text, search_start=prev_chunk.start_char)
    pos += step
```

**Why `search_start = chunk.start_char` (not `end_char`)?**
With overlap, the next chunk starts with a sentence that was also in the
current chunk. Setting `search_start` to `start_char` lets `str.find()`
locate the overlapping sentence correctly without jumping past it.

---

### Class 3: `ParagraphChunker`

**`chunker_type = "paragraph"`**

Splits on blank lines (`\n\n`) and merges short paragraphs into the next
one until the combined length reaches `min_chars`.

```
Parameters:  min_chars (default 100)
```

**Algorithm in `chunk()`:**
```
raw_paragraphs = re.split(r"\n\s*\n", text)    ← handles \n\n, \n   \n, etc.
paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

accumulator = ""
for para in paragraphs:
    accumulator += "\n\n" + para   (or just para if accumulator is empty)
    if len(accumulator) >= min_chars:
        flush accumulator → merged list
        reset accumulator = ""

# Remaining accumulator after loop:
if merged:
    merged[-1] += "\n\n" + accumulator   ← attach orphan to last chunk
else:
    merged.append(accumulator)            ← edge case: whole doc is one block
```

**Why attach orphan to the last chunk?**
A tiny trailing chunk would create a low-quality graph node with almost no
embedding information.

---

### Class 4: `RecursiveChunker`

**`chunker_type = "recursive"`**

Tries delimiters from coarse to fine. If splitting on the current delimiter
still produces an oversized piece, it recurses into that piece with the next
delimiter.

```
Parameters:  max_chunk_size (default 512)
             delimiters     (default ["\n\n", "\n", ". ", " "])
```

**Public method `chunk()`:**
Calls `_split(text, 0)`, then builds `Chunk` objects from the returned
pieces. After filtering empty strings, re-numbers `chunk_id` sequentially.

**Private method `_split(text, delimiter_index)`:**

```
Base case 1: len(text) <= max_chunk_size → return [text]
Base case 2: delimiter_index out of range → return [text]   (last resort)

delimiter = delimiters[delimiter_index]
parts = text.split(delimiter)   (delimiter re-attached to each part except last)

# Greedy packing:
buffer = ""
for part in parts:
    candidate = buffer + part
    if len(candidate) <= max_chunk_size:
        buffer = candidate        ← still fits, keep accumulating
    else:
        flush buffer
        if len(part) > max_chunk_size:
            recurse: _split(part, delimiter_index + 1)
        else:
            buffer = part         ← start fresh with this part
flush remaining buffer
```

**Guarantee:** every returned piece satisfies `len(piece) ≤ max_chunk_size`
except for a single word longer than the limit (pathological case — the
recursion bottoms out and returns it as-is rather than splitting mid-char).

---

### Class 5: `SlidingWindowChunker`

**`chunker_type = "sliding_window"`**

A window of `window_size` sentences slides forward by `step_size` sentences
each step. Overlap = `window_size - step_size` sentences.

```
Parameters:  window_size (default 8)
             step_size   (default 4, must be > 0 and ≤ window_size)
```

**Algorithm in `chunk()`:**
```
sentences = _split_sentences(text)
i = 0  (sentence index)
while i < len(sentences):
    window = sentences[i : i + window_size]
    chunk  = _make_chunk(" ".join(window), ...)
    i += step_size
```

**Why heavy overlap?**
When a question involves a concept straddling two chunks, at least one chunk
captures the full context due to overlap — improves retrieval recall.

**Metadata stored per chunk:**
```python
{
    "window_size"      : self.window_size,
    "step_size"        : self.step_size,
    "overlap_sentences": self.window_size - self.step_size,
    "sentence_count"   : len(window),
}
```

**Constructor guards:**
```python
if step_size <= 0: raise ValueError(...)     # would never advance → infinite loop
if step_size > window_size: raise ValueError(...)  # no overlap would occur
```

---

### Class 6: `SemanticChunker` (Default)

**`chunker_type = "semantic"`**

The most sophisticated chunker. Uses transformer embeddings to detect
topic shifts between adjacent sentences and places chunk boundaries there.

```
Parameters:  model_name               (default "sentence-transformers/all-MiniLM-L6-v2")
             threshold                (default 0.5)
             max_sentences_per_chunk  (default 20)
             min_sentences_per_chunk  (default 2)
             batch_size               (default 64)
```

**`_load_model()`** — lazy singleton:
```python
if self._model is None:
    from sentence_transformers import SentenceTransformer
    self._model = SentenceTransformer(self.model_name)
# Model is loaded at most once per instance.
# Other chunkers can be used without triggering this import.
```

**`_embed(sentences) -> np.ndarray`:**
```python
embeddings = self._model.encode(
    sentences,
    normalize_embeddings=True,   # L2-normalise → each vector has ||v|| = 1
    convert_to_numpy=True,
)
return embeddings.astype("float32")
# All sentences encoded in ONE batched forward pass (orders of magnitude
# faster than calling encode() N times).
```

**`_cosine_similarity(a, b) -> float`:**
```python
sim = float(np.dot(a, b))        # dot product of two unit vectors = cosine sim
return max(-1.0, min(1.0, sim))  # clip for floating-point safety
# L2-normalisation in _embed() makes this O(dimensions) instead of the full
# cosine formula which would also need two norm computations.
```

**`chunk()` — six steps:**

```
Step 1: clean(text)
Step 2: _split_sentences(text) → sentences list
Step 3: _embed(sentences)      → embeddings array, shape (N, 384)

Step 4: boundary detection
    groups = [[0]]   ← first group starts with sentence 0
    for i in 1..N:
        sim = _cosine_similarity(embeddings[i-1], embeddings[i])
        if sim < threshold  OR  current group size >= max_sentences_per_chunk:
            groups.append([i])   ← new group
        else:
            groups[-1].append(i) ← continue current group

Step 5: _merge_small_groups(groups, embeddings, sentences)
    (see below)

Step 6: build Chunk objects
    for each group:
        chunk_text = " ".join(sentences in group)
        avg_sim    = mean of pairwise similarities inside the group
        store avg_sim in metadata["avg_internal_similarity"] as a graph-edge quality signal
```

**`_merge_small_groups()`:**

Repeatedly scans groups. When it finds one with `< min_sentences_per_chunk`
sentences, it merges it into the neighbour with the highest centroid
similarity:

```
centroid       = mean(embeddings[group_indices])
left_centroid  = mean(embeddings[left_neighbour_indices])
right_centroid = mean(embeddings[right_neighbour_indices])

if sim(centroid, left_centroid) >= sim(centroid, right_centroid):
    merge into left
else:
    merge into right

repeat until no small groups remain
```

**Why centroids?** Computing the mean embedding of a group is an O(n × d)
operation (n = sentences, d = 384). This is far cheaper than all pairwise
similarity comparisons and gives an accurate proxy for topical closeness.

---

## 4. `chunking.py` — Public API

### Role
The only file users import from. It:
1. Re-exports `Chunk` from `chunk_base.py` (so `from chunking import Chunk` works)
2. Imports all six chunker classes from `chunk_implementations.py`
3. Defines `__all__` to control `from chunking import *`
4. Holds the `_CHUNKER_REGISTRY` dict and `DEFAULT_CHUNKER`
5. Exposes two public functions: `get_chunker()` and `chunk_text()`

### `__all__`
```python
__all__ = [
    "Chunk", "get_chunker", "chunk_text", "DEFAULT_CHUNKER",
    "FixedSizeChunker", "SentenceChunker", "ParagraphChunker",
    "RecursiveChunker", "SlidingWindowChunker", "SemanticChunker",
]
# TextCleaner and BaseChunker are deliberately absent — they are internal.
```

### `_CHUNKER_REGISTRY`
```python
_CHUNKER_REGISTRY = {
    "fixed_size"    : FixedSizeChunker,
    "sentence"      : SentenceChunker,
    "paragraph"     : ParagraphChunker,
    "recursive"     : RecursiveChunker,
    "sliding_window": SlidingWindowChunker,
    "semantic"      : SemanticChunker,
}
DEFAULT_CHUNKER = "semantic"
```
A plain dict is used instead of a class hierarchy or plugin system — simple,
fast, and easy to extend by adding one line.

### `get_chunker(chunker_type, **kwargs) -> BaseChunker`
```python
key = chunker_type.lower().strip()   # normalise case and whitespace
cls = _CHUNKER_REGISTRY.get(key)
if cls is None:
    raise ValueError(f"Unknown chunker type '{chunker_type}'. Valid: ...")
return cls(**kwargs)
# **kwargs are forwarded directly to __init__, so the caller controls
# chunk_size, threshold, overlap, etc. without the factory needing to know them.
```

### `chunk_text(text, chunker_type, **kwargs) -> List[Chunk]`
```python
chunker = get_chunker(chunker_type, **kwargs)
return chunker.chunk(text)
# Two-liner convenience: no need to instantiate manually for one-off use.
```

---

## Import Chain Summary

```
User code
    │
    │   from chunking import get_chunker, chunk_text, Chunk
    ▼
chunking.py
    │   from chunk_base import Chunk, BaseChunker
    │   from chunk_implementations import FixedSizeChunker, ..., SemanticChunker
    ▼
chunk_implementations.py
    │   from chunk_base import BaseChunker, Chunk
    ▼
chunk_base.py
    │   from text_cleaner import TextCleaner
    ▼
text_cleaner.py
        import re   ← only standard library
```

Each layer only imports from the layer directly below it.
No circular imports are possible because the arrow is strictly one-directional.

---

## What the Test File Should Import

```python
# Everything the test file needs comes from a single import:
from chunking import (
    Chunk,
    get_chunker,
    chunk_text,
    DEFAULT_CHUNKER,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    RecursiveChunker,
    SlidingWindowChunker,
    SemanticChunker,
)

# TextCleaner is internal — if tests need it, import from its own file:
from text_cleaner import TextCleaner
```

`SimpleExtractorWrapper` is imported separately from the extractors module
and is not part of the chunking package at all.
