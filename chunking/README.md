# GraphRAG Chunking Module

> **File:** `chunking.py`  |  **Tests:** `chunking_test.py`
>
> Part of the **GraphRAG_Library** pipeline — sits between the Extractors module
> and the Embedding / Graph-Building phase.

---

## 1. Why Chunking Matters in GraphRAG

A GraphRAG system stores knowledge in a **graph** where:

- **Nodes** = chunks of text (semantic units)
- **Edges** = relationships between chunks (co-occurrence, citation, entailment …)

The quality of every downstream phase — embedding, graph construction, retrieval, and answer generation — depends directly on chunk quality:

| Chunk too large | Chunk too small |
|---|---|
| Multiple unrelated topics conflated into one node | Insufficient context for meaningful embeddings |
| Edges connect unrelated concepts | Too many nodes → graph becomes sparse and noisy |
| LLM context window exceeded | Answer generation lacks surrounding context |

The six strategies in this module cover a spectrum from **simple/fast** to **smart/slower**, so you can choose the right trade-off for your data and latency budget.

---

## 2. Module Architecture

```
chunking.py
│
├── Chunk                    ← dataclass returned by every chunker
├── TextCleaner              ← cleans raw extractor output before splitting
├── BaseChunker              ← abstract base with shared helpers
│   ├── FixedSizeChunker     ← split by character count + overlap
│   ├── SentenceChunker      ← split by sentence count + overlap
│   ├── ParagraphChunker     ← split on blank-line boundaries
│   ├── RecursiveChunker     ← delimiter hierarchy with greedy packing
│   ├── SlidingWindowChunker ← sliding sentence window with overlap
│   └── SemanticChunker      ← embedding-similarity boundary detection (DEFAULT)
│
├── get_chunker(type, **kw)  ← factory — instantiate any chunker by name
└── chunk_text(text, type)   ← one-liner: clean + chunk in a single call
```

**Pipeline position:**

```
Source file
    │
    ▼
SimpleExtractorWrapper.extract_auto()   [extractors module]
    │  raw_text : str
    ▼
TextCleaner.clean()                     [inside each chunker]
    │  cleaned_text : str
    ▼
<Chunker>.chunk()                       [chunking.py]
    │  List[Chunk]
    ▼
Embedding phase → Graph building → Retrieval → Answer generation
```

---

## 3. The `Chunk` Dataclass

```python
@dataclass
class Chunk:
    text       : str
    chunk_id   : int
    start_char : int
    end_char   : int
    metadata   : dict = field(default_factory=dict)
```

### Field-by-field explanation

| Field | Type | Purpose |
|---|---|---|
| `text` | `str` | The actual text content of the chunk. Fed to the embedding model. |
| `chunk_id` | `int` | Zero-based position within the document. Used to reconstruct reading order during retrieval. |
| `start_char` | `int` | Character offset in the **cleaned** source text where this chunk begins. Enables precise retrieval provenance. |
| `end_char` | `int` | Character offset where this chunk ends. `end_char - start_char == len(text)`. |
| `metadata` | `dict` | Extensible key-value store. Chunker type, overlap settings, similarity scores, etc. Downstream phases add their own keys (embedding vector, graph node ID, …) without breaking this contract. |

### Properties

```python
@property
def word_count(self) -> int:
    return len(self.text.split())
    # Split on whitespace and count tokens.
    # Used by tests and downstream to filter trivially short chunks.

@property
def char_count(self) -> int:
    return len(self.text)
    # Convenience alias for len(text).
```

### Why `start_char` / `end_char`?

These offsets allow the retrieval phase to **highlight the exact passage** in the original document, which is essential for answer generation with citations. They also allow the graph-builder to detect when two chunks from different chunking strategies overlap, which can be used to add edges.

---

## 4. TextCleaner

```python
class TextCleaner:
    _PAGE_MARKER  = re.compile(r"-{3,}\s*Page\s+\d+\s*-{3,}", re.IGNORECASE)
    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _MULTI_SPACE  = re.compile(r"[ \t]+")
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
```

### Regex patterns

| Pattern | Matches | Action |
|---|---|---|
| `_PAGE_MARKER` | `--- Page 3 ---` (inserted by `PDFExtractor`) | Replace with `\n` |
| `_MULTI_NEWLINE` | Three or more consecutive newlines | Collapse to `\n\n` |
| `_MULTI_SPACE` | Spaces or tabs in a run | Collapse to single space |
| `_CONTROL_CHARS` | Null bytes, bell, carriage return, etc. | Replace with space |

### `clean(text)` — step by step

```python
def clean(self, text: str) -> str:
    if not text:           # Guard: return immediately if text is None or ""
        return ""

    # Step 1 — remove control characters (null bytes from PDFs, \x01 from audio transcripts…)
    text = self._CONTROL_CHARS.sub(" ", text)

    # Step 2 — remove PDF page markers ("--- Page 3 ---")
    # These are added by PDFExtractor and would create noise in chunks.
    text = self._PAGE_MARKER.sub("\n", text)

    # Step 3 — collapse triple+ newlines to a double newline
    # Preserves paragraph breaks (double newline) but removes extra blank lines.
    text = self._MULTI_NEWLINE.sub("\n\n", text)

    # Step 4 — collapse runs of spaces/tabs to a single space
    # DOCX and PPTX extractors sometimes produce double spaces around bullets.
    text = self._MULTI_SPACE.sub(" ", text)

    # Step 5 — per-line strip (remove leading/trailing spaces on every line)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Step 6 — strip the whole document
    return text.strip()
```

---

## 5. BaseChunker

```python
class BaseChunker:
    chunker_type: str = "base"
```

### `__init__`

```python
def __init__(self):
    self._cleaner = TextCleaner()
    # Each chunker instance owns one TextCleaner.
    # Instantiated once and reused across multiple .chunk() calls.
```

### `clean(text)`

```python
def clean(self, text: str) -> str:
    return self._cleaner.clean(text)
    # Delegates to TextCleaner.clean().
    # Every subclass calls this first inside .chunk().
```

### `_make_chunk(text, chunk_id, source_text, search_start, extra_meta)`

```python
def _make_chunk(self, text, chunk_id, source_text, search_start=0, extra_meta=None):
    start = source_text.find(text, search_start)
    # str.find() searches for 'text' inside 'source_text' starting from
    # 'search_start'. The 'search_start' hint avoids matching an earlier
    # occurrence of the same substring (important for overlapping chunkers).

    if start == -1:
        start = search_start   # Fallback: use the hint position
    end = start + len(text)

    metadata = {"chunker": self.chunker_type}
    # Every chunk records which chunker produced it.
    if extra_meta:
        metadata.update(extra_meta)   # Merge chunker-specific metadata

    return Chunk(text=text, chunk_id=chunk_id,
                 start_char=start, end_char=end, metadata=metadata)
```

### `_split_sentences(text)` — static method

This is the sentence tokeniser shared by `SentenceChunker`, `SlidingWindowChunker`, and `SemanticChunker`.

```python
@staticmethod
def _split_sentences(text: str) -> List[str]:

    abbreviations = ["Dr", "Mr", "Mrs", "Ms", "Prof", ...]
    placeholder = "ABBR_DOT"

    for abbr in abbreviations:
        # Replace "Dr." with "DrABBR_DOT" so the dot is not treated as
        # a sentence ender. Uses \b (word boundary) to avoid partial matches.
        text = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}{placeholder}", text)

    # Split AFTER punctuation (.!?) that is followed by whitespace and then
    # an uppercase letter, digit, or opening quote/bracket.
    # (?<=[.!?]) — lookbehind: must be preceded by sentence-ending punct
    # \s+        — one or more whitespace characters (the split point)
    # (?=[A-Z0-9\"\'(]) — lookahead: next char starts a sentence
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\(])", text)

    # Restore protected abbreviation dots
    sentences = [p.replace(placeholder, ".").strip() for p in parts]

    # Filter out empty strings produced by leading/trailing whitespace
    return [s for s in sentences if s]
```

**Why no NLTK/spaCy?** Those libraries add hundreds of MB of dependencies and multi-second import times. The regex above handles ~95% of English academic text correctly and is instantaneous.

---

## 6. FixedSizeChunker

### Concept

Slice the text into windows of exactly `chunk_size` characters. Each window overlaps the previous by `overlap` characters.

```
Text: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
       |<─ 512 ─>|
              |<─ 512 ─>|
                     |<─ 512 ─>|
       |<30>|  ← overlap
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `chunk_size` | 512 | Target characters per chunk |
| `overlap` | 50 | Characters shared between consecutive chunks |

### `__init__`

```python
def __init__(self, chunk_size: int = 512, overlap: int = 50):
    super().__init__()          # calls BaseChunker.__init__() → creates TextCleaner
    if overlap >= chunk_size:
        raise ValueError(...)   # Prevents infinite loops (step would be ≤ 0)
    self.chunk_size = chunk_size
    self.overlap = overlap
```

### `chunk(text)`

```python
def chunk(self, text: str) -> List[Chunk]:
    text = self.clean(text)     # Always clean first
    if not text:
        return []

    chunks: List[Chunk] = []
    step = self.chunk_size - self.overlap  # How far the window slides each step
    pos = 0                                # Current position in text (in chars)
    chunk_id = 0

    while pos < len(text):
        window = text[pos : pos + self.chunk_size]
        # Slice Python string — O(chunk_size) per iteration, not O(N²)

        chunk = Chunk(
            text=window,
            chunk_id=chunk_id,
            start_char=pos,               # Exact offset — no search needed
            end_char=pos + len(window),
            metadata={
                "chunker"   : self.chunker_type,
                "chunk_size": self.chunk_size,
                "overlap"   : self.overlap,
            },
        )
        chunks.append(chunk)
        pos += step        # Slide forward by (chunk_size - overlap)
        chunk_id += 1

    return chunks
```

**Why inline `Chunk()` instead of `_make_chunk()`?**
`FixedSizeChunker` knows the exact `start_char` from `pos`, so the `str.find()` search inside `_make_chunk()` is unnecessary work. Direct construction is O(1) per chunk.

### Time complexity

`O(N / step)` chunks × O(chunk_size) per slice = **O(N)** total.

---

## 7. SentenceChunker

### Concept

Group `sentences_per_chunk` consecutive sentences into one chunk. Overlap by `overlap_sentences` sentences.

```
Sentences: [S1, S2, S3, S4, S5, S6, S7, S8]
sentences_per_chunk=3, overlap_sentences=1

Chunk 0: [S1, S2, S3]
Chunk 1: [S3, S4, S5]   ← S3 repeated (overlap)
Chunk 2: [S5, S6, S7]
Chunk 3: [S7, S8]
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `sentences_per_chunk` | 5 | Sentences per chunk |
| `overlap_sentences` | 1 | Sentences repeated in next chunk |

### `chunk(text)`

```python
def chunk(self, text: str) -> List[Chunk]:
    text = self.clean(text)
    if not text:
        return []

    sentences = self._split_sentences(text)
    # Returns List[str], each element is one sentence.
    if not sentences:
        return []

    chunks: List[Chunk] = []
    step = self.sentences_per_chunk - self.overlap_sentences
    # step is how many sentences the window advances.
    # With sentences_per_chunk=5 and overlap=1, step=4.

    search_start = 0   # Tracks where to start str.find() in _make_chunk
    chunk_id = 0
    pos = 0            # Current sentence index

    while pos < len(sentences):
        window_sents = sentences[pos : pos + self.sentences_per_chunk]
        # Slice the sentence list — last window may have fewer sentences.
        window_text = " ".join(window_sents)
        # Join with a space — the split already stripped trailing whitespace.

        chunk = self._make_chunk(
            text=window_text,
            chunk_id=chunk_id,
            source_text=text,
            search_start=search_start,  # Avoids re-matching earlier chunks
            extra_meta={
                "sentences_per_chunk": self.sentences_per_chunk,
                "overlap_sentences"  : self.overlap_sentences,
                "sentence_count"     : len(window_sents),
            },
        )
        chunks.append(chunk)
        search_start = chunk.start_char  # Next search starts here
        pos += step
        chunk_id += 1

    return chunks
```

### Why `search_start = chunk.start_char` (not `chunk.end_char`)?

With overlap, the next chunk starts with a sentence that also appears in the current chunk. Setting `search_start` to `chunk.start_char` allows `str.find()` to locate the next chunk's text beginning at or after where the current chunk started — correctly handling the re-used sentence.

---

## 8. ParagraphChunker

### Concept

Use blank lines (`\n\n`) as natural boundaries. Merge short paragraphs into the next one to avoid tiny orphan chunks.

```
Text:
  Para A (20 chars)      ← too short, merge into B
  
  Para B (150 chars)     ← now A+B = 170 chars ≥ min_chars → flush as chunk 0
  
  Para C (200 chars)     ← chunk 1
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `min_chars` | 100 | Merge paragraphs shorter than this |

### `chunk(text)`

```python
def chunk(self, text: str) -> List[Chunk]:
    text = self.clean(text)
    if not text:
        return []

    # Split on one or more blank lines (handles \n\n, \n   \n, etc.)
    raw_paragraphs = re.split(r"\n\s*\n", text)

    # Remove empty strings from split artefacts
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    merged: List[str] = []
    accumulator = ""   # Buffer for merging short paragraphs

    for para in paragraphs:
        if accumulator:
            # Append this para to the accumulator, keeping the blank-line sep
            accumulator += "\n\n" + para
        else:
            accumulator = para

        if len(accumulator) >= self.min_chars:
            # Accumulator is big enough — flush it as a finished chunk
            merged.append(accumulator)
            accumulator = ""

    # Flush any remaining text after the loop
    if accumulator:
        if merged:
            # Attach orphan tail to the last chunk rather than creating a tiny one
            merged[-1] += "\n\n" + accumulator
        else:
            merged.append(accumulator)   # Edge case: whole doc is one short block

    # Build Chunk objects with correct offsets
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
        search_start = chunk.end_char   # Next search after this chunk ends

    return chunks
```

### Time complexity

**O(N)** — one pass through paragraphs + one pass to build Chunks.

---

## 9. RecursiveChunker

### Concept

Try delimiters from coarse to fine. If a piece is still too large after splitting on the current delimiter, recurse with the next one. Greedily pack small pieces to avoid orphan chunks.

```
Delimiter hierarchy: ["\n\n", "\n", ". ", " "]

Step 1: split on "\n\n"
  → Piece A: 600 chars   (too large → recurse)
  → Piece B: 200 chars   (fits → keep)

Step 2 (recursing Piece A): split on "\n"
  → A1: 300 chars   (too large → recurse)
  → A2: 250 chars   (fits → keep)

Step 3 (recursing A1): split on ". "
  → A1a: 120 chars  (fits)
  → A1b: 180 chars  (fits)
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_chunk_size` | 512 | Hard upper bound in characters |
| `delimiters` | `["\n\n", "\n", ". ", " "]` | Ordered delimiter hierarchy |

### `_split(text, delimiter_index)` — the core recursion

```python
def _split(self, text: str, delimiter_index: int) -> List[str]:

    # Base case 1: text already fits — return it as-is
    if len(text) <= self.max_chunk_size:
        return [text]

    # Base case 2: no more delimiters — return as-is (last resort, oversized)
    if delimiter_index >= len(self.delimiters):
        return [text]

    delimiter = self.delimiters[delimiter_index]

    # Split and re-attach the delimiter to each piece (except the last)
    # so the text can be reassembled faithfully.
    raw_parts = text.split(delimiter)
    parts = []
    for i, part in enumerate(raw_parts):
        if i < len(raw_parts) - 1:
            parts.append(part + delimiter)   # Restore delimiter
        else:
            parts.append(part)               # Last part: no delimiter

    # Greedy packing: accumulate small pieces until they would overflow
    packed: List[str] = []
    buffer = ""

    for part in parts:
        candidate = buffer + part

        if len(candidate) <= self.max_chunk_size:
            buffer = candidate     # Fits → keep accumulating

        else:
            if buffer:
                packed.append(buffer)   # Flush the buffer

            if len(part) > self.max_chunk_size:
                # This single part is oversized → recurse deeper
                sub = self._split(part, delimiter_index + 1)
                packed.extend(sub)
                buffer = ""
            else:
                buffer = part       # Start fresh accumulation

    if buffer:
        packed.append(buffer)       # Flush the final buffer

    return packed
```

### `chunk(text)` — public entry point

```python
def chunk(self, text: str) -> List[Chunk]:
    text = self.clean(text)
    if not text:
        return []

    pieces = self._split(text, delimiter_index=0)   # Start with coarsest delimiter

    chunks: List[Chunk] = []
    search_start = 0

    for chunk_id, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue   # Skip empty artefacts

        chunk = self._make_chunk(
            text=piece, chunk_id=chunk_id,
            source_text=text, search_start=search_start,
            extra_meta={"max_chunk_size": self.max_chunk_size,
                        "delimiters"    : self.delimiters},
        )
        chunks.append(chunk)
        search_start = chunk.end_char

    # Re-number after empty pieces are filtered out
    for i, c in enumerate(chunks):
        c.chunk_id = i

    return chunks
```

### Guarantee

Every chunk satisfies `len(chunk.text) ≤ max_chunk_size` — **except** in the pathological case where a single word is longer than `max_chunk_size`. The recursion bottoms out when delimiters are exhausted and returns that word as-is rather than splitting it mid-character.

---

## 10. SlidingWindowChunker

### Concept

A window of `window_size` sentences slides forward by `step_size` sentences at a time. The overlap is `window_size - step_size` sentences.

```
Sentences: [S1  S2  S3  S4  S5  S6  S7  S8  S9  S10]
window_size=5, step_size=3

Chunk 0: [S1  S2  S3  S4  S5]
Chunk 1: [S4  S5  S6  S7  S8]   ← S4 and S5 are the overlap
Chunk 2: [S7  S8  S9  S10]
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `window_size` | 8 | Sentences per chunk |
| `step_size` | 4 | Sentences advanced per step |

### `chunk(text)`

```python
def chunk(self, text: str) -> List[Chunk]:
    text = self.clean(text)
    if not text:
        return []

    sentences = self._split_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    search_start = 0
    chunk_id = 0
    i = 0   # Sentence index (start of current window)

    while i < len(sentences):
        window = sentences[i : i + self.window_size]
        # Slice: last window may have fewer than window_size sentences (that is fine)
        window_text = " ".join(window)

        chunk = self._make_chunk(
            text=window_text, chunk_id=chunk_id,
            source_text=text, search_start=search_start,
            extra_meta={
                "window_size"      : self.window_size,
                "step_size"        : self.step_size,
                "overlap_sentences": self.window_size - self.step_size,
                "sentence_count"   : len(window),
            },
        )
        chunks.append(chunk)
        search_start = chunk.start_char   # Hint for next search
        i += self.step_size               # Slide the window
        chunk_id += 1

    return chunks
```

### Why heavy overlap is good for GraphRAG

When a question involves a concept that straddles two chunks, at least one chunk will contain the full context because of the overlap. This significantly improves **retrieval recall** — the single most important metric for RAG quality.

---

## 11. SemanticChunker (Default)

### Concept

Embed each sentence using a transformer model. Compare adjacent sentences by **cosine similarity**. Insert a chunk boundary whenever similarity drops below `threshold` — this signals a **topic shift**.

```
Sentences: [S1, S2, S3, S4, S5, S6]
Similarities: [0.82, 0.78, 0.31, 0.75, 0.69]
                              ↑
                        threshold=0.5
                        Topic shift here!

Result:
  Chunk 0: [S1, S2, S3]   ← high internal similarity
  Chunk 1: [S4, S5, S6]   ← new topic
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `threshold` | 0.5 | Similarity below this → new chunk |
| `max_sentences_per_chunk` | 20 | Hard cap on chunk size |
| `min_sentences_per_chunk` | 2 | Merge chunks smaller than this |
| `batch_size` | 64 | Sentences per embedding batch |

### Why `all-MiniLM-L6-v2`?

- **384-dimensional** embeddings — small enough for fast dot products
- **~14,000 sentences/second** on CPU (no GPU required)
- **Excellent cosine-similarity quality** for English academic/technical text
- Freely available via HuggingFace, ~80 MB download

### `_load_model()`

```python
def _load_model(self):
    if self._model is None:          # Only load once (singleton pattern)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        # Model cached in self._model — subsequent calls to _load_model()
        # are instant (just the `if` check).
```

Lazy loading means that importing `chunking.py` does **not** trigger a model download. The model is only loaded when `SemanticChunker.chunk()` is first called.

### `_embed(sentences)`

```python
def _embed(self, sentences: List[str]):
    import numpy as np
    self._load_model()

    embeddings = self._model.encode(
        sentences,
        batch_size=self.batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalise → each vector has ||v|| = 1
        convert_to_numpy=True,       # Return numpy array (not torch tensor)
    )
    return embeddings.astype("float32")
    # float32 uses half the memory of float64 with negligible precision loss.
```

**Key insight:** `normalize_embeddings=True` L2-normalises each vector so that `cosine(a, b) = dot(a, b)`. This turns the expensive cosine similarity formula:

```
cos(a, b) = (a · b) / (||a|| × ||b||)
```

into the cheap dot product:

```
cos(a, b) = a · b    (since ||a|| = ||b|| = 1)
```

All N sentences are encoded in **one batched call** — this is orders of magnitude faster than calling `encode()` N times.

### `_cosine_similarity(a, b)`

```python
@staticmethod
def _cosine_similarity(a, b) -> float:
    import numpy as np
    sim = float(np.dot(a, b))      # Fast dot product on numpy arrays
    return max(-1.0, min(1.0, sim)) # Clip to [-1, 1] for float rounding safety
```

### `chunk(text)` — step by step

```python
def chunk(self, text: str) -> List[Chunk]:
    import numpy as np

    text = self.clean(text)        # Step 1: clean
    if not text:
        return []

    sentences = self._split_sentences(text)   # Step 2: sentence-tokenise
    n = len(sentences)

    if n == 0:
        return []
    if n == 1:
        # Single sentence → trivially one chunk (no similarity comparison possible)
        return [self._make_chunk(sentences[0], 0, text, ...)]

    # Step 3: embed ALL sentences in one forward pass
    embeddings = self._embed(sentences)   # shape: (n, 384)

    # Step 4: detect boundaries
    groups: List[List[int]] = [[0]]       # Start first group with sentence 0

    for i in range(1, n):
        # Cosine similarity between consecutive sentences
        sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

        current_group_size = len(groups[-1])
        topic_shift = sim < self.threshold                        # Low similarity
        too_long = current_group_size >= self.max_sentences_per_chunk  # Hard cap

        if topic_shift or too_long:
            groups.append([i])    # Open a new group
        else:
            groups[-1].append(i)  # Continue current group

    # Step 5: merge tiny groups
    groups = self._merge_small_groups(groups, embeddings, sentences)

    # Step 6: build Chunk objects
    chunks: List[Chunk] = []
    search_start = 0

    for chunk_id, group in enumerate(groups):
        chunk_sentences = [sentences[idx] for idx in group]
        chunk_text = " ".join(chunk_sentences)

        # Compute average pairwise similarity inside this chunk
        # This is a quality signal for the graph-builder:
        # high avg_sim → tight topic → strong self-edge weight
        if len(group) > 1:
            sims = [
                self._cosine_similarity(embeddings[group[j]], embeddings[group[j + 1]])
                for j in range(len(group) - 1)
            ]
            avg_sim = float(np.mean(sims))
        else:
            avg_sim = 1.0    # A single sentence is perfectly self-coherent

        chunk = self._make_chunk(
            text=chunk_text, chunk_id=chunk_id,
            source_text=text, search_start=search_start,
            extra_meta={
                "model"                   : self.model_name,
                "threshold"               : self.threshold,
                "sentence_count"          : len(group),
                "avg_internal_similarity" : round(avg_sim, 4),
            },
        )
        chunks.append(chunk)
        search_start = chunk.end_char

    return chunks
```

### `_merge_small_groups(groups, embeddings, sentences)`

```python
def _merge_small_groups(self, groups, embeddings, sentences):
    import numpy as np

    if self.min_sentences_per_chunk <= 1:
        return groups   # No merging needed

    changed = True
    while changed:              # Iterate until no more merges needed
        changed = False
        new_groups: List[List[int]] = []
        skip = False

        for g_idx in range(len(groups)):
            if skip:
                skip = False    # This group was already merged into the previous one
                continue

            group = groups[g_idx]

            if len(group) >= self.min_sentences_per_chunk:
                new_groups.append(group)   # Big enough — keep as-is
                continue

            # Group is too small — find best merge direction
            has_left  = len(new_groups) > 0
            has_right = g_idx + 1 < len(groups)

            if not has_left and not has_right:
                new_groups.append(group)   # Only group — nothing to merge with
            elif not has_left:
                # Prepend to the next group
                groups[g_idx + 1] = group + groups[g_idx + 1]
                changed = True
            elif not has_right:
                # Append to the previous group
                new_groups[-1] = new_groups[-1] + group
                changed = True
            else:
                # Compare centroid similarity to both neighbours
                centroid       = embeddings[group].mean(axis=0)
                # numpy fancy indexing: embeddings[group] selects the rows
                # corresponding to the sentence indices in 'group'.
                left_centroid  = embeddings[new_groups[-1]].mean(axis=0)
                right_centroid = embeddings[groups[g_idx + 1]].mean(axis=0)

                sim_left  = float(np.dot(centroid, left_centroid))
                sim_right = float(np.dot(centroid, right_centroid))

                if sim_left >= sim_right:
                    new_groups[-1] = new_groups[-1] + group   # Merge into left
                else:
                    groups[g_idx + 1] = group + groups[g_idx + 1]  # Merge into right

                changed = True

        groups = new_groups

    return groups
```

**Why centroids?** The centroid (mean embedding) of a group of sentences is a good proxy for the group's overall topic. Comparing centroids is far faster than computing all pairwise similarities between the small group and its neighbour.

---

## 12. Factory Functions

### `get_chunker(chunker_type, **kwargs)`

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

def get_chunker(chunker_type: str = DEFAULT_CHUNKER, **kwargs) -> BaseChunker:
    key = chunker_type.lower().strip()
    # Normalise: case-insensitive, strip accidental whitespace
    cls = _CHUNKER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown chunker type '{chunker_type}'. Valid: {list(_CHUNKER_REGISTRY)}")
    return cls(**kwargs)
    # **kwargs forwarded to __init__ → caller controls chunk_size, threshold, etc.
```

### `chunk_text(text, chunker_type, **kwargs)`

```python
def chunk_text(text: str, chunker_type: str = DEFAULT_CHUNKER, **kwargs) -> List[Chunk]:
    chunker = get_chunker(chunker_type, **kwargs)
    return chunker.chunk(text)
    # One-liner for the common case: no need to instantiate manually.
```

---

## 13. Design Decisions & Pipeline Fit

### Why no LangChain / LlamaIndex?

Both frameworks add heavy dependencies (100 MB+), frequent breaking API changes, and black-box behaviour. Building from scratch gives us:
- Full control over every splitting decision
- Easy debugging when chunk quality is wrong
- No dependency conflicts with the rest of GraphRAG_Library

### Why is SemanticChunker the default?

GraphRAG graph nodes should represent **coherent topics**. A fixed-size chunker blindly cuts mid-sentence; a paragraph chunker is at the mercy of document formatting. SemanticChunker detects actual topic shifts, producing nodes with the highest embedding quality and the most meaningful edges.

### Why carry `start_char` / `end_char`?

The retrieval phase needs to highlight exact passages in the original document. The graph-builder uses offsets to detect overlapping chunks (potential edge candidates). The answer-generation phase uses offsets to cite sources precisely.

### Why `metadata` is a `dict` (not typed fields)?

Downstream phases (embedding, graph-builder, retrieval) each add their own keys:
- Embedding: `"embedding": np.ndarray`
- Graph-builder: `"node_id": str`, `"edge_ids": List[str]`
- Retrieval: `"retrieval_score": float`

A `dict` lets each phase extend the chunk without modifying the dataclass — open/closed principle.

### Thread safety

Each `BaseChunker` instance owns its own `TextCleaner`. The `SemanticChunker` loads the model once per instance. If you need to chunk in parallel, create one chunker instance per thread.

---

## 14. Quick-Start Examples

### Basic usage

```python
from chunking import chunk_text, get_chunker

# One-liner with default (semantic) chunker
chunks = chunk_text(raw_text)

# One-liner with a specific chunker
chunks = chunk_text(raw_text, chunker_type="paragraph", min_chars=200)

# Manual instantiation for repeated use (avoids re-loading the model)
chunker = get_chunker("semantic", threshold=0.45)
chunks_a = chunker.chunk(text_a)
chunks_b = chunker.chunk(text_b)
```

### Connecting to the extractors module

```python
from extractors.extractors_wrapper import SimpleExtractorWrapper
from chunking import get_chunker

wrapper = SimpleExtractorWrapper()
raw_text = wrapper.extract_auto("lecture.pdf")  # or .docx, .pptx, .mp4 …

chunker = get_chunker("semantic", threshold=0.5)
chunks = chunker.chunk(raw_text)

for chunk in chunks:
    print(chunk)
    # Chunk(id=0, chars=312, words=52, preview='Graph Neural Networks are a class...')
```

### Inspecting chunks for downstream phases

```python
for chunk in chunks:
    print(f"ID       : {chunk.chunk_id}")
    print(f"Text     : {chunk.text[:100]}")
    print(f"Offset   : [{chunk.start_char}:{chunk.end_char}]")
    print(f"Words    : {chunk.word_count}")
    print(f"Chunker  : {chunk.metadata['chunker']}")
    # SemanticChunker also provides:
    if "avg_internal_similarity" in chunk.metadata:
        print(f"Coherence: {chunk.metadata['avg_internal_similarity']:.4f}")
```

### Choosing a chunker

| Use case | Recommended chunker |
|---|---|
| Uniform token budgets for LLM | `FixedSizeChunker` |
| Academic papers, articles | `SemanticChunker` (default) |
| Well-structured documents with clear paragraphs | `ParagraphChunker` |
| Transcripts, audio, video | `SentenceChunker` or `SlidingWindowChunker` |
| Mixed content (slides + prose + code) | `RecursiveChunker` |
| High retrieval recall needed | `SlidingWindowChunker` (heavy overlap) |

---
