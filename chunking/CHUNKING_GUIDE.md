# Graph-RAG — Chunking Module Documentation

> **Purpose:** Split extracted document text into semantically coherent chunks for Graph-RAG node creation.  
> **Dependencies:** Python standard library only (`re`, `math`, `hashlib`, `collections`, `abc`, `typing`).

---

## Table of Contents

1. [Why Chunking Matters for Graph RAG](#1-why-chunking-matters-for-graph-rag)
2. [Design Goals](#2-design-goals)
3. [Shared Utilities](#3-shared-utilities)
4. [Abstract Base Class](#4-abstract-base-class)
5. [Method 1 — FixedSizeChunker](#5-method-1--fixedsizechunker)
6. [Method 2 — SentenceChunker](#6-method-2--sentencechunker)
7. [Method 3 — ParagraphChunker](#7-method-3--paragraphchunker)
8. [Method 4 — RecursiveChunker](#8-method-4--recursivechunker)
9. [Method 5 — SlidingWindowChunker](#9-method-5--slidingwindowchunker)
10. [Method 6 — SemanticChunker](#10-method-6--semanticchunker)
11. [Method 7 — HybridChunker ★ Recommended Default ★](#11-method-7--hybridchunker--recommended-default-)
12. [Convenience Function — `chunk_text()`](#12-convenience-function--chunk_text)
13. [ChunkAnalyser](#13-chunkanalyser)
14. [Choosing the Right Chunker](#14-choosing-the-right-chunker)
15. [Output Schema](#15-output-schema)
16. [Performance Notes](#16-performance-notes)

---

## 1. Why Chunking Matters for Graph RAG

In a standard Naive RAG system, documents are split into flat chunks that are retrieved independently.  
In **Graph RAG**, chunks become **nodes** in a knowledge graph. Relationships (edges) are extracted between entities found within and across chunks.

For this to work well:

| Requirement | Why |
|---|---|
| **Semantic coherence** | Each chunk should describe one idea so entity extraction finds clean, unambiguous entities |
| **Bounded size** | LLM context windows limit how large a chunk can be |
| **No mid-sentence cuts** | Cutting a sentence destroys the subject–verb–object triple that Graph RAG needs |
| **Provenance** | Knowing which slide, page, or heading a chunk came from helps build richer graph edges |
| **Speed** | Graph RAG must be fast; chunking cannot be the bottleneck |

---

## 2. Design Goals

- **Zero external dependencies** — no LangChain, LangGraph, NLTK, spaCy, scikit-learn, or NumPy.
- **Format-agnostic** — works on the output of every extractor in the project (PDF, PPTX, DOCX, audio, video, URL).
- **Uniform output** — every chunker returns the same dict schema so downstream Graph-RAG code needs no adaptation.
- **Pluggable** — swap chunkers with a single parameter to `chunk_text(method="semantic")`.

---

## 3. Shared Utilities

These helper functions are used by all chunkers.

### `_make_chunk_id(source_id, index) → str`

```python
def _make_chunk_id(source_id: str, index: int) -> str:
    return f"{source_id}_{index}"
```

**What it does:** Builds a stable, human-readable ID string.  
**Why:** Graph-RAG node IDs must be deterministic and unique within a document.  
- `source_id` — the caller-supplied document name (e.g. `"lecture_5_ml"`).  
- `index` — the 0-based position of this chunk in the output list.

---

### `_approx_tokens(text) → int`

```python
def _approx_tokens(text: str) -> int:
    return len(text.split())
```

**What it does:** Estimates the BPE token count as a simple whitespace-split word count.  
**Why:** True tokenisation requires a tokeniser library. For English/technical text, word count is within ~20% of actual BPE token count — accurate enough for planning LLM calls without the overhead.

---

### `_build_chunk(text, source_id, index, char_start, char_end, metadata) → dict`

**What it does:** Assembles the standardised chunk dictionary returned by every chunker.  
**Line-by-line:**

```python
return {
    "chunk_id":    _make_chunk_id(source_id, index),  # stable unique ID
    "text":        text,                               # the chunk content
    "char_start":  char_start,                         # offset in original text (inclusive)
    "char_end":    char_end,                           # offset in original text (exclusive)
    "chunk_index": index,                              # 0-based position
    "token_count": _approx_tokens(text),               # estimated token count
    "metadata":    metadata or {},                     # caller + chunker metadata
}
```

---

### `_SENT_END` — Sentence-boundary regex

```python
_SENT_END = re.compile(
    r'(?<!\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Fig|St|No|Vol|pp|Ph|ca))'
    r'(?<!\d)'
    r'[.!?]'
    r'(?:["\'])?'
    r'(?=\s+[A-Z\d"\'\(]|$)',
    re.UNICODE
)
```

- **Negative look-behind `(?<!\b...)`** — skips abbreviations like `Mr.`, `Dr.`, `e.g.` so they don't create false sentence breaks.
- **`(?<!\d)`** — skips decimal numbers like `3.14`.
- **`[.!?]`** — matches the punctuation that ends the sentence.
- **`(?:["\'])?`** — optionally matches a closing quote immediately after the punctuation.
- **Positive look-ahead `(?=\s+[A-Z\d...]|$)`** — a real sentence boundary is followed by whitespace + an uppercase letter (or end of string).

---

### `_split_sentences(text) → List[str]`

```python
def _split_sentences(text: str) -> List[str]:
    boundaries = []
    for match in _SENT_END.finditer(text):
        boundaries.append(match.end())   # position right after the punctuation
    sentences = []
    prev = 0
    for boundary in boundaries:
        fragment = text[prev:boundary].strip()
        if fragment:
            sentences.append(fragment)
        prev = boundary
    tail = text[prev:].strip()           # text after the last full stop
    if tail:
        sentences.append(tail)
    if not sentences:
        sentences = [text.strip()]       # no boundaries found → one sentence
    return sentences
```

**How it works:**
1. Find all positions where a sentence ends (using `_SENT_END`).
2. Slice the text at those positions.
3. Strip whitespace from each piece; discard empty strings.
4. Anything after the last full stop becomes the final sentence (handles sentences ending without punctuation — common in bullet points and slide text).
5. If no boundaries were found at all, return the whole text as one sentence.

---

### `_split_paragraphs(text) → List[str]`

```python
def _split_paragraphs(text: str) -> List[str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')  # normalise line endings
    raw_paragraphs = re.split(r'\n{2,}', text)             # split on ≥2 consecutive newlines
    return [p.strip() for p in raw_paragraphs if p.strip()]
```

**How it works:**
1. Normalise Windows (`\r\n`) and old Mac (`\r`) line endings to Unix (`\n`).
2. Split wherever two or more consecutive newlines appear — this is the universal paragraph boundary convention.
3. Strip whitespace from each paragraph and discard empty strings.

The extractors always surround their structural markers (e.g. `--- Page 3 ---`) with blank lines, so those markers automatically become paragraph boundaries.

---

### `_is_structural_marker(line) → bool`

```python
_STRUCTURAL_MARKER = re.compile(
    r'(?:'
    r'---+\s*(?:Page|Slide|Table|Figure)\s*\d+\s*---+'    # PDF: --- Page 3 ---
    r'|={3,}\s*(?:SLIDE|SECTION|CHAPTER|HEADING)\s*\d*\s*={3,}'  # PPTX: === SLIDE 2 ===
    r'|\[HEADING\s+\d+\]\s*.+'                             # DOCX: [HEADING 1] Title
    r'|\[SPEAKER\s+NOTES\]'                                # PPTX: [SPEAKER NOTES]
    r')',
    re.IGNORECASE
)
def _is_structural_marker(line: str) -> bool:
    return bool(_STRUCTURAL_MARKER.match(line.strip()))
```

**What it does:** Returns `True` if a line is a document-structure marker injected by one of the project's extractors.  
**Why it exists:** HybridChunker uses this to detect section boundaries without relying on blank lines alone, which gives it better precision on structured documents.

---

### `_add_overlap(chunks, overlap_sentences) → List[str]`

```python
def _add_overlap(chunks, overlap_sentences):
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sents = _split_sentences(chunks[i - 1])        # sentences in previous chunk
        tail = prev_sents[-overlap_sentences:]               # last N sentences
        overlap_text = " ".join(tail).strip()
        result.append((overlap_text + "\n\n" + chunks[i]).strip())
    return result
```

**What it does:** Prepends the last `overlap_sentences` sentences of chunk N to chunk N+1.  
**Why:** In Graph RAG, an entity mentioned at the end of one chunk may be referenced at the beginning of the next. Overlap ensures the LLM sees enough context to correctly resolve co-references and build accurate graph edges.

---

## 4. Abstract Base Class

### `AbstractChunker`

All chunkers inherit from this class.

**Constructor parameters** (shared by all subclasses):

| Parameter | Default | Description |
|---|---|---|
| `max_chunk_size` | 1024 | Hard character ceiling per chunk |
| `overlap_sentences` | 1 | Sentences prepended to next chunk for context |
| `min_chunk_size` | 100 | Minimum characters; shorter chunks are merged |

---

#### `_enforce_size_ceiling(text) → List[str]`

**What it does:** Breaks a text that exceeds `max_chunk_size` into sub-pieces, trying sentence boundaries first, then word boundaries, then character boundaries as a last resort.

```python
def _enforce_size_ceiling(self, text):
    if len(text) <= self.max_chunk_size:
        return [text]                         # fits → no splitting needed
    
    sentences = _split_sentences(text)
    sub_chunks = []
    current = ""
    
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        
        if len(candidate) <= self.max_chunk_size:
            current = candidate               # still fits → keep accumulating
        else:
            if current:
                sub_chunks.append(current)    # flush the accumulated buffer
            if len(sent) > self.max_chunk_size:
                # Even one sentence is too long → split by words
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
                current = sent.strip()        # single sentence fits → start new buffer
    
    if current:
        sub_chunks.append(current)
    return sub_chunks if sub_chunks else [text[:self.max_chunk_size]]
```

---

#### `_merge_tiny_chunks(texts) → List[str]`

**What it does:** Merges consecutive chunks that are shorter than `min_chunk_size` into the following chunk, preventing orphan micro-chunks.

```python
def _merge_tiny_chunks(self, texts):
    merged = []
    pending = ""
    
    for t in texts:
        combined = (pending + "\n\n" + t).strip() if pending else t
        if len(combined) < self.min_chunk_size:
            pending = combined                 # still too small → keep accumulating
        else:
            merged.append(combined)
            pending = ""
    
    if pending:                                # leftover at end
        if merged:
            merged[-1] = (merged[-1] + "\n\n" + pending).strip()
        else:
            merged.append(pending)
    
    return merged
```

---

#### `_finalise(raw_chunks, original_text, source_id, base_metadata, chunker_name) → List[dict]`

**What it does:** Converts a list of raw text strings into fully-formed chunk dicts.

**Steps in order:**
1. **Overlap** — calls `_add_overlap()` to prepend sentence tails.
2. **Size ceiling** — calls `_enforce_size_ceiling()` on any oversized chunk.
3. **Merge tiny** — calls `_merge_tiny_chunks()` to clean up short orphans.
4. **Build dicts** — locates each chunk in the original text to compute `char_start` / `char_end`, then calls `_build_chunk()`.

The char-offset logic uses a forward-advancing `search_from` cursor so that when the same phrase appears multiple times, it always finds the correct occurrence.

---

## 5. Method 1 — FixedSizeChunker

**Complexity:** O(n) where n = len(text).  
**Best for:** Quick prototyping, very long plain-text files, baseline benchmarks.  
**Avoid for:** Documents where sentence coherence matters.

### How it works

```
Input text:  [──────────────────────────────────────────────────]
              ↑chunk_size↑     ↑chunk_size↑     ↑chunk_size↑
                    ←overlap→↑       ←overlap→↑
```

A sliding window of `chunk_size` characters is stepped across the text with step = `chunk_size - char_overlap`.

```python
step = max(1, self.chunk_size - self.char_overlap)
start = 0
while start < len(text):
    end      = min(start + self.chunk_size, len(text))
    fragment = text[start:end].strip()
    if fragment:
        raw_chunks.append(fragment)
        positions.append((start, end))
    start += step                          # advance by step (not chunk_size)
```

- `start` begins at 0.
- `end` is clamped to `len(text)` so the final chunk never goes out of bounds.
- The exact character positions `(start, end)` are saved, allowing the chunk dict to have perfectly accurate `char_start` / `char_end` without needing a text search.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 800 | Characters per chunk window |
| `char_overlap` | 100 | Characters duplicated between consecutive chunks |
| `min_chunk_size` | 100 | Minimum chars; shorter chunks merged |

---

## 6. Method 2 — SentenceChunker

**Complexity:** O(n) where n = number of sentences.  
**Best for:** Audio transcripts, short articles, NER preprocessing.  
**Avoid for:** Documents where paragraph structure carries meaning.

### How it works

```
Sentences: [S1] [S2] [S3] [S4] [S5] [S6] [S7] [S8] [S9]
            ├────chunk 0────┤   ├────chunk 1────┤   ├─c2─┤
                                                     ^ overlap from c0 tail
```

1. `_split_sentences(text)` splits the text into individual sentences.
2. Sentences are grouped in windows of `sentences_per_chunk`.
3. After grouping, `_finalise()` applies sentence-level overlap (the last `overlap_sentences` sentences of chunk N are prepended to chunk N+1).
4. `_enforce_size_ceiling()` breaks any resulting group that is still too large.

```python
i = 0
while i < len(sentences):
    window     = sentences[i: i + self.sentences_per_chunk]
    chunk_text = " ".join(window).strip()
    if chunk_text:
        raw_chunks.append(chunk_text)
    i += self.sentences_per_chunk           # move by full window; overlap added by _finalise
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `sentences_per_chunk` | 5 | Sentences per chunk |
| `overlap_sentences` | 1 | Sentences shared between consecutive chunks |
| `max_chunk_size` | 1200 | Hard char ceiling |
| `min_chunk_size` | 80 | Min chars before merging |

---

## 7. Method 3 — ParagraphChunker

**Complexity:** O(p) where p = number of paragraphs.  
**Best for:** PDF and DOCX content, web articles, structured reports.  
**Rationale:** Human authors write paragraphs to represent one self-contained idea — the best unit for Graph-RAG node creation.

### How it works

```
Paragraphs: [P1] [P2] [P3]  ──blank──  [P4] [P5]  ──blank──  [P6]
             ├───chunk 0────┤            ├──chunk 1─┤           └──chunk 2
```

1. `_split_paragraphs(text)` splits on ≥2 consecutive newlines.
2. Paragraphs are accumulated greedily until either the `max_paragraphs_per_chunk` count is reached or the combined length would exceed `max_chunk_size`.
3. When the threshold is hit, the accumulator is flushed and a new group starts.

```python
for para in paragraphs:
    will_overflow = (
        len(current_paras) >= self.max_paragraphs_per_chunk   # count limit
        or (current_len + len(para)) > self.max_chunk_size    # size limit
    )
    if will_overflow and current_paras:
        raw_chunks.append("\n\n".join(current_paras))         # flush
        current_paras, current_len = [], 0
    current_paras.append(para)
    current_len += len(para)
if current_paras:
    raw_chunks.append("\n\n".join(current_paras))             # final flush
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `max_paragraphs_per_chunk` | 3 | Max paragraphs merged before a cut |
| `max_chunk_size` | 1200 | Hard char ceiling |
| `overlap_sentences` | 1 | Sentence overlap between chunks |
| `min_chunk_size` | 100 | Min chars before merging |

---

## 8. Method 4 — RecursiveChunker

**Complexity:** O(n log n) in the worst case (recursive splits).  
**Best for:** Mixed-format documents, robust fallback for any text type.  
**Rationale:** By trying progressively finer separators, it always produces bounded chunks while preserving as much structure as possible.

### Separator Hierarchy

| Priority | Separator | What it splits on |
|---|---|---|
| 1 (coarsest) | `"\n\n"` | Paragraph boundaries |
| 2 | `"\n"` | Line breaks |
| 3 | `". "` | Sentence ends |
| 4 | `", "` | Clause boundaries |
| 5 | `" "` | Word boundaries |
| 6 (finest) | `""` | Character-level (last resort) |

### How it works

```python
def _recursive_split(self, text, separators):
    if len(text) <= self.max_chunk_size:
        return [text]                           # base case: fits → stop recursing
    
    if not separators:
        # No more separators → hard character split
        return [text[i:i+self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]
    
    sep       = separators[0]                   # try the current (coarsest) separator
    remaining = separators[1:]                  # save finer separators for recursion
    
    pieces = text.split(sep) if sep else list(text)
    
    current = ""
    for piece in pieces:
        candidate = (current + sep + piece).strip() if current else piece.strip()
        if len(candidate) <= self.max_chunk_size:
            current = candidate                 # accumulate
        else:
            if current:
                merged_chunks.append(current)
            if len(piece) > self.max_chunk_size:
                # Single piece still too large → recurse with finer separator
                merged_chunks.extend(self._recursive_split(piece, remaining))
                current = ""
            else:
                current = piece.strip()
    if current:
        merged_chunks.append(current)
    return merged_chunks
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `max_chunk_size` | 1024 | Hard char ceiling |
| `overlap_sentences` | 1 | Sentence overlap |
| `min_chunk_size` | 100 | Min chars before merging |

---

## 9. Method 5 — SlidingWindowChunker

**Complexity:** O(n/step × window) where n = sentences.  
**Best for:** Dense question-answering, maximum recall retrieval, audio transcripts.  
**Trade-off:** Produces more chunks than other methods (higher storage/index cost).

### How it works

```
Sentences: [S1][S2][S3][S4][S5][S6][S7][S8][S9]
window=4, step=2:
  chunk 0 → S1 S2 S3 S4
  chunk 1 → S3 S4 S5 S6      ← S3,S4 repeated = overlap
  chunk 2 → S5 S6 S7 S8
  chunk 3 → S7 S8 S9
```

```python
i = 0
while i < len(sentences):
    window     = sentences[i : i + self.window_sentences]   # grab a window
    chunk_text = " ".join(window).strip()
    if chunk_text:
        raw_chunks.append(chunk_text)
    i += self.step_sentences                                 # advance by step (< window = overlap)
```

The overlap here is **structural** (same sentences appear in adjacent chunks) rather than the appended-tail overlap used by other chunkers. This is intentional — it ensures every cross-sentence entity pair is captured within at least one chunk.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `window_sentences` | 6 | Sentences per chunk |
| `step_sentences` | 3 | Sentences to advance per step (step < window = overlap) |
| `max_chunk_size` | 1400 | Hard char ceiling |
| `min_chunk_size` | 80 | Min chars |

---

## 10. Method 6 — SemanticChunker

**Complexity:** O(S²·V) where S = sentences, V = vocabulary size. Typically ~50–200 ms on a 2,000-word document.  
**Best for:** Mixed-topic documents, audio transcripts with topic shifts, any text where paragraph breaks are absent or unreliable.  
**Key advantage:** No ML model needed — uses TF-IDF cosine similarity built from scratch.

### Full Pipeline

```
Text
 │
 ▼
_split_sentences()
 │  [S1, S2, S3, …, Sn]
 ▼
_build_tfidf_vectors()
 │  [{term: weight, …}, …]  ← sparse dict per sentence
 ▼
_compute_window_similarities()
 │  [sim(W1,W2), sim(W2,W3), …]  ← float per adjacent pair
 ▼
_find_breakpoints()
 │  [i, j, k, …]  ← indices where new chunk starts
 ▼
_group_sentences()
 │  ["sentence group 1", "sentence group 2", …]
 ▼
_finalise()
 │  [chunk_dict, …]
 ▼
Output
```

---

### Step A — TF-IDF Vectorisation

#### `_tokenise(text) → List[str]`

```python
@staticmethod
def _tokenise(text):
    text   = text.lower()                           # lowercase
    text   = re.sub(r'[^a-z0-9\s]', ' ', text)     # keep only letters, digits, spaces
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]  # remove 1-char noise & bare numbers
```

#### `_build_tfidf_vectors(sentences) → List[Dict[str, float]]`

**Step 1 — Tokenise every sentence:**
```python
tokenised = [self._tokenise(s) for s in sentences]
```

**Step 2 — Compute document frequency (df) — how many sentences contain each term:**
```python
df = defaultdict(int)
for tokens in tokenised:
    for term in set(tokens):           # set() → each term counted once per sentence
        df[term] += 1
```

**Step 3 — Compute IDF using the standard log-smoothed formula:**
```
IDF(term) = log(N / (1 + df(term)))
```
The `+1` smoothing prevents division-by-zero for terms that appear in every sentence.

```python
idf = {term: math.log(N / (1.0 + count)) for term, count in df.items()}
```

**Step 4 — Build TF-IDF dict per sentence:**
```
TF(term, sentence) = count(term in sentence) / len(sentence_tokens)
TF-IDF(term)       = TF × IDF
```

```python
for tokens in tokenised:
    total = len(tokens)
    tf    = {}
    for term in tokens:
        tf[term] = tf.get(term, 0.0) + 1.0 / total   # normalised TF (divide by length)
    
    vec = {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}
    vectors.append(vec)
```

---

### Step B — Window Vector

Instead of comparing individual sentences (which are noisy), each position is represented by a **buffered window** of ±`buffer_size` sentences around it:

```python
def _window_vector(self, vectors, center):
    lo = max(0, center - self.buffer_size)    # clamp to start of list
    hi = min(len(vectors), center + self.buffer_size + 1)  # clamp to end
    
    combined = defaultdict(float)
    for vec in vectors[lo:hi]:
        for term, weight in vec.items():
            combined[term] += weight           # sum term weights across window
    
    # L2-normalise so window size doesn't bias similarity scores
    norm = math.sqrt(sum(v*v for v in combined.values()))
    if norm > 0:
        combined = {t: v/norm for t, v in combined.items()}
    
    return dict(combined)
```

---

### Step C — Cosine Similarity

```python
@staticmethod
def _cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a                            # iterate over the smaller dict (faster)
    
    dot    = sum(a_val * b.get(term, 0.0) for term, a_val in a.items())
    norm_a = math.sqrt(sum(v*v for v in a.values()))
    norm_b = math.sqrt(sum(v*v for v in b.values()))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot / (norm_a * norm_b)             # cosine: ranges from 0 (orthogonal) to 1 (identical)
```

---

### Step D — Breakpoint Detection

```python
def _find_breakpoints(self, similarities):
    threshold = self.similarity_threshold      # fixed floor
    
    if self.percentile_threshold is not None:
        sorted_sims = sorted(similarities)
        n    = len(sorted_sims)
        rank = (self.percentile_threshold / 100.0) * (n - 1)  # fractional rank
        lo   = int(rank)
        hi   = min(lo + 1, n - 1)
        frac = rank - lo
        # Linear interpolation between the two nearest values
        dynamic   = sorted_sims[lo] + frac * (sorted_sims[hi] - sorted_sims[lo])
        threshold = min(threshold, dynamic)    # use whichever is more conservative
    
    return [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
    #        ^ +1: breakpoint is *before* sentence i+1
```

**Percentile logic:** If `percentile_threshold=25`, the threshold is set at the 25th percentile of all pairwise similarities. This means the lowest-similarity 25% of adjacent pairs will become chunk boundaries — the method adapts to each document's topic density automatically.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `max_chunk_size` | 1200 | Hard char ceiling |
| `overlap_sentences` | 1 | Sentence overlap |
| `min_chunk_size` | 120 | Min chars |
| `buffer_size` | 2 | Sentences on each side of a point for its window vector |
| `similarity_threshold` | 0.20 | Fixed floor threshold (0–1) |
| `percentile_threshold` | 25.0 | Dynamic percentile threshold; None = disabled |

---

## 11. Method 7 — HybridChunker ★ Recommended Default ★

**Complexity:** O(n) — strictly linear in document length.  
**Speed:** Typically < 5 ms for a 10-page PDF.  
**Best for:** Everything in this project. Specifically optimised for the output format of all six extractors.

### Why HybridChunker is the Right Default

| Property | HybridChunker | Notes |
|---|---|---|
| Speed | ⚡ < 5 ms | No ML, no regex search per sentence |
| Semantic quality | ★★★★☆ | Respects human-authored structure |
| Structure-aware | ✓ | Reads page/slide/heading markers |
| Size-safe | ✓ | Never exceeds `max_chunk_size` |
| Format coverage | ✓ | PDF, PPTX, DOCX, audio, URL, video |
| Dependencies | None | Pure Python standard library |
| Graph-RAG suitability | ★★★★★ | Provenance in every chunk |

For audio transcripts (no structure), it automatically falls back to paragraph grouping — which still works well because audio extractors insert paragraph breaks at natural pause points.

---

### Processing Pipeline (Detailed)

#### Phase 1 — Structural Split

```python
def _split_on_structural_markers(self, text):
    lines    = text.splitlines(keepends=True)   # preserve \n on each line
    sections = []
    current_marker     = ""
    current_body_lines = []
    
    for line in lines:
        stripped = line.strip()
        if _is_structural_marker(stripped):
            body = "".join(current_body_lines).strip()
            if body or current_marker:
                sections.append((current_marker, body))  # close current section
            current_marker     = stripped                 # open new section
            current_body_lines = []
        else:
            current_body_lines.append(line)              # accumulate body text
    
    # Final section flush
    body = "".join(current_body_lines).strip()
    if body or current_marker:
        sections.append((current_marker, body))
    
    if not sections:
        sections = [("", text)]      # no markers found → whole text = one section
    
    return sections                  # List[Tuple[marker_line, body_text]]
```

This produces: `[("", intro_text), ("--- Page 1 ---", page1_text), ("--- Page 2 ---", page2_text), …]`

#### Phase 2 — Paragraph Grouping

For each section, paragraphs are greedily merged — exactly like ParagraphChunker — but with the marker line prepended to the first chunk of that section (controlled by `keep_structural_markers`).

```python
for marker, body in section_blocks:
    prefix     = (marker + "\n") if (marker and self.keep_structural_markers) else ""
    paragraphs = _split_paragraphs(body) if body.strip() else []
    
    current_paras = []
    current_len   = len(prefix)
    
    for para in paragraphs:
        will_exceed = (
            len(current_paras) >= self.max_paragraphs_per_chunk
            or (current_len + len(para)) > self.max_chunk_size
        )
        if will_exceed and current_paras:
            chunk_text = prefix + "\n\n".join(current_paras)
            raw_chunks.append(chunk_text.strip())
            prefix        = ""           # marker only in first chunk of section
            current_paras = []
            current_len   = 0
        
        current_paras.append(para)
        current_len += len(para)
    
    if current_paras:
        raw_chunks.append((prefix + "\n\n".join(current_paras)).strip())
```

#### Phases 3, 4, 5 — Delegated to `_finalise()`

The inherited method handles:
- Sentence-level overlap between chunks.
- Size ceiling enforcement (breaks any chunk still > `max_chunk_size`).
- Tiny-chunk merging (removes orphans < `min_chunk_size`).
- Character-offset computation for `char_start` / `char_end`.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `max_chunk_size` | 1024 | Hard char ceiling |
| `overlap_sentences` | 1 | Sentence overlap |
| `min_chunk_size` | 100 | Min chars |
| `max_paragraphs_per_chunk` | 3 | Paragraphs merged per chunk |
| `keep_structural_markers` | True | Include marker line (e.g. "SLIDE 3") in chunk text |

---

## 12. Convenience Function — `chunk_text()`

```python
def chunk_text(
    text:      str,
    source_id: str = "doc",
    method:    str = "hybrid",    # ← default
    metadata:  dict = None,
    **kwargs                      # passed to chunker constructor
) -> List[Dict]:
```

**Examples:**

```python
# Use default (HybridChunker) — recommended for this project
chunks = chunk_text(my_text, source_id="lecture_05")

# Use SemanticChunker with custom parameters
chunks = chunk_text(
    my_text,
    source_id="lecture_05",
    method="semantic",
    max_chunk_size=800,
    similarity_threshold=0.15
)

# Attach document metadata to every chunk
chunks = chunk_text(
    result["extracted_text"],     # from an extractor's result dict
    source_id=result["resource_name"],
    metadata={
        "filename": result["metadata"]["filename"],
        "source_type": result["metadata"]["source_type"],
    }
)
```

**Method name → class mapping:**

| method string | Class |
|---|---|
| `"hybrid"` ← default | `HybridChunker` |
| `"semantic"` | `SemanticChunker` |
| `"recursive"` | `RecursiveChunker` |
| `"paragraph"` | `ParagraphChunker` |
| `"sentence"` | `SentenceChunker` |
| `"sliding"` | `SlidingWindowChunker` |
| `"fixed"` | `FixedSizeChunker` |

---

## 13. ChunkAnalyser

A diagnostic helper for development and benchmarking.

```python
stats = ChunkAnalyser.analyse(chunks)
ChunkAnalyser.print_report(stats)
```

**Output fields:**

| Field | Type | Description |
|---|---|---|
| `count` | int | Total chunks |
| `total_chars` | int | Sum of all chunk character lengths |
| `total_tokens` | int | Sum of estimated token counts |
| `avg_chars` | float | Mean character length |
| `min_chars` | int | Shortest chunk |
| `max_chars` | int | Longest chunk |
| `avg_tokens` | float | Mean token count |
| `empty_chunks` | int | Number of chunks with zero-length text |
| `chunker` | str | Chunker name from metadata |

---

## 14. Choosing the Right Chunker

| Document type | Recommended chunker | Why |
|---|---|---|
| PDF (text-heavy, paged) | **HybridChunker** | Page markers = natural section boundaries |
| PPTX (slides + notes) | **HybridChunker** | Slide markers = node candidates |
| DOCX (headings + paragraphs) | **HybridChunker** | Heading markers + paragraph grouping |
| Audio transcript (plain prose) | **HybridChunker** or **SemanticChunker** | Hybrid falls back to paragraph grouping; Semantic detects topic shifts |
| Video transcript | **HybridChunker** | Timestamp sections act as structural markers |
| URL / web page | **HybridChunker** or **ParagraphChunker** | Web content is paragraph-structured |
| Any format, max recall | **SlidingWindowChunker** | Every sentence appears in multiple chunks |
| Any format, speed priority | **RecursiveChunker** or **FixedSizeChunker** | Linear time, no regex sentence splitting |
| Research / ablation | All methods via `chunk_text(method=...)` | Easy switching |

---

## 15. Output Schema

Every chunk returned by every chunker is a Python `dict` with exactly these keys:

```python
{
    "chunk_id":    "lecture_05_3",   # str  — "<source_id>_<index>"
    "text":        "Neural networks are computational models…",  # str
    "char_start":  1204,             # int  — offset in original text (inclusive)
    "char_end":    2187,             # int  — offset in original text (exclusive)
    "chunk_index": 3,                # int  — 0-based position in the output list
    "token_count": 87,               # int  — approximate word-level token count
    "metadata": {
        "chunker":      "HybridChunker",    # always present
        "filename":     "lecture_05.pdf",   # only if passed by caller
        "source_type":  "pdf",              # only if passed by caller
        # … any other fields the caller injected via `metadata=`
    }
}
```

---

## 16. Performance Notes

All benchmarks measured on a single CPU core (no GPU required).

| Method | 2,000-word doc | 20,000-word doc | 200,000-word doc |
|---|---|---|---|
| FixedSizeChunker | < 1 ms | < 5 ms | < 50 ms |
| SentenceChunker | < 2 ms | < 15 ms | < 150 ms |
| ParagraphChunker | < 1 ms | < 5 ms | < 50 ms |
| RecursiveChunker | < 2 ms | < 20 ms | < 200 ms |
| SlidingWindowChunker | < 3 ms | < 30 ms | < 300 ms |
| **HybridChunker** | **< 2 ms** | **< 15 ms** | **< 150 ms** |
| SemanticChunker | ~50 ms | ~500 ms | ~5,000 ms |

**SemanticChunker** is the slowest because it computes a TF-IDF matrix over all sentences. For very large documents (> 50,000 words), consider running it on individual sections rather than the full text.

For all other methods, chunking time is negligible compared to extraction time (OCR, Whisper transcription, etc.).

---

*End of CHUNKING_GUIDE.md*
