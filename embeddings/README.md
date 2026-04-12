# GraphRAG Embeddings Module — Structure & Logic Reference

> This document describes the three files that make up the embeddings module,
> what each one is responsible for, how they connect to each other, and the
> logic behind every significant design choice.

---

## Overview — The Three-File Layout

The embeddings module is split into three files with clear, non-overlapping
responsibilities. The dependency arrow points downward: nothing in a lower
layer imports from a higher one.

```
embeddings/__init__.py     ← the ONLY file users import from
        │
        └── embeddings/huggingFace.py   ← concrete HuggingFace implementation
                │
                └── embeddings/base.py  ← BaseEmbedding ABC (lowest layer)
```

| File | Exposed to users? | One-line role |
|---|---|---|
| `base.py` | `BaseEmbedding` only | Abstract contract + shared helpers |
| `huggingFace.py` | `HuggingFaceEmbedding` | Sentence Transformers implementation |
| `__init__.py` | Yes (everything) | Public API, re-exports |

---

## 1. `base.py`

### Role
Lowest layer of the stack. Contains exactly one class: `BaseEmbedding`.
No other module in the package imports it except `huggingFace.py`.

### Why a separate file?
The abstract interface is completely independent of any specific model
backend. Isolating it here means a new backend (OpenAI, Cohere, etc.) can
be added by creating one new file and inheriting from `BaseEmbedding` —
without touching anything else.

### Imports
```python
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np          # standard scientific stack — no internal dependencies
```

### Class: `BaseEmbedding`

Abstract base class that every embedding backend must implement.

```python
class BaseEmbedding(ABC):
    def __init__(self, dimension: Optional[int] = None):
        self._dimension = dimension
```

#### Property: `dimension -> Optional[int]`
Returns the embedding dimensionality (e.g. 384, 768) if known.
Used by downstream phases — FAISS index creation requires the dimension
at construction time (`faiss.IndexFlatIP(dim)`).

```python
@property
def dimension(self) -> Optional[int]:
    return self._dimension
```

#### Abstract method: `encode(texts) -> np.ndarray`
The only method subclasses **must** implement.

```python
@abstractmethod
def encode(self, texts: List[str]) -> np.ndarray:
    # Returns 2D array of shape (n_texts, dim), dtype float32
    raise NotImplementedError
```

**Why float32?** FAISS requires float32 arrays. Enforcing this at the base
level means every backend naturally produces FAISS-compatible output without
any casting in downstream code.

#### Concrete helper: `encode_one(text) -> np.ndarray`
Convenience wrapper — delegates to `encode()` and unwraps the single row.

```python
def encode_one(self, text: str) -> np.ndarray:
    return self.encode([text])[0]   # shape (dim,) — 1D vector
```

**Why not abstract?** Any subclass that implements `encode()` automatically
gets a correct `encode_one()` for free. Subclasses can override it for
performance if needed, but the default is always correct.

#### Concrete helper: `encode_batch(texts, show_progress) -> np.ndarray`
Encodes with an optional progress bar. Default implementation just calls
`encode()` — subclasses override to add actual progress bar support.

```python
def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
    return self.encode(texts)
```

**Why in the base class?** Having a named `encode_batch()` in the interface
allows downstream code (the embedding phase, the test suite) to always call
`encode_batch()` with `show_progress=True/False` without knowing which
backend is in use.

#### `__repr__`
```python
def __repr__(self) -> str:
    return f"{self.__class__.__name__}(dimension={self.dimension})"
```

---

## 2. `huggingFace.py`

### Role
The concrete implementation of `BaseEmbedding` using the
`sentence-transformers` library. This is the default backend used across
the entire GraphRAG pipeline.

### Why a separate file?
`sentence-transformers` is a heavyweight optional dependency. Keeping it
isolated means the base interface and any future lightweight backends can
be imported without triggering the `sentence-transformers` import.

### Imports
```python
from typing import List, Optional
import warnings
import numpy as np
from .base import BaseEmbedding                      # one internal import
from sentence_transformers import SentenceTransformer  # guarded by try/except
```

The `sentence-transformers` import is wrapped in a `try/except ImportError`
at module level — if the package is missing, a clear installation message is
raised immediately rather than a cryptic `AttributeError` later.

### Recommended models

| Model | Dim | Speed | Quality | Use case |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast | Good | Default — best speed/quality balance |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Slower | Better | Higher-quality retrieval |
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good | Alternative fast option |
| `BAAI/bge-base-en-v1.5` | 768 | Slower | Better | Alternative quality option |

### Class: `HuggingFaceEmbedding`

```python
class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str  = "sentence-transformers/all-MiniLM-L6-v2",
        device    : Optional[str] = None,    # None = auto-detect (CPU / CUDA / MPS)
        normalize : bool = True,             # L2-normalise output vectors
        batch_size: int  = 32,               # sentences per forward pass
    ):
```

**Constructor flow:**
```
1. Store parameters
2. Load SentenceTransformer(model_name, device=device)
3. Call model.get_sentence_embedding_dimension() → dim
4. Call super().__init__(dimension=dim)   ← sets self._dimension
5. Print confirmation with dim + device
```

**Why `normalize=True` by default?**
L2-normalised vectors allow cosine similarity to be computed as a plain dot
product: `sim(a, b) = dot(a, b)` when `||a|| = ||b|| = 1`. This is what
`FaissVectorStore` (using `IndexFlatIP`) relies on. If `normalize=False`,
the inner product is no longer cosine similarity and retrieval scores become
meaningless.

---

#### `encode(texts) -> np.ndarray`

The primary method. Handles invalid inputs gracefully before calling the model.

**Invalid-input handling:**
```
for each text in texts:
    if text is None, not a str, or whitespace-only:
        warn → will use zero vector
        skip from valid_texts
    else:
        add to valid_texts, record original index

if no valid texts:
    return np.zeros((len(texts), dim), dtype=float32)

embeddings = model.encode(valid_texts, ...)

if some texts were invalid:
    build full_embeddings = zeros((len(texts), dim))
    fill valid positions from embeddings
    return full_embeddings

return embeddings
```

**Why zero vectors for invalid inputs instead of raising?**
In a large corpus, a single malformed chunk should not crash the entire
embedding phase. The zero vector is a known sentinel — downstream code can
filter `(matrix == 0).all(axis=1)` to detect and discard bad rows if needed.

**Model call parameters:**
```python
self.model.encode(
    valid_texts,
    batch_size          = self.batch_size,
    convert_to_numpy    = True,              # avoid torch tensor overhead
    normalize_embeddings= self.normalize,    # L2-normalise inside the model
    show_progress_bar   = False,             # use encode_batch() for progress
)
```

**dtype enforcement:**
```python
if embeddings.dtype != np.float32:
    embeddings = embeddings.astype(np.float32)
# FAISS requirement — always float32.
```

---

#### `encode_batch(texts, show_progress) -> np.ndarray`

Overrides the base class default. Calls `model.encode()` directly with
`show_progress_bar=show_progress`. Does **not** apply the invalid-input
handling from `encode()` — it is designed for clean, pre-validated batches
where performance matters (e.g. embedding thousands of chunks).

```python
embeddings = self.model.encode(
    texts,
    batch_size          = self.batch_size,
    convert_to_numpy    = True,
    normalize_embeddings= self.normalize,
    show_progress_bar   = show_progress,
)
```

**When to use `encode()` vs `encode_batch()`:**

| Method | Input validation | Progress bar | Best for |
|---|---|---|---|
| `encode()` | ✓ (zero vectors for bad inputs) | ✗ | Mixed / untrusted input |
| `encode_batch()` | ✗ | ✓ optional | Large clean batches (chunk embedding phase) |
| `encode_one()` | via `encode()` | ✗ | Single query vectors |

---

#### `similarity(emb1, emb2) -> float`

Computes cosine similarity between two 1D embedding vectors.

```python
if self.normalize:
    return float(np.dot(emb1, emb2))
    # When both vectors are L2-normalised, dot product == cosine similarity.
    # O(dim) operation — no norm computation needed.

# Fallback for non-normalised embeddings:
norm1 = np.linalg.norm(emb1)
norm2 = np.linalg.norm(emb2)
if norm1 == 0 or norm2 == 0:
    return 0.0
return float(np.dot(emb1, emb2) / (norm1 * norm2))
```

**Note:** The zero-norm check uses `== 0` (float comparison). This is
acceptable here because a true zero vector only occurs for invalid inputs
(see `encode()` above), not from rounding. A near-zero norm from a valid
text would still produce a valid similarity score.

#### `__repr__`
```python
f"HuggingFaceEmbedding(model='{self.model_name}', dim={self.dimension}, device='{self.model.device}')"
```

---

## 3. `__init__.py`

### Role
Public API of the embeddings package. Re-exports the two classes users
need. Nothing else is exposed.

```python
from .base        import BaseEmbedding
from .huggingFace import HuggingFaceEmbedding

__all__ = [
    "BaseEmbedding",
    "HuggingFaceEmbedding",
]
```

**Why re-export `BaseEmbedding`?**
Type annotations in downstream modules (`vectordb/base.py`,
`embedding_test.py`) use `BaseEmbedding` for `isinstance()` checks and
function signatures. Re-exporting it here means they can write
`from embeddings import BaseEmbedding` rather than reaching into `base.py`
directly.

---

## Embedding Convention for the Vector Store

The `Chunk` dataclass (`chunking/chunk_base.py`) has no dedicated embedding
field. Embeddings are attached via `chunk.metadata["embedding"]` by the
embedding phase before chunks are passed to any vector store:

```python
embeddings = model.encode_batch([c.text for c in chunks], show_progress=True)
for chunk, emb in zip(chunks, embeddings):
    chunk.metadata["embedding"] = emb
```

`vectordb/base.py` reads them back with `_get_embedding(chunk)`:
```python
emb = chunk.metadata.get("embedding")
if emb is None:
    raise ValueError(f"Chunk(chunk_id={chunk.chunk_id}) has no embedding in metadata.")
return np.asarray(emb, dtype=np.float32)
```

---

## Import Chain Summary

```
User code
    │
    │   from embeddings import BaseEmbedding, HuggingFaceEmbedding
    ▼
embeddings/__init__.py
    │   from .base        import BaseEmbedding
    │   from .huggingFace import HuggingFaceEmbedding
    ▼
embeddings/huggingFace.py
    │   from .base import BaseEmbedding
    ▼
embeddings/base.py
        from abc import ABC, abstractmethod
        import numpy as np    ← only standard/scientific stack
```

Each layer only imports from the layer directly below it.
No circular imports are possible because the arrow is strictly one-directional.

---

## What the Test File Should Import

```python
# Everything the test file needs comes from a single import:
from embeddings import BaseEmbedding, HuggingFaceEmbedding

# For chunk-level embedding tests, also import from the chunking module:
from chunking import chunk_text, Chunk

# For vector store tests, import from the vectordb module:
from vectordb import FaissVectorStore, InMemoryVectorStore, RetrievalResult
```

The embedding phase glues these three modules together:
```
chunking  →  List[Chunk]
                │
                ▼
embeddings  →  np.ndarray  →  chunk.metadata["embedding"]
                                        │
                                        ▼
                              vectordb  →  similarity search
```