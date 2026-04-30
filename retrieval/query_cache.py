"""
Semantic LRU cache for the retrieval pipeline.

Why cosine similarity, not exact string matching
-------------------------------------------------
"What is gradient descent?" and "Explain gradient descent" are semantically
identical — exact string matching would miss this and re-run all three
retrievers.  Cosine similarity between query embeddings correctly identifies
near-duplicate queries and returns cached results without any retriever calls.

Design
------
- Fixed-capacity LRU cache (default 128 entries).
- Cache key: query embedding vector.  Hit condition: cosine similarity ≥ threshold.
- Linear scan over cached embeddings (acceptable for ≤ 200 entries;
  for larger caches a FAISS flat index would be more efficient).
- Stores GradedResult objects — the full pipeline output, not just chunks.
  This means a cache hit is zero-cost: no retrievers, no reranker, no grader.
- Thread-safe via threading.Lock.
- Optional disk persistence: cache survives restarts (pickle of numpy arrays).

Threshold guidance
------------------
0.98 : Very tight — only catches near-verbatim paraphrases.
0.95 : Recommended default — catches most query reformulations.
0.90 : Looser — may over-generalise ("what is X" vs "how does X work").
"""
from __future__ import annotations

import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .retrieval_context import GradedResult


class QueryCache:
    """
    Semantic LRU cache for retrieval results.

    Args
    ----
    capacity        : Maximum number of entries (default 128).
    similarity_threshold : Cosine similarity cutoff for a cache hit (default 0.95).
    cache_file      : Optional path to persist cache across restarts.
    verbose         : Print cache hits/misses to console.
    """

    def __init__(
        self,
        capacity             : int   = 128,
        similarity_threshold : float = 0.95,
        cache_file           : Optional[str] = None,
        verbose              : bool  = True,
    ):
        self.capacity    = capacity
        self.threshold   = similarity_threshold
        self.cache_file  = Path(cache_file) if cache_file else None
        self.verbose     = verbose

        # OrderedDict: key = int (insertion index), value = (embedding, GradedResult, timestamp)
        self._cache  : OrderedDict[int, Tuple[np.ndarray, GradedResult, float]] = OrderedDict()
        self._counter : int          = 0
        self._lock    : threading.Lock = threading.Lock()

        # Stats
        self._hits   : int = 0
        self._misses : int = 0

        if self.cache_file and self.cache_file.exists():
            self._load()

    # Public API 

    def get(self, query_embedding: np.ndarray) -> Optional[GradedResult]:
        """
        Look up a cached result by embedding similarity.

        Returns the GradedResult if a similar query was cached, else None.
        Moves the hit entry to the end of the LRU order (most recently used).
        """
        if len(self._cache) == 0:
            self._misses += 1
            return None

        with self._lock:
            best_key  : Optional[int] = None
            best_sim  : float         = -1.0
            q_norm    = self._l2norm(query_embedding)

            for key, (cached_emb, _, _) in self._cache.items():
                sim = float(np.dot(q_norm, self._l2norm(cached_emb)))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_sim >= self.threshold and best_key is not None:
                # Move to end (most recently used)
                self._cache.move_to_end(best_key)
                _, result, _ = self._cache[best_key]
                self._hits += 1
                if self.verbose:
                    print(
                        f"  ★ Cache HIT  (cosine={best_sim:.4f} ≥ {self.threshold})  "
                        f"[hits={self._hits} misses={self._misses}]"
                    )
                return result

        self._misses += 1
        if self.verbose:
            print(
                f"  ○ Cache MISS (best_cosine={best_sim:.4f} < {self.threshold})  "
                f"[hits={self._hits} misses={self._misses}]"
            )
        return None

    def put(self, query_embedding: np.ndarray, result: GradedResult) -> None:
        """
        Store a retrieval result, evicting the LRU entry if at capacity.
        """
        with self._lock:
            if len(self._cache) >= self.capacity:
                # Evict the oldest (first) entry
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]

            self._cache[self._counter] = (
                query_embedding.copy(),
                result,
                time.time(),
            )
            self._cache.move_to_end(self._counter)
            self._counter += 1

        if self.cache_file:
            self._save()

    def invalidate(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._counter = 0
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()
        if self.verbose:
            print("  ○ Query cache invalidated")

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size"      : self.size,
            "capacity"  : self.capacity,
            "hits"      : self._hits,
            "misses"    : self._misses,
            "hit_rate"  : round(self.hit_rate, 3),
            "threshold" : self.threshold,
        }

    # Persistence 

    def _save(self) -> None:
        """Pickle the cache to disk (numpy arrays + GradedResult objects)."""
        try:
            tmp = self.cache_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({
                    "cache"   : dict(self._cache),
                    "counter" : self._counter,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self.cache_file)
        except Exception as e:
            if self.verbose:
                print(f"  Cache save failed: {e}")

    def _load(self) -> None:
        """Restore cache from disk."""
        try:
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)
            raw = data.get("cache", {})
            self._cache   = OrderedDict(
                (k, v) for k, v in sorted(raw.items(), key=lambda x: x[0])
            )
            self._counter = data.get("counter", len(self._cache))
            # Enforce capacity on load
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
            if self.verbose and self._cache:
                print(f"  ○ Query cache loaded ({len(self._cache)} entries)")
        except Exception as e:
            if self.verbose:
                print(f"  Cache load failed ({e}) — starting empty")

    # Utility

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        """Return L2-normalised vector (safe against zero vectors)."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v

    def __repr__(self) -> str:
        return (
            f"QueryCache(size={self.size}/{self.capacity}, "
            f"threshold={self.threshold}, hit_rate={self.hit_rate:.1%})"
        )