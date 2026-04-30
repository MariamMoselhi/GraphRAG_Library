"""
BM25 keyword retriever for the GraphRAG pipeline.

BM25 (Best Match 25) is a probabilistic term-frequency ranking model that
substantially outperforms TF-IDF for keyword-heavy queries.  It is the
backbone of Elasticsearch and Lucene.

Scientific basis
----------------
BM25 score(q, D) = Σ_t IDF(t) · (tf(t,D) · (k1+1)) / (tf(t,D) + k1·(1 - b + b·|D|/avgdl))

  t    = query term
  IDF  = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  tf   = term frequency in document D
  |D|  = document length in tokens
  avgdl= average document length across corpus
  k1   = term-frequency saturation (default 1.5)
  b    = length normalisation (default 0.75)

We use rank-bm25 (pip install rank-bm25) which implements this exactly.
Fallback: if rank-bm25 is unavailable, we use a simple TF-IDF implementation.

Index is built at ingestion time and can be serialised to disk.

Usage
-----
    retriever = BM25Retriever()
    retriever.build(chunks)
    retriever.save("bm25_index")

    retriever = BM25Retriever.load("bm25_index")
    results   = retriever.search(keywords=["gradient", "descent"], top_k=10)
"""
from __future__ import annotations

import math
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

from .retrieval_context import PhaseStats, QueryRepresentation, RetrievedChunk, RetrievalTrace
from .retrieval_logger import RetrievalLogger

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# Stop words (same set as query_engine for consistency)
_STOP_WORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","have","has","had","do",
    "does","did","will","would","could","should","may","might","can","what",
    "which","who","this","that","it","how","why","when","where","not","no",
    "so","as","about","just","more","also","than","then","into","through",
}


def _tokenise(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, filter stop words ≥ 2 chars."""
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOP_WORDS]


class BM25Retriever:
    """
    BM25 keyword retriever with pre-built index and disk persistence.

    Args
    ----
    k1      : Term saturation parameter (default 1.5).
    b       : Length normalisation parameter (default 0.75).
    logger  : RetrievalLogger instance.
    verbose : Print phase details.
    """

    def __init__(
        self,
        k1      : float = 1.5,
        b       : float = 0.75,
        logger  : Optional[RetrievalLogger] = None,
        verbose : bool  = True,
    ):
        self.k1      = k1
        self.b       = b
        self.logger  = logger or RetrievalLogger(verbose=verbose)
        self.verbose = verbose

        self._bm25   = None        # rank_bm25.BM25Okapi or _FallbackBM25
        self._meta   : list = []   # parallel list: _meta[i] → chunk info


    def build(self, chunks: list) -> int:
        """
        Build the BM25 index from a list of Chunk objects.

        Args
        ----
        chunks : Chunk objects with .text, .chunk_id, .metadata attributes.

        Returns
        -------
        Number of documents indexed.
        """
        if not chunks:
            return 0

        t0       = time.time()
        tokenised = [_tokenise(c.text) for c in chunks]

        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(tokenised, k1=self.k1, b=self.b)
        else:
            self.logger.warn(
                "rank-bm25 not installed — using fallback TF-IDF BM25. "
                "pip install rank-bm25 for the full implementation."
            )
            self._bm25 = _FallbackBM25(tokenised, k1=self.k1, b=self.b)

        self._meta = [
            {
                "chunk_id": str(getattr(c, "chunk_id", i)),
                "text"    : c.text,
                "source"  : c.metadata.get("source", "unknown") if hasattr(c, "metadata") else "unknown",
                "metadata": dict(getattr(c, "metadata", {})),
            }
            for i, c in enumerate(chunks)
        ]

        elapsed = (time.time() - t0) * 1000
        if self.verbose:
            print(
                f"  BM25: indexed {len(chunks)} documents in {elapsed:.0f} ms "
                f"(avg tokens/doc={sum(len(t) for t in tokenised)//max(len(tokenised),1)})"
            )
        return len(chunks)


    def search(
        self,
        rep     : QueryRepresentation,
        top_k   : int                   = 10,
        trace   : Optional[RetrievalTrace] = None,
    ) -> List[RetrievedChunk]:
        """
        Run BM25 search using query keywords.

        Args
        ----
        rep   : QueryRepresentation with .keywords populated by query_engine.py.
        top_k : Maximum results to return.
        trace : RetrievalTrace to update.

        Returns
        -------
        List of RetrievedChunk sorted by descending BM25 score.
        """
        t0 = time.time()
        self.logger.phase_start("BM25 Retriever")

        if self._bm25 is None or not self._meta:
            self.logger.warn("BM25 index not built — call build() first")
            return []

        if not rep.keywords:
            self.logger.warn("No keywords provided — BM25 returning empty results")
            return []

        # BM25 expects a tokenised query
        query_tokens = []
        for kw in rep.keywords:
            query_tokens.extend(_tokenise(kw))

        if not query_tokens:
            return []

        # Get scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Build sorted results
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            m = self._meta[idx]
            results.append(RetrievedChunk(
                chunk_id  = m["chunk_id"],
                text      = m["text"],
                source    = m["source"],
                score     = score,
                retriever = "bm25",
                metadata  = m.get("metadata", {}),
            ))

        elapsed_ms = (time.time() - t0) * 1000

        if trace:
            trace.bm25_count = len(results)
            trace.add_phase(PhaseStats(
                phase_name   = "BM25 Retriever",
                elapsed_ms   = elapsed_ms,
                input_count  = len(query_tokens),
                output_count = len(results),
                notes        = f"keywords={rep.keywords[:5]}",
            ))

        self.logger.print_bm25_results(results)
        self.logger.phase_end("BM25 Retriever", count=len(results), elapsed_ms=elapsed_ms)
        return results


    def save(self, directory: str) -> None:
        """Pickle the BM25 index and metadata to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self._bm25, "meta": self._meta}, f)
        if self.verbose:
            print(f"  BM25: saved {len(self._meta)} docs to '{directory}/'")

    @classmethod
    def load(cls, directory: str, verbose: bool = True) -> "BM25Retriever":
        """Restore a BM25Retriever from disk."""
        path = Path(directory) / "bm25.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index found in '{directory}'")
        instance = cls(verbose=verbose)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance._bm25 = data["bm25"]
        instance._meta = data["meta"]
        if verbose:
            print(f"  BM25: loaded {len(instance._meta)} docs from '{directory}/'")
        return instance

    def __repr__(self) -> str:
        impl = "BM25Okapi" if _BM25_AVAILABLE else "FallbackBM25"
        return f"BM25Retriever(docs={len(self._meta)}, impl={impl})"



class _FallbackBM25:
    """
    Pure-Python BM25Okapi implementation used when rank-bm25 is not installed.
    Identical formula, slower for large corpora.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1       = k1
        self.b        = b
        self.corpus   = corpus
        self.n        = len(corpus)
        self.avgdl    = sum(len(d) for d in corpus) / max(self.n, 1)

        # Document frequency
        self.df: Counter = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

        # IDF
        self.idf: dict = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query: List[str]) -> List[float]:
        scores = []
        for doc in self.corpus:
            tf  = Counter(doc)
            dl  = len(doc)
            score = 0.0
            for term in query:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * num / den
            scores.append(score)
        return scores