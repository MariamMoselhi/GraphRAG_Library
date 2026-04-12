from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chunking"))

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Chunk lives in chunking/chunk_base.py
# sys.path must include the chunking/ folder (handled by the project entry point)
from chunking.chunk_base import Chunk

# ---------------------------------------------------------------------------
# RetrievalResult — defined here since it belongs to the retrieval layer
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single result returned by any vector store search.

    Attributes
    ----------
    chunk : The matched Chunk object (text, offsets, metadata).
    score : Similarity score — cosine similarity for InMemoryVectorStore,
            inner-product score for FaissVectorStore (equivalent to cosine
            when embeddings are L2-normalised).
    """
    chunk: Chunk
    score: float

    def __repr__(self) -> str:
        preview = self.chunk.text[:60].replace("\n", " ")
        return (
            f"RetrievalResult(score={self.score:.4f}, "
            f"chunk_id={self.chunk.chunk_id}, preview='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Helper: get embedding stored in chunk.metadata["embedding"]
# ---------------------------------------------------------------------------

def _get_embedding(chunk: Chunk) -> np.ndarray:
    """
    Retrieve the embedding vector attached to a Chunk.

    The Chunk dataclass (chunk_base.py) has no dedicated embedding field,
    so embeddings are stored in chunk.metadata["embedding"] by the
    embedding phase before chunks are passed to the vector store.
    """
    emb = chunk.metadata.get("embedding")
    if emb is None:
        raise ValueError(
            f"Chunk(chunk_id={chunk.chunk_id}) has no embedding in metadata. "
            "Make sure to run the embedding phase before adding chunks to the store."
        )
    return np.asarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    """
    Abstract base class for all vector stores (FAISS, in-memory, etc.).
    Stores Chunk objects + their embeddings and supports similarity search.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the store.
        Each chunk must have its embedding stored in chunk.metadata["embedding"].
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Search the most similar chunks for a given query embedding.
        Returns a list of RetrievalResult (chunk + score), sorted by
        descending score.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# In-memory implementation (numpy cosine similarity)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(BaseVectorStore):
    """
    Simple in-memory vector store using numpy.
    Useful for small corpora and unit tests before switching to FAISS.
    """

    def __init__(self):
        self._embeddings: Optional[np.ndarray] = None  # shape (n, dim)
        self._chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        new_embs = [_get_embedding(c) for c in chunks]
        new_embs = np.vstack(new_embs).astype(np.float32)

        if self._embeddings is None:
            self._embeddings = new_embs
        else:
            self._embeddings = np.vstack([self._embeddings, new_embs])

        self._chunks.extend(chunks)

    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        docs  = self._embeddings  # (n, dim)

        # cosine similarity
        dot   = np.dot(docs, query.T).reshape(-1)                        # (n,)
        norms = np.linalg.norm(docs, axis=1) * (np.linalg.norm(query) + 1e-12)
        sims  = dot / (norms + 1e-12)

        k       = min(k, len(self._chunks))
        top_idx = np.argsort(sims)[::-1][:k]

        return [
            RetrievalResult(chunk=self._chunks[int(i)], score=float(sims[int(i)]))
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._chunks)