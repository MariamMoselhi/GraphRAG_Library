from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chunking"))

from typing import List

import numpy as np

# Chunk lives in chunking/chunk_base.py
from chunking.chunk_base import Chunk
from .base import BaseVectorStore, RetrievalResult, _get_embedding

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError(
        "faiss is required for FaissVectorStore.\n"
        "Install it with: pip install faiss-cpu"
    ) from e


class FaissVectorStore(BaseVectorStore):
    """
    FAISS-based vector store using IndexFlatIP (inner product).

    Works as cosine similarity when embeddings are L2-normalised
    (HuggingFaceEmbedding uses normalize=True by default, so this
    is already handled by the embedding phase).

    Args
    ----
    dim : Embedding dimensionality (must match model.dimension).
    """

    def __init__(self, dim: int):
        self.dim    = dim
        self.index  = faiss.IndexFlatIP(dim)
        self._chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        embs = []
        for c in chunks:
            emb = _get_embedding(c)
            if emb.shape[-1] != self.dim:
                raise ValueError(
                    f"Chunk(chunk_id={c.chunk_id}) embedding dim={emb.shape[-1]} "
                    f"does not match index dim={self.dim}."
                )
            embs.append(emb)

        if not embs:
            return

        embs_matrix = np.vstack(embs).astype(np.float32)
        self.index.add(embs_matrix)
        self._chunks.extend(chunks)

    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        if len(self._chunks) == 0:
            return []

        q = query_embedding.astype(np.float32).reshape(1, -1)
        if q.shape[-1] != self.dim:
            raise ValueError(
                f"Query embedding dim={q.shape[-1]} does not match index dim={self.dim}."
            )

        k = min(k, len(self._chunks))
        distances, indices = self.index.search(q, k)  # (1, k), (1, k)

        results: List[RetrievalResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[int(idx)],
                    score=float(score),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)