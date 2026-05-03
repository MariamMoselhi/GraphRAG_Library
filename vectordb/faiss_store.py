from __future__ import annotations

import sys
import pickle
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

    def __init__(self, dim: int, verbose: bool = True):
        self.dim     = dim
        self.verbose = verbose
        self.index   = faiss.IndexFlatIP(int(dim))
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

    # Persistence

    def save(self, directory: str) -> None:
        """
        Save the FAISS index and chunk list to disk.

        Creates two files inside {directory}:
          faiss.index    — FAISS binary index (vectors only)
          faiss_meta.pkl — pickled _chunks list + dim integer

        Call this once after add_chunks() at ingestion time so the
        store can be restored on the next program start without
        re-embedding every chunk.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "faiss_meta.pkl", "wb") as f:
            pickle.dump(
                {"chunks": self._chunks, "dim": self.dim},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, directory: str) -> "FaissVectorStore":
        """
        Restore a FaissVectorStore previously saved with save().

        Args
        ----
        directory : Directory that contains faiss.index and faiss_meta.pkl.

        Returns
        -------
        Fully restored FaissVectorStore ready for similarity_search_by_vector().

        Raises
        ------
        FileNotFoundError if faiss.index is not found in directory.
        """
        path = Path(directory)

        if not (path / "faiss.index").exists():
            raise FileNotFoundError(
                f"No faiss.index found in '{directory}'. "
                "Call save() before load()."
            )

        with open(path / "faiss_meta.pkl", "rb") as f:
            data = pickle.load(f)

        store         = cls(dim=int(data["dim"]))
        store.index   = faiss.read_index(str(path / "faiss.index"))
        store._chunks = data["chunks"]
        return store