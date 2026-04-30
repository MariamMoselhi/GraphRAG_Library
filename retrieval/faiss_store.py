"""
FAISS vector index wrapper for the GraphRAG retrieval pipeline.

Responsibilities
----------------
- Build a FAISS flat-L2 index from chunk embeddings at ingestion time.
- Serialize the index + chunk metadata to disk so it survives restarts.
- Expose a fast ANN search method used by vector_retriever.py.
- Provide an incremental add() method for adding new chunks without rebuilding.

Why FAISS Flat (IndexFlatIP) vs IVF or HNSW
--------------------------------------------
For corpora up to ~100K chunks, the exact flat index is fast enough
(<50 ms for top-10 on 50K vectors) and gives perfect recall.
IndexFlatIP uses inner product (= cosine similarity when vectors are
L2-normalised, which HuggingFaceEmbedding does by default).
For larger corpora, swap to IndexHNSWFlat for sub-millisecond search.

Usage
-----
    store = FAISSStore(embedding_model)
    store.build_from_chunks(chunks)   # at ingestion time
    store.save("faiss_index")

    store = FAISSStore.load("faiss_index", embedding_model)
    results = store.search(query_vector, top_k=10)
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from .retrieval_context import RetrievedChunk


class FAISSStore:
    """
    FAISS-backed vector store for chunk embeddings.

    Args
    ----
    embedding_model : Any object with an encode(texts: List[str]) → np.ndarray method.
                      Compatible with HuggingFaceEmbedding from your embeddings package.
    dimension       : Embedding dimensionality.  Auto-detected on first add if None.
    verbose         : Print build/search stats.
    """

    def __init__(
        self,
        embedding_model,
        dimension : Optional[int] = None,
        verbose   : bool          = True,
    ):
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is required.\n"
                "pip install faiss-cpu"
            )

        self.embedding_model = embedding_model
        self.dimension       = dimension
        self.verbose         = verbose

        # FAISS index — built lazily
        self._index          : Optional[faiss.IndexFlatIP] = None

        # Parallel list: _meta[i] corresponds to FAISS vector i
        self._meta           : List[Dict[str, Any]]        = []


    def build_from_chunks(self, chunks: list) -> int:
        """
        Build the FAISS index from a list of Chunk objects.

        Embeds all chunk texts in one batched call, then adds them to the index.

        Args
        ----
        chunks : List of Chunk objects with .text, .chunk_id, .metadata fields.

        Returns
        -------
        Number of vectors added.
        """
        if not chunks:
            return 0

        t0    = time.time()
        texts = [c.text for c in chunks]

        if self.verbose:
            print(f"  FAISS: embedding {len(texts)} chunks…", end=" ", flush=True)

        embeddings = self.embedding_model.encode(texts).astype(np.float32)
        self._ensure_l2norm(embeddings)

        self.dimension = embeddings.shape[1]
        self._index    = faiss.IndexFlatIP(self.dimension)
        self._index.add(embeddings)

        self._meta = [
            {
                "chunk_id" : str(getattr(c, "chunk_id", i)),
                "text"     : c.text,
                "source"   : c.metadata.get("source", "unknown"),
                "metadata" : dict(getattr(c, "metadata", {})),
            }
            for i, c in enumerate(chunks)
        ]

        elapsed = (time.time() - t0) * 1000
        if self.verbose:
            print(f"done — {len(texts)} vectors, dim={self.dimension}, {elapsed:.0f} ms")

        return len(texts)

    def add(self, chunk, embedding: Optional[np.ndarray] = None) -> None:
        """
        Incrementally add a single chunk to an existing index.

        If embedding is None, the chunk's text is embedded on the fly.
        Call save() after a batch of incremental adds.
        """
        if embedding is None:
            embedding = self.embedding_model.encode([chunk.text]).astype(np.float32)
        else:
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

        self._ensure_l2norm(embedding)

        if self._index is None:
            self.dimension = embedding.shape[1]
            self._index    = faiss.IndexFlatIP(self.dimension)

        self._index.add(embedding)
        self._meta.append({
            "chunk_id" : str(getattr(chunk, "chunk_id", len(self._meta))),
            "text"     : chunk.text,
            "source"   : chunk.metadata.get("source", "unknown") if hasattr(chunk, "metadata") else "unknown",
            "metadata" : dict(getattr(chunk, "metadata", {})),
        })


    def search(
        self,
        query_vector : np.ndarray,
        top_k        : int = 10,
    ) -> List[RetrievedChunk]:
        """
        Run ANN search and return ranked RetrievedChunk objects.

        Args
        ----
        query_vector : 1-D numpy array of the query embedding (will be L2-normalised).
        top_k        : Number of results to return.

        Returns
        -------
        List of RetrievedChunk sorted by descending cosine similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query_vector.astype(np.float32).reshape(1, -1)
        self._ensure_l2norm(q)

        k       = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[idx]
            results.append(RetrievedChunk(
                chunk_id  = m["chunk_id"],
                text      = m["text"],
                source    = m["source"],
                score     = float(score),
                retriever = "vector",
                metadata  = m.get("metadata", {}),
            ))

        return results


    def save(self, directory: str) -> None:
        """
        Save the FAISS index and metadata to disk.

        Creates:
          {directory}/faiss.index   — FAISS binary index
          {directory}/faiss_meta.pkl — chunk metadata list
        """
        if self._index is None:
            raise RuntimeError("No index to save — call build_from_chunks() first")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "faiss_meta.pkl", "wb") as f:
            pickle.dump({
                "meta"     : self._meta,
                "dimension": self.dimension,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            print(f"  FAISS: saved {self._index.ntotal} vectors to '{directory}/'")

    @classmethod
    def load(cls, directory: str, embedding_model, verbose: bool = True) -> "FAISSStore":
        """
        Load a previously saved FAISS index from disk.

        Args
        ----
        directory      : Directory containing faiss.index and faiss_meta.pkl.
        embedding_model: Embedding model (needed for incremental adds and search).
        """
        path = Path(directory)
        if not (path / "faiss.index").exists():
            raise FileNotFoundError(f"No faiss.index found in '{directory}'")

        store        = cls(embedding_model=embedding_model, verbose=verbose)
        store._index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "faiss_meta.pkl", "rb") as f:
            data = pickle.load(f)
        store._meta      = data["meta"]
        store.dimension  = data["dimension"]

        if verbose:
            print(
                f"  FAISS: loaded {store._index.ntotal} vectors "
                f"(dim={store.dimension}) from '{directory}/'"
            )
        return store


    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def __repr__(self) -> str:
        return (
            f"FAISSStore(vectors={self.total_vectors}, dim={self.dimension})"
        )


    @staticmethod
    def _ensure_l2norm(matrix: np.ndarray) -> None:
        """In-place L2 normalisation (row-wise). Safe against zero vectors."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        matrix /= norms