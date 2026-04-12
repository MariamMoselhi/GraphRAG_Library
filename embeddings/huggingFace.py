# embeddings/huggingface.py
from __future__ import annotations

from typing import List, Optional
import warnings

import numpy as np

from .base import BaseEmbedding

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required for HuggingFaceEmbedding.\n"
        "Install it with: pip install sentence-transformers"
    ) from e


class HuggingFaceEmbedding(BaseEmbedding):
    """
    HuggingFace Sentence Transformers embedding model.
    
    Recommended models:
        - "sentence-transformers/all-MiniLM-L6-v2": Fast, 384 dim (RECOMMENDED)
        - "sentence-transformers/all-mpnet-base-v2": Better quality, 768 dim
        - "BAAI/bge-small-en-v1.5": Fast, 384 dim
        - "BAAI/bge-base-en-v1.5": Better quality, 768 dim
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to use ("cpu", "cuda", "mps", or None for auto)
        normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
        batch_size: Batch size for encoding
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device

        # Load model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get dimension and pass to parent
        dim = self.model.get_sentence_embedding_dimension()
        super().__init__(dimension=dim)
        
        print(f"✓ Model loaded (dimension: {dim}, device: {self.model.device})")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            2D numpy array of shape (len(texts), dimension)
        """
        if not texts:
            # Return empty array with correct shape
            return np.empty((0, self.dimension or 0), dtype=np.float32)

        # Filter out None and empty strings, but track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                warnings.warn(f"Empty or invalid text at index {i}, will use zero vector")
        
        # If all texts were invalid, return zero vectors
        if not valid_texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        # Encode valid texts
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        # Ensure float32 (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # If some texts were invalid, create full array with zeros
        if len(valid_texts) < len(texts):
            full_embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, valid_idx in enumerate(valid_indices):
                full_embeddings[valid_idx] = embeddings[i]
            return full_embeddings
        
        return embeddings
    
    def encode_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts with optional progress bar.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            2D numpy array of embeddings
        """
        if not texts:
            return np.empty((0, self.dimension or 0), dtype=np.float32)
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        return embeddings
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding (1D array)
            emb2: Second embedding (1D array)
            
        Returns:
            Similarity score (0 to 1 if normalized, -1 to 1 otherwise)
        """
        # If normalized, dot product = cosine similarity
        if self.normalize:
            return float(np.dot(emb1, emb2))
        
        # Otherwise compute cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0: #not the best way to compare float with  0 
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def __repr__(self) -> str:
        return (
            f"HuggingFaceEmbedding("
            f"model='{self.model_name}', "
            f"dim={self.dimension}, "
            f"device='{self.model.device}')"
        )