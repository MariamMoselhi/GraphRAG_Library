# embeddings/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEmbedding(ABC):
    """
    Abstract base class for all embedding backends.

    Every embedding model in your library should implement this interface.
    """

    def __init__(self, dimension: Optional[int] = None):
        self._dimension = dimension

    @property
    def dimension(self) -> Optional[int]:
        """
        Returns embedding dimensionality if known (e.g. 384, 768).
        Useful for FAISS/Chroma.
        """
        return self._dimension

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into a 2D numpy array of shape (n_texts, dim).
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            2D numpy array of shape (n_texts, dimension) with dtype float32
        """
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        """
        Convenience helper to encode a single text.
        Returns a 1D numpy vector of shape (dim,).
        
        Args:
            text: Single text string to encode
            
        Returns:
            1D numpy array of shape (dimension,)
        """
        return self.encode([text])[0]
    
    def encode_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts with optional progress bar.
        Default implementation just calls encode(), but subclasses
        can override for actual progress bar support.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            2D numpy array of embeddings
        """
        return self.encode(texts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension})"