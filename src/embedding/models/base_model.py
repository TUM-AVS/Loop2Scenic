"""
Base class for embedding models.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        pass
