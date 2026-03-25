"""
Base class for embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        f"""
        Encode arbitrary inputs into embeddings.
        
        Args:
            inputs: List of inputs to encode. Each input should be a dictionary with "text", "image", "video" keys.
            
        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        pass