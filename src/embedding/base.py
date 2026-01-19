"""
Base class for embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_documents(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            inputs: List of inputs to embed
            
        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, input: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            input: Query input to embed
            
        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        pass

    @property
    @abstractmethod
    def embeddings(self):
        """
        Get the underlying embeddings object (for compatibility with LangChain).
        
        Returns:
            Embeddings object
        """
        pass
