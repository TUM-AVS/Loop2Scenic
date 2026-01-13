"""
OpenAI embedding model.
"""

import logging
from typing import List

from openai import OpenAI

from .base_model import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbeddingModel):
    """OpenAI embedding model."""

    DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model_name: str = "text-embedding-ada-002", **kwargs):
        """
        Initialize OpenAI embedder.
        
        Args:
            model_name: OpenAI model name
            **kwargs: Additional arguments for OpenAI client
        """
        self.model_name = model_name
        self.client = OpenAI(**kwargs)
        self._dimension = self.DIMENSIONS.get(model_name, 1536)
        logger.info(f"Initialized OpenAI embedder: {model_name}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        if not texts:
            return []
        
        logger.debug(f"Encoding {len(texts)} texts with OpenAI {self.model_name}")
        
        # OpenAI API call
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
