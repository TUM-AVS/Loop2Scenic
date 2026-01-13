"""
HuggingFace sentence-transformers embedding model.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from .base_model import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(BaseEmbeddingModel):
    """HuggingFace sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize HuggingFace embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu', 'cuda', 'mps')
            **kwargs: Additional arguments for SentenceTransformer
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading HuggingFace model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded, dimension: {self._dimension}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        if not texts:
            return []
        
        logger.debug(f"Encoding {len(texts)} texts with {self.model_name}")
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
