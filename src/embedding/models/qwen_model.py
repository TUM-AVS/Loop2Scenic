"""
Qwen embedding model (Alibaba Cloud).
"""

import logging
from typing import List

try:
    from dashscope import TextEmbedding
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

from .base_model import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class QwenEmbedder(BaseEmbeddingModel):
    """Qwen (Alibaba Cloud) embedding model."""

    DIMENSIONS = {
        "text-embedding-v1": 1536,
        "text-embedding-v2": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-v2", api_key: str = None, **kwargs):
        """
        Initialize Qwen embedder.
        
        Args:
            model_name: Qwen model name
            api_key: Alibaba Cloud API key
            **kwargs: Additional arguments
        """
        if not QWEN_AVAILABLE:
            raise ImportError(
                "DashScope library not installed. "
                "Install with: pip install dashscope"
            )
        
        self.model_name = model_name
        self._dimension = self.DIMENSIONS.get(model_name, 1536)
        self.api_key = api_key
        
        logger.info(f"Initialized Qwen embedder: {model_name}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        if not texts:
            return []
        
        logger.debug(f"Encoding {len(texts)} texts with Qwen {self.model_name}")
        
        response = TextEmbedding.call(
            model=self.model_name,
            input=texts,
            api_key=self.api_key
        )
        
        embeddings = [item['embedding'] for item in response.output['embeddings']]
        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
