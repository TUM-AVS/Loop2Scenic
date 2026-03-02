"""
Qwen embedding model (offline/local models only).
Uses Qwen inference engine for local model execution.
"""

import logging
from typing import List, Optional, Dict, Any

from src.embedding.models.qwen3_vl_model import Qwen3VLEmbedder

from .base_model import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class QwenEmbedder(BaseEmbeddingModel):
    """
    Qwen embedding model (offline/local).
    
    Uses local inference engine for fast, offline embedding generation.
    Requires local model files downloaded from HuggingFace or Qwen repository.
    """

    # Model dimensions
    DIMENSIONS = {
        "Qwen3-VL-Embedding-2B": 2048,
        "Qwen3-VL-Embedding-8B": 4096,
        "Qwen2.5-Embedding": 1024,
        "Qwen2-Embedding": 768,
    }

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Qwen embedder with local model.
        
        Args:
            model_path: Path to local model directory (required)
            model_name: Model identifier for dimension lookup (optional)
            dtype: Data type for inference (float16, float32, etc.)
            runner: Runner type (pooling for embeddings)
            **kwargs: Additional EngineArgs arguments
        """        
        if not model_path:
            raise ValueError("model_path is required for offline Qwen models")
        
        self.model_path = model_path
        self.model_name = model_name or "Qwen3-VL-Embedding"
        
        logger.info(f"Initializing Qwen engine with model: {model_path}")
        
        # Initialize embedding model
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            **kwargs
        )
        
        # Get dimension from model name or default
        self._dimension = self.DIMENSIONS.get(self.model_name, 1024)
        
        logger.info(f"Qwen embedder initialized, dimension: {self._dimension}")

    def encode(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """Encode texts into embeddings using local model."""
        if not inputs:
            return []
        
        logger.debug(f"Encoding {len(inputs)} inputs with Qwen engine")
        
        embeddings = self.embedder.process(inputs)
        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def to(self, device: str):
        """Move the underlying model to a specific device."""
        logger.info(f"Moving QwenEmbedder to {device}")
        # Check if the internal embedder has a .model attribute (common in these wrappers)
        self.embedder.to(device)
        return self