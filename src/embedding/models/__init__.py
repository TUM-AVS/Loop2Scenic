"""
Embedding models.
"""

from .base_model import BaseEmbeddingModel
from .qwen_model import QwenEmbedder

__all__ = ["BaseEmbeddingModel", "QwenEmbedder"]
