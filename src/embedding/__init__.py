"""
Embedding generation module.
"""

from .base import BaseEmbedder
from .embedder import Embedder, get_embedder

__all__ = [
    "BaseEmbedder",
    "Embedder",
    "get_embedder",
]
