from .base import BaseEmbeddingModel
from .providers import Qwen3VLEmbedding

__all__ = ["BaseEmbeddingModel", "Qwen3VLEmbedding", "get_embedder"]

def get_embedder(provider: str, **kwargs) -> BaseEmbeddingModel:
    """
    Get an embedder based on the provider.
    """
    if provider == "qwen3vl":
        return Qwen3VLEmbedding(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: qwen3vl"
        )