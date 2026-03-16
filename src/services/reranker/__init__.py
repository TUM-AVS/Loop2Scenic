from .base import BaseReranker
from .providers import QwenVLReranker

__all__ = ["BaseReranker", "QwenVLReranker", "get_reranker"]

def get_reranker(provider: str, **kwargs) -> BaseReranker:
    """
    Get a reranker based on the provider.
    """
    if provider == "qwen3vl":
        return QwenVLReranker(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: qwen3vl"
        )