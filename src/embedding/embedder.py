"""
Main embedder that wraps embedding models.
"""

import logging
from typing import List, Optional, Dict, Any

from langchain.embeddings.base import Embeddings

from .base import BaseEmbedder
from .models.base_model import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class LangChainEmbeddingsWrapper(Embeddings):
    """Wrapper to make our models compatible with LangChain."""
    
    def __init__(self, model: BaseEmbeddingModel):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0]


class Embedder(BaseEmbedder):
    """Main embedder that uses embedding models."""

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize embedder.
        
        Args:
            provider: Model provider ('huggingface', 'openai', 'gemini', 'qwen')
            model_name: Model name (provider-specific)
            device: Device to run on ('cpu', 'cuda', 'mps') - for HuggingFace
            batch_size: Batch size for encoding
            **kwargs: Additional model-specific arguments
        """
        self.provider = provider.lower()
        self.batch_size = batch_size
        
        # Initialize the appropriate model
        if self.provider == "huggingface":
            pass
            # model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            # self.model = HuggingFaceEmbedder(
            #     model_name=model_name,
            #     device=device,
            #     **kwargs
            # )
            
        elif self.provider == "openai":
            pass
            # model_name = model_name or "text-embedding-ada-002"
            # self.model = OpenAIEmbedder(
            #     model_name=model_name,
            #     **kwargs
            # )
            
        elif self.provider == "gemini":
            pass
            # from .models.gemini_model import GeminiEmbedder
            # model_name = model_name or "models/embedding-001"
            # self.model = GeminiEmbedder(
            #     model_name=model_name,
            #     **kwargs
            # )
            
        elif self.provider == "qwen":
            from .models.qwen_model import QwenEmbedder
            # Qwen requires model_path for offline models
            if "model_path" not in kwargs:
                raise ValueError(
                    "model_path is required for Qwen offline models. "
                    "Example: model_path='./models/Qwen3-VL-Embedding'"
                )
            self.model = QwenEmbedder(
                model_path=kwargs.pop("model_path"),
                model_name=model_name,
                **kwargs
            )
            
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: huggingface, openai, gemini, qwen"
            )
        
        # Create LangChain wrapper
        self._embeddings = LangChainEmbeddingsWrapper(self.model)
        
        logger.info(f"Embedder initialized with {provider} model")

    def embed_documents(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """Encode documents in batches."""
        if not inputs:
            return []
        
        logger.info(f"Encoding {len(inputs)} documents")
        
        # Batch processing
        all_embeddings = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            embeddings = self.model.encode(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings

    def embed_query(self, input: Dict[str, Any]) -> List[float]:
        """Encode a single query."""
        return self.model.encode([input])[0]

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.dimension

    @property
    def embeddings(self):
        """Get LangChain-compatible embeddings object."""
        return self._embeddings


def get_embedder(
    provider: str = "huggingface",
    model_name: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 32,
    **kwargs
) -> BaseEmbedder:
    """
    Factory function to create an embedder.
    
    Args:
        provider: Model provider ('huggingface', 'openai', 'gemini', 'qwen')
        model_name: Model name
        device: Device ('cpu', 'cuda', 'mps')
        batch_size: Batch size
        **kwargs: Additional arguments
        
    Returns:
        Embedder instance
    """
    return Embedder(
        provider=provider,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        **kwargs
    )
