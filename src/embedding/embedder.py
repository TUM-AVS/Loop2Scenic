"""
Main embedder that wraps embedding models.
"""

import logging
import torch
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
        
        # Check if any inputs contain videos (which require more memory)
        has_videos = any("video" in inp and inp.get("video") for inp in inputs)
        
        # For video inputs, process one at a time to avoid OOM
        # For text/image only, use batch_size
        effective_batch_size = 1 if has_videos else self.batch_size
        
        if has_videos:
            logger.info("Detected video inputs - processing one at a time to manage GPU memory")
        
        # Batch processing
        all_embeddings = []
        for i in range(0, len(inputs), effective_batch_size):
            batch = inputs[i:i + effective_batch_size]
            
            # Process batch
            embeddings = self.model.encode(batch)
            
            # Convert PyTorch tensors to CPU lists to free GPU memory immediately
            processed_embeddings = []
            for emb in embeddings:
                if isinstance(emb, torch.Tensor):
                    processed_embeddings.append(emb.cpu().tolist())
                elif hasattr(emb, 'tolist'):
                    processed_embeddings.append(emb.tolist())
                else:
                    processed_embeddings.append(emb)
            
            all_embeddings.extend(processed_embeddings)
            
            # Clear CUDA cache after each batch to free fragmented memory
            if torch.cuda.is_available() and has_videos:
                torch.cuda.empty_cache()
        
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

    def to(self, device: str):
        """Pass the device move command down to the underlying model provider."""
        if hasattr(self, 'model') and hasattr(self.model, 'to'):
            self.model.to(device)
            # If moving to CPU, clear cache immediately
            if device == 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.error(f"Provider {self.provider} model does not support .to() method")
        return self


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
