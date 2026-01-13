"""
Google Gemini embedding model.
"""

import logging
from typing import List

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .base_model import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class GeminiEmbedder(BaseEmbeddingModel):
    """Google Gemini embedding model."""

    DIMENSIONS = {
        "models/embedding-001": 768,
        "models/text-embedding-004": 768,
    }

    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None, **kwargs):
        """
        Initialize Gemini embedder.
        
        Args:
            model_name: Gemini model name
            api_key: Google API key
            **kwargs: Additional arguments
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.model_name = model_name
        self._dimension = self.DIMENSIONS.get(model_name, 768)
        
        if api_key:
            genai.configure(api_key=api_key)
        
        logger.info(f"Initialized Gemini embedder: {model_name}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        if not texts:
            return []
        
        logger.debug(f"Encoding {len(texts)} texts with Gemini {self.model_name}")
        
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        
        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
