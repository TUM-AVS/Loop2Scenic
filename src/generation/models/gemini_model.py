"""
Google Gemini model.
"""

import logging
from typing import Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .base_model import BaseLLMModel

logger = logging.getLogger(__name__)


class GeminiGenerator(BaseLLMModel):
    """Google Gemini model."""

    def __init__(
        self,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemini generator.
        
        Args:
            model: Model name (gemini-pro, gemini-pro-vision, etc.)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            api_key: Google API key
            **kwargs: Additional parameters
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model)
        
        logger.info(f"Initialized Gemini model: {model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Generating with {self._model_name}")
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
