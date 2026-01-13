"""
OpenAI LLM model.
"""

import logging
from typing import Optional

from openai import OpenAI

from .base_model import BaseLLMModel

logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseLLMModel):
    """OpenAI GPT model."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize OpenAI generator.
        
        Args:
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
        """
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(**kwargs)
        
        logger.info(f"Initialized OpenAI model: {model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Generating with {self._model_name}")
        
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
