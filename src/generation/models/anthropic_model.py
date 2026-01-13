"""
Anthropic Claude model.
"""

import logging
from typing import Optional

from anthropic import Anthropic

from .base_model import BaseLLMModel

logger = logging.getLogger(__name__)


class AnthropicGenerator(BaseLLMModel):
    """Anthropic Claude model."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize Anthropic generator.
        
        Args:
            model: Model name (claude-3-opus, claude-3-sonnet, etc.)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters
        """
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Anthropic(**kwargs)
        
        logger.info(f"Initialized Anthropic model: {model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Generating with {self._model_name}")
        
        response = self.client.messages.create(
            model=self._model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
