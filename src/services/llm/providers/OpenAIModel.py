import logging
from typing import List, Dict

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)

class OpenAIModel(BaseLLMModel):
    """OpenAI GPT model."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize OpenAI model.
        
        Args:
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI client parameters
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(**kwargs)
        
        logger.info(f"Initialized OpenAI model: {model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat with OpenAI model."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Chatting with {self._model_name}")
        
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
        )
        
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
