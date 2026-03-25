import logging
from typing import List, Dict

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)

class GeminiModel(BaseLLMModel):
    """Google Gemini model."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 512000,
        **kwargs
    ):
        """
        Initialize Gemini model.
        
        Args:
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini client parameters
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure Gemini
        if 'api_key' in kwargs:
            genai.configure(api_key=kwargs.pop('api_key'))
        
        self.model = genai.GenerativeModel(model)
        
        logger.info(f"Initialized Gemini model: {model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat with Gemini model."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Chatting with {self._model_name}")
        
        # Gemini uses different format - combine messages into a single prompt
        # For simplicity, we'll use the last user message or combine all
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts)
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
        )
        
        return response.text

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
