"""
LLM model implementations.
"""

import logging
from typing import List, Dict, Optional

from .base import BaseLLMModel

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


class AnthropicModel(BaseLLMModel):
    """Anthropic Claude model."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize Anthropic model.
        
        Args:
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic client parameters
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Anthropic(**kwargs)
        
        logger.info(f"Initialized Anthropic model: {model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat with Anthropic model."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Chatting with {self._model_name}")
        
        # Anthropic uses different message format
        # Convert from standard format to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = content
            else:
                # Anthropic uses "user" and "assistant" roles
                anthropic_role = "user" if role == "user" else "assistant"
                anthropic_messages.append({"role": anthropic_role, "content": content})
        
        response = self.client.messages.create(
            model=self._model_name,
            messages=anthropic_messages,
            system=system_message if system_message else None,
            temperature=temperature,
            max_tokens=max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
        )
        
        return response.content[0].text

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


class GeminiModel(BaseLLMModel):
    """Google Gemini model."""

    def __init__(
        self,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 512,
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
