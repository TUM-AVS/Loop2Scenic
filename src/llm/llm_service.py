"""
Centralized LLM service for chat functionality.
"""

import logging
from typing import List, Dict, Optional

from .base import BaseLLMModel
from .models import OpenAIModel, AnthropicModel, GeminiModel

logger = logging.getLogger(__name__)


class LLMService:
    """
    Centralized LLM service that provides a simple chat interface.
    
    This service handles LLM calls across ingestion, query, and generation parts.
    Template preparation and message formatting should be done by the caller.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize LLM service.
        
        Args:
            provider: Model provider ('openai', 'anthropic', 'gemini')
            model: Model name (provider-specific)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific arguments
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the appropriate model
        if self.provider == "openai":
            model = model or "gpt-3.5-turbo"
            self.model: BaseLLMModel = OpenAIModel(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        elif self.provider == "anthropic":
            model = model or "claude-3-sonnet-20240229"
            self.model: BaseLLMModel = AnthropicModel(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        elif self.provider == "gemini":
            model = model or "gemini-pro"
            self.model: BaseLLMModel = GeminiModel(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: openai, anthropic, gemini"
            )
        
        logger.info(f"LLMService initialized with {provider} model: {self.model.model_name}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat with the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Example: [
                         {"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": "Hello"}
                     ]
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response string
        """
        logger.debug(f"Chatting with {self.model.model_name}")
        response = self.model.chat(messages, **kwargs)
        logger.debug("Response received")
        return response
    
    def chat_simple(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simple chat interface with user and optional system message.
        
        Args:
            user_message: User's message
            system_message: Optional system message
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response string
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        
        return self.chat(messages, **kwargs)
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model.model_name


def get_llm_service(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs
) -> LLMService:
    """
    Factory function to create an LLM service.
    
    Args:
        provider: Model provider ('openai', 'anthropic', 'gemini')
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments
        
    Returns:
        LLMService instance
    """
    return LLMService(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
