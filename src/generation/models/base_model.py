"""
Base class for LLM models.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMModel(ABC):
    """Abstract base class for LLM models."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
