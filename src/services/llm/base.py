"""
Base class for LLM models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLLMModel(ABC):
    """Abstract base class for LLM models."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat with the LLM using messages format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello"}]
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response string
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        Optional metrics API implemented by concrete providers.
        Returns cumulative call/time/token stats when available.
        """
        return {}
