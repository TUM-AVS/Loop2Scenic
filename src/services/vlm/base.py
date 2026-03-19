"""
Base class for VLM models.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional


class BaseVLMModel(ABC):
    """Abstract base class for Vision Language Models."""

    @abstractmethod
    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Chat with the VLM using multimodal inputs.
        
        Args:
            text: Text input (description, question, etc.)
            image: Path to image file
            video: Path to video file
            instruction: Optional instruction text
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response string
        """
        pass

    @abstractmethod
    def chat_with_content(
        self, 
        contents: Any, 
        system_instruction: Optional[str] = None, 
        **kwargs
    ) -> str:
        """
        Chat with the VLM using formatted content inputs.

        Args:
            contents: The content to be formatted and sent to the VLM.
            system_instruction: Optional system instruction to be sent to the VLM.
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
