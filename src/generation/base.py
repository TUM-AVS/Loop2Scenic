"""
Base class for LLM generators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from .prompts import PromptTemplate


class BaseGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context_documents: List[Document],
        prompt_template: Optional[PromptTemplate] = None,
        **kwargs
    ) -> str:
        """
        Generate a response based on query and context.
        
        Args:
            query: User query
            context_documents: Retrieved context documents
            prompt_template: Custom prompt template
            **kwargs: Additional arguments for the prompt
            
        Returns:
            Generated response string
        """
        pass

    @abstractmethod
    def generate_with_metadata(
        self,
        query: str,
        context_documents: List[Document],
        prompt_template: Optional[PromptTemplate] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with additional metadata.
        
        Args:
            query: User query
            context_documents: Retrieved context documents
            prompt_template: Custom prompt template
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with response and metadata
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
