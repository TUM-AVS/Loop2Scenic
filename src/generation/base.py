"""
Base class for LLM generators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from langchain.schema import Document

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

    @abstractmethod
    def batch_generate(
        self,
        queries: List[str],
        context_documents_list: List[List[Document]],
        prompt_template: Optional[PromptTemplate] = None
    ) -> List[str]:
        """
        Generate responses for multiple queries in batch.
        
        Args:
            queries: List of queries
            context_documents_list: List of context document lists
            prompt_template: Custom prompt template
            
        Returns:
            List of generated responses
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
