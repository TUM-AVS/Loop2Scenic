"""
Generator that uses centralized LLM service.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from .base import BaseGenerator
from .prompts import DEFAULT_QA_PROMPT, PromptTemplate
from ..llm import LLMService, get_llm_service

logger = logging.getLogger(__name__)


class Generator(BaseGenerator):
    """
    Generator that uses the centralized LLM service.
    
    Handles prompt template preparation and message formatting,
    then delegates to LLMService for actual LLM calls.
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize generator.
        
        Args:
            llm_service: Optional LLMService instance. If provided, other args are ignored.
            provider: Model provider ('openai', 'anthropic', 'gemini')
            model: Model name (provider-specific)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific arguments
        """
        if llm_service:
            self.llm_service = llm_service
        else:
            self.llm_service = get_llm_service(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        logger.info(f"Generator initialized with LLM service: {self.llm_service.model_name}")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.llm_service.model_name

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
        if not context_documents:
            logger.warning("No context documents provided")
            return "I don't have enough information to answer this question."
        
        # Format prompt using template
        template = prompt_template or DEFAULT_QA_PROMPT
        formatted_prompt = template.format_with_context(
            query=query,
            context_documents=context_documents,
            question=query,
            **kwargs
        )
        
        logger.info(f"Generating response for query: '{query[:50]}...'")
        
        # Use LLM service to generate response
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.llm_service.chat(messages, **kwargs)
        
        logger.info("Response generated successfully")
        return response

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
        response = self.generate(query, context_documents, prompt_template, **kwargs)
        
        sources = []
        for doc in context_documents:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "tags": doc.metadata.get("tags", []),
            }
            if source_info not in sources:
                sources.append(source_info)
        
        return {
            "query": query,
            "response": response,
            "sources": sources,
            "num_context_docs": len(context_documents),
            "model": self.llm_service.model_name
        }

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
        if len(queries) != len(context_documents_list):
            raise ValueError("Number of queries must match number of context document lists")
        
        logger.info(f"Generating responses for {len(queries)} queries")
        
        responses = [
            self.generate(query, context_docs, prompt_template)
            for query, context_docs in zip(queries, context_documents_list)
        ]
        
        logger.info(f"Generated {len(responses)} responses")
        return responses


def get_generator(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs
) -> BaseGenerator:
    """
    Factory function to create a generator.
    
    Args:
        provider: Model provider ('openai', 'anthropic', 'gemini')
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments
        
    Returns:
        Generator instance
    """
    return Generator(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
