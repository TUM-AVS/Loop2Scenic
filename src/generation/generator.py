"""
Main generator that wraps LLM models.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from .base import BaseGenerator
from .prompts import DEFAULT_QA_PROMPT, PromptTemplate
from .models import OpenAIGenerator, AnthropicGenerator
from .models.base_model import BaseLLMModel

logger = logging.getLogger(__name__)


class Generator(BaseGenerator):
    """Main generator that uses LLM models."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize generator.
        
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
            self.model = OpenAIGenerator(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
        elif self.provider == "anthropic":
            model = model or "claude-3-sonnet-20240229"
            self.model = AnthropicGenerator(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
        elif self.provider == "gemini":
            from .models.gemini_model import GeminiGenerator
            model = model or "gemini-pro"
            self.model = GeminiGenerator(
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
        
        logger.info(f"Generator initialized with {provider} model")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model.model_name

    def generate(
        self,
        query: str,
        context_documents: List[Document],
        prompt_template: Optional[PromptTemplate] = None,
        **kwargs
    ) -> str:
        """Generate a response based on query and context."""
        if not context_documents:
            logger.warning("No context documents provided")
            return "I don't have enough information to answer this question."
        
        # Format prompt
        template = prompt_template or DEFAULT_QA_PROMPT
        formatted_prompt = template.format_with_context(
            query=query,
            context_documents=context_documents,
            question=query,
            **kwargs
        )
        
        logger.info(f"Generating response for query: '{query[:50]}...'")
        
        # Call model.generate()
        response = self.model.generate(formatted_prompt, **kwargs)
        
        logger.info("Response generated successfully")
        return response

    def generate_with_metadata(
        self,
        query: str,
        context_documents: List[Document],
        prompt_template: Optional[PromptTemplate] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response with additional metadata."""
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
            "model": self.model.model_name
        }

    def batch_generate(
        self,
        queries: List[str],
        context_documents_list: List[List[Document]],
        prompt_template: Optional[PromptTemplate] = None
    ) -> List[str]:
        """Generate responses for multiple queries in batch."""
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
