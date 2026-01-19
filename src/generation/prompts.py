"""
Prompt templates for the RAG pipeline.
"""

from typing import Dict, List
from langchain_core.documents import Document


class PromptTemplate:
    """Template for formatting prompts."""

    def __init__(self, template: str):
        """
        Initialize prompt template.
        
        Args:
            template: Template string with placeholders
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """
        Format the template with provided arguments.
        
        Args:
            **kwargs: Arguments to fill in the template
            
        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)

    def format_with_context(
        self,
        query: str,
        context_documents: List[Document],
        **kwargs
    ) -> str:
        """
        Format template with query and context documents.
        
        Args:
            query: User query
            context_documents: Retrieved context documents
            **kwargs: Additional arguments
            
        Returns:
            Formatted prompt string
        """
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(context_documents)
        ])
        
        return self.format(query=query, context=context, **kwargs)


# Default QA prompt template
DEFAULT_QA_PROMPT = PromptTemplate(
    template="""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear and accurate answer based on the context
- If the context doesn't contain enough information, say so
- Cite the relevant parts of the context when appropriate
- Be concise but comprehensive

Answer:"""
)


# Prompt with source citation
CITATION_QA_PROMPT = PromptTemplate(
    template="""You are a helpful AI assistant. Answer the question based on the provided context and cite your sources.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear and accurate answer based on the context
- Cite specific documents using [Document N] notation
- If the context doesn't contain enough information, explicitly state this
- Be objective and stick to the information provided

Answer:"""
)


# Conversational prompt
CONVERSATIONAL_PROMPT = PromptTemplate(
    template="""You are a knowledgeable AI assistant engaged in a conversation. Use the provided context to answer the question.

Chat History:
{chat_history}

Context:
{context}

Current Question: {question}

Provide a helpful, conversational answer:"""
)


# Summary prompt
SUMMARY_PROMPT = PromptTemplate(
    template="""Summarize the following documents:

{context}

Provide a concise summary that captures the key points:"""
)
