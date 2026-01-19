"""
Document chunking functionality.
"""

import logging
from typing import List

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits documents into smaller chunks for embedding."""

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize document chunker.
        
        Args:
            strategy: Chunking strategy ('recursive' or 'fixed')
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators for recursive splitting
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators or ["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        elif strategy == "fixed":
            self.splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n",
                length_function=len,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        logger.info(
            f"Split {len(documents)} document(s) into {len(chunks)} chunks "
            f"(strategy: {self.strategy}, size: {self.chunk_size}, "
            f"overlap: {self.chunk_overlap})"
        )
        
        return chunks

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split a single text string into chunks.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Document objects
        """
        chunks = self.splitter.split_text(text)
        
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    **(metadata or {}),
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        logger.info(f"Split text into {len(documents)} chunks")
        
        return documents
