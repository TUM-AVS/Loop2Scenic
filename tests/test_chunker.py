"""
Tests for document chunking.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document
from src.ingestion import DocumentChunker


class TestDocumentChunker:
    """Test DocumentChunker class."""
    
    def test_chunker_initialization(self):
        """Test chunker initialization."""
        chunker = DocumentChunker()
        assert chunker.strategy == "recursive"
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
    
    def test_chunk_short_text(self):
        """Test chunking short text."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."
        
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].page_content == text
    
    def test_chunk_long_text(self):
        """Test chunking long text."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a longer text. " * 20
        
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        
        # Check metadata
        for chunk in chunks:
            assert 'chunk_id' in chunk.metadata
            assert 'chunk_size' in chunk.metadata
    
    def test_chunk_documents(self):
        """Test chunking documents."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        docs = [
            Document(page_content="Text " * 50, metadata={"source": "doc1"}),
            Document(page_content="More text " * 50, metadata={"source": "doc2"})
        ]
        
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 2
        
        # Check metadata is preserved
        for chunk in chunks:
            assert 'source' in chunk.metadata
            assert 'chunk_id' in chunk.metadata
    
    def test_empty_documents(self):
        """Test chunking empty document list."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents([])
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
