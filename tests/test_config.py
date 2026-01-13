"""
Tests for configuration management.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Config,
    VectorDBConfig,
    EmbeddingConfig,
    ChunkingConfig,
    RetrievalConfig,
    LLMConfig,
    get_config
)


class TestConfig:
    """Test configuration classes."""
    
    def test_vector_db_config_defaults(self):
        """Test VectorDBConfig default values."""
        config = VectorDBConfig()
        assert config.provider == "chromadb"
        assert config.collection_name == "documents"
        assert config.distance_metric == "cosine"
    
    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.device == "cpu"
    
    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        config = ChunkingConfig()
        assert config.strategy == "recursive"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert isinstance(config.separators, list)
    
    def test_retrieval_config_defaults(self):
        """Test RetrievalConfig default values."""
        config = RetrievalConfig()
        assert config.top_k == 5
        assert config.similarity_threshold == 0.7
        assert config.enable_reranking == False
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
    
    def test_main_config_defaults(self):
        """Test main Config with default values."""
        config = Config()
        assert isinstance(config.vector_db, VectorDBConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.llm, LLMConfig)
    
    def test_config_from_env(self):
        """Test loading config from environment."""
        config = Config.from_env()
        assert config is not None
        assert isinstance(config, Config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
