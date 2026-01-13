"""
Configuration management for the RAG pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    provider: str = "milvus"
    collection_name: str = "documents"
    distance_metric: str = "cosine"
    
    # Milvus settings
    host: str = "localhost"
    port: str = "19530"
    index_type: str = "IVF_FLAT"
    nlist: int = 1024
    nprobe: int = 10
    
    # ChromaDB settings
    persist_directory: str = "./data/vector_db"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: str = "huggingface"  # huggingface, openai
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"  # For HuggingFace models


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = 5
    similarity_threshold: float = 0.7
    enable_reranking: bool = False
    rerank_top_k: int = 10


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 512
    streaming: bool = False


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""
    supported_formats: List[str] = Field(
        default_factory=lambda: ['.txt', '.pdf', '.docx', '.md', '.html']
    )
    batch_size: int = 10
    max_workers: int = 4


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/rag_pipeline.log"


class Config(BaseModel):
    """Main configuration class."""
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if api_key := os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        
        if model := os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = model
        
        if llm_model := os.getenv("LLM_MODEL"):
            config.llm.model = llm_model
        
        return config


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration instance.
    
    Args:
        config_path: Optional path to YAML configuration file
        
    Returns:
        Config instance
    """
    if config_path:
        return Config.from_yaml(config_path)
    
    # Try to load from default YAML, fallback to environment
    try:
        return Config.from_yaml()
    except FileNotFoundError:
        return Config.from_env()
