"""
Configuration management for the RAG pipeline.
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class VectorDBConfig(BaseModel):
    """Milvus vector database configuration."""
    collection_name: str = "documents"
    distance_metric: str = "cosine"
    
    # Milvus mode selection
    use_lite: bool = False  # True: Milvus Lite (no Docker), False: Milvus Server (Docker)
    lite_db_path: str = "./data/vector_db/milvus.db"  # Used when use_lite=True
    
    # Milvus Server settings (used when use_lite=False)
    host: str = "localhost"
    port: str = "19530"  # Accepts int or str, converts to str
    index_type: str = "IVF_FLAT"
    nlist: int = 1024
    nprobe: int = 10
    
    @field_validator('port', mode='before')
    @classmethod
    def convert_port_to_string(cls, v):
        """Convert port to string if it's an integer."""
        if isinstance(v, int):
            return str(v)
        return v


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: str = "huggingface"  # huggingface, openai, gemini, qwen
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_path: Optional[str] = None  # Required for Qwen models, optional for others
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"  # For HuggingFace models (cpu, cuda, mps)


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
    enable_reranking: bool = True
    rerank_top_k: int = 10


class RerankingConfig(BaseModel):
    """Reranking model configuration."""
    provider: str = "qwen"  # Options: qwen, openai, gemini
    model_path: Optional[str] = None  # Path to local reranker model directory
    model_name: Optional[str] = None  # Model name identifier
    device: Optional[str] = None  # Device to run the model on (cpu, cuda, etc.)
    torch_dtype: Optional[str] = None  # Data type for the model (fp16, fp32, bf16, etc.)
    instruction: str = "Retrieval relevant image or text with user's query"  # Instruction text
    fps: float = 1.0  # Frames per second for video processing


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None  # Added API key
    temperature: float = 0.7
    max_tokens: int = 512
    streaming: bool = False

    @model_validator(mode='after')
    def load_api_key(self) -> "LLMConfig":
        """Automatically load the correct API key based on the provider."""
        if not self.api_key:
            provider_lower = self.provider.lower()
            if provider_lower in ["openai"]:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif provider_lower in ["gemini", "google"]:
                self.api_key = os.getenv("GOOGLE_API_KEY")
        return self


class VLMConfig(BaseModel):
    """VLM (Vision Language Model) configuration."""
    provider: str = "gemini"  # Options: qwen3vl, openai_vision, gemini
    api_key: Optional[str] = None  # Added API key
    model_path: Optional[str] = None  # Required for Qwen3VL, path to local model directory
    model: Optional[str] = "gemini-1.5-pro"  # Model name
    device: str = "cuda"  # Device to run on (cpu, cuda, etc.)
    torch_dtype: str = "fp16"  # Data type (fp16, fp32, bf16)
    temperature: float = 0.7  # Sampling temperature
    max_tokens: int = 512  # Maximum tokens to generate
    fps: float = 1.0  # Frames per second for video processing
    max_frames: int = 64  # Maximum frames to extract from video
    default_instruction: str = "You are a helpful AI assistant."  # Default instruction text

    @model_validator(mode='after')
    def load_api_key(self) -> "VLMConfig":
        """Automatically load the correct API key based on the provider."""
        # Only attempt to load if an API key wasn't explicitly provided in the YAML
        if not self.api_key:
            provider_lower = self.provider.lower()
            if provider_lower in ["gemini", "google"]:
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif provider_lower in ["openai", "openai_vision"]:
                self.api_key = os.getenv("OPENAI_API_KEY")
        return self


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
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
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

if __name__ == "__main__":
    config = get_config()
    print(f"Using config: {config}")