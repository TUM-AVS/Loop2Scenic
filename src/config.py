"""
Configuration management for the ADS-MRAG pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# Load environment variables from .env
load_dotenv()


class VectorDBConfig(BaseModel):
    collection_name: str = "avs_new"
    snippets_collection_name: str = "scenario_components"
    distance_metric: str = "cosine"

    host: str = "localhost"
    port: str = "19530"
    index_type: str = "IVF_FLAT"
    nlist: int = 1024
    nprobe: int = 10

    @field_validator("port", mode="before")
    @classmethod
    def _port_to_str(cls, value):
        return str(value)


class EmbeddingConfig(BaseModel):
    provider: str = "qwen"  # huggingface, openai, gemini, qwen
    model_name: str = "Qwen3-VL-Embedding-2B"
    model_path: Optional[str] = "./models/Qwen3-VL-Embedding-2B"
    batch_size: int = 4
    device: str = "cuda"

    # Optional fallback for code that still references embedding.dimension
    dimension: int = 384


class RetrievalConfig(BaseModel):
    top_k: int = 3
    similarity_threshold: float = 0.0
    enable_reranking: bool = True
    rerank_top_k: int = 3


class RerankingConfig(BaseModel):
    provider: str = "qwen"
    model_path: Optional[str] = "./models/Qwen3-VL-Reranker-2B"
    model_name: Optional[str] = "Qwen3-VL-Reranker-2B"
    device: str = "cuda"

    # Optional fields kept for provider compatibility
    torch_dtype: Optional[str] = None
    instruction: str = "Retrieval relevant image or text with user's query"
    fps: float = 1.0


class LLMConfig(BaseModel):
    provider: str = "gemini"  # openai, anthropic, gemini
    model: str = "gemini-2.5-pro"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = False

    @model_validator(mode="after")
    def _load_api_key(self) -> "LLMConfig":
        if not self.api_key:
            p = self.provider.lower()
            if p == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif p in {"gemini", "google"}:
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif p == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        return self


class VLMConfig(BaseModel):
    provider: str = "gemini"  # qwen, openai, gemini
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 10_000_000

    # Optional compatibility fields
    device: str = "cuda"

    @model_validator(mode="after")
    def _load_api_key(self) -> "VLMConfig":
        if not self.api_key:
            p = self.provider.lower()
            if p in {"gemini", "google"}:
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif p in {"openai", "openai_vision"}:
                self.api_key = os.getenv("OPENAI_API_KEY")
        return self


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/rag_pipeline.log"


class Config(BaseModel):
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> "Config":
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_file.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "Config":
        config = cls()

        if model := os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = model

        if llm_model := os.getenv("LLM_MODEL"):
            config.llm.model = llm_model

        return config


def get_config(config_path: Optional[str] = None) -> Config:
    if config_path:
        return Config.from_yaml(config_path)

    try:
        return Config.from_yaml()
    except FileNotFoundError:
        return Config.from_env()


if __name__ == "__main__":
    cfg = get_config()
    print(cfg.model_dump_json(indent=2))