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
    provider: str = "gemini"  # openai, anthropic, gemini, qwen, deepseek, ollama
    model: str = "gemini-2.5-pro"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # e.g. http://10.147.17.29:11434 for ollama
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
            elif p == "ollama":
                self.api_key = os.getenv("OLLAMA_API_KEY") or "ollama"
        if not self.base_url and self.provider.lower() == "ollama":
            self.base_url = os.getenv("OLLAMA_URL") or None
        return self


class VLMConfig(BaseModel):
    provider: str = "gemini"  # qwen, openai, gemini, ollama
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # e.g. http://10.147.17.29:11434 for ollama
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
            elif p in {"qwen", "dashscope"}:
                self.api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            elif p == "ollama":
                self.api_key = os.getenv("OLLAMA_API_KEY") or "ollama"
        if not self.base_url and self.provider.lower() == "ollama":
            self.base_url = os.getenv("OLLAMA_URL") or None
        return self


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/rag_pipeline.log"


class CriticConfig(BaseModel):
    """VLM evaluator (critic) ablation settings."""

    # Prompt file stem under src/prompt/ (without .txt)
    prompt_name: str = "vlm_eval/evaluate_with_vlm"
    # Whether to attach the generated BEV video to the critic inputs
    include_bev_video: bool = True
    # Whether to attach the generated Scenic code to the critic inputs
    include_scenic_code: bool = False


class CodegenConfig(BaseModel):
    """Scenic component-generator (codegen) ablation settings.

    Prompt groups live under ``src/prompt/gen_eval/`` (see MANIFEST.txt).
    Factors: CP (contextual), CoT, ICL, snippet retrieval.

      g1_zeroshot         — none (vanilla zeroshot)
      g2_cp               — CP
      g3_cp_cot           — CP + CoT
      g4_cp_icl           — CP + ICL
      g5_cp_icl_cot       — CP + ICL + CoT
      g6_cp_snippets      — CP + snippets
      g7_cp_cot_snippets  — CP + CoT + snippets

    Empty ``prompt_group`` uses the legacy full prompts at
    ``src/prompt/component_generator_*.txt`` (CP+CoT+ICL+snippets).
    """

    # Directory under src/prompt/ containing component_generator_*.txt
    # e.g. "gen_eval/g1_zeroshot". Empty = legacy root prompts.
    prompt_group: str = ""
    # Runtime feature flags. When None, inferred from ``FLAGS.txt`` in the
    # prompt group (or all-True for legacy root prompts).
    use_contextual: Optional[bool] = None
    use_cot: Optional[bool] = None
    use_icl: Optional[bool] = None
    use_snippet_retrieval: Optional[bool] = None


class CarlaSimulationConfig(BaseModel):
    """CARLA server and connection settings for run_scenic_batch.sh."""

    binary_dir: str = ""
    cmd: str = (
        "./CarlaUE4.sh -quality-level=High -nosound -RenderOffScreen -carla-rpc-port=2000"
    )
    host: str = "localhost"
    port: int = 2000
    rpc_port: Optional[int] = None
    log_dir: str = "logs/carla"
    auto_start: bool = True
    check_connection: bool = True
    restart_every: int = 25
    cooldown_sec: int = 5
    force_restart: bool = False
    start_timeout_sec: int = 60
    stop_grace_sec: int = 10
    stop_term_grace_sec: int = 5
    kill_process_name_pattern: str = "CARLA_Shipping"


class ScenicSimulationConfig(BaseModel):
    """Scenic simulator settings for run_scenic_batch.sh."""

    workdir: str = "Scenic"
    map_root: str = "Scenic/assets/maps/CARLA"
    version: int = 3
    conda_env: str = "chenli"
    scenic2_conda_env: str = "scenic2.0"
    scenic3_venv_activate: Optional[str] = None
    duration_sec: int = 25
    time_sec: int = 25
    time_steps: Optional[int] = None
    timestep: float = 0.1
    count: int = 1
    use_2d: bool = True
    render: bool = True
    show_params: bool = False
    auto_map: bool = True
    seed: Optional[int] = None


class RecorderSimulationConfig(BaseModel):
    """Recorder settings for run_scenic_batch.sh."""

    grace_sec: int = 10
    start_delay_sec: int = 1
    ready_timeout_sec: int = 8
    ego_alive_threshold_sec: float = 0.5
    disabled: bool = False


class SimulationConfig(BaseModel):
    """Batch simulation (CARLA + Scenic + recorder) configuration."""

    timeout_sec: int = 1000
    batch_script: str = "src/utils/run_scenic_batch.sh"
    recorder_script: str = "src/utils/recorder_scenic.py"
    scenarios_data_dir: str = "data/scenarios"
    carla: CarlaSimulationConfig = Field(default_factory=CarlaSimulationConfig)
    scenic: ScenicSimulationConfig = Field(default_factory=ScenicSimulationConfig)
    recorder: RecorderSimulationConfig = Field(default_factory=RecorderSimulationConfig)


class Config(BaseModel):
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    codegen: CodegenConfig = Field(default_factory=CodegenConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

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