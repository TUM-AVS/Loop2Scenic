from __future__ import annotations

import logging
import csv
import json
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import re

from src.agents import InterpreterAgent
from src.config import get_config
from src.schema import MultimodalQuery
from src.services import (
    Retriever,
    MilvusVectorStore,
    get_embedder,
    get_reranker,
    get_vlm_service,
)
from src.utils.logger import setup_logging

FOLDER_PATH = "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios"


class QueryMode(str, Enum):
    TEXT_ONLY = "text-only"
    TEXT_IMAGE = "text-image"
    TEXT_VIDEO = "text-video"
    TEXT_IMAGE_VIDEO = "text-image-video"

class EvalRetrieval:
    """
    Helper class for retrieval evaluation data preparation.

    Responsibilities:
    1) Initialize required runtime components from config/config.yaml.
    2) Build MultimodalQuery objects from scenario subfolders.
    """

    def __init__(
        self,
        folder_path: Path | str = FOLDER_PATH,
        config_path: Optional[str] = None,
    ) -> None:
        self.folder_path = Path(folder_path)
        self.config = get_config(config_path)

        setup_logging(
            level=self.config.logging.level,
            log_format=self.config.logging.format,
        )
        self.logger = logging.getLogger(__name__)

        self.interpreter_agent: InterpreterAgent | None = None
        self.embedder = None
        self.vector_db: MilvusVectorStore | None = None
        self.reranker = None
        self.vlm_service = None
        self.retriever: Retriever | None = None

        self.initialize_components()

    def initialize_components(self) -> None:
        """Initialize interpreter_agent, embedder, vector_db, reranker, vlm_service."""
        self.logger.info("Initializing eval retrieval components...")

        vlm_kwargs = {
            "provider": self.config.vlm.provider,
            "model": self.config.vlm.model,
            "temperature": self.config.vlm.temperature,
            "max_tokens": self.config.vlm.max_tokens,
            "api_key": self.config.vlm.api_key,
        }
        if self.config.vlm.model_path:
            vlm_kwargs["model_path"] = self.config.vlm.model_path
        self.vlm_service = get_vlm_service(**vlm_kwargs)

        embedder_kwargs = {
            "provider": self.config.embedding.provider,
            "model_name": self.config.embedding.model_name,
        }
        if self.config.embedding.model_path:
            embedder_kwargs["model_path"] = self.config.embedding.model_path
        self.embedder = get_embedder(**embedder_kwargs)

        connection_args = {
            "host": self.config.vector_db.host,
            "port": self.config.vector_db.port,
        }
        index_params = {
            "metric_type": self.config.vector_db.distance_metric.upper(),
            "index_type": self.config.vector_db.index_type,
            "params": {"nlist": self.config.vector_db.nlist},
        }
        search_params = {
            "metric_type": self.config.vector_db.distance_metric.upper(),
            "params": {"nprobe": self.config.vector_db.nprobe},
        }
        self.vector_db = MilvusVectorStore(
            embedding_dim=self.embedder.dimension,
            collection_name=self.config.vector_db.collection_name,
            snippets_collection_name=self.config.vector_db.snippets_collection_name,
            connection_args=connection_args,
            index_params=index_params,
            search_params=search_params,
        )

        if self.config.retrieval.enable_reranking:
            reranker_kwargs = {
                "provider": self.config.reranking.provider,
                "device": self.config.reranking.device,
                "model_name": self.config.reranking.model_name,
            }
            if self.config.reranking.model_path:
                reranker_kwargs["model_path"] = self.config.reranking.model_path
            self.reranker = get_reranker(**reranker_kwargs)
        else:
            self.reranker = None

        self.retriever = Retriever(
            vectorstore=self.vector_db,
            reranker=self.reranker,
            top_k=self.config.retrieval.top_k,
            similarity_threshold=self.config.retrieval.similarity_threshold,
        )
        self.interpreter_agent = InterpreterAgent(vlm_service=self.vlm_service)
        self.logger.info("Eval retrieval components initialized.")

    def build_multimodal_queries(
        self,
        mode: QueryMode = QueryMode.TEXT_IMAGE_VIDEO,
    ) -> list[dict[str, Any]]:
        """
        Scan immediate subfolders and build query records.

        Mapping per subfolder:
        - text: content of description.txt if present, else None
        - image_path: absolute path to image.png (when mode includes image)
        - video_path: absolute path to BEV.mp4 (when mode includes video)

        Returns:
        - list of {"ground_truth": str, "query": MultimodalQuery}
        """
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Expected directory, got: {self.folder_path}")

        queries: list[dict[str, Any]] = []
        for subfolder in sorted(self.folder_path.iterdir()):
            if not subfolder.is_dir():
                continue

            description_path = subfolder / "description.txt"
            text = description_path.read_text(encoding="utf-8").strip() if description_path.is_file() else None

            image_path = None
            video_path = None
            if mode in (QueryMode.TEXT_IMAGE, QueryMode.TEXT_IMAGE_VIDEO):
                image_path = str((subfolder / "image.png").resolve())
            if mode in (QueryMode.TEXT_VIDEO, QueryMode.TEXT_IMAGE_VIDEO):
                video_path = str((subfolder / "BEV.mp4").resolve())

            query = MultimodalQuery(
                text=text,
                image_path=image_path,
                video_path=video_path,
            )
            queries.append(
                {
                    "ground_truth": subfolder.name,
                    "query": query,
                }
            )

        self.logger.info(
            "Built %d multimodal queries from %s (mode=%s)",
            len(queries),
            self.folder_path,
            mode.value,
        )
        return queries

    def embed_query(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate-style embed_query.

        Works like workflow.embed_query, but returns a new dict instead of mutating workflow state.
        """
        query = state.get("user_query")
        if not query:
            self.logger.error("No user query provided")
            return {}
        if self.interpreter_agent is None or self.embedder is None:
            self.logger.error("Components not initialized: interpreter_agent/embedder")
            return {}

        dsl, flattened_text = self.interpreter_agent.generate_dsl_from_user_query(query)
        query_to_embed = MultimodalQuery(
            text=flattened_text,
            image_path=query.image_path,
            video_path=query.video_path,
        )
        query_embeddings = self.embedder.encode([query_to_embed.model_dump()])
        if query_embeddings is None:
            self.logger.error("Failed to embed query (got None)")
            return {}

        try:
            batch_size = len(query_embeddings)
        except TypeError:
            batch_size = int(getattr(query_embeddings, "shape", [0])[0] or 0)
        if batch_size == 0:
            self.logger.error("Failed to embed query")
            return {}

        return {
            "user_query": query,
            "scenario_dsl": dsl,
            "flattened_dsl": flattened_text,
            "query_embedding": query_embeddings[0],
        }

    def retrieve_base_scenario(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate-style retrieve_base_scenario.

        Works like workflow.retrieve_base_scenario, but returns an output dict.
        """
        user_query = state.get("user_query")
        query_embedding = state.get("query_embedding")
        if user_query is None or query_embedding is None:
            self.logger.error("No user query or query embedding provided")
            return {}
        if self.retriever is None:
            self.logger.error("Retriever is not initialized")
            return {}

        best_scenarios = self.retriever.retrieve(
            original_query=user_query,
            query_embedding=query_embedding,
        )
        if not best_scenarios:
            self.logger.error("No scenarios found for query")
            return {}

        best_scenario_ids = [scenario.scenario_id for scenario in best_scenarios]
        base_scenario_id = best_scenarios[0].scenario_id

        return {
            "base_scenario_id": base_scenario_id,
            "best_scenario_ids": best_scenario_ids,
        }

    def eval_text_only(self, mode: QueryMode = QueryMode.TEXT_ONLY) -> Path:
        """
        Run retrieval evaluation for a selected query mode and save results to CSV.

        Output file:
        - eval/results/eval_<mode>_<timestamp>.csv
        """
        query_records = self.build_multimodal_queries(mode=mode)

        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_slug = mode.value.replace("-", "_")
        output_csv_path = results_dir / f"eval_{mode_slug}_{timestamp}.csv"

        fieldnames = [
            "ground_truth",
            "user_query",
            "scenario_dsl",
            "flattened_dsl",
            "base_scenario_id",
            "best_scenario_ids",
            "error_message",
        ]

        with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in query_records:
                self.logger.info("Evaluating query for ground_truth=%s", record["ground_truth"])
                ground_truth = record["ground_truth"]
                query = record["query"]
                error_message = ""
                scenario_dsl = None
                flattened_dsl = None
                base_scenario_id = None
                best_scenario_ids = None

                try:
                    embedded_state = self.embed_query({"user_query": query})
                    retrieved_state = self.retrieve_base_scenario(embedded_state) if embedded_state else {}

                    scenario_dsl = embedded_state.get("scenario_dsl") if embedded_state else None
                    flattened_dsl = embedded_state.get("flattened_dsl") if embedded_state else None
                    base_scenario_id = retrieved_state.get("base_scenario_id") if retrieved_state else None
                    best_scenario_ids = retrieved_state.get("best_scenario_ids") if retrieved_state else None
                except Exception as exc:
                    error_message = str(exc)
                    self.logger.exception(
                        "Failed to evaluate query for ground_truth=%s",
                        ground_truth,
                    )

                writer.writerow(
                    {
                        "ground_truth": str(ground_truth),
                        "user_query": query.model_dump_json(),
                        "scenario_dsl": json.dumps(scenario_dsl, ensure_ascii=False) if scenario_dsl is not None else "",
                        "flattened_dsl": str(flattened_dsl) if flattened_dsl is not None else "",
                        "base_scenario_id": str(base_scenario_id) if base_scenario_id is not None else "",
                        "best_scenario_ids": json.dumps(best_scenario_ids, ensure_ascii=False)
                        if best_scenario_ids is not None
                        else "",
                        "error_message": error_message,
                    }
                )

        self.logger.info("Saved retrieval evaluation (%s) to %s", mode.value, output_csv_path)
        return output_csv_path

if __name__ == "__main__":
    eval_retrieval = EvalRetrieval()
    eval_retrieval.eval_text_only(mode=QueryMode.TEXT_ONLY)
    eval_retrieval.eval_text_only(mode=QueryMode.TEXT_VIDEO)