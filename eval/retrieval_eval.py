from __future__ import annotations

import logging
import csv
import json
import time
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

FOLDER_PATH = str(Path(__file__).resolve().parent.parent / "data" / "scenarios")


class QueryMode(str, Enum):
    TEXT_ONLY = "text-only"
    TEXT_IMAGE = "text-image"
    TEXT_VIDEO = "text-video"
    VIDEO_ONLY = "video-only"
    TEXT_IMAGE_VIDEO = "text-image-video"


class QueryTextSource(str, Enum):
    """Where the query text comes from.

    INDEXED_DESCRIPTION (new_description.txt) is the text embedded in the vector DB —
    querying with it is self-retrieval (leaks the indexed document into the query).
    MLLM_CAPTION (text_query_description.txt) is an independent VLM description of the
    scenario video, i.e. a fair, deployment-faithful query.
    """
    INDEXED_DESCRIPTION = "new_description.txt"
    MLLM_CAPTION = "text_query_description.txt"
    VIDEO_ONLY_CAPTION = "video_only_description.txt"  # pure-video caption (qwen3.6-plus, no GT text)
    COMBINED_CAPTION = "__combined__"  # text caption + video-only caption (the "two texts" bridge)

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
        query_text_source: QueryTextSource = QueryTextSource.MLLM_CAPTION,
    ) -> list[dict[str, Any]]:
        """
        Scan immediate subfolders and build query records.

        Mapping per subfolder:
        - text: content of the query_text_source file (None for VIDEO_ONLY mode)
        - image_path: absolute path to image.png (when mode includes image)
        - video_path: absolute path to BEV.mp4 (when mode includes video)

        Args:
        - mode: which modalities feed the query
        - query_text_source: MLLM_CAPTION (default, fair independent query) or
          INDEXED_DESCRIPTION (self-retrieval upper bound — the text embedded in the DB)

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

            text = None
            if mode != QueryMode.VIDEO_ONLY:
                if query_text_source == QueryTextSource.COMBINED_CAPTION:
                    # Bridge "two texts": text caption + video-only caption concatenated.
                    tpath = subfolder / QueryTextSource.MLLM_CAPTION.value
                    vpath = subfolder / QueryTextSource.VIDEO_ONLY_CAPTION.value
                    if not (tpath.is_file() and vpath.is_file()):
                        self.logger.warning("Missing caption(s) in %s — skipping", subfolder.name)
                        continue
                    text = (tpath.read_text(encoding="utf-8").strip() + "\n"
                            + vpath.read_text(encoding="utf-8").strip())
                else:
                    description_path = subfolder / query_text_source.value
                    if not description_path.is_file():
                        self.logger.warning(
                            "Missing %s in %s — skipping scenario (generate captions first)",
                            query_text_source.value, subfolder.name,
                        )
                        continue
                    text = description_path.read_text(encoding="utf-8").strip()

            image_path = None
            video_path = None
            if mode in (QueryMode.TEXT_IMAGE, QueryMode.TEXT_IMAGE_VIDEO):
                image_path = str((subfolder / "image.png").resolve())
            if mode in (QueryMode.TEXT_VIDEO, QueryMode.VIDEO_ONLY, QueryMode.TEXT_IMAGE_VIDEO):
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
            self.logger.info(f"The query is: {query}")

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

        if getattr(self, "query_repr", "raw") == "dsl":
            # F1: production symmetric path — VLM interprets the query into the structured DSL,
            # the flattened DSL text is what gets embedded (mirrors workflow.embed_query).
            dsl, flattened_text = self.interpreter_agent.generate_dsl_from_user_query(query)
        else:
            dsl, flattened_text = query.text, query.text
        query_to_embed = {
            "text": flattened_text,
            "image": query.image_path,
            "video": query.video_path,
        }
        query_embeddings = self.embedder.encode([query_to_embed])
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

        retrieval = self.retriever.retrieve(
            original_query=user_query,
            query_embedding=query_embedding,
        )
        if not retrieval.scenarios:
            self.logger.error("No scenarios found for query")
            return {}

        best_scenario_ids = [scenario.scenario_id for scenario in retrieval.scenarios]
        base_scenario_id = retrieval.scenarios[0].scenario_id

        return {
            "base_scenario_id": base_scenario_id,
            "best_scenario_ids": best_scenario_ids,
            "best_similarity_score": retrieval.best_similarity_score,
            "best_rerank_score": retrieval.best_rerank_score,
        }

    def eval_text_only(
        self,
        mode: QueryMode = QueryMode.TEXT_ONLY,
        query_text_source: QueryTextSource = QueryTextSource.MLLM_CAPTION,
    ) -> Path:
        """
        Run retrieval evaluation for a selected query mode and save results to CSV.

        Output file:
        - eval/results/eval_<mode>_<source>_<timestamp>.csv
        """
        query_records = self.build_multimodal_queries(mode=mode, query_text_source=query_text_source)

        # Variant stamping: arms must be self-identifying (see ABLATION_PLAN.md F3)
        rerank_on = self.reranker is not None
        embedder_name = getattr(self.config.embedding, "model_name", "unknown")

        # Structured output: eval/results/rag/runs/<embedder-slug>/<mode>__<source>__<rerank>__<ts>.csv
        embedder_slug = re.sub(r"[^A-Za-z0-9.]+", "-", str(embedder_name)).strip("-").lower()
        results_dir = Path(__file__).resolve().parent / "results" / "rag" / "runs" / embedder_slug
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_slug = mode.value.replace("-", "_")
        source_slug = {
            QueryTextSource.MLLM_CAPTION: "caption",
            QueryTextSource.INDEXED_DESCRIPTION: "indexed",
            QueryTextSource.VIDEO_ONLY_CAPTION: "vcaption",
            QueryTextSource.COMBINED_CAPTION: "combined",
        }[query_text_source]
        rerank_slug = "rerank" if rerank_on else "norerank"
        repr_slug = getattr(self, "query_repr", "raw")
        output_csv_path = results_dir / f"{mode_slug}__{source_slug}__{repr_slug}__{rerank_slug}__{timestamp}.csv"

        fieldnames = [
            "ground_truth",
            "user_query",
            "scenario_dsl",
            "flattened_dsl",
            "base_scenario_id",
            "best_scenario_ids",
            "best_similarity_score",
            "best_rerank_score",
            "response_time_sec",
            "error_message",
            "mode",
            "query_text_source",
            "query_repr",
            "rerank_on",
            "embedder",
            "top_k",
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
                best_similarity_score = None
                best_rerank_score = None
                response_time_sec = None

                start_time = time.perf_counter()
                try:
                    embedded_state = self.embed_query({"user_query": query})
                    retrieved_state = self.retrieve_base_scenario(embedded_state) if embedded_state else {}

                    scenario_dsl = embedded_state.get("scenario_dsl") if embedded_state else None
                    flattened_dsl = embedded_state.get("flattened_dsl") if embedded_state else None
                    base_scenario_id = retrieved_state.get("base_scenario_id") if retrieved_state else None
                    best_scenario_ids = retrieved_state.get("best_scenario_ids") if retrieved_state else None
                    best_similarity_score = (
                        retrieved_state.get("best_similarity_score") if retrieved_state else None
                    )
                    best_rerank_score = (
                        retrieved_state.get("best_rerank_score") if retrieved_state else None
                    )
                except Exception as exc:
                    error_message = str(exc)
                    self.logger.exception(
                        "Failed to evaluate query for ground_truth=%s",
                        ground_truth,
                    )
                finally:
                    response_time_sec = time.perf_counter() - start_time

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
                        "best_similarity_score": (
                            f"{best_similarity_score:.6f}"
                            if best_similarity_score is not None
                            else ""
                        ),
                        "best_rerank_score": (
                            f"{best_rerank_score:.6f}" if best_rerank_score is not None else ""
                        ),
                        "response_time_sec": f"{response_time_sec:.4f}",
                        "error_message": error_message,
                        "mode": mode.value,
                        "query_text_source": query_text_source.value,
                        "query_repr": getattr(self, "query_repr", "raw"),
                        "rerank_on": str(rerank_on),
                        "embedder": str(embedder_name),
                        "top_k": str(getattr(self.retriever, "top_k", self.config.retrieval.top_k)),
                    }
                )

        self.logger.info("Saved retrieval evaluation (%s) to %s", mode.value, output_csv_path)
        return output_csv_path

def analyze_retrieval_csv(csv_path: str | Path) -> dict[str, float | list[str]]:
    """
    Analyze retrieval/eval CSV with columns:
    - ground_truth
    - base_scenario_id
    - best_scenario_ids
    - error_message

    Returns rates in percentage:
    1) no_error_rate
    2) ground_truth_eq_base_scenario_id_rate_among_no_error
    3) ground_truth_in_best_scenario_ids_rate_among_no_error

    Also returns average response time in seconds:
    4) avg_response_time_sec
    5) avg_response_time_sec_among_no_error

    Also returns average scores among successful rows (when present in CSV):
    6) avg_best_similarity_score_among_no_error
    7) avg_best_rerank_score_among_no_error
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    total_count = 0
    no_error_count = 0
    base_match_count = 0
    in_best_ids_count = 0
    total_response_time_sec = 0.0
    no_error_response_time_sec = 0.0
    similarity_score_sum = 0.0
    similarity_score_count = 0
    rerank_score_sum = 0.0
    rerank_score_count = 0
    reciprocal_rank_sum = 0.0
    recall_at = {1: 0, 3: 0, 5: 0, 10: 0}  # counts; only meaningful up to the stored list length
    failed_ground_truths: list[str] = []

    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_count += 1
            ground_truth = str(row.get("ground_truth", "")).strip()
            error_message = str(row.get("error_message", "")).strip()
            response_time_raw = str(row.get("response_time_sec", "")).strip()
            response_time_sec = float(response_time_raw) if response_time_raw else 0.0
            total_response_time_sec += response_time_sec

            if error_message and error_message.lower() not in ["none", "nan", "null", "false"]:
                failed_ground_truths.append(ground_truth)
                continue

            no_error_count += 1
            no_error_response_time_sec += response_time_sec

            base_scenario_id = str(row.get("base_scenario_id", "")).strip()
            if ground_truth == base_scenario_id:
                base_match_count += 1

            best_scenario_ids_raw = row.get("best_scenario_ids", "")
            best_scenario_ids: list[str] = []
            if isinstance(best_scenario_ids_raw, str) and best_scenario_ids_raw.strip():
                raw = best_scenario_ids_raw.strip()
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        best_scenario_ids = [str(x).strip() for x in parsed]
                    else:
                        best_scenario_ids = [str(parsed).strip()]
                except Exception:
                    # Fallback for non-JSON list formatting (comma-separated or plain text)
                    best_scenario_ids = [part.strip() for part in raw.split(",") if part.strip()]

            if ground_truth in best_scenario_ids:
                in_best_ids_count += 1
                rank = best_scenario_ids.index(ground_truth) + 1  # 1-indexed
                # MRR over the ordered list (rank position, not just membership).
                reciprocal_rank_sum += 1.0 / rank
                # Recall@k at multiple cutoffs (needs the stored list length >= k).
                for _k in recall_at:
                    if rank <= _k:
                        recall_at[_k] += 1

            similarity_raw = str(row.get("best_similarity_score", "")).strip()
            if similarity_raw and similarity_raw.lower() not in ("none", "nan", "null"):
                try:
                    similarity_score_sum += float(similarity_raw)
                    similarity_score_count += 1
                except ValueError:
                    pass

            rerank_raw = str(row.get("best_rerank_score", "")).strip()
            if rerank_raw and rerank_raw.lower() not in ("none", "nan", "null"):
                try:
                    rerank_score_sum += float(rerank_raw)
                    rerank_score_count += 1
                except ValueError:
                    pass

    print(f"DEBUG: total_count = {total_count}")
    print(f"DEBUG: no_error_count = {no_error_count}")
    print(f"DEBUG: base_match_count = {base_match_count}")
    print(f"DEBUG: in_best_ids_count = {in_best_ids_count}")
    print(f"DEBUG: skipped strings = {failed_ground_truths[:5]}") # See what got treated as an error

    no_error_rate = (no_error_count / total_count * 100.0) if total_count > 0 else 0.0
    base_match_rate = (base_match_count / no_error_count * 100.0) if no_error_count > 0 else 0.0
    in_best_ids_rate = (in_best_ids_count / no_error_count * 100.0) if no_error_count > 0 else 0.0
    avg_response_time_sec = (total_response_time_sec / total_count) if total_count > 0 else 0.0
    avg_response_time_sec_among_no_error = (
        no_error_response_time_sec / no_error_count if no_error_count > 0 else 0.0
    )
    avg_best_similarity_score_among_no_error = (
        similarity_score_sum / similarity_score_count if similarity_score_count > 0 else 0.0
    )
    avg_best_rerank_score_among_no_error = (
        rerank_score_sum / rerank_score_count if rerank_score_count > 0 else 0.0
    )
    mrr_among_no_error = (reciprocal_rank_sum / no_error_count) if no_error_count > 0 else 0.0
    recall_rates = {
        f"recall@{k}_among_no_error": (recall_at[k] / no_error_count * 100.0) if no_error_count > 0 else 0.0
        for k in recall_at
    }

    return {
        "no_error_rate": no_error_rate,
        "ground_truth_eq_base_scenario_id_rate_among_no_error": base_match_rate,
        "ground_truth_in_best_scenario_ids_rate_among_no_error": in_best_ids_rate,
        "mrr_among_no_error": mrr_among_no_error,
        **recall_rates,
        "avg_response_time_sec": avg_response_time_sec,
        "avg_response_time_sec_among_no_error": avg_response_time_sec_among_no_error,
        "avg_best_similarity_score_among_no_error": avg_best_similarity_score_among_no_error,
        "avg_best_rerank_score_among_no_error": avg_best_rerank_score_among_no_error,
        "failed_ground_truths": failed_ground_truths,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval evaluation / CSV analysis")
    parser.add_argument("--mode", choices=[m.value for m in QueryMode], default=None,
                        help="Run retrieval eval for this query mode")
    parser.add_argument("--query-text-source", choices=[s.name.lower() for s in QueryTextSource],
                        default="mllm_caption",
                        help="mllm_caption (fair, default) or indexed_description (self-retrieval upper bound)")
    parser.add_argument("--folder", default=FOLDER_PATH, help="Scenario folder path")
    parser.add_argument("--config", default=None, help="Config yaml path (e.g. config/config_norerank.yaml)")
    parser.add_argument("--query-repr", choices=["raw", "dsl"], default="raw",
                        help="raw = embed query text directly; dsl = production symmetric VLM->DSL->flatten path")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override retrieval top_k (e.g. 20 to enable Recall@5/@10; top-1/@3 are unchanged)")
    parser.add_argument("--analyze", default=None, help="Only analyze an existing CSV, no retrieval run")
    args = parser.parse_args()

    if args.analyze:
        print(analyze_retrieval_csv(args.analyze))
    elif args.mode:
        source = QueryTextSource[args.query_text_source.upper()]
        eval_retrieval = EvalRetrieval(folder_path=args.folder, config_path=args.config)
        eval_retrieval.query_repr = args.query_repr
        if args.top_k is not None:
            eval_retrieval.retriever.top_k = args.top_k
        csv_path = eval_retrieval.eval_text_only(mode=QueryMode(args.mode), query_text_source=source)
        print(analyze_retrieval_csv(csv_path))
    else:
        parser.print_help()