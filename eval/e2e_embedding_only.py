from __future__ import annotations

import csv
import json
import logging
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.app import ChatbotWorkflow
from src.schema import MultimodalQuery
from src.workflow.scenario_workflow_state import CLEAN_STATE

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from e2e_metrics import extract_vlm_llm_metrics_rows, normalize_model_metrics_blob

FOLDER_PATH = "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/test"
SAVE_PATH = "eval/results"


class QueryMode(str, Enum):
    TEXT_ONLY = "text-only"
    TEXT_IMAGE = "text-image"
    TEXT_VIDEO = "text-video"
    TEXT_IMAGE_VIDEO = "text-image-video"


class EvalE2EEmbeddingOnlyWorkflow:
    """
    Evaluation helper that only runs:
      1) embed_query
      2) retrieve_base_scenario

    It scans scenario subfolders from folder_path, builds MultimodalQuery records,
    runs the two nodes, writes per-record outputs, and generates a batch CSV.
    """

    def __init__(
        self,
        folder_path: Path | str = FOLDER_PATH,
        config_path: Optional[str] = None,
        save_path: Path | str = SAVE_PATH,
    ) -> None:
        self.folder_path = Path(folder_path)
        self.config_path = config_path
        self.save_path = Path(save_path)
        self.logger = logging.getLogger(__name__)
        self.chatbot_workflow = ChatbotWorkflow()
        self.workflow = self.chatbot_workflow.initialize_system(config_path=config_path)

    def build_multimodal_queries(
        self,
        mode: QueryMode = QueryMode.TEXT_IMAGE_VIDEO,
    ) -> list[dict[str, Any]]:
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
            queries.append({"ground_truth": subfolder.name, "query": query})

        self.logger.info(
            "Built %d multimodal queries from %s (mode=%s)",
            len(queries),
            self.folder_path,
            mode.value,
        )
        return queries

    def _to_jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
        return str(value)

    def run_query_embedding_and_retrieval(
        self,
        query: MultimodalQuery,
    ) -> dict[str, Any]:
        state = dict(CLEAN_STATE)
        state["user_query"] = query

        state_after_embed = self.workflow.embed_query(state)
        if not state_after_embed:
            raise RuntimeError("embed_query failed to produce state updates.")
        state.update(state_after_embed)

        state_after_retrieve = self.workflow.retrieve_base_scenario(state)
        if not state_after_retrieve:
            raise RuntimeError("retrieve_base_scenario failed to produce state updates.")
        state.update(state_after_retrieve)

        current_scenic_scenario = state.get("current_scenic_scenario")
        return {
            "base_scenario_id": state.get("base_scenario_id"),
            "retrieved_scenic_code": getattr(current_scenic_scenario, "scenic_code", None),
            "scenario_dsl": state.get("scenario_dsl"),
            "header_settings": state.get("header_settings"),
            "model_metrics": normalize_model_metrics_blob(state.get("model_metrics")) or {},
            "state": state,
        }

    def run_batch(
        self,
        mode: QueryMode = QueryMode.TEXT_IMAGE_VIDEO,
    ) -> Path:
        query_records = self.build_multimodal_queries(mode=mode)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = self.save_path / f"e2e_embedding_only_{timestamp}"
        results_root.mkdir(parents=True, exist_ok=True)
        output_csv_path = results_root / "batch_results.csv"

        fieldnames = [
            "ground_truth",
            "user_query",
            "best_scenario_id",
            "base_scenario_id",
            "generation_count",
            "vlm_calls",
            "vlm_prompt_tokens",
            "vlm_completion_tokens",
            "vlm_total_tokens",
            "vlm_response_time_ms",
            "llm_calls",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_total_tokens",
            "llm_response_time_ms",
            "record_total_time_ms",
            "error_message",
        ]

        with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in query_records:
                ground_truth = record["ground_truth"]
                query = record["query"]
                error_message = ""
                base_scenario_id = ""
                generation_count: int | str = 0  # no loop generation in this evaluator
                best_scenario_id = ""  # keep column for compatibility with e2e_eval CSV
                vlm_calls = 0
                vlm_prompt_tokens = 0
                vlm_completion_tokens = 0
                vlm_total_tokens = 0
                vlm_response_time_ms = 0.0
                llm_calls = 0
                llm_prompt_tokens = 0
                llm_completion_tokens = 0
                llm_total_tokens = 0
                llm_response_time_ms = 0.0

                record_start = time.perf_counter()
                try:
                    self.logger.info(
                        "[START E2E-EMBED-ONLY] Running query for ground_truth=%s",
                        ground_truth,
                    )
                    result = self.run_query_embedding_and_retrieval(query=query)
                    self.logger.info(
                        "[END E2E-EMBED-ONLY] Query for ground_truth=%s completed",
                        ground_truth,
                    )

                    base_scenario_id = str(result.get("base_scenario_id") or "")
                    best_scenario_id = base_scenario_id
                    st = result.get("state")
                    model_metrics = (
                        result.get("model_metrics")
                        or (st.get("model_metrics") if isinstance(st, dict) else None)
                        or {}
                    )
                    vlm_metrics, llm_metrics = extract_vlm_llm_metrics_rows(model_metrics)
                    vlm_calls = int(vlm_metrics["calls"])
                    vlm_prompt_tokens = int(vlm_metrics["prompt_tokens"])
                    vlm_completion_tokens = int(vlm_metrics["completion_tokens"])
                    vlm_total_tokens = int(vlm_metrics["total_tokens"])
                    vlm_response_time_ms = float(vlm_metrics["response_time_ms"])
                    llm_calls = int(llm_metrics["calls"])
                    llm_prompt_tokens = int(llm_metrics["prompt_tokens"])
                    llm_completion_tokens = int(llm_metrics["completion_tokens"])
                    llm_total_tokens = int(llm_metrics["total_tokens"])
                    llm_response_time_ms = float(llm_metrics["response_time_ms"])

                    query_result_dir = results_root / str(ground_truth)
                    query_result_dir.mkdir(parents=True, exist_ok=True)

                    dsl_path = query_result_dir / "scenario_dsl.json"
                    dsl_payload = self._to_jsonable(result.get("scenario_dsl"))
                    dsl_path.write_text(
                        json.dumps(dsl_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    header_settings_path = query_result_dir / "header_settings.json"
                    header_payload = self._to_jsonable(result.get("header_settings"))
                    header_settings_path.write_text(
                        json.dumps(header_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    error_message = str(exc)
                    self.logger.exception(
                        "Failed to run e2e embedding-only query for ground_truth=%s",
                        ground_truth,
                    )

                record_total_time_ms = (time.perf_counter() - record_start) * 1000.0
                writer.writerow(
                    {
                        "ground_truth": str(ground_truth),
                        "user_query": query.model_dump_json(),
                        "best_scenario_id": best_scenario_id,
                        "base_scenario_id": base_scenario_id,
                        "generation_count": generation_count,
                        "vlm_calls": vlm_calls,
                        "vlm_prompt_tokens": vlm_prompt_tokens,
                        "vlm_completion_tokens": vlm_completion_tokens,
                        "vlm_total_tokens": vlm_total_tokens,
                        "vlm_response_time_ms": f"{vlm_response_time_ms:.2f}",
                        "llm_calls": llm_calls,
                        "llm_prompt_tokens": llm_prompt_tokens,
                        "llm_completion_tokens": llm_completion_tokens,
                        "llm_total_tokens": llm_total_tokens,
                        "llm_response_time_ms": f"{llm_response_time_ms:.2f}",
                        "record_total_time_ms": f"{record_total_time_ms:.2f}",
                        "error_message": error_message,
                    }
                )

        self.logger.info("Saved e2e embedding-only results to %s", output_csv_path)
        return output_csv_path


if __name__ == "__main__":
    evaluator = EvalE2EEmbeddingOnlyWorkflow()
    output_csv = evaluator.run_batch(mode=QueryMode.TEXT_IMAGE_VIDEO)
    print(f"Batch done. CSV: {output_csv}")
