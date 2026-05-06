from __future__ import annotations

import csv
import json
import logging
import shutil
import time
import uuid
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path
from typing import Any, Optional

from src.app import ChatbotWorkflow
from src.schema import MultimodalQuery

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from e2e_metrics import extract_vlm_llm_metrics_rows, normalize_model_metrics_blob

FOLDER_PATH = "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/random_100"


def _try_move_temp_bev_to_eval_result(
    best_scenario_id: str,
    query_result_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Move ``temp/<best_scenario_id>/video/BEV.mp4`` (relative to repo root) into
    ``query_result_dir / generated_video.mp4``. No-op if source missing; log on failure.
    """
    sid = (best_scenario_id or "").strip()
    if not sid:
        return
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "temp" / sid / "video" / "BEV.mp4"
    dest = query_result_dir / "generated_video.mp4"
    try:
        if not src.is_file():
            logger.warning("Generated BEV not found, skip move: %s", src)
            return
        query_result_dir.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            dest.unlink()
        shutil.move(str(src), str(dest))
        logger.info("Moved generated BEV to %s", dest)
    except OSError as exc:
        logger.warning("Could not move BEV %s -> %s: %s", src, dest, exc)


class QueryMode(str, Enum):
    TEXT_ONLY = "text-only"
    TEXT_IMAGE = "text-image"
    TEXT_VIDEO = "text-video"
    TEXT_IMAGE_VIDEO = "text-image-video"


class EvalE2EWorkflow:
    """
    Helper class for end-to-end workflow evaluation.

    Responsibilities:
    1) Build MultimodalQuery objects from scenario subfolders.
    2) Run full ScenarioWorkflow from user query to output_best_scenario.
    3) Skip manual human review by auto-setting user_satisfied=True.
    """

    def __init__(
        self,
        folder_path: Path | str = FOLDER_PATH,
        config_path: Optional[str] = None,
    ) -> None:
        self.folder_path = Path(folder_path)
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.chatbot_workflow = ChatbotWorkflow()
        self.workflow = self.chatbot_workflow.initialize_system(config_path=config_path)

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

    def run_query_to_output_best_scenario(
        self,
        query: MultimodalQuery,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run workflow from user query to output_best_scenario.

        The graph pauses before human_review by design. This function immediately
        auto-accepts review (user_satisfied=True) to avoid manual intervention.
        """
        current_thread_id = thread_id or f"e2e_eval_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": current_thread_id}}
        user_message = {"role": "user", "content": query.model_dump_json()}

        for _event in self.workflow.app.stream(
            {"user_query": query, "messages": [user_message]},
            config=config,
        ):
            pass

        # Graph is expected to pause here (interrupt_before=["human_review"]).
        paused_state = self.workflow.app.get_state(config)
        if paused_state.next:
            self.workflow.app.update_state(
                config,
                {
                    "user_satisfied": True,
                    "user_modification": None,
                },
            )
            for _event in self.workflow.app.stream(None, config=config):
                pass

        final_state = self.workflow.app.get_state(config).values
        best_scenario = final_state.get("best_scenario")

        return {
            "thread_id": current_thread_id,
            "best_scenario_id": getattr(best_scenario, "scenario_id", None),
            "best_scenic_code": getattr(best_scenario, "scenic_code", None),
            "best_score": getattr(best_scenario, "score", None),
            "best_error": getattr(best_scenario, "error", None),
            "model_metrics": normalize_model_metrics_blob(final_state.get("model_metrics")) or {},
            "state": final_state,
        }

    def _to_jsonable(self, value: Any) -> Any:
        """Convert pydantic/custom objects to JSON-serializable data."""
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
        return str(value)

    def run_batch(
        self,
        mode: QueryMode = QueryMode.TEXT_IMAGE_VIDEO,
    ) -> Path:
        """
        Run end-to-end evaluation for all built queries.

        For each query:
        - run workflow with per-query try/except
        - write scenic code to:
          eval/results/e2e_<timestamp>/<best_scenario_id>/<best_scenario_id>.scenic
        - write final_state["scenario_dsl"] to:
          eval/results/e2e_<timestamp>/<best_scenario_id>/<best_scenario_id>.json
        - write final_state["header_settings"] to:
          eval/results/e2e_<timestamp>/<best_scenario_id>/header_settings.json
        - append batch row to CSV (including model_metrics_json: full metrics object)
        """
        query_records = self.build_multimodal_queries(mode=mode)
        n_records = len(query_records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = Path(__file__).resolve().parent / "results" / f"e2e_{timestamp}"
        results_root.mkdir(parents=True, exist_ok=True)
        output_csv_path = results_root / "batch_results.csv"

        self.logger.info(
            "e2e run_batch: built %d record(s), mode=%s, folder_path=%s",
            n_records,
            mode.value,
            self.folder_path,
        )
        self.logger.info("e2e run_batch: results_root=%s", results_root)
        self.logger.info("e2e run_batch: output_csv_path=%s", output_csv_path)
        print(
            f"[e2e_batch] start: {n_records} record(s), mode={mode.value!r}, "
            f"csv={output_csv_path}"
        )

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
            "model_metrics_json",
            "record_total_time_ms",
            "error_message",
        ]

        with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            csvfile.flush()
            hdr_bytes = output_csv_path.stat().st_size
            self.logger.info("e2e run_batch: CSV opened, header written+flush, size_bytes=%s", hdr_bytes)
            print(f"[e2e_batch] CSV header flushed, size_bytes={hdr_bytes}")

            for idx, record in enumerate(query_records, start=1):
                ground_truth = record["ground_truth"]
                query = record["query"]
                error_message = ""
                best_scenario_id = ""
                base_scenario_id = ""
                generation_count: int | str = ""
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
                model_metrics_json = "{}"
                record_total_time_ms = 0.0
                record_start_time = time.perf_counter()

                self.logger.info(
                    "e2e run_batch: (%d/%d) begin ground_truth=%s",
                    idx,
                    n_records,
                    ground_truth,
                )
                print(f"[e2e_batch] ({idx}/{n_records}) BEGIN ground_truth={ground_truth!r}")

                try:
                    self.logger.info(f"[START E2E] Running query for ground_truth={ground_truth}")
                    result = self.run_query_to_output_best_scenario(
                        query=query,
                        thread_id=f"e2e_eval_{ground_truth}_{uuid.uuid4().hex}",
                    )
                    self.logger.info(f"[END E2E] Query for ground_truth={ground_truth} completed")
                    final_state = result.get("state", {})
                    best_scenic_code = result.get("best_scenic_code")
                    best_scenario_id = str(result.get("best_scenario_id") or "")
                    base_scenario_id = str(final_state.get("base_scenario_id") or "")
                    generation_count = final_state.get("generation_count", "")
                    model_metrics = (
                        result.get("model_metrics")
                        or (final_state.get("model_metrics") if isinstance(final_state, dict) else None)
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

                    mm_norm = normalize_model_metrics_blob(model_metrics) or {}
                    model_metrics_json = json.dumps(
                        self._to_jsonable(mm_norm),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

                    if not best_scenario_id:
                        best_scenario_id = f"no_best_scenario_{ground_truth}"

                    query_result_dir = results_root / str(ground_truth)
                    query_result_dir.mkdir(parents=True, exist_ok=True)

                    scenic_path = query_result_dir / f"{best_scenario_id}.scenic"
                    scenic_path.write_text(best_scenic_code or "", encoding="utf-8")

                    dsl_path = query_result_dir / f"{best_scenario_id}.json"
                    dsl_payload = self._to_jsonable(final_state.get("scenario_dsl"))
                    dsl_path.write_text(
                        json.dumps(dsl_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    header_settings_path = query_result_dir / "header_settings.json"
                    header_payload = self._to_jsonable(final_state.get("header_settings"))
                    header_settings_path.write_text(
                        json.dumps(header_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    _try_move_temp_bev_to_eval_result(
                        best_scenario_id=best_scenario_id,
                        query_result_dir=query_result_dir,
                        logger=self.logger,
                    )
                except Exception as exc:
                    error_message = str(exc)
                    self.logger.exception(
                        "Failed to run e2e query for ground_truth=%s",
                        ground_truth,
                    )
                finally:
                    record_total_time_ms = (time.perf_counter() - record_start_time) * 1000.0

                self.logger.info(
                    "e2e run_batch: (%d/%d) writing CSV row ground_truth=%s error_message_len=%d",
                    idx,
                    n_records,
                    ground_truth,
                    len(error_message or ""),
                )
                print(
                    f"[e2e_batch] ({idx}/{n_records}) writing CSV row "
                    f"(record_total_time_ms={record_total_time_ms:.1f}, "
                    f"has_error={bool(error_message)})"
                )

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
                        "model_metrics_json": model_metrics_json,
                        "record_total_time_ms": f"{record_total_time_ms:.2f}",
                        "error_message": error_message,
                    }
                )
                csvfile.flush()
                size_after = output_csv_path.stat().st_size
                self.logger.info(
                    "e2e run_batch: (%d/%d) CSV row flushed ground_truth=%s size_bytes=%s",
                    idx,
                    n_records,
                    ground_truth,
                    size_after,
                )
                print(f"[e2e_batch] ({idx}/{n_records}) row flushed, csv size_bytes={size_after}")

        final_bytes = output_csv_path.stat().st_size
        self.logger.info(
            "Saved e2e batch results to %s (final_size_bytes=%s, records=%d)",
            output_csv_path,
            final_bytes,
            n_records,
        )
        print(f"[e2e_batch] done: csv={output_csv_path} size_bytes={final_bytes}")
        return output_csv_path

if __name__ == "__main__":
    evaluator = EvalE2EWorkflow()
    output_csv2 = evaluator.run_batch(mode=QueryMode.TEXT_VIDEO)
    print(f"Batch done. CSV: {output_csv2}")