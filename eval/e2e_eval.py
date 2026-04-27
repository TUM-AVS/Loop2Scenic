from __future__ import annotations

import csv
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.app import ChatbotWorkflow
from src.schema import MultimodalQuery

FOLDER_PATH = "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/test"


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
        - append batch row to CSV with user_query, base_scenario_id, generation_count
        """
        query_records = self.build_multimodal_queries(mode=mode)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = Path(__file__).resolve().parent / "results" / f"e2e_{timestamp}"
        results_root.mkdir(parents=True, exist_ok=True)
        output_csv_path = results_root / "batch_results.csv"

        fieldnames = [
            "ground_truth",
            "user_query",
            "best_scenario_id",
            "base_scenario_id",
            "generation_count",
            "error_message",
        ]

        with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in query_records:
                ground_truth = record["ground_truth"]
                query = record["query"]
                error_message = ""
                best_scenario_id = ""
                base_scenario_id = ""
                generation_count: int | str = ""

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
                except Exception as exc:
                    error_message = str(exc)
                    self.logger.exception(
                        "Failed to run e2e query for ground_truth=%s",
                        ground_truth,
                    )

                writer.writerow(
                    {
                        "ground_truth": str(ground_truth),
                        "user_query": query.model_dump_json(),
                        "best_scenario_id": best_scenario_id,
                        "base_scenario_id": base_scenario_id,
                        "generation_count": generation_count,
                        "error_message": error_message,
                    }
                )

        self.logger.info("Saved e2e batch results to %s", output_csv_path)
        return output_csv_path

if __name__ == "__main__":
    evaluator = EvalE2EWorkflow()
    # output_csv1 = evaluator.run_batch(mode=QueryMode.TEXT_ONLY)
    # print(f"Batch done. CSV: {output_csv1}")
    output_csv2 = evaluator.run_batch(mode=QueryMode.TEXT_VIDEO)
    print(f"Batch done. CSV: {output_csv2}")