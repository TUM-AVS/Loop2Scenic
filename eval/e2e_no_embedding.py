from __future__ import annotations

import csv
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.agents import CriticAgent, InterpreterAgent, ScenicCoderAgent
from src.config import get_config
from src.schema import HeaderSetting, MultimodalQuery, ScenicScenario
from src.services import MilvusVectorStore, get_embedder, get_llm_service, get_vlm_service
from src.utils import (
    find_scenic_code_with_scenario_id,
    flatten_dsl_to_text,
    get_scenario_document_with_scenario_id,
    setup_logging,
)
from src.workflow.scenario_workflow_state import CLEAN_STATE
from src.workflow.workflow import ScenarioWorkflow

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from e2e_metrics import extract_vlm_llm_metrics_rows, normalize_model_metrics_blob

FOLDER_PATH = "data/test"
BASE_SCENARIO_MAP_CSV_PATH = "eval/result/base_scenario_map.csv"


class _NoopRetriever:
    def retrieve(self, *args, **kwargs):
        return []


class _NoopEmbedder:
    dimension = 1

    def encode(self, *args, **kwargs):
        return []


class EvalE2ENoEmbeddingWorkflow:
    """
    End-to-end evaluation pipeline without embed_query/retrieve_base_scenario.

    Inputs are loaded from a folder and a CSV mapping file:
    - For each subfolder under folder_path:
      - ground_truth = subfolder name
      - scenario_dsl = JSON loaded from subfolder DSL file
      - header_settings = loaded from <subfolder>/header_settings.json (if present)
    - base_scenario_id is looked up from csv_path by matching ground_truth.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = get_config(config_path)
        setup_logging(
            level=self.config.logging.level,
            log_format=self.config.logging.format,
        )
        self.logger = logging.getLogger(__name__)
        self.workflow = self._initialize_workflow_without_embed_retrieve()

    def _initialize_workflow_without_embed_retrieve(self) -> ScenarioWorkflow:
        llm_service = get_llm_service(
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            api_key=self.config.llm.api_key,
        )
        vlm_service = get_vlm_service(
            provider=self.config.vlm.provider,
            model=self.config.vlm.model,
            temperature=self.config.vlm.temperature,
            max_tokens=self.config.vlm.max_tokens,
            api_key=self.config.vlm.api_key,
            model_path=self.config.vlm.model_path,
        )

        # Still required for ScenicCoderAgent snippet retrieval.
        snippets_embedder = get_embedder(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda",
        )

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
        vector_db = MilvusVectorStore(
            embedding_dim=snippets_embedder.dimension,
            collection_name=self.config.vector_db.collection_name,
            snippets_collection_name=self.config.vector_db.snippets_collection_name,
            connection_args=connection_args,
            index_params=index_params,
            search_params=search_params,
        )

        interpreter_agent = InterpreterAgent(vlm_service=vlm_service)
        scenic_coder_agent = ScenicCoderAgent(
            llm_service=llm_service,
            vector_store=vector_db,
            snippets_embedder=snippets_embedder,
        )
        critic_agent = CriticAgent(vlm_service=vlm_service)

        return ScenarioWorkflow(
            interpreter=interpreter_agent,
            coder=scenic_coder_agent,
            critic=critic_agent,
            retriever=_NoopRetriever(),
            embedder=_NoopEmbedder(),
            logger=self.logger,
        )

    def _to_jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
        return str(value)

    def _load_base_scenario_map(self, csv_path: Path | str) -> dict[str, str]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV mapping file not found: {path}")

        mapping: dict[str, str] = {}
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if "ground_truth" not in (reader.fieldnames or []) or "base_scenario_id" not in (reader.fieldnames or []):
                raise ValueError("CSV mapping must contain 'ground_truth' and 'base_scenario_id' columns.")
            for row in reader:
                ground_truth = str(row.get("ground_truth") or "").strip()
                base_scenario_id = str(row.get("base_scenario_id") or "").strip()
                if not ground_truth:
                    continue
                if ground_truth in mapping:
                    raise ValueError(f"Duplicate ground_truth in mapping CSV: {ground_truth}")
                mapping[ground_truth] = base_scenario_id
        return mapping

    def _load_json_dict(self, path: Path, context: str) -> dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse {context} JSON at {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{context} JSON must be an object at {path}")
        return raw

    def _find_scenario_dsl_file(self, subfolder: Path) -> Path:
        preferred = [
            "scenario_dsl.json",
            "dsl.json",
            "scenario.json",
            "input_dsl.json",
        ]
        for name in preferred:
            candidate = subfolder / name
            if candidate.is_file():
                return candidate

        json_candidates = sorted(
            p
            for p in subfolder.glob("*.json")
            if p.name != "header_settings.json"
        )
        if len(json_candidates) == 1:
            return json_candidates[0]
        if len(json_candidates) > 1:
            raise ValueError(
                f"Multiple DSL json files found in {subfolder}. "
                f"Please keep one, or use preferred names {preferred}."
            )
        raise FileNotFoundError(
            f"No scenario DSL JSON file found in {subfolder}. "
            f"Tried {preferred} and generic *.json."
        )

    def build_records_from_folder(
        self,
        folder_path: Path | str,
        csv_path: Path | str,
    ) -> list[dict[str, Any]]:
        root = Path(folder_path)
        if not root.exists():
            raise FileNotFoundError(f"Folder path not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Expected folder path, got: {root}")

        base_scenario_map = self._load_base_scenario_map(csv_path)
        records: list[dict[str, Any]] = []

        for subfolder in sorted(root.iterdir()):
            if not subfolder.is_dir():
                continue

            ground_truth = subfolder.name
            base_scenario_id = str(base_scenario_map.get(ground_truth) or "").strip()
            if not base_scenario_id:
                raise ValueError(
                    f"Missing base_scenario_id for ground_truth={ground_truth} in CSV mapping."
                )

            dsl_file = self._find_scenario_dsl_file(subfolder)
            scenario_dsl = self._load_json_dict(dsl_file, context="scenario_dsl")

            header_settings_file = subfolder / "header_settings.json"
            header_settings = None
            if header_settings_file.is_file():
                header_settings = self._load_json_dict(header_settings_file, context="header_settings")

            records.append(
                {
                    "ground_truth": ground_truth,
                    "base_scenario_id": base_scenario_id,
                    "scenario_dsl": scenario_dsl,
                    "header_settings": header_settings,
                }
            )

        self.logger.info("Built %d no-embedding records from %s", len(records), root)
        return records

    def _parse_header_settings(self, record: dict[str, Any]) -> HeaderSetting | None:
        raw = record.get("header_settings")
        if not raw:
            return None
        if isinstance(raw, HeaderSetting):
            return raw
        if isinstance(raw, dict):
            try:
                return HeaderSetting(**raw)
            except Exception:
                self.logger.warning("Invalid header_settings format. Ignoring.")
                return None
        return None

    def _parse_user_query(self, record: dict[str, Any], scenario_dsl: dict[str, Any]) -> MultimodalQuery:
        raw = record.get("user_query")
        if isinstance(raw, MultimodalQuery):
            return raw
        if isinstance(raw, dict):
            return MultimodalQuery(
                text=raw.get("text"),
                image_path=raw.get("image_path"),
                video_path=raw.get("video_path"),
            )

        text = flatten_dsl_to_text(scenario_dsl) or json.dumps(scenario_dsl, ensure_ascii=False)
        return MultimodalQuery(text=text, image_path=None, video_path=None)

    def _build_initial_state(self, record: dict[str, Any]) -> dict[str, Any]:
        scenario_dsl = record.get("scenario_dsl")
        if not isinstance(scenario_dsl, dict):
            raise ValueError("Each record must include 'scenario_dsl' as a JSON object.")

        base_scenario_id = str(record.get("base_scenario_id") or "").strip()
        if not base_scenario_id:
            raise ValueError("Each record must include non-empty 'base_scenario_id'.")

        scenic_code = find_scenic_code_with_scenario_id(base_scenario_id)
        if not scenic_code:
            raise ValueError(f"Cannot find scenic code for base_scenario_id={base_scenario_id}")

        scenario_doc = get_scenario_document_with_scenario_id(base_scenario_id)
        base_scenario = ScenicScenario(
            scenario_id=base_scenario_id,
            scenic_code=scenic_code,
            description=getattr(scenario_doc, "description", None),
        )

        initial_state = deepcopy(CLEAN_STATE)
        initial_state.update(
            {
                "user_query": self._parse_user_query(record, scenario_dsl),
                "scenario_dsl": scenario_dsl,
                "header_settings": self._parse_header_settings(record),
                "base_scenario_id": base_scenario_id,
                "current_scenic_scenario": base_scenario,
                "scenic_scenarios_list": [base_scenario],
            }
        )
        return initial_state

    def run_record(self, record: dict[str, Any]) -> dict[str, Any]:
        state = self._build_initial_state(record)
        loop_guard = 0
        while True:
            loop_guard += 1
            if loop_guard > 10:
                raise RuntimeError("Unexpected loop overflow in no-embedding pipeline.")

            state.update(self.workflow.run_simulation(state))
            state.update(self.workflow.evaluate_with_vlm(state))

            route = self.workflow.route_after_vlm_evaluation(state)
            if route == "output_best_scenario":
                state.update(self.workflow.output_best_scenario(state))
                break

            # Keep same node sequence as original workflow.
            state.update(self.workflow.interpret(state))
            state.update(self.workflow.adapt_code(state))

        best_scenario = state.get("best_scenario")
        return {
            "best_scenario_id": getattr(best_scenario, "scenario_id", None),
            "best_scenic_code": getattr(best_scenario, "scenic_code", None),
            "best_score": getattr(best_scenario, "score", None),
            "best_error": getattr(best_scenario, "error", None),
            "model_metrics": normalize_model_metrics_blob(state.get("model_metrics")) or {},
            "state": state,
        }

    def run_batch(
        self,
        folder_path: Path | str = FOLDER_PATH,
        csv_path: Path | str = BASE_SCENARIO_MAP_CSV_PATH,
    ) -> Path:
        records = self.build_records_from_folder(folder_path=folder_path, csv_path=csv_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = Path(__file__).resolve().parent / "results" / f"e2e_no_embedding_{timestamp}"
        results_root.mkdir(parents=True, exist_ok=True)
        output_csv_path = results_root / "batch_results.csv"

        fieldnames = [
            "ground_truth",
            "base_scenario_id",
            "best_scenario_id",
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

            for record in records:
                ground_truth = str(record.get("ground_truth") or record.get("base_scenario_id") or "")
                base_scenario_id = str(record.get("base_scenario_id") or "")
                error_message = ""
                best_scenario_id = ""
                generation_count: int | str = ""
                vlm_calls = vlm_prompt_tokens = vlm_completion_tokens = vlm_total_tokens = 0
                llm_calls = llm_prompt_tokens = llm_completion_tokens = llm_total_tokens = 0
                vlm_response_time_ms = 0.0
                llm_response_time_ms = 0.0
                record_start = time.perf_counter()

                try:
                    self.logger.info("[START E2E-NO-EMBED] ground_truth=%s", ground_truth)
                    result = self.run_record(record)
                    final_state = result.get("state", {})
                    best_scenic_code = result.get("best_scenic_code")
                    best_scenario_id = str(result.get("best_scenario_id") or "")
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
                        "Failed to run e2e_no_embedding for ground_truth=%s",
                        ground_truth,
                    )
                finally:
                    record_total_time_ms = (time.perf_counter() - record_start) * 1000.0

                writer.writerow(
                    {
                        "ground_truth": ground_truth,
                        "base_scenario_id": base_scenario_id,
                        "best_scenario_id": best_scenario_id,
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

        self.logger.info("Saved e2e no-embedding results to %s", output_csv_path)
        return output_csv_path


if __name__ == "__main__":
    evaluator = EvalE2ENoEmbeddingWorkflow()
    output_csv = evaluator.run_batch(FOLDER_PATH, BASE_SCENARIO_MAP_CSV_PATH)
    print(f"Batch done. CSV: {output_csv}")
