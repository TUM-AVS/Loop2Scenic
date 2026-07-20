from __future__ import annotations

import argparse
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
from typing import Any, Optional, Sequence

_EVAL_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.app import ChatbotWorkflow
from src.schema import MultimodalQuery

from e2e_metrics import extract_vlm_llm_metrics_rows, normalize_model_metrics_blob

BENCHMARK_ROOT = _REPO_ROOT / "data" / "benchmark"
# Legacy single-folder default (kept for backward compatibility).
FOLDER_PATH = str(BENCHMARK_ROOT)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _resolve_media_path(subfolder: Path, preferred_name: str, extensions: set[str]) -> Optional[str]:
    """Prefer ``preferred_name`` if present; else first matching file in the subfolder (sorted)."""
    preferred = subfolder / preferred_name
    if preferred.is_file():
        return str(preferred.resolve())
    matches = sorted(
        p for p in subfolder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    return str(matches[0].resolve()) if matches else None


def _extract_final_candidate_vlm_score(final_state: dict[str, Any], result: dict[str, Any]) -> str:
    """VLM critic score for the scenario actually returned as the final output.

    Prefers ``best_scenario.score`` / ``result["best_score"]``. If that is missing
    (e.g. baseline fallback without a fresh critic pass), looks up the same
    ``scenario_id`` in ``scenic_scenarios_list``. Does not take max over other
    candidates. Treats score ``0`` as valid.
    """
    def _fmt(score: Any) -> str:
        return f"{float(score):.2f}"

    best_score = result.get("best_score")
    if best_score is not None:
        return _fmt(best_score)

    best_scenario = None
    if isinstance(final_state, dict):
        best_scenario = final_state.get("best_scenario")
    score = getattr(best_scenario, "score", None) if best_scenario is not None else None
    if score is not None:
        return _fmt(score)

    best_id = (
        str(result.get("best_scenario_id") or "").strip()
        or str(getattr(best_scenario, "scenario_id", "") or "").strip()
    )
    if not best_id:
        return ""

    for scenario in (final_state.get("scenic_scenarios_list") or []) if isinstance(final_state, dict) else []:
        if str(getattr(scenario, "scenario_id", "") or "").strip() != best_id:
            continue
        score = getattr(scenario, "score", None)
        if score is not None:
            return _fmt(score)
        break
    return ""


def _try_copy_temp_bev_to_eval_result(
    best_scenario_id: str,
    query_result_dir: Path,
    logger: logging.Logger,
    *,
    base_scenario_id: str = "",
    final_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    Copy a generated (or baseline) BEV into ``query_result_dir / generated_video.mp4``.

    Tries, in order:
    1. ``temp/<best_scenario_id>/video/BEV.mp4``
    2. ``temp/<base_scenario_id>/video/BEV.mp4`` (baseline fallback)
    3. Any ``temp/<id>/video/BEV.mp4`` for ids in ``scenic_scenarios_list`` (newest first)
    4. ``data/scenarios/<id>/BEV.mp4`` for best/base ids (corpus baseline clip)

    Uses copy (not move) so ``temp/`` keeps the original for later inspection.
    """
    repo_root = Path(__file__).resolve().parent.parent
    dest = query_result_dir / "generated_video.mp4"

    candidate_ids: list[str] = []
    for sid in (best_scenario_id, base_scenario_id):
        sid = (sid or "").strip()
        if sid and sid not in candidate_ids and not sid.startswith("no_best_scenario_"):
            candidate_ids.append(sid)

    if isinstance(final_state, dict):
        scenarios = final_state.get("scenic_scenarios_list") or []
        # Prefer later (more adapted) sims first when hunting for a rendered BEV.
        for scenario in reversed(list(scenarios)):
            sid = str(getattr(scenario, "scenario_id", "") or "").strip()
            if sid and sid not in candidate_ids:
                candidate_ids.append(sid)

    sources: list[Path] = []
    for sid in candidate_ids:
        sources.append(repo_root / "temp" / sid / "video" / "BEV.mp4")
    for sid in candidate_ids:
        # Corpus baseline video (useful when fallback is library base and temp was cleaned).
        sources.append(repo_root / "data" / "scenarios" / sid / "BEV.mp4")

    src: Optional[Path] = next((p for p in sources if p.is_file()), None)
    if src is None:
        logger.warning(
            "Generated/baseline BEV not found for ids=%s; skip video copy",
            candidate_ids or ["(none)"],
        )
        return

    try:
        query_result_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
        logger.info("Copied BEV %s -> %s", src, dest)
    except OSError as exc:
        logger.warning("Could not copy BEV %s -> %s: %s", src, dest, exc)


class QueryMode(str, Enum):
    TEXT_ONLY = "text-only"
    TEXT_IMAGE = "text-image"
    TEXT_VIDEO = "text-video"
    IMAGE_ONLY = "image-only"
    VIDEO_ONLY = "video-only"
    TEXT_IMAGE_VIDEO = "text-image-video"


# Five modality folders under data/benchmark (folder name == QueryMode.value).
BENCHMARK_CATEGORIES: tuple[QueryMode, ...] = (
    QueryMode.TEXT_ONLY,
    QueryMode.TEXT_IMAGE,
    QueryMode.TEXT_VIDEO,
    QueryMode.IMAGE_ONLY,
    QueryMode.VIDEO_ONLY,
)


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
        folder_path: Path | str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Scan immediate subfolders and build query records.

        Mapping per subfolder:
        - text: content of description.txt if present, else None
        - image_path: image.png if present, else any image in the subfolder (when mode includes image)
        - video_path: BEV.mp4 if present, else any video in the subfolder (when mode includes video)

        Returns:
        - list of {"ground_truth": str, "query": MultimodalQuery}
        """
        source_folder = Path(folder_path) if folder_path is not None else self.folder_path
        if not source_folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {source_folder}")
        if not source_folder.is_dir():
            raise NotADirectoryError(f"Expected directory, got: {source_folder}")

        queries: list[dict[str, Any]] = []
        for subfolder in sorted(source_folder.iterdir()):
            if not subfolder.is_dir():
                continue

            if mode in (QueryMode.TEXT_ONLY, QueryMode.TEXT_IMAGE, QueryMode.TEXT_VIDEO, QueryMode.TEXT_IMAGE_VIDEO):
                description_path = subfolder / "description.txt"
                text = description_path.read_text(encoding="utf-8").strip() if description_path.is_file() else None
            elif mode == QueryMode.IMAGE_ONLY:
                text = (
                    "Generate an autonomous driving test scenario that reproduces the situation "
                    "shown in the attached image. Infer the ego vehicle behavior, other road users, "
                    "road layout, and spatial relationships entirely from the image; treat the image "
                    "as the complete scenario specification."
                )
            elif mode == QueryMode.VIDEO_ONLY:
                text = (
                    "Generate an autonomous driving test scenario that reproduces the situation "
                    "shown in the attached video. Infer the ego vehicle behavior, other road users, "
                    "road layout, and how the scene evolves over time entirely from the video; treat "
                    "the video as the complete scenario specification."
                )

            image_path = None
            video_path = None
            if mode in (QueryMode.TEXT_IMAGE, QueryMode.IMAGE_ONLY, QueryMode.TEXT_IMAGE_VIDEO):
                image_path = _resolve_media_path(subfolder, "image.png", _IMAGE_EXTS)
            if mode in (QueryMode.TEXT_VIDEO, QueryMode.VIDEO_ONLY, QueryMode.TEXT_IMAGE_VIDEO):
                video_path = _resolve_media_path(subfolder, "BEV.mp4", _VIDEO_EXTS)

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
            source_folder,
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

        If the graph aborts mid-run (e.g. coder/debug failure), recover the
        checkpointed state and fall back to the retrieved baseline scenario so
        eval still has scenic code + video to write.
        """
        current_thread_id = thread_id or f"e2e_eval_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": current_thread_id}}
        user_message = {"role": "user", "content": query.model_dump_json()}
        workflow_error = ""

        try:
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
        except Exception as exc:
            workflow_error = str(exc)
            self.logger.exception(
                "Workflow stream failed (thread_id=%s): %s",
                current_thread_id,
                exc,
            )

        try:
            final_state = self.workflow.app.get_state(config).values or {}
        except Exception:
            final_state = {}

        best_scenario = final_state.get("best_scenario") if isinstance(final_state, dict) else None
        if best_scenario is None and isinstance(final_state, dict):
            best_scenario = self._fallback_best_scenario_from_state(final_state)
            if best_scenario is not None:
                final_state = dict(final_state)
                final_state["best_scenario"] = best_scenario
                self.logger.warning(
                    "Recovered baseline scenario after workflow failure: %s",
                    getattr(best_scenario, "scenario_id", None),
                )

        return {
            "thread_id": current_thread_id,
            "best_scenario_id": getattr(best_scenario, "scenario_id", None),
            "best_scenic_code": getattr(best_scenario, "scenic_code", None),
            "best_score": getattr(best_scenario, "score", None),
            "best_error": getattr(best_scenario, "error", None),
            "model_metrics": normalize_model_metrics_blob(
                final_state.get("model_metrics") if isinstance(final_state, dict) else None
            )
            or {},
            "state": final_state if isinstance(final_state, dict) else {},
            "workflow_error": workflow_error,
        }

    def _collect_error_message(
        self,
        *,
        workflow_error: str,
        best_error: Any,
        best_score: Any,
        best_scenario_id: str,
        base_scenario_id: str,
        final_state: dict[str, Any],
    ) -> str:
        """Build CSV error_message from hard failures and soft unresolved issues."""
        parts: list[str] = []
        if workflow_error:
            parts.append(str(workflow_error))

        for message in final_state.get("messages") or []:
            content = ""
            if isinstance(message, dict):
                content = str(message.get("content") or "")
            else:
                content = str(getattr(message, "content", "") or "")
            if content.startswith("[workflow_warning]"):
                parts.append(content.removeprefix("[workflow_warning]").strip())

        used_baseline = (
            bool(base_scenario_id)
            and best_scenario_id == base_scenario_id
            and (best_score is None or best_error)
        )
        unresolved = bool(workflow_error) or bool(best_error) or used_baseline or bool(parts)

        # Only attach sim traceback history when the run did not fully resolve.
        if unresolved:
            for scenario in final_state.get("scenic_scenarios_list") or []:
                err = getattr(scenario, "error", None)
                sid = getattr(scenario, "scenario_id", "") or ""
                if err:
                    parts.append(f"{sid}: {err}" if sid else str(err))
            if best_error:
                parts.append(f"best_scenario_error: {best_error}")
            if used_baseline:
                parts.append(
                    f"Fell back to baseline scenario {base_scenario_id} "
                    "(no successful adapted candidate)"
                )

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_parts: list[str] = []
        for part in parts:
            text = str(part).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique_parts.append(text)
        return " | ".join(unique_parts)

    def _fallback_best_scenario_from_state(self, final_state: dict[str, Any]):
        """Mirror workflow.output_best_scenario baseline fallback for aborted runs."""
        from src.schema import ScenicScenario
        from src.utils import find_scenic_code_with_scenario_id

        scenic_scenarios_list = final_state.get("scenic_scenarios_list") or []
        base_id = str(final_state.get("base_scenario_id") or "").strip()

        if base_id:
            for scenario in scenic_scenarios_list:
                if (
                    str(getattr(scenario, "scenario_id", "") or "").strip() == base_id
                    and getattr(scenario, "scenic_code", None)
                ):
                    return scenario

        for scenario in scenic_scenarios_list:
            if getattr(scenario, "scenic_code", None):
                return scenario

        if not base_id:
            return None

        scenic_code = find_scenic_code_with_scenario_id(base_id)
        if not scenic_code:
            return None
        return ScenicScenario(scenario_id=base_id, scenic_code=scenic_code)

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
        *,
        results_root: Optional[Path] = None,
        output_csv_path: Optional[Path] = None,
        limit: Optional[int] = None,
        write_header: bool = True,
        category: Optional[str] = None,
    ) -> Path:
        """
        Run end-to-end evaluation for all built queries in ``self.folder_path``.

        For each query:
        - run workflow with per-query try/except
        - write scenic / dsl / header under:
          ``results_root / <category> / <ground_truth> /``
        - append batch row to CSV (including model_metrics_json: full metrics object)

        Args:
            mode: modality mode used to build queries from ``self.folder_path``
            results_root: optional shared results directory (created if missing)
            output_csv_path: optional shared CSV path (append if write_header=False)
            limit: optional max number of scenarios to process in this folder
            write_header: whether to write the CSV header (False when appending)
            category: optional category label stored in CSV / result subdir
                (defaults to ``mode.value``)
        """
        category_name = category or mode.value
        query_records = self.build_multimodal_queries(mode=mode)
        if limit is not None:
            query_records = query_records[:limit]
        n_records = len(query_records)

        if results_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_root = Path(__file__).resolve().parent / "results" / f"e2e_{timestamp}"
        results_root = Path(results_root)
        results_root.mkdir(parents=True, exist_ok=True)

        if output_csv_path is None:
            output_csv_path = results_root / "batch_results.csv"
        else:
            output_csv_path = Path(output_csv_path)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        batch_folder = Path(folder_path) if folder_path is not None else self.folder_path
        self.logger.info(
            "e2e run_batch: built %d record(s), mode=%s, category=%s, folder_path=%s",
            n_records,
            mode.value,
            category_name,
            self.folder_path,
        )
        self.logger.info("e2e run_batch: results_root=%s", results_root)
        self.logger.info("e2e run_batch: output_csv_path=%s", output_csv_path)
        print(
            f"[e2e_batch] start: {n_records} record(s), mode={mode.value!r}, "
            f"category={category_name!r}, csv={output_csv_path}"
        )

        fieldnames = [
            "category",
            "ground_truth",
            "user_query",
            "best_scenario_id",
            "best_vlm_eval_score",
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

        csv_mode = "w" if write_header else "a"
        with output_csv_path.open(csv_mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                csvfile.flush()
                hdr_bytes = output_csv_path.stat().st_size
                self.logger.info(
                    "e2e run_batch: CSV opened, header written+flush, size_bytes=%s",
                    hdr_bytes,
                )
                print(f"[e2e_batch] CSV header flushed, size_bytes={hdr_bytes}")

            for idx, record in enumerate(query_records, start=1):
                ground_truth = record["ground_truth"]
                query = record["query"]
                error_message = ""
                best_scenario_id = ""
                best_vlm_eval_score = ""
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
                    "e2e run_batch: [%s] (%d/%d) begin ground_truth=%s",
                    category_name,
                    idx,
                    n_records,
                    ground_truth,
                )
                print(
                    f"[e2e_batch] [{category_name}] ({idx}/{n_records}) "
                    f"BEGIN ground_truth={ground_truth!r}"
                )

                try:
                    self.logger.info(
                        "[START E2E] Running query for category=%s ground_truth=%s",
                        category_name,
                        ground_truth,
                    )
                    result = self.run_query_to_output_best_scenario(
                        query=query,
                        thread_id=f"e2e_eval_{category_name}_{ground_truth}_{uuid.uuid4().hex}",
                    )
                    self.logger.info(
                        "[END E2E] Query for category=%s ground_truth=%s completed",
                        category_name,
                        ground_truth,
                    )
                    final_state = result.get("state", {}) or {}
                    best_scenic_code = result.get("best_scenic_code")
                    best_scenario_id = str(result.get("best_scenario_id") or "")
                    best_vlm_eval_score = _extract_final_candidate_vlm_score(final_state, result)
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
                        # Prefer baseline id for artifact naming when no scored best exists.
                        best_scenario_id = base_scenario_id or f"no_best_scenario_{ground_truth}"
                    if not best_scenic_code and base_scenario_id:
                        from src.utils import find_scenic_code_with_scenario_id

                        best_scenic_code = find_scenic_code_with_scenario_id(base_scenario_id) or ""

                    # Soft + hard failures: keep unresolved issues in CSV even when
                    # baseline scenic/video were still written.
                    error_message = self._collect_error_message(
                        workflow_error=str(result.get("workflow_error") or ""),
                        best_error=result.get("best_error"),
                        best_score=result.get("best_score"),
                        best_scenario_id=best_scenario_id,
                        base_scenario_id=base_scenario_id,
                        final_state=final_state if isinstance(final_state, dict) else {},
                    )

                    query_result_dir = results_root / category_name / str(ground_truth)
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

                    _try_copy_temp_bev_to_eval_result(
                        best_scenario_id=best_scenario_id,
                        query_result_dir=query_result_dir,
                        logger=self.logger,
                        base_scenario_id=base_scenario_id,
                        final_state=final_state if isinstance(final_state, dict) else None,
                    )
                except Exception as exc:
                    error_message = str(exc)
                    self.logger.exception(
                        "Failed to run e2e query for category=%s ground_truth=%s",
                        category_name,
                        ground_truth,
                    )
                finally:
                    record_total_time_ms = (time.perf_counter() - record_start_time) * 1000.0

                self.logger.info(
                    "e2e run_batch: [%s] (%d/%d) writing CSV row ground_truth=%s error_message_len=%d",
                    category_name,
                    idx,
                    n_records,
                    ground_truth,
                    len(error_message or ""),
                )
                print(
                    f"[e2e_batch] [{category_name}] ({idx}/{n_records}) writing CSV row "
                    f"(record_total_time_ms={record_total_time_ms:.1f}, "
                    f"has_error={bool(error_message)})"
                )

                writer.writerow(
                    {
                        "category": category_name,
                        "ground_truth": str(ground_truth),
                        "user_query": query.model_dump_json(),
                        "best_scenario_id": best_scenario_id,
                        "best_vlm_eval_score": best_vlm_eval_score,
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
                    "e2e run_batch: [%s] (%d/%d) CSV row flushed ground_truth=%s size_bytes=%s",
                    category_name,
                    idx,
                    n_records,
                    ground_truth,
                    size_after,
                )
                print(
                    f"[e2e_batch] [{category_name}] ({idx}/{n_records}) "
                    f"row flushed, csv size_bytes={size_after}"
                )

        final_bytes = output_csv_path.stat().st_size
        self.logger.info(
            "Saved e2e batch results to %s (final_size_bytes=%s, records=%d, category=%s)",
            output_csv_path,
            final_bytes,
            n_records,
            category_name,
        )
        print(
            f"[e2e_batch] done category={category_name!r}: "
            f"csv={output_csv_path} size_bytes={final_bytes}"
        )
        return output_csv_path

    def run_benchmark_categories(
        self,
        categories: Sequence[QueryMode] = BENCHMARK_CATEGORIES,
        *,
        benchmark_root: Path | str = BENCHMARK_ROOT,
        limit: Optional[int] = None,
    ) -> Path:
        """
        Run e2e evaluation for each modality folder under ``data/benchmark``.

        Uses one shared ``eval/results/e2e_<timestamp>/`` tree and one CSV:
          - results: ``e2e_<ts>/<category>/<scenario>/...``
          - CSV: ``batch_results.csv`` with a ``category`` column
        """
        benchmark_root = Path(benchmark_root)
        if not benchmark_root.is_dir():
            raise NotADirectoryError(f"Benchmark root does not exist: {benchmark_root}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = Path(__file__).resolve().parent / "results" / f"e2e_{timestamp}"
        results_root.mkdir(parents=True, exist_ok=True)
        output_csv_path = results_root / "batch_results.csv"

        print(
            f"[e2e_batch] benchmark start: root={benchmark_root}, "
            f"categories={[c.value for c in categories]}, results={results_root}"
        )
        self.logger.info(
            "e2e run_benchmark_categories: root=%s categories=%s results=%s",
            benchmark_root,
            [c.value for c in categories],
            results_root,
        )

        for idx, mode in enumerate(categories):
            category_dir = benchmark_root / mode.value
            if not category_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing benchmark category folder: {category_dir}"
                )
            self.folder_path = category_dir
            self.run_batch(
                mode=mode,
                results_root=results_root,
                output_csv_path=output_csv_path,
                limit=limit,
                write_header=(idx == 0),
                category=mode.value,
            )

        print(f"[e2e_batch] benchmark done: csv={output_csv_path}")
        return output_csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end ADS-MRAG evaluation on data/benchmark modality folders "
            "(text-only, text-image, text-video, image-only, video-only)."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=BENCHMARK_ROOT,
        help=f"Parent of the five modality folders (default: {BENCHMARK_ROOT})",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in BENCHMARK_CATEGORIES],
        default=[c.value for c in BENCHMARK_CATEGORIES],
        help="Subset of categories to run (default: all five)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N scenarios per category (smoke test)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to config YAML",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    categories = tuple(QueryMode(c) for c in args.categories)
    evaluator = EvalE2EWorkflow(
        folder_path=args.benchmark_root,
        config_path=args.config_path,
    )
    output_csv = evaluator.run_benchmark_categories(
        categories=categories,
        benchmark_root=args.benchmark_root,
        limit=args.limit,
    )
    print(f"Batch done. CSV: {output_csv}")
