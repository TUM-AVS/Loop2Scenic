#!/usr/bin/env python3
"""
Test ``describe_in_layer_model`` DSL extraction on ``data/benchmark/test_dsl``.

Uses the same fixed image-only / video-only instruction text as e2e_eval, then
calls the configured VLM via ``InterpreterAgent.generate_dsl_from_user_query``.
Each case writes the original DSL JSON to ``dsl.json`` (no flattened text).

Usage:
    conda activate ads-mrag
    python scripts/test_dsl_vlm.py
    python scripts/test_dsl_vlm.py --config config/config.yaml
    python scripts/test_dsl_vlm.py --limit 1   # smoke test one case
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents import InterpreterAgent
from src.config import get_config
from src.schema import MultimodalQuery
from src.services import get_vlm_service
from src.utils import setup_logging

DEFAULT_TEST_DSL_ROOT = _REPO_ROOT / "data" / "benchmark" / "test_dsl"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

IMAGE_ONLY_TEXT = (
    "Generate an autonomous driving test scenario that reproduces the situation "
    "shown in the attached image. Infer the ego vehicle behavior, other road users, "
    "road layout, and spatial relationships entirely from the image; treat the image "
    "as the complete scenario specification."
)
VIDEO_ONLY_TEXT = (
    "Generate an autonomous driving test scenario that reproduces the situation "
    "shown in the attached video. Infer the ego vehicle behavior, other road users, "
    "road layout, and how the scene evolves over time entirely from the video; treat "
    "the video as the complete scenario specification."
)


def _resolve_media_path(
    subfolder: Path, preferred_name: str, extensions: set[str]
) -> Optional[str]:
    preferred = subfolder / preferred_name
    if preferred.is_file():
        return str(preferred.resolve())
    matches = sorted(
        p for p in subfolder.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    return str(matches[0].resolve()) if matches else None


def _init_vlm(config) -> Any:
    kwargs: dict[str, Any] = {
        "provider": config.vlm.provider,
        "model": config.vlm.model,
        "temperature": config.vlm.temperature,
        "max_tokens": config.vlm.max_tokens,
        "api_key": config.vlm.api_key,
    }
    if config.vlm.model_path:
        kwargs["model_path"] = config.vlm.model_path
    if getattr(config.vlm, "base_url", None):
        kwargs["base_url"] = config.vlm.base_url
    return get_vlm_service(**kwargs)


def _discover_cases(test_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for modality, text, preferred, exts in (
        ("image-only", IMAGE_ONLY_TEXT, "image.png", _IMAGE_EXTS),
        ("video-only", VIDEO_ONLY_TEXT, "BEV.mp4", _VIDEO_EXTS),
    ):
        modality_dir = test_root / modality
        if not modality_dir.is_dir():
            continue
        for subfolder in sorted(p for p in modality_dir.iterdir() if p.is_dir()):
            media = _resolve_media_path(subfolder, preferred, exts)
            cases.append(
                {
                    "modality": modality,
                    "name": subfolder.name,
                    "folder": subfolder,
                    "text": text,
                    "image_path": media if modality == "image-only" else None,
                    "video_path": media if modality == "video-only" else None,
                }
            )
    return cases


def _dsl_to_dict(dsl: Any) -> Optional[dict[str, Any]]:
    if dsl is None:
        return None
    if isinstance(dsl, dict):
        return dict(dsl)
    if hasattr(dsl, "model_dump"):
        return dsl.model_dump()
    if hasattr(dsl, "dict"):
        return dsl.dict()
    return None


def _summarize_adversarials(dsl_dict: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not dsl_dict:
        return {"count": 0, "objects": []}
    adversarials = dsl_dict.get("adversarials") or []
    if not isinstance(adversarials, list):
        adversarials = [adversarials]
    objects = []
    for item in adversarials:
        if isinstance(item, dict):
            objects.append(
                {
                    "object": item.get("object"),
                    "behavior": item.get("behavior"),
                }
            )
        else:
            objects.append({"object": str(item), "behavior": None})
    return {"count": len(objects), "objects": objects}


def run_case(interpreter: InterpreterAgent, case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    case_out = out_dir / case["modality"] / case["name"]
    case_out.mkdir(parents=True, exist_ok=True)

    query = MultimodalQuery(
        text=case["text"],
        image_path=case["image_path"],
        video_path=case["video_path"],
    )
    logger.info(
        "Running %s/%s (image=%s video=%s)",
        case["modality"],
        case["name"],
        case["image_path"],
        case["video_path"],
    )

    error: Optional[str] = None
    dsl = None
    try:
        dsl, _flattened = interpreter.generate_dsl_from_user_query(query)
    except Exception as exc:
        error = str(exc)
        logger.exception("DSL generation failed for %s/%s", case["modality"], case["name"])

    dsl_dict = _dsl_to_dict(dsl)
    adv_summary = _summarize_adversarials(dsl_dict)

    if dsl_dict is not None:
        (case_out / "dsl.json").write_text(
            json.dumps(dsl_dict, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Copy folder captions for side-by-side review (not used as VLM input).
    for fname in ("description.txt", "scenario_text_description_by_vlm.txt"):
        src = case["folder"] / fname
        if src.is_file():
            (case_out / f"ref_{fname}").write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8"
            )

    record = {
        "modality": case["modality"],
        "name": case["name"],
        "image_path": case["image_path"],
        "video_path": case["video_path"],
        "ok": dsl_dict is not None and error is None,
        "error": error,
        "adversarial_count": adv_summary["count"],
        "adversarials": adv_summary["objects"],
        "ego": (dsl_dict or {}).get("ego"),
        "spatial_relation": (dsl_dict or {}).get("spatial_relation"),
        "road_side_structures": (dsl_dict or {}).get("road_side_structures"),
        "temporary_modifications": (dsl_dict or {}).get("temporary_modifications"),
        "output_dir": str(case_out),
    }
    (case_out / "summary.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test describe_in_layer_model DSL extraction on benchmark/test_dsl"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML (default: config/config.yaml via get_config)",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_TEST_DSL_ROOT,
        help=f"Folder with image-only/ and video-only/ (default: {DEFAULT_TEST_DSL_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write results (default: eval/results/test_dsl_<timestamp>)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of cases (after discovery order)",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(level=config.logging.level, log_format=config.logging.format)
    logger = logging.getLogger(__name__)
    logger.info(
        "VLM provider=%s model=%s",
        config.vlm.provider,
        config.vlm.model,
    )

    test_root = args.test_root.resolve()
    if not test_root.is_dir():
        logger.error("test_dsl root not found: %s", test_root)
        return 1

    cases = _discover_cases(test_root)
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        logger.error("No cases found under %s", test_root)
        return 1

    out_dir = args.output_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = str(config.vlm.model or config.vlm.provider or "vlm").replace("/", "-").replace(":", "-")
        out_dir = _REPO_ROOT / "eval" / "results" / f"test_dsl_{stamp}_{model_tag}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vlm = _init_vlm(config)
    interpreter = InterpreterAgent(vlm_service=vlm)

    records: list[dict[str, Any]] = []
    for case in cases:
        records.append(run_case(interpreter, case, out_dir))

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "test_root": str(test_root),
                "vlm_provider": config.vlm.provider,
                "vlm_model": config.vlm.model,
                "n_cases": len(records),
                "n_ok": sum(1 for r in records if r["ok"]),
                "cases": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {len(records)} case result(s) under {out_dir}")
    print(f"Batch summary: {summary_path}")
    for r in records:
        status = "OK" if r["ok"] else "FAIL"
        print(
            f"  [{status}] {r['modality']}/{r['name']}: "
            f"adversarials={r['adversarial_count']}"
            + (f" error={r['error']}" if r["error"] else "")
        )
    return 0 if all(r["ok"] for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
