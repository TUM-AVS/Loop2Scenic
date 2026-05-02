"""Shared helpers for reading workflow ``model_metrics`` in eval CSV writers."""

from __future__ import annotations

import json
from typing import Any


def _zero_metric_row() -> dict[str, int | float]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "response_time_ms": 0.0,
    }


def normalize_model_metrics_blob(model_metrics: Any) -> dict[str, Any]:
    if model_metrics is None:
        return {}
    if isinstance(model_metrics, str):
        try:
            model_metrics = json.loads(model_metrics)
        except json.JSONDecodeError:
            return {}
    if hasattr(model_metrics, "model_dump"):
        model_metrics = model_metrics.model_dump()
    if not isinstance(model_metrics, dict):
        return {}
    return model_metrics


def _slice_metric_dict(raw: Any) -> dict[str, int | float]:
    if not isinstance(raw, dict):
        return _zero_metric_row()
    return {
        "calls": int(raw.get("calls", 0) or 0),
        "prompt_tokens": int(raw.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(raw.get("completion_tokens", 0) or 0),
        "total_tokens": int(raw.get("total_tokens", 0) or 0),
        "response_time_ms": float(raw.get("response_time_ms", 0.0) or 0.0),
    }


def _add_metric_rows(a: dict[str, int | float], b: dict[str, int | float]) -> dict[str, int | float]:
    return {
        "calls": int(a["calls"]) + int(b["calls"]),
        "prompt_tokens": int(a["prompt_tokens"]) + int(b["prompt_tokens"]),
        "completion_tokens": int(a["completion_tokens"]) + int(b["completion_tokens"]),
        "total_tokens": int(a["total_tokens"]) + int(b["total_tokens"]),
        "response_time_ms": float(a["response_time_ms"]) + float(b["response_time_ms"]),
    }


def _metric_strength(m: dict[str, int | float]) -> int:
    return int(m["calls"]) * 1_000_000 + int(m["total_tokens"])


def extract_vlm_llm_metrics_rows(model_metrics: Any) -> tuple[dict[str, int | float], dict[str, int | float]]:
    """
    Prefer workflow ``totals_by_type``; if missing or all zeros, sum per-node
    ``vlm`` / ``llm`` blocks under ``by_node`` (matches logged state shape).
    """
    mm = normalize_model_metrics_blob(model_metrics)
    if not mm:
        z = _zero_metric_row()
        return z, z

    tbt = mm.get("totals_by_type")
    if isinstance(tbt, dict):
        vlm = _slice_metric_dict(tbt.get("vlm"))
        llm = _slice_metric_dict(tbt.get("llm"))
        if _metric_strength(vlm) > 0 or _metric_strength(llm) > 0:
            return vlm, llm

    by_node = mm.get("by_node")
    if isinstance(by_node, dict) and by_node:
        vlm_acc = _zero_metric_row()
        llm_acc = _zero_metric_row()
        for node_block in by_node.values():
            if isinstance(node_block, dict):
                vlm_acc = _add_metric_rows(vlm_acc, _slice_metric_dict(node_block.get("vlm")))
                llm_acc = _add_metric_rows(llm_acc, _slice_metric_dict(node_block.get("llm")))
        if _metric_strength(vlm_acc) > 0 or _metric_strength(llm_acc) > 0:
            return vlm_acc, llm_acc

    if isinstance(tbt, dict):
        return _slice_metric_dict(tbt.get("vlm")), _slice_metric_dict(tbt.get("llm"))

    z = _zero_metric_row()
    return z, z
