"""Schema for the full VLM critic output (evaluate_with_vlm 3-stage pipeline)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class VLMEvaluationFeedback(TypedDict, total=False):
    event_verification: str
    per_item_notes: str


class VLMFieldEvaluation(TypedDict, total=False):
    """Per-field comparison block (Stage 2)."""

    scenario: bool
    ego: bool
    adversarials: List[bool]
    spatial_relation: bool
    requirements_and_restrictions: bool
    road_side_structures: List[bool]
    temporary_modifications: List[bool]


class VLMKpiMatches(TypedDict, total=False):
    scenario: bool
    ego: bool
    adversarials: bool
    spatial_relation: bool
    requirements_and_restrictions: bool
    road_side_structures: bool
    temporary_modifications: bool


class VLMEvaluationResult(TypedDict, total=False):
    """
    Full critic JSON from evaluate_with_vlm.txt.

    Stage 1 → observed_dsl
    Stage 2 → evaluation (per-field / boolean arrays)
    Stage 3 → kpi_matches, kpi_passed, score
    """

    observed_dsl: Dict[str, Any]
    evaluation: VLMFieldEvaluation
    kpi_matches: VLMKpiMatches
    kpi_passed: int
    feedback: VLMEvaluationFeedback
    score: float | int


def get_comparison_evaluation(
    evaluation_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return the Stage-2 comparison dict used by adapt_code.

    Supports:
    - new full payload: ``{"observed_dsl": ..., "evaluation": {...}, "score": ...}``
    - legacy flat comparison: ``{"scenario": bool, "ego": bool, ...}``
    """
    if not evaluation_result:
        return {}
    nested = evaluation_result.get("evaluation")
    if isinstance(nested, dict):
        return nested
    return evaluation_result
