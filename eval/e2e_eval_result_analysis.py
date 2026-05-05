from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class E2EEvalResultAnalysis:
    """Analyze visual scoring CSV results."""

    FIELD_ROAD_TOPOLOGY = "Road topology"
    FIELD_TRAFFIC_INFRASTRUCTURE = "Traffic infrastructure"
    FIELD_TEMPORAL_MODIFICATIONS = "Temporal modifications"
    FIELD_EGO_DYNAMIC_BEHAVIORS = "Ego dynamic behaviors"
    FIELD_DYNAMIC_OBJECTS_BESIDES_EGO = "Dynamic objects besides ego"
    FIELD_ENVIRONMENT = "Environment"
    FIELD_OVERALL_SCORE = "Overall score"

    def __init__(self, csv_file_path: str | Path) -> None:
        self.csv_file_path = Path(csv_file_path)
        self._rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, str]]:
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        with self.csv_file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    @staticmethod
    def _parse_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value

        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y"}

    def count_error_false_and_error_true_ground_truths(self) -> Tuple[int, List[str]]:
        """
        Return:
        1) number of records where `error` is false
        2) list of `ground_truth` values where `error` is true
        """
        error_false_count = 0
        error_true_ground_truths: List[str] = []

        for row in self._rows:
            is_error = self._parse_bool(row.get("error", ""))
            if is_error:
                gt = str(row.get("ground_truth", "")).strip()
                if gt:
                    error_true_ground_truths.append(gt)
            else:
                error_false_count += 1

        return error_false_count, error_true_ground_truths

    def _iter_rows_with_error_false(self) -> Iterable[Dict[str, str]]:
        for row in self._rows:
            if not self._parse_bool(row.get("error", "")):
                yield row

    def _average_field_error_false_ignore_zero(self, field_name: str) -> Optional[float]:
        """
        Average one scoring field over rows where `error` is false.
        Values equal to "0" (or numeric 0) are ignored.

        Returns None when there is no valid value to average.
        """
        values: List[float] = []
        for row in self._iter_rows_with_error_false():
            raw = str(row.get(field_name, "")).strip()
            if not raw:
                continue

            try:
                score = float(raw)
            except ValueError:
                continue

            if score == 0:
                continue
            values.append(score)

        if not values:
            return None
        return sum(values) / len(values)

    def average_road_topology(self) -> Optional[float]:
        return self._average_field_error_false_ignore_zero(self.FIELD_ROAD_TOPOLOGY)

    def average_traffic_infrastructure(self) -> Optional[float]:
        return self._average_field_error_false_ignore_zero(self.FIELD_TRAFFIC_INFRASTRUCTURE)

    def average_temporal_modifications(self) -> Optional[float]:
        return self._average_field_error_false_ignore_zero(self.FIELD_TEMPORAL_MODIFICATIONS)

    def average_ego_dynamic_behaviors(self) -> Optional[float]:
        return self._average_field_error_false_ignore_zero(self.FIELD_EGO_DYNAMIC_BEHAVIORS)

    def average_dynamic_objects_besides_ego(self) -> Optional[float]:
        return self._average_field_error_false_ignore_zero(self.FIELD_DYNAMIC_OBJECTS_BESIDES_EGO)

    def average_environment(self) -> Optional[float]:
        return self._average_field_error_false_ignore_zero(self.FIELD_ENVIRONMENT)

    def average_overall_score(self) -> Optional[float]:
        return self._average_field_error_false_ignore_zero(self.FIELD_OVERALL_SCORE)


def _format_average(value: Optional[float]) -> str:
    return "N/A (no valid non-zero values)" if value is None else f"{value:.4f}"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python eval/e2e_eval_result_analysis.py <path_to_csv>")
        return

    csv_path = Path(sys.argv[1])
    analysis = E2EEvalResultAnalysis(csv_path)

    error_false_count, error_true_ground_truths = (
        analysis.count_error_false_and_error_true_ground_truths()
    )

    print(f"CSV: {csv_path}")
    print(f"Records with error=false: {error_false_count}")
    print(
        f"ground_truth list where error=true ({len(error_true_ground_truths)}): "
        f"{error_true_ground_truths}"
    )
    print("Averages on rows where error=false (ignoring score=0):")
    print(f"  - Road topology: {_format_average(analysis.average_road_topology())}")
    print(
        "  - Traffic infrastructure: "
        f"{_format_average(analysis.average_traffic_infrastructure())}"
    )
    print(
        "  - Temporal modifications: "
        f"{_format_average(analysis.average_temporal_modifications())}"
    )
    print(
        "  - Ego dynamic behaviors: "
        f"{_format_average(analysis.average_ego_dynamic_behaviors())}"
    )
    print(
        "  - Dynamic objects besides ego: "
        f"{_format_average(analysis.average_dynamic_objects_besides_ego())}"
    )
    print(f"  - Environment: {_format_average(analysis.average_environment())}")
    print(f"  - Overall score: {_format_average(analysis.average_overall_score())}")


if __name__ == "__main__":
    main()
