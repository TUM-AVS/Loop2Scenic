from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class E2EEvalResultAnalysis:
    """Analyze visual scoring CSV results."""

    FIELD_L1_SCORE = "l1_score"
    FIELD_L2_SCORE = "l2_score"
    FIELD_L3_SCORE = "l3_score"
    FIELD_L4_SCORE = "l4_score"
    FIELD_L5_SCORE = "l5_score"

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

    def _average_field_error_false_ignore_minus_one(self, field_name: str) -> Optional[float]:
        """
        Average one scoring field over rows where `error` is false.
        Values equal to "-1" (or numeric -1) are ignored.

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

            if score == -1:
                continue
            values.append(score)

        if not values:
            return None
        return sum(values) / len(values)

    def average_l1_score(self) -> Optional[float]:
        return self._average_field_error_false_ignore_minus_one(self.FIELD_L1_SCORE)

    def average_l2_score(self) -> Optional[float]:
        return self._average_field_error_false_ignore_minus_one(self.FIELD_L2_SCORE)

    def average_l3_score(self) -> Optional[float]:
        return self._average_field_error_false_ignore_minus_one(self.FIELD_L3_SCORE)

    def average_l4_score(self) -> Optional[float]:
        return self._average_field_error_false_ignore_minus_one(self.FIELD_L4_SCORE)

    def average_l5_score(self) -> Optional[float]:
        return self._average_field_error_false_ignore_minus_one(self.FIELD_L5_SCORE)

    def average_overall_layer_score(self) -> Optional[float]:
        """
        Compute overall layer score by:
        1) For each record, average available l1..l5 values
           (skip non-numeric values and -1).
        2) Average these per-record averages.
        """
        per_record_averages: List[float] = []
        layer_fields = [
            self.FIELD_L1_SCORE,
            self.FIELD_L2_SCORE,
            self.FIELD_L3_SCORE,
            self.FIELD_L4_SCORE,
            self.FIELD_L5_SCORE,
        ]

        for row in self._iter_rows_with_error_false():
            values: List[float] = []
            for field_name in layer_fields:
                raw = str(row.get(field_name, "")).strip()
                if not raw:
                    continue
                try:
                    score = float(raw)
                except ValueError:
                    continue
                if score == -1:
                    continue
                values.append(score)

            if values:
                per_record_averages.append(sum(values) / len(values))

        if not per_record_averages:
            return None
        return sum(per_record_averages) / len(per_record_averages)


def _format_average(value: Optional[float]) -> str:
    return "N/A (no valid non--1 values)" if value is None else f"{value:.4f}"


def _count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return sum(1 for _ in reader)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python eval/e2e_eval_result_analysis.py "
            "<path_to_scored_csv> [path_to_reference_csv_for_success_rate]"
        )
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
    l1_avg = analysis.average_l1_score()
    l2_avg = analysis.average_l2_score()
    l3_avg = analysis.average_l3_score()
    l4_avg = analysis.average_l4_score()
    l5_avg = analysis.average_l5_score()
    overall_avg = analysis.average_overall_layer_score()

    print("Averages on rows where error=false (ignoring score=-1):")
    print(f"  - l1_score: {_format_average(l1_avg)}")
    print(f"  - l2_score: {_format_average(l2_avg)}")
    print(f"  - l3_score: {_format_average(l3_avg)}")
    print(f"  - l4_score: {_format_average(l4_avg)}")
    print(f"  - l5_score: {_format_average(l5_avg)}")
    print(f"  - overall_layer_score: {_format_average(overall_avg)}")

    if len(sys.argv) >= 3:
        reference_csv_path = Path(sys.argv[2])
        reference_total = _count_csv_rows(reference_csv_path)
        if reference_total == 0:
            print("Success rate: N/A (reference CSV has 0 rows)")
        else:
            success_rate = error_false_count / reference_total
            print(
                "Success rate "
                f"(error=false count / reference rows): {success_rate:.4f} "
                f"({error_false_count}/{reference_total})"
            )


if __name__ == "__main__":
    main()
