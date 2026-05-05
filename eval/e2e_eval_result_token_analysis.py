from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class E2EEvalResultTokenAnalysis:
    """Analyze token/time metrics from e2e evaluation batch result CSV."""

    AVERAGE_FIELDS = [
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
    ]

    def __init__(self, csv_file_path: str | Path) -> None:
        self.csv_file_path = Path(csv_file_path)
        self._rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, str]]:
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        with self.csv_file_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]

    @staticmethod
    def _has_error_message(row: Dict[str, str]) -> bool:
        return bool(str(row.get("error_message", "")).strip())

    def split_by_error_message(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split rows into 2 groups:
        1) rows with non-empty error_message
        2) rows with empty error_message
        """
        with_error: List[Dict[str, str]] = []
        without_error: List[Dict[str, str]] = []

        for row in self._rows:
            if self._has_error_message(row):
                with_error.append(row)
            else:
                without_error.append(row)

        return with_error, without_error

    def print_error_messages(self) -> None:
        """Print each non-empty error message line by line."""
        with_error, _ = self.split_by_error_message()
        if not with_error:
            print("No records with error_message.")
            return

        print(f"Records with error_message: {len(with_error)}")
        for idx, row in enumerate(with_error, start=1):
            ground_truth = str(row.get("ground_truth", "")).strip()
            message = str(row.get("error_message", "")).strip()
            print(f"{idx}. ground_truth={ground_truth} | error_message={message}")

    @staticmethod
    def _parse_float(value: object) -> Optional[float]:
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def calculate_averages_without_error_message(self) -> Dict[str, Optional[float]]:
        """
        For rows without error_message, calculate averages for required numeric fields.
        Returns:
            dict[field_name, average_value_or_none]
        """
        _, without_error = self.split_by_error_message()
        averages: Dict[str, Optional[float]] = {}

        for field in self.AVERAGE_FIELDS:
            values: List[float] = []
            for row in without_error:
                numeric_value = self._parse_float(row.get(field, ""))
                if numeric_value is not None:
                    values.append(numeric_value)

            averages[field] = (sum(values) / len(values)) if values else None

        return averages

    def count_generation_count_equal_one(self) -> int:
        """Count records without error_message where generation_count equals 1."""
        count = 0
        _, without_error = self.split_by_error_message()
        for row in without_error:
            value = self._parse_float(row.get("generation_count", ""))
            if value == 1:
                count += 1
        return count

    def count_generation_count_equal_zero(self) -> int:
        """Count records without error_message where generation_count equals 0."""
        count = 0
        _, without_error = self.split_by_error_message()
        for row in without_error:
            value = self._parse_float(row.get("generation_count", ""))
            if value == 0:
                count += 1
        return count


def _format_average(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python eval/e2e_eval_result_token_analysis.py <path_to_csv>")
        return

    csv_path = Path(sys.argv[1])
    analysis = E2EEvalResultTokenAnalysis(csv_path)

    with_error, without_error = analysis.split_by_error_message()
    print(f"CSV: {csv_path}")
    print(f"Total records: {len(with_error) + len(without_error)}")
    print(f"With error_message: {len(with_error)}")
    print(f"Without error_message: {len(without_error)}")
    print(f"generation_count == 0: {analysis.count_generation_count_equal_zero()}")
    print(f"generation_count == 1: {analysis.count_generation_count_equal_one()}")
    print()

    print("Error messages:")
    analysis.print_error_messages()
    print()

    print("Averages for rows without error_message:")
    averages = analysis.calculate_averages_without_error_message()
    for field in analysis.AVERAGE_FIELDS:
        print(f"  - {field}: {_format_average(averages.get(field))}")


if __name__ == "__main__":
    main()
