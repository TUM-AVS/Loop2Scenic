from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for


def load_records_without_error(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Read a CSV file and keep only records without an error message.

    A record is considered "no error message" when the `error_message` field is
    missing, empty, or whitespace-only.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    clean_records: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            error_message = (row.get("error_message") or "").strip()
            if error_message == "":
                clean_records.append(row)

    return clean_records


def write_records_without_error(
    input_csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
) -> int:
    """
    Read records from input CSV, keep rows without error messages, and write them
    to a separate output CSV file.

    Returns the number of filtered data rows written.
    """
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)
    clean_records = load_records_without_error(input_path)

    # Use input headers when available to preserve the original column order.
    with input_path.open("r", newline="", encoding="utf-8") as in_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames or []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_records)

    return len(clean_records)


class VisualScoringWebApp:
    SCORE_DIMENSIONS = [
        "Road topology",
        "Traffic infrastructure",
        "Temporal modifications",
        "Ego dynamic behaviors",
        "Dynamic objects besides ego",
        "Environment",
        "Overall score",
    ]
    SCORE_LEVELS = [
        ("5", "Totally fit"),
        ("4", "Mostly fit"),
        ("3", "Partially fit"),
        ("2", "Slightly fit"),
        ("1", "Not fit at all"),
        ("0", "Not mentioned"),
    ]

    def __init__(
        self,
        csv_path: Union[str, Path],
        generated_video_folder_path: Union[str, Path],
        score_output_csv_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.folder_path = Path(generated_video_folder_path)
        self.project_root = Path(__file__).resolve().parents[1]
        self.score_output_csv_path = (
            Path(score_output_csv_path)
            if score_output_csv_path is not None
            else self.csv_path.with_name(f"{self.csv_path.stem}_scores.csv")
        )
        self.records = self._build_records()
        self._scores = self._load_existing_scores()
        self.app = Flask(__name__)
        self._register_routes()

    def _parse_user_query(self, raw_user_query: str) -> Dict[str, Optional[str]]:
        if not raw_user_query:
            return {"text": "", "image_path": None, "video_path": None}

        try:
            parsed = json.loads(raw_user_query)
        except json.JSONDecodeError:
            parsed = {"text": raw_user_query, "image_path": None, "video_path": None}

        if not isinstance(parsed, dict):
            return {"text": "", "image_path": None, "video_path": None}

        return {
            "text": parsed.get("text", ""),
            "image_path": parsed.get("image_path"),
            "video_path": parsed.get("video_path"),
        }

    def _build_records(self) -> List[Dict[str, str]]:
        rows = load_records_without_error(self.csv_path)
        records: List[Dict[str, str]] = []
        for row in rows:
            ground_truth = (row.get("ground_truth") or "").strip()
            best_scenario_id = (row.get("best_scenario_id") or "").strip()
            user_query = self._parse_user_query(row.get("user_query") or "")

            user_query_image = str(user_query.get("image_path") or "")
            user_query_video_path = str(Path("data") / "scenarios" / ground_truth / "video.mp4")
            generated_video_path = str(
                self.folder_path / best_scenario_id / "video" / "BEV.mp4"
            )

            records.append(
                {
                    "ground_truth": ground_truth,
                    "best_scenario_id": best_scenario_id,
                    "user_query_text": user_query.get("text") or "",
                    "user_query_image": user_query_image,
                    "user_query_image_exists": str(self._media_exists(user_query_image)).lower(),
                    "user_query_video_path": user_query_video_path,
                    "user_query_video_exists": str(self._media_exists(user_query_video_path)).lower(),
                    "generated_video_path": generated_video_path,
                    "generated_video_exists": str(self._media_exists(generated_video_path)).lower(),
                }
            )
        return records

    def _score_key(self, record: Dict[str, str]) -> str:
        return f"{record['ground_truth']}::{record['best_scenario_id']}"

    def _load_existing_scores(self) -> Dict[str, Dict[str, str]]:
        if not self.score_output_csv_path.exists():
            return {}

        existing: Dict[str, Dict[str, str]] = {}
        with self.score_output_csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = f"{row.get('ground_truth', '')}::{row.get('best_scenario_id', '')}"
                existing[key] = row
        return existing

    def _write_scores(self) -> None:
        self.score_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["ground_truth", "best_scenario_id", "user_query_text", "error"] + self.SCORE_DIMENSIONS
        with self.score_output_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(self._scores):
                writer.writerow(self._scores[key])

    def _remove_scored_record_from_source(self, scored_record: Dict[str, str]) -> None:
        """
        Remove one scored record from the source CSV so remaining rows represent
        unscored progress.
        """
        if not self.csv_path.exists():
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        removed = False
        kept_rows: List[Dict[str, str]] = []
        for row in rows:
            is_target = (
                (row.get("ground_truth") or "").strip() == scored_record["ground_truth"]
                and (row.get("best_scenario_id") or "").strip() == scored_record["best_scenario_id"]
            )
            if is_target and not removed:
                removed = True
                continue
            kept_rows.append(row)

        if not removed:
            return

        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

    def _safe_resolve(self, raw_path: str) -> Path:
        requested = Path(raw_path)
        if requested.is_absolute():
            resolved = requested.resolve()
        else:
            resolved = (self.project_root / requested).resolve()

        root = self.project_root.resolve()
        if not str(resolved).startswith(str(root)):
            raise PermissionError("Requested file is outside project root")
        return resolved

    def _media_exists(self, raw_path: str) -> bool:
        if not raw_path:
            return False
        try:
            return self._safe_resolve(raw_path).exists()
        except PermissionError:
            return False

    def _register_routes(self) -> None:
        template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>E2E Visual Scoring</title>
  <style>
    :root {
      --bg: #f4f6fb;
      --card: #ffffff;
      --card-border: #e3e8f3;
      --text: #1f2937;
      --muted: #667085;
      --brand: #4f46e5;
      --brand-soft: #eef0ff;
      --accent: #10b981;
    }
    body {
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }
    .container { max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
    .header {
      background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%);
      color: #fff;
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .header h2 { margin: 0; font-size: 24px; }
    .header .subtitle { margin-top: 8px; font-size: 15px; opacity: 0.95; }
    .nav {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 10px 14px;
      box-shadow: 0 4px 14px rgba(31, 41, 55, 0.06);
    }
    .nav a { color: var(--brand); font-weight: 600; text-decoration: none; }
    .nav a:hover { text-decoration: underline; }
    .media-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 4px 14px rgba(31, 41, 55, 0.06);
    }
    .title {
      font-weight: 700;
      margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 2px solid var(--brand-soft);
      color: #111827;
    }
    .path {
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
      background: #f8faff;
      border: 1px solid #e8ecf8;
      border-radius: 8px;
      padding: 6px 8px;
    }
    .placeholder {
      margin-top: 8px;
      padding: 10px;
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      color: #9a3412;
      font-weight: 600;
      width: fit-content;
    }
    video, img {
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      border-radius: 10px;
      border: 1px solid #dbe2f0;
      margin-top: 8px;
      background: #000;
    }
    .scoring-panel { padding-top: 4px; }
    table { width: 100%; border-collapse: separate; border-spacing: 0; }
    thead th {
      background: #f0f4ff;
      color: #273449;
      font-size: 13px;
      padding: 8px 6px;
      border-top: 1px solid #d9e2fb;
      border-bottom: 1px solid #d9e2fb;
    }
    th:first-child { border-left: 1px solid #d9e2fb; border-top-left-radius: 10px; }
    th:last-child { border-right: 1px solid #d9e2fb; border-top-right-radius: 10px; }
    td {
      border-left: 1px solid #e5eaf5;
      border-bottom: 1px solid #e5eaf5;
      text-align: center;
      font-size: 13px;
      padding: 8px 6px;
      background: #fff;
    }
    tbody tr td:last-child { border-right: 1px solid #e5eaf5; }
    .dimension {
      text-align: left;
      font-weight: 600;
      min-width: 260px;
      background: #fafcff;
    }
    input[type="radio"] { accent-color: var(--accent); transform: scale(1.08); }
    .actions { margin-top: 12px; }
    .save-btn {
      background: linear-gradient(120deg, #10b981 0%, #059669 100%);
      border: none;
      color: #fff;
      font-weight: 700;
      border-radius: 10px;
      padding: 10px 18px;
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(5, 150, 105, 0.24);
    }
    .save-btn:hover { filter: brightness(1.05); }
    .error-toggle {
      margin-bottom: 12px;
      padding: 10px 12px;
      background: #fff1f2;
      border: 1px solid #fecdd3;
      border-radius: 10px;
      color: #9f1239;
      font-weight: 600;
    }
    .error-toggle input[type="checkbox"] {
      margin-right: 8px;
      transform: scale(1.15);
      accent-color: #e11d48;
    }
    @media (max-width: 1100px) {
      .media-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>{{ record["ground_truth"] }}</h2>
      <div class="subtitle">E2E Visual Scoring</div>
    </div>

    <div class="nav">
      <strong>Record {{ index + 1 }} / {{ total }}</strong>
      {% if index > 0 %}
        | <a href="{{ url_for('index', i=index-1) }}">Previous</a>
      {% endif %}
      {% if index + 1 < total %}
        | <a href="{{ url_for('index', i=index+1) }}">Next</a>
      {% endif %}
    </div>

    <form method="post" action="{{ url_for('submit_score') }}">
      <input type="hidden" name="index" value="{{ index }}" />
      <div class="media-row">
        <div class="panel">
          <div class="title">Original Query</div>
          <div>{{ record["user_query_text"] }}</div>
          <div class="path"><strong>Image path:</strong> {{ record["user_query_image"] if record["user_query_image"] else "not provided" }}</div>
          {% if record["user_query_image_exists"] == "true" %}
            <img src="{{ url_for('media_file') }}?path={{ record['user_query_image'] }}" alt="user query image" />
          {% else %}
            <div class="placeholder">not provided</div>
          {% endif %}
          <div class="title" style="margin-top:10px;">Ground Truth Video</div>
          <div class="path"><strong>Video path:</strong> {{ record["user_query_video_path"] if record["user_query_video_path"] else "not provided" }}</div>
          {% if record["user_query_video_exists"] == "true" %}
            <video controls>
              <source src="{{ url_for('media_file') }}?path={{ record['user_query_video_path'] }}" type="video/mp4">
            </video>
          {% else %}
            <div class="placeholder">not provided</div>
          {% endif %}
        </div>

        <div class="panel">
          <div class="title">Generated Video</div>
          <div class="path"><strong>Video path:</strong> {{ record["generated_video_path"] if record["generated_video_path"] else "not provided" }}</div>
          {% if record["generated_video_exists"] == "true" %}
            <video controls>
              <source src="{{ url_for('media_file') }}?path={{ record['generated_video_path'] }}" type="video/mp4">
            </video>
          {% else %}
            <div class="placeholder">not provided</div>
          {% endif %}
        </div>
      </div>

      <div class="panel scoring-panel">
        <div class="title">Scoring</div>
        <label class="error-toggle">
          <input
            type="checkbox"
            name="error"
            value="true"
            {% if existing_scores.get("error") == "true" %}checked{% endif %}
          />
          error scenario
        </label>
        <table>
          <thead>
            <tr>
              <th class="dimension">Dimension</th>
              {% for _, label in levels %}
                <th>{{ label }}</th>
              {% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for dim in dimensions %}
            <tr>
              <td class="dimension">{{ dim }}</td>
              {% for value, _ in levels %}
              <td>
                <input
                  type="radio"
                  name="{{ dim }}"
                  value="{{ value }}"
                  {% if existing_scores.get(dim) == value or (existing_scores.get("error") == "true" and value == "0") %}checked{% endif %}
                  required
                />
              </td>
              {% endfor %}
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>

      <div class="actions">
        <button type="submit" class="save-btn">Save Score</button>
      </div>
    </form>
  </div>
  <script>
    (function () {
      const errorCheckbox = document.querySelector('input[name="error"]');
      if (!errorCheckbox) {
        return;
      }

      const dimensionNames = {{ dimensions|tojson }};
      const notMentionedValue = "0";

      function applyNotMentionedSelections() {
        for (const dim of dimensionNames) {
          const selector = `input[type="radio"][name="${dim}"][value="${notMentionedValue}"]`;
          const radio = document.querySelector(selector);
          if (radio) {
            radio.checked = true;
          }
        }
      }

      errorCheckbox.addEventListener("change", function () {
        if (errorCheckbox.checked) {
          applyNotMentionedSelections();
        }
      });

      if (errorCheckbox.checked) {
        applyNotMentionedSelections();
      }
    })();
  </script>
</body>
</html>
"""

        @self.app.get("/")
        def index():
            if not self.records:
                return "No records available after filtering error rows."

            total = len(self.records)
            index_value = request.args.get("i", default=0, type=int)
            index_value = min(max(index_value, 0), total - 1)
            record = self.records[index_value]
            existing_scores = self._scores.get(self._score_key(record), {})
            return render_template_string(
                template,
                record=record,
                index=index_value,
                total=total,
                dimensions=self.SCORE_DIMENSIONS,
                levels=self.SCORE_LEVELS,
                existing_scores=existing_scores,
            )

        @self.app.post("/submit-score")
        def submit_score():
            index_value = request.form.get("index", type=int, default=0)
            if index_value < 0 or index_value >= len(self.records):
                abort(400, "Invalid record index")

            record = self.records[index_value]
            allowed_values = {value for value, _ in self.SCORE_LEVELS}
            score_row: Dict[str, str] = {
                "ground_truth": record["ground_truth"],
                "best_scenario_id": record["best_scenario_id"],
                "user_query_text": record["user_query_text"],
                "error": "true" if request.form.get("error") == "true" else "false",
            }
            if score_row["error"] == "true":
                for dim in self.SCORE_DIMENSIONS:
                    score_row[dim] = "0"
            else:
                for dim in self.SCORE_DIMENSIONS:
                    value = request.form.get(dim, "")
                    if value not in allowed_values:
                        abort(400, f"Invalid score for {dim}")
                    score_row[dim] = value

            self._scores[self._score_key(record)] = score_row
            self._write_scores()
            self._remove_scored_record_from_source(record)

            # Keep in-memory list in sync so progress updates immediately.
            self.records.pop(index_value)
            if not self.records:
                return "All records have been scored."

            next_index = min(index_value, len(self.records) - 1)
            return redirect(url_for("index", i=next_index))

        @self.app.get("/media")
        def media_file():
            requested_path = request.args.get("path", default="", type=str)
            if not requested_path:
                abort(400, "Missing media path")
            try:
                resolved_path = self._safe_resolve(requested_path)
            except PermissionError:
                abort(403, "Forbidden path")
            if not resolved_path.exists():
                abort(404, f"File not found: {resolved_path}")
            return send_file(resolved_path)

    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    app = VisualScoringWebApp(
        csv_path=Path("eval") / "result" / "visual_scoring.csv",
        generated_video_folder_path=Path("data") / "eval_res",
        score_output_csv_path=Path("eval") / "result" / "visual_scoring_results.csv",
    )
    app.run()