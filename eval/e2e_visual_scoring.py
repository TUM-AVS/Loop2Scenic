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
    TEMPORAL_DIMENSION_KEY = "Temporal modifications"
    DYNAMIC_OBJECTS_DIMENSION_KEY = "Dynamic objects besides ego"
    SCORE_DIMENSIONS = [
        {
            "key": "Overall score",
            "label": "Overall score (first impression)",
            "layer": "Overall score",
            "levels": [
                ("1", "Totally fit"),
                ("0.5", "Partly fit"),
                ("0", "Not fit at all"),
            ],
            "not_mentioned": "1",
        },
        {
            "key": "road_topology",
            "label": "Road topology (e.g. intersection, straight, curve)",
            "layer": "Layer 1 - road",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "road_lane",
            "label": "Lane lines (e.g. 4-way road)",
            "layer": "Layer 1 - road",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "road_sidewalk",
            "label": "Sidewalks (e.g. pedestrian crosswalk)",
            "layer": "Layer 1 - road",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "traffic_light",
            "label": "Traffic light",
            "layer": "Layer 2 - traffic infrastructure",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "traffic_sign",
            "label": "Traffic sign",
            "layer": "Layer 2 - traffic infrastructure",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "Temporal modifications",
            "label": "Temporal modifications (e.g. traffic cone, construction cone, traffic warning sign, debris)",
            "layer": "Layer 3 - temporal modifications",
            "levels": [("1", "fit"), ("0", "not fit")],
            "not_mentioned": "-1",
        },
        {
            "key": "ego_object",
            "label": "Ego object type (e.g. truck, motorcycle, car)",
            "layer": "Layer 4 - Dynamic objects",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "Ego dynamic behaviors",
            "label": "Ego dynamic behaviors (e.g. accelerating, yielding, remaining stationary)",
            "layer": "Layer 4 - Dynamic objects",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "Dynamic objects besides ego",
            "label": "Dynamic objects besides ego",
            "layer": "Layer 4 - Dynamic objects",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "environment_map_type",
            "label": "Environment - map type (urban/rural/highway)",
            "layer": "Layer 5 - environment",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "environment_illumination",
            "label": "Environment - illumination (e.g. day, night)",
            "layer": "Layer 5 - environment",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
        {
            "key": "environment_weather",
            "label": "Environment - weather (e.g. sunny, rainy, snowy)",
            "layer": "Layer 5 - environment",
            "levels": [("1", "fit"), ("0", "not fit"), ("-1", "not mentioned")],
            "not_mentioned": "-1",
        },
    ]

    @staticmethod
    def _to_float(value: object) -> Optional[float]:
        text = str(value).strip()
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _format_score(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.4f}"

    @staticmethod
    def _to_int(value: object, default: int = 0) -> int:
        text = str(value).strip()
        if text == "":
            return default
        try:
            return int(text)
        except ValueError:
            return default

    def _mean_ignore_value(self, values: List[Optional[float]], ignored: float) -> Optional[float]:
        filtered: List[float] = [v for v in values if v is not None and v != ignored]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    def _apply_layer_scores(self, row: Dict[str, str]) -> Dict[str, str]:
        # Layer 1: average of road fields, ignoring "not mentioned" (-1).
        l1_values = [
            self._to_float(row.get("road_topology", "")),
            self._to_float(row.get("road_lane", "")),
            self._to_float(row.get("road_sidewalk", "")),
        ]
        l1_score = self._mean_ignore_value(l1_values, ignored=-1.0)

        # Layer 2: average of traffic fields, ignoring "not mentioned" (-1).
        l2_values = [
            self._to_float(row.get("traffic_light", "")),
            self._to_float(row.get("traffic_sign", "")),
        ]
        l2_score = self._mean_ignore_value(l2_values, ignored=-1.0)

        # Layer 3: mean of temporal modification items (fit=1, not fit=0).
        temporal_count = self._to_int(row.get("temporal_modifications_count", "0"), default=0)
        temporal_fit_count = self._to_int(row.get("temporal_modifications_fit_count", "0"), default=0)
        if temporal_count > 0:
            l3_score: Optional[float] = temporal_fit_count / temporal_count
        else:
            l3_score = None

        # Layer 4:
        # 1) total agents = dynamic objects count + 1 ego
        # 2) ego score: mean of ego fields, ignoring not mentioned (-1)
        # 3) each dynamic agent score: mean of presence/action/object_type,
        #    ignoring not mentioned (-1)
        # 4) l4_score = (sum of ego + all dynamic agent scores) / total agents
        dynamic_count = self._to_int(row.get("dynamic_objects_count", "0"), default=0)
        total_agents = dynamic_count + 1

        ego_values = [
            self._to_float(row.get("Ego dynamic behaviors", "")),
            self._to_float(row.get("ego_object", "")),
        ]
        ego_score = self._mean_ignore_value(ego_values, ignored=-1.0)
        if ego_score is None:
            ego_score = 0.0

        agent_scores_sum = 0.0
        try:
            agent_scores_raw = json.loads(row.get("dynamic_objects_agent_scores_json", "[]") or "[]")
            if not isinstance(agent_scores_raw, list):
                agent_scores_raw = []
        except json.JSONDecodeError:
            agent_scores_raw = []

        for agent in agent_scores_raw:
            if not isinstance(agent, dict):
                continue
            per_agent_values = [
                self._to_float(agent.get("presence", "")),
                self._to_float(agent.get("action", "")),
                self._to_float(agent.get("object_type", "")),
            ]
            per_agent_score = self._mean_ignore_value(per_agent_values, ignored=-1.0)
            if per_agent_score is None:
                per_agent_score = 0.0
            agent_scores_sum += per_agent_score

        l4_score = (ego_score + agent_scores_sum) / total_agents if total_agents > 0 else None

        # Layer 5: average of environment fields, ignoring not mentioned (-1).
        l5_values = [
            self._to_float(row.get("environment_map_type", "")),
            self._to_float(row.get("environment_illumination", "")),
            self._to_float(row.get("environment_weather", "")),
        ]
        l5_score = self._mean_ignore_value(l5_values, ignored=-1.0)

        row["l1_score"] = self._format_score(l1_score)
        row["l2_score"] = self._format_score(l2_score)
        row["l3_score"] = -1 if l3_score is None else self._format_score(l3_score)
        row["l4_score"] = self._format_score(l4_score)
        row["l5_score"] = self._format_score(l5_score)
        return row

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
            user_query_video_path = str(Path("data") / "scenarios" / ground_truth / "BEV.mp4")
            generated_video_path = str(
                self.folder_path / ground_truth / "generated_video.mp4"
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
        dimension_keys = [dim["key"] for dim in self.SCORE_DIMENSIONS]
        fieldnames = [
            "ground_truth",
            "best_scenario_id",
            "user_query_text",
            "error",
            "l1_score",
            "l2_score",
            "l3_score",
            "l4_score",
            "l5_score",
            "temporal_modifications_count",
            "temporal_modifications_fit_count",
            "temporal_modifications_not_fit_count",
            "dynamic_objects_count",
            "dynamic_objects_presence_fit_count",
            "dynamic_objects_presence_not_fit_count",
            "dynamic_objects_presence_not_mentioned_count",
            "dynamic_objects_action_fit_count",
            "dynamic_objects_action_not_fit_count",
            "dynamic_objects_action_not_mentioned_count",
            "dynamic_objects_object_type_fit_count",
            "dynamic_objects_object_type_not_fit_count",
            "dynamic_objects_object_type_not_mentioned_count",
            "dynamic_objects_agent_scores_json",
        ] + dimension_keys
        with self.score_output_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for key in sorted(self._scores):
                row = dict(self._scores[key])
                writer.writerow(self._apply_layer_scores(row))

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
      text-align: left;
      font-size: 13px;
      padding: 8px 6px;
      background: #fff;
    }
    tbody tr td:last-child { border-right: 1px solid #e5eaf5; }
    .layer {
      text-align: left;
      font-weight: 600;
      min-width: 260px;
      background: #f6f9ff;
      color: #334155;
    }
    .dimension {
      text-align: left;
      font-weight: 600;
      min-width: 220px;
      background: #fafcff;
    }
    input[type="radio"] { accent-color: var(--accent); transform: scale(1.08); }
    .choices { display: flex; flex-wrap: wrap; gap: 12px; }
    .choice { display: inline-flex; align-items: center; gap: 6px; }
    .temporal-box { display: flex; flex-direction: column; gap: 10px; }
    .temporal-count { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }
    .temporal-count input {
      width: 88px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid #d1d9ee;
      font-size: 14px;
    }
    .temporal-items { display: flex; flex-direction: column; gap: 8px; }
    .temporal-item { display: inline-flex; align-items: center; gap: 12px; }
    .temporal-item-label { min-width: 120px; font-weight: 600; color: #475569; }
    .dynamic-box { display: flex; flex-direction: column; gap: 10px; }
    .dynamic-count { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }
    .dynamic-count input {
      width: 88px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid #d1d9ee;
      font-size: 14px;
    }
    .dynamic-items { display: flex; flex-direction: column; gap: 10px; }
    .dynamic-item {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 10px;
      padding: 8px 10px;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      background: #fcfdff;
    }
    .dynamic-item-label { font-weight: 700; color: #334155; }
    .dynamic-field {
      display: inline-flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }
    .dynamic-field-name { font-weight: 600; color: #334155; min-width: 72px; }
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
              <th class="layer">Layer</th>
              <th class="dimension">Dimension</th>
              <th>Choices</th>
            </tr>
          </thead>
          <tbody>
            {% for dim in dimensions %}
            <tr>
              <td class="layer">{{ dim["layer"] }}</td>
              <td class="dimension">{{ dim["label"] }}</td>
              <td>
                {% if dim["key"] == temporal_dimension_key %}
                <div class="temporal-box">
                  <label class="temporal-count">
                    <span>mentioned count:</span>
                    <input
                      type="number"
                      min="0"
                      step="1"
                      name="temporal_modifications_count"
                      id="temporal_modifications_count"
                      value="{{ existing_scores.get('temporal_modifications_count', '0') }}"
                    />
                  </label>
                  <div id="temporal-modification-items" class="temporal-items"></div>
                </div>
                {% elif dim["key"] == dynamic_objects_dimension_key %}
                <div class="dynamic-box">
                  <label class="dynamic-count">
                    <span>agent count:</span>
                    <input
                      type="number"
                      min="0"
                      step="1"
                      name="dynamic_objects_count"
                      id="dynamic_objects_count"
                      value="{{ existing_scores.get('dynamic_objects_count', '0') }}"
                    />
                  </label>
                  <div id="dynamic-object-items" class="dynamic-items"></div>
                </div>
                {% else %}
                <div class="choices">
                  {% for value, option_label in dim["levels"] %}
                    <label class="choice">
                      <input
                        type="radio"
                        name="{{ dim['key'] }}"
                        value="{{ value }}"
                        {% if existing_scores.get(dim['key']) == value or (existing_scores.get("error") == "true" and value == dim["not_mentioned"]) %}checked{% endif %}
                        required
                      />
                      <span>{{ option_label }}</span>
                    </label>
                  {% endfor %}
                </div>
                {% endif %}
              </td>
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

      const dimensions = {{ dimensions|tojson }};
      const temporalDimensionKey = {{ temporal_dimension_key|tojson }};
      const dynamicObjectsDimensionKey = {{ dynamic_objects_dimension_key|tojson }};
      const temporalCountInput = document.getElementById("temporal_modifications_count");
      const temporalItemsContainer = document.getElementById("temporal-modification-items");
      const dynamicCountInput = document.getElementById("dynamic_objects_count");
      const dynamicItemsContainer = document.getElementById("dynamic-object-items");
      const existingScores = {{ existing_scores|tojson }};

      function buildTemporalItems(count) {
        if (!temporalItemsContainer) {
          return;
        }
        temporalItemsContainer.innerHTML = "";
        for (let i = 1; i <= count; i += 1) {
          const row = document.createElement("div");
          row.className = "temporal-item";

          const label = document.createElement("span");
          label.className = "temporal-item-label";
          label.textContent = `item ${i}`;
          row.appendChild(label);

          const fitId = `temporal_mod_item_${i}_fit`;
          const notFitId = `temporal_mod_item_${i}_not_fit`;
          const name = `temporal_mod_item_${i}`;

          const fitLabel = document.createElement("label");
          fitLabel.className = "choice";
          fitLabel.innerHTML = `<input type="radio" id="${fitId}" name="${name}" value="1" required /> <span>fit</span>`;

          const notFitLabel = document.createElement("label");
          notFitLabel.className = "choice";
          notFitLabel.innerHTML = `<input type="radio" id="${notFitId}" name="${name}" value="0" required /> <span>not fit</span>`;

          row.appendChild(fitLabel);
          row.appendChild(notFitLabel);
          temporalItemsContainer.appendChild(row);
        }

        const fitCount = parseInt(existingScores.temporal_modifications_fit_count || "0", 10);
        const notFitCount = parseInt(existingScores.temporal_modifications_not_fit_count || "0", 10);
        if (Number.isFinite(fitCount) && Number.isFinite(notFitCount) && (fitCount + notFitCount) === count) {
          for (let i = 1; i <= count; i += 1) {
            const preferred = i <= fitCount ? "1" : "0";
            const radio = temporalItemsContainer.querySelector(`input[name="temporal_mod_item_${i}"][value="${preferred}"]`);
            if (radio) {
              radio.checked = true;
            }
          }
        }
      }

      function applyNotMentionedSelections() {
        for (const dim of dimensions) {
          if (dim.key === temporalDimensionKey) {
            if (temporalCountInput) {
              temporalCountInput.value = "0";
              buildTemporalItems(0);
            }
            continue;
          }
          if (dim.key === dynamicObjectsDimensionKey) {
            if (dynamicCountInput) {
              dynamicCountInput.value = "0";
              buildDynamicObjectItems(0);
            }
            continue;
          }
          const selector = `input[type="radio"][name="${dim.key}"][value="${dim.not_mentioned}"]`;
          const radio = document.querySelector(selector);
          if (radio) {
            radio.checked = true;
          }
        }
      }

      function sanitizeNonNegativeCount(rawValue) {
        const parsed = Number.parseInt(rawValue, 10);
        if (!Number.isFinite(parsed) || parsed < 0) {
          return 0;
        }
        return parsed;
      }

      function makeChoiceRadio(name, value, label, required) {
        const choiceLabel = document.createElement("label");
        choiceLabel.className = "choice";
        choiceLabel.innerHTML = `<input type="radio" name="${name}" value="${value}" ${required ? "required" : ""} /> <span>${label}</span>`;
        return choiceLabel;
      }

      function buildDynamicObjectItems(count) {
        if (!dynamicItemsContainer) {
          return;
        }
        dynamicItemsContainer.innerHTML = "";

        let existingAgentScores = [];
        try {
          existingAgentScores = JSON.parse(existingScores.dynamic_objects_agent_scores_json || "[]");
          if (!Array.isArray(existingAgentScores)) {
            existingAgentScores = [];
          }
        } catch (e) {
          existingAgentScores = [];
        }

        for (let i = 1; i <= count; i += 1) {
          const row = document.createElement("div");
          row.className = "dynamic-item";

          const itemLabel = document.createElement("span");
          itemLabel.className = "dynamic-item-label";
          itemLabel.textContent = `agent ${i}`;
          row.appendChild(itemLabel);

          const presenceField = document.createElement("div");
          presenceField.className = "dynamic-field";
          presenceField.innerHTML = `<span class="dynamic-field-name">presence</span>`;
          const presenceName = `dynamic_obj_agent_${i}_presence`;
          presenceField.appendChild(makeChoiceRadio(presenceName, "1", "fit", true));
          presenceField.appendChild(makeChoiceRadio(presenceName, "0", "not fit", true));
          presenceField.appendChild(makeChoiceRadio(presenceName, "-1", "not mentioned", true));
          row.appendChild(presenceField);

          const actionField = document.createElement("div");
          actionField.className = "dynamic-field";
          actionField.innerHTML = `<span class="dynamic-field-name">action</span>`;
          const actionName = `dynamic_obj_agent_${i}_action`;
          actionField.appendChild(makeChoiceRadio(actionName, "1", "fit", true));
          actionField.appendChild(makeChoiceRadio(actionName, "0", "not fit", true));
          actionField.appendChild(makeChoiceRadio(actionName, "-1", "not mentioned", true));
          row.appendChild(actionField);

          const objectTypeField = document.createElement("div");
          objectTypeField.className = "dynamic-field";
          objectTypeField.innerHTML = `<span class="dynamic-field-name">object type</span>`;
          const objectTypeName = `dynamic_obj_agent_${i}_object_type`;
          objectTypeField.appendChild(makeChoiceRadio(objectTypeName, "1", "fit", true));
          objectTypeField.appendChild(makeChoiceRadio(objectTypeName, "0", "not fit", true));
          objectTypeField.appendChild(makeChoiceRadio(objectTypeName, "-1", "not mentioned", true));
          row.appendChild(objectTypeField);

          dynamicItemsContainer.appendChild(row);

          const saved = existingAgentScores[i - 1];
          if (saved && typeof saved === "object") {
            const savedPresence = String(saved.presence ?? "");
            const savedAction = String(saved.action ?? "");
            const savedObjectType = String(saved.object_type ?? "");
            const presenceRadio = dynamicItemsContainer.querySelector(
              `input[name="${presenceName}"][value="${savedPresence}"]`
            );
            const actionRadio = dynamicItemsContainer.querySelector(
              `input[name="${actionName}"][value="${savedAction}"]`
            );
            const objectTypeRadio = dynamicItemsContainer.querySelector(
              `input[name="${objectTypeName}"][value="${savedObjectType}"]`
            );
            if (presenceRadio) {
              presenceRadio.checked = true;
            }
            if (actionRadio) {
              actionRadio.checked = true;
            }
            if (objectTypeRadio) {
              objectTypeRadio.checked = true;
            }
          }
        }
      }

      if (temporalCountInput) {
        const initialCount = sanitizeNonNegativeCount(temporalCountInput.value);
        temporalCountInput.value = String(initialCount);
        buildTemporalItems(initialCount);
        temporalCountInput.addEventListener("input", function () {
          const count = sanitizeNonNegativeCount(temporalCountInput.value);
          temporalCountInput.value = String(count);
          buildTemporalItems(count);
        });
      }

      if (dynamicCountInput) {
        const initialCount = sanitizeNonNegativeCount(dynamicCountInput.value);
        dynamicCountInput.value = String(initialCount);
        buildDynamicObjectItems(initialCount);
        dynamicCountInput.addEventListener("input", function () {
          const count = sanitizeNonNegativeCount(dynamicCountInput.value);
          dynamicCountInput.value = String(count);
          buildDynamicObjectItems(count);
        });
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
                existing_scores=existing_scores,
                temporal_dimension_key=self.TEMPORAL_DIMENSION_KEY,
                dynamic_objects_dimension_key=self.DYNAMIC_OBJECTS_DIMENSION_KEY,
            )

        @self.app.post("/submit-score")
        def submit_score():
            index_value = request.form.get("index", type=int, default=0)
            if index_value < 0 or index_value >= len(self.records):
                abort(400, "Invalid record index")

            record = self.records[index_value]
            score_row: Dict[str, str] = {
                "ground_truth": record["ground_truth"],
                "best_scenario_id": record["best_scenario_id"],
                "user_query_text": record["user_query_text"],
                "error": "true" if request.form.get("error") == "true" else "false",
            }
            if score_row["error"] == "true":
                for dim in self.SCORE_DIMENSIONS:
                    score_row[dim["key"]] = dim["not_mentioned"]
                score_row["temporal_modifications_count"] = "0"
                score_row["temporal_modifications_fit_count"] = "0"
                score_row["temporal_modifications_not_fit_count"] = "0"
                score_row["dynamic_objects_count"] = "0"
                score_row["dynamic_objects_presence_fit_count"] = "0"
                score_row["dynamic_objects_presence_not_fit_count"] = "0"
                score_row["dynamic_objects_presence_not_mentioned_count"] = "0"
                score_row["dynamic_objects_action_fit_count"] = "0"
                score_row["dynamic_objects_action_not_fit_count"] = "0"
                score_row["dynamic_objects_action_not_mentioned_count"] = "0"
                score_row["dynamic_objects_object_type_fit_count"] = "0"
                score_row["dynamic_objects_object_type_not_fit_count"] = "0"
                score_row["dynamic_objects_object_type_not_mentioned_count"] = "0"
                score_row["dynamic_objects_agent_scores_json"] = "[]"
            else:
                for dim in self.SCORE_DIMENSIONS:
                    if dim["key"] == self.TEMPORAL_DIMENSION_KEY:
                        raw_count = request.form.get("temporal_modifications_count", "0")
                        try:
                            mention_count = int(raw_count)
                        except ValueError:
                            abort(400, "Temporal modifications count must be an integer")
                        if mention_count < 0:
                            abort(400, "Temporal modifications count cannot be negative")

                        fit_count = 0
                        not_fit_count = 0
                        for i in range(1, mention_count + 1):
                            item_value = request.form.get(f"temporal_mod_item_{i}", "")
                            if item_value not in {"1", "0"}:
                                abort(400, f"Invalid temporal modification score for item {i}")
                            if item_value == "1":
                                fit_count += 1
                            else:
                                not_fit_count += 1

                        score_row["temporal_modifications_count"] = str(mention_count)
                        score_row["temporal_modifications_fit_count"] = str(fit_count)
                        score_row["temporal_modifications_not_fit_count"] = str(not_fit_count)
                        score_row[dim["key"]] = "1" if mention_count > 0 and not_fit_count == 0 else "0"
                        continue
                    if dim["key"] == self.DYNAMIC_OBJECTS_DIMENSION_KEY:
                        raw_count = request.form.get("dynamic_objects_count", "0")
                        try:
                            agent_count = int(raw_count)
                        except ValueError:
                            abort(400, "Dynamic objects count must be an integer")
                        if agent_count < 0:
                            abort(400, "Dynamic objects count cannot be negative")

                        presence_fit = 0
                        presence_not_fit = 0
                        presence_not_mentioned = 0
                        action_fit = 0
                        action_not_fit = 0
                        action_not_mentioned = 0
                        object_type_fit = 0
                        object_type_not_fit = 0
                        object_type_not_mentioned = 0
                        agent_scores: List[Dict[str, int]] = []

                        for i in range(1, agent_count + 1):
                            presence_value = request.form.get(f"dynamic_obj_agent_{i}_presence", "")
                            action_value = request.form.get(f"dynamic_obj_agent_{i}_action", "")
                            object_type_value = request.form.get(f"dynamic_obj_agent_{i}_object_type", "")
                            if presence_value not in {"1", "0", "-1"}:
                                abort(400, f"Invalid dynamic object presence score for agent {i}")
                            if action_value not in {"1", "0", "-1"}:
                                abort(400, f"Invalid dynamic object action score for agent {i}")
                            if object_type_value not in {"1", "0", "-1"}:
                                abort(400, f"Invalid dynamic object type score for agent {i}")

                            if presence_value == "1":
                                presence_fit += 1
                            elif presence_value == "0":
                                presence_not_fit += 1
                            else:
                                presence_not_mentioned += 1

                            if action_value == "1":
                                action_fit += 1
                            elif action_value == "0":
                                action_not_fit += 1
                            else:
                                action_not_mentioned += 1

                            if object_type_value == "1":
                                object_type_fit += 1
                            elif object_type_value == "0":
                                object_type_not_fit += 1
                            else:
                                object_type_not_mentioned += 1

                            agent_scores.append(
                                {
                                    "agent_index": i,
                                    "presence": int(presence_value),
                                    "action": int(action_value),
                                    "object_type": int(object_type_value),
                                }
                            )

                        score_row["dynamic_objects_count"] = str(agent_count)
                        score_row["dynamic_objects_presence_fit_count"] = str(presence_fit)
                        score_row["dynamic_objects_presence_not_fit_count"] = str(presence_not_fit)
                        score_row["dynamic_objects_presence_not_mentioned_count"] = str(presence_not_mentioned)
                        score_row["dynamic_objects_action_fit_count"] = str(action_fit)
                        score_row["dynamic_objects_action_not_fit_count"] = str(action_not_fit)
                        score_row["dynamic_objects_action_not_mentioned_count"] = str(action_not_mentioned)
                        score_row["dynamic_objects_object_type_fit_count"] = str(object_type_fit)
                        score_row["dynamic_objects_object_type_not_fit_count"] = str(object_type_not_fit)
                        score_row["dynamic_objects_object_type_not_mentioned_count"] = str(object_type_not_mentioned)
                        score_row["dynamic_objects_agent_scores_json"] = json.dumps(agent_scores)
                        all_fit = (
                            presence_not_fit == 0
                            and action_not_fit == 0
                            and presence_not_mentioned == 0
                            and action_not_mentioned == 0
                            and object_type_not_fit == 0
                            and object_type_not_mentioned == 0
                        )
                        score_row[dim["key"]] = "-1" if agent_count == 0 else ("1" if all_fit else "0")
                        continue

                    allowed_values = {value for value, _ in dim["levels"]}
                    value = request.form.get(dim["key"], "")
                    if value not in allowed_values:
                        abort(400, f"Invalid score for {dim['label']}")
                    score_row[dim["key"]] = value

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
        csv_path=Path("eval/visual_scoring_csv/C1_vanilla/input.csv"), # the path for the original csv file, recommend to make a copy of the original one
        generated_video_folder_path=Path("data") / "C1_vanilla", # the path for the folder that contains all the generated results
        score_output_csv_path=Path("eval/visual_scoring_csv/C1_vanilla/output_scores.csv"), # the path for the output scroing file
    )
    app.run()