"""
B1b: Reciprocal Rank Fusion of two retrieval runs (dense + BM25) — pure offline post-processing.

Reads two stamped run CSVs, fuses their ranked best_scenario_ids lists per query with
RRF (score = sum 1/(k + rank)), and emits a fused CSV into eval/results/rag/runs/rrf-hybrid/
in the same stamped format (aggregator-compatible).

Usage:
    python eval/rrf_fusion.py --dense <dense.csv> --sparse <bm25.csv> [--k 60] [--top-k 3]
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def load_rankings(path: str) -> dict[str, list[str]]:
    out = {}
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if row.get("error_message", "").strip():
            continue
        try:
            ids = json.loads(row["best_scenario_ids"]) if row.get("best_scenario_ids") else []
        except Exception:
            ids = []
        out[row["ground_truth"]] = [str(x) for x in ids]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RRF fusion of two run CSVs")
    parser.add_argument("--dense", required=True)
    parser.add_argument("--sparse", required=True)
    parser.add_argument("--k", type=int, default=60, help="RRF constant")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    dense = load_rankings(args.dense)
    sparse = load_rankings(args.sparse)
    common = sorted(set(dense) & set(sparse))
    print(f"fusing {len(common)} queries (dense={len(dense)}, sparse={len(sparse)})")

    dense_path = Path(args.dense)
    # Unique slug so rapid successive runs don't overwrite each other, and so
    # aggregate_rag_results.py keeps one summary row per dense parent (not one global rrf-hybrid).
    dense_slug = f"{dense_path.parent.name}__{dense_path.stem}"
    dense_slug = "".join(c if c.isalnum() or c in "-._" else "-" for c in dense_slug)[:120]
    # Infer mode stamp from dense filename when possible (text_only / text_video / video_only).
    mode = "text-only"
    stem_lower = dense_path.name.lower()
    if "text_video" in stem_lower or "text-video" in stem_lower:
        mode = "text-video"
    elif "video_only" in stem_lower or "video-only" in stem_lower:
        mode = "video-only"

    out_dir = Path("eval/results/rag/runs/rrf-hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # microseconds → unique even in the same second
    out_path = out_dir / f"{mode.replace('-', '_')}__fused__{dense_slug}__norerank__{ts}.csv"
    embedder_label = f"rrf-hybrid::{dense_slug}"

    fieldnames = ["ground_truth", "user_query", "scenario_dsl", "flattened_dsl", "base_scenario_id",
                  "best_scenario_ids", "best_similarity_score", "best_rerank_score", "response_time_sec",
                  "error_message", "mode", "query_text_source", "query_repr", "rerank_on", "embedder", "top_k"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for gt in common:
            scores: dict[str, float] = {}
            for ranking in (dense[gt], sparse[gt]):
                for rank, sid in enumerate(ranking):
                    scores[sid] = scores.get(sid, 0.0) + 1.0 / (args.k + rank + 1)
            fused = sorted(scores, key=scores.get, reverse=True)[: args.top_k]
            writer.writerow({
                "ground_truth": gt,
                "user_query": json.dumps({"fusion": [args.dense, args.sparse]}),
                "scenario_dsl": "", "flattened_dsl": "",
                "base_scenario_id": fused[0] if fused else "",
                "best_scenario_ids": json.dumps(fused),
                "best_similarity_score": f"{scores.get(fused[0], 0):.6f}" if fused else "",
                "best_rerank_score": "",
                "response_time_sec": "0",
                "error_message": "",
                "mode": mode,
                "query_text_source": f"fused(dense+bm25)::{dense_slug}",
                "query_repr": "raw",
                "rerank_on": "False",
                "embedder": embedder_label,
                "top_k": str(args.top_k),
            })
    print(f"RRF_DONE -> {out_path}")


if __name__ == "__main__":
    main()
