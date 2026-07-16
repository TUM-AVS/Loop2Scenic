"""
Aggregate all structured RAG eval CSVs into one summary table.

Walks eval/results/rag/runs/**/*.csv, scores each with analyze_retrieval_csv, reads the
stamped variant columns (mode, query_text_source, rerank_on, embedder, top_k), and writes
eval/results/rag/summaries/summary.csv — one row per run, ready for the paper table.

Usage:
    python eval/aggregate_rag_results.py [--runs-dir eval/results/rag/runs] [--markdown]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.retrieval_eval import analyze_retrieval_csv  # noqa: E402


def derive_reranker(name: str) -> str:
    """Which reranker a run used, recovered from the filename (rerank_on alone can't tell 2B/8B/text)."""
    n = name.lower()
    if "norerank" in n:
        return "none"
    for suf, lab in [("txt06rr", "txt-0.6B"), ("txt4rr", "txt-4B"), ("txt8rr", "txt-8B"),
                     ("2brr", "vl-2B"), ("8brr", "vl-8B"),
                     ("phase2_validation", "vl-8B"), ("phase2", "vl-8B")]:
        if suf in n:
            return lab
    if "rerank" in n:  # live originals (Table-2 2B reranker)
        return "vl-2B"
    return "none"


def derive_ts(path: Path) -> str:
    """Sortable run timestamp from the filename (YYYYMMDD[_HHMMSS]); fall back to file mtime."""
    m = re.findall(r"(\d{8}_\d{6}|\d{8})", path.name)
    if m:
        t = max(m)
        return t if "_" in t else t + "_000000"
    return "00000000_000000"

# Embedding dimension per model (matches image-16's "dimensions" column).
EMBED_DIM = {
    "Qwen3-VL-Embedding-2B": 2048, "Qwen3-VL-Embedding-8B": 4096,
    "gemini-embedding-2": 3072, "BAAI/bge-m3": 1024,
    "./models/Qwen3-Embedding-0.6B": 1024, "./models/Qwen3-Embedding-4B": 2560,
    "./models/Qwen3-Embedding-8B": 4096, "bm25": "n/a", "rrf-hybrid": "n/a",
}


def first_row_stamp(csv_path: Path) -> dict:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            return {
                "embedder": row.get("embedder", csv_path.parent.name),
                "mode": row.get("mode", ""),
                "query_text_source": row.get("query_text_source", ""),
                "query_repr": row.get("query_repr", "raw"),
                "rerank_on": row.get("rerank_on", ""),
                "top_k": row.get("top_k", ""),
            }
    return {"embedder": csv_path.parent.name, "mode": "", "query_text_source": "", "query_repr": "raw", "rerank_on": "", "top_k": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate structured RAG eval runs")
    parser.add_argument("--runs-dir", default="eval/results/rag/runs")
    parser.add_argument("--markdown", action="store_true", help="Also print a markdown table")
    parser.add_argument("--no-dedup", action="store_true",
                        help="keep every run (default: collapse re-runs of the same variant, keep latest)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = runs_dir.parent / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(runs_dir.glob("**/*.csv")):
        try:
            stats = analyze_retrieval_csv(csv_path)
        except Exception as exc:
            print(f"SKIP {csv_path}: {exc}", file=sys.stderr)
            continue
        stamp = first_row_stamp(csv_path)
        rows.append({
            **stamp,
            "reranker": derive_reranker(csv_path.name),
            "run_ts": derive_ts(csv_path),
            "dimension": EMBED_DIM.get(stamp["embedder"], ""),
            "hit@1": round(stats["ground_truth_eq_base_scenario_id_rate_among_no_error"], 2),
            "hit@3": round(stats["ground_truth_in_best_scenario_ids_rate_among_no_error"], 2),
            "mrr@3": round(stats["mrr_among_no_error"], 4),
            "no_error_rate": round(stats["no_error_rate"], 2),
            "avg_latency_sec": round(stats["avg_response_time_sec_among_no_error"], 4),
            "avg_similarity": round(stats["avg_best_similarity_score_among_no_error"], 4),
            "avg_rerank_score": round(stats["avg_best_rerank_score_among_no_error"], 4),
            "csv": str(csv_path),
        })

    if not rows:
        raise SystemExit(f"No CSVs found under {runs_dir}")

    # Dedup: one row per full variant (embedder, mode, query source, repr, reranker, top_k),
    # keeping the latest run. rerank_on is redundant given `reranker` but kept in the key for safety.
    if not args.no_dedup:
        def variant_key(r):
            return (r["embedder"], r["mode"], r["query_text_source"], r["query_repr"],
                    r["rerank_on"], r["reranker"], r["top_k"])
        best = {}
        for r in rows:
            k = variant_key(r)
            if k not in best or r["run_ts"] > best[k]["run_ts"]:
                best[k] = r
        collapsed = len(rows) - len(best)
        rows = list(best.values())
        print(f"Dedup: kept {len(rows)} variants (collapsed {collapsed} re-run duplicates; "
              f"--no-dedup to keep all)")

    rows.sort(key=lambda r: (r["embedder"], r["mode"], r["query_text_source"], r["reranker"], r["rerank_on"]))
    fieldnames = list(rows[0].keys())
    out_path = out_dir / "summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} runs)")

    if args.markdown:
        cols = ["embedder", "mode", "query_text_source", "rerank_on", "hit@1", "hit@3", "mrr@3", "avg_latency_sec"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
