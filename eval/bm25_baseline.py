"""
B1: BM25 sparse/lexical retrieval baseline over the flattened-DSL corpus.

Corpus doc = new_description.txt per scenario (same text the dense index embeds).
Query = text_query_description.txt (caption, default) or video_only_description.txt.
Emits a CSV in the same stamped format as retrieval_eval.py into eval/results/rag/runs/bm25/,
so eval/aggregate_rag_results.py picks it up automatically.

Usage:
    python eval/bm25_baseline.py [--folder data/scenarios] [--source mllm_caption|video_only_caption] [--top-k 3]
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

from rank_bm25 import BM25Okapi

SOURCE_FILES = {
    "mllm_caption": "text_query_description.txt",
    "video_only_caption": "video_only_description.txt",
    "indexed_description": "new_description.txt",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 retrieval baseline")
    parser.add_argument("--folder", default="data/scenarios")
    parser.add_argument("--source", choices=list(SOURCE_FILES), default="mllm_caption")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.folder)
    qfile = SOURCE_FILES[args.source]

    ids, corpus = [], []
    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        doc = sub / "new_description.txt"
        if doc.is_file():
            ids.append(sub.name)
            corpus.append(tokenize(doc.read_text(encoding="utf-8")))
    bm25 = BM25Okapi(corpus)
    print(f"BM25 index: {len(ids)} docs")

    out_dir = Path("eval/results/rag/runs/bm25")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_slug = {"mllm_caption": "caption", "video_only_caption": "vcaption", "indexed_description": "indexed"}[args.source]
    out_path = out_dir / f"text_only__{src_slug}__raw__norerank__{ts}.csv"

    fieldnames = ["ground_truth", "user_query", "scenario_dsl", "flattened_dsl", "base_scenario_id",
                  "best_scenario_ids", "best_similarity_score", "best_rerank_score", "response_time_sec",
                  "error_message", "mode", "query_text_source", "query_repr", "rerank_on", "embedder", "top_k"]

    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sub in sorted(d for d in root.iterdir() if d.is_dir()):
            qpath = sub / qfile
            if not qpath.is_file():
                continue
            qtext = qpath.read_text(encoding="utf-8")
            t0 = time.perf_counter()
            scores = bm25.get_scores(tokenize(qtext))
            ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[: args.top_k]
            dt = time.perf_counter() - t0
            top_ids = [ids[i] for i in ranked]
            writer.writerow({
                "ground_truth": sub.name,
                "user_query": json.dumps({"text": qtext[:500]}),
                "scenario_dsl": "", "flattened_dsl": "",
                "base_scenario_id": top_ids[0] if top_ids else "",
                "best_scenario_ids": json.dumps(top_ids),
                "best_similarity_score": f"{scores[ranked[0]]:.6f}" if ranked else "",
                "best_rerank_score": "",
                "response_time_sec": f"{dt:.6f}",
                "error_message": "",
                "mode": "text-only",
                "query_text_source": qfile,
                "query_repr": "raw",
                "rerank_on": "False",
                "embedder": "bm25",
                "top_k": str(args.top_k),
            })
            n += 1
    print(f"BM25_DONE queries={n} -> {out_path}")


if __name__ == "__main__":
    main()
