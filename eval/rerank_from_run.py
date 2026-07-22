"""Phase-2 OFFLINE rerank: re-score a saved dense-retrieval run with a reranker loaded ALONE.

Why: 8B embedder (15.2 GiB) + 8B reranker (16.3 GiB) can't co-reside on a 32 GB card.
But reranking only reorders the dense top-k shortlist — it never needs the embedder in memory.
So we reuse the *already-saved* dense top-k (from a `norerank` run) and apply the reranker in a
separate process. This is mathematically identical to the co-resident pipeline (HIT@3 invariant;
only the top-k ordering changes), at FULL bf16 precision — no quantization, no OOM, no asterisk.

Usage:
  python -m eval.rerank_from_run --source-csv <norerank_run.csv> --config config/config_8b_rerank8b.yaml \
      --out-csv <out.csv> --embedder-label Qwen3-VL-Embedding-8B
"""
import argparse, csv, json, sys
from pathlib import Path

from src.config import get_config
from src.services.reranker import get_reranker
from src.services.retriever.retriever import Retriever
from src.schema import MultimodalQuery
from eval.retrieval_eval import analyze_retrieval_csv

FIELDNAMES = [
    "ground_truth", "user_query", "scenario_dsl", "flattened_dsl", "base_scenario_id",
    "best_scenario_ids", "best_similarity_score", "best_rerank_score", "response_time_sec",
    "error_message", "mode", "query_text_source", "query_repr", "rerank_on", "embedder", "top_k",
]


def rerank_one(reranker, retr, src: Path, out: Path, embedder_label: str, top_k: int, pool_k: int):
    """Rerank ONE saved norerank run. pool_k truncates the dense candidate list before reranking
    (=3 to match the deployed dense-top-3 -> rerank funnel and keep HIT@3 invariant)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open() as fin, out.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in reader:
            gt = row["ground_truth"]
            err, new_ids, base_id, rerank_score = "", None, None, None
            try:
                uq = json.loads(row["user_query"])
                query = MultimodalQuery(
                    text=uq.get("text"), image_path=uq.get("image_path"), video_path=uq.get("video_path"),
                )
                dense_ids = json.loads(row["best_scenario_ids"]) if row.get("best_scenario_ids") else []
                dense_ids = dense_ids[:pool_k]  # deployed funnel reranks the dense top-3
                docs = [retr._get_original_scenario(cid) for cid in dense_ids]
                docs = [d for d in docs if d is not None]
                if not docs:
                    raise ValueError(f"no candidate docs reconstructed for {dense_ids}")
                scored = reranker.rerank_with_scores(query, docs, top_k=top_k)
                new_ids = [d.scenario_id for d, _ in scored]
                base_id = new_ids[0] if new_ids else None
                rerank_score = scored[0][1] if scored else None
            except Exception as exc:
                err = str(exc)
                print(f"  [err] gt={gt}: {exc}", flush=True)

            writer.writerow({
                "ground_truth": str(gt),
                "user_query": row.get("user_query", ""),
                "scenario_dsl": "", "flattened_dsl": "",
                "base_scenario_id": str(base_id) if base_id is not None else "",
                "best_scenario_ids": json.dumps(new_ids, ensure_ascii=False) if new_ids is not None else "",
                "best_similarity_score": "",
                "best_rerank_score": f"{rerank_score:.6f}" if rerank_score is not None else "",
                "response_time_sec": "0", "error_message": err,
                "mode": row.get("mode", ""),
                "query_text_source": row.get("query_text_source", ""),
                "query_repr": row.get("query_repr", "raw"),
                "rerank_on": "True", "embedder": embedder_label, "top_k": str(top_k),
            })
            n += 1
            if n % 50 == 0:
                print(f"  {src.name}: reranked {n} ...", flush=True)

    metrics = analyze_retrieval_csv(out)
    h1 = metrics["ground_truth_eq_base_scenario_id_rate_among_no_error"]
    h3 = metrics["ground_truth_in_best_scenario_ids_rate_among_no_error"]
    mrr = metrics["mrr_among_no_error"]
    print(f"[RESULT] {embedder_label:24s} {row.get('mode',''):11s} -> {h1:.2f} / {h3:.2f} / {mrr:.4f}   ({out.name})",
          flush=True)


def _default_out(src: Path, suffix: str) -> Path:
    stem = src.stem.replace("norerank", "rerank")
    return src.parent / f"{stem}__phase2_{suffix}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", nargs="+", required=True, help="one or more norerank run CSVs")
    ap.add_argument("--config", required=True, help="config with the reranking: section (model_path)")
    ap.add_argument("--out-csv", default=None, help="explicit output (single source only)")
    ap.add_argument("--out-suffix", default="phase2", help="tag for auto-named outputs (e.g. 2brr, 8brr)")
    ap.add_argument("--embedder-label", default=None, help="override; else read from each CSV's embedder col")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--pool-k", type=int, default=3, help="truncate dense candidates before rerank")
    ap.add_argument("--reranker-kind", choices=["vl", "text"], default="vl",
                    help="vl = Qwen3-VL-Reranker (multimodal); text = Qwen3-Reranker (text-only)")
    ap.add_argument("--reranker-path", default=None, help="model path for --reranker-kind text")
    args = ap.parse_args()

    cfg = get_config(args.config)  # sets the global config used by _get_original_scenario
    if args.reranker_kind == "text":
        from eval.text_reranker import TextReranker
        rr_path = args.reranker_path or cfg.reranking.model_path
        reranker = TextReranker(rr_path)
        print(f"[rerank_from_run] loaded TEXT reranker: {rr_path}", flush=True)
    else:
        reranker = get_reranker(
            provider=cfg.reranking.provider, device=cfg.reranking.device,
            model_name=cfg.reranking.model_name, model_path=cfg.reranking.model_path,
        )
        print(f"[rerank_from_run] loaded VL reranker: {cfg.reranking.model_name} ({cfg.reranking.model_path})", flush=True)
    retr = Retriever(vectorstore=None, reranker=reranker, top_k=args.top_k)

    for i, s in enumerate(args.source_csv):
        src = Path(s)
        out = Path(args.out_csv) if (args.out_csv and len(args.source_csv) == 1) else _default_out(src, args.out_suffix)
        # embedder label: CLI override, else the source CSV's own embedder column
        label = args.embedder_label
        if label is None:
            with src.open() as f:
                r0 = next(csv.DictReader(f), {})
                label = r0.get("embedder", "unknown")
        print(f"[{i+1}/{len(args.source_csv)}] rerank {src.name}  (embedder={label})", flush=True)
        rerank_one(reranker, retr, src, out, label, args.top_k, args.pool_k)


if __name__ == "__main__":
    main()
