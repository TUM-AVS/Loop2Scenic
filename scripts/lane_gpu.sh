#!/bin/bash
# GPU lane: funnel-width sweep (C1) + 8B rerank arms (A1 generalization) + RRF fusion (B1b).
# Sequential to avoid VRAM contention. Log: eval/results/rag/logs/lane_gpu.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

run() { local label=$1; shift; echo "== [$(date '+%H:%M:%S')] $label =="; timeout 5400 "$@" && echo "[OK] $label" || echo "[FAILED] $label"; }

# Funnel-width sweep (2B, text-only): dense recall curve + rerank pool width
run k5_dense   python -m eval.retrieval_eval --mode text-only --config config/config_norerank_k5.yaml
run k10_dense  python -m eval.retrieval_eval --mode text-only --config config/config_norerank_k10.yaml
run k5_rerank  python -m eval.retrieval_eval --mode text-only --config config/config_rerank_k5.yaml
run k10_rerank python -m eval.retrieval_eval --mode text-only --config config/config_rerank_k10.yaml

# Does the A1 rerank finding generalize to the 8B backbone?
run 8b_rerank_text_only  python -m eval.retrieval_eval --mode text-only  --config config/config_qwen8b_rerank.yaml
run 8b_rerank_text_video python -m eval.retrieval_eval --mode text-video --config config/config_qwen8b_rerank.yaml

# B1b: RRF fusion of dense k10 x BM25 k10 (offline; both produced by the lanes)
DENSE_K10=$(ls -t eval/results/rag/runs/qwen3-vl-embedding-2b/text_only__caption__raw__norerank__*.csv 2>/dev/null | head -1)
for i in $(seq 1 60); do BM25_K10=$(ls -t eval/results/rag/runs/bm25/text_only__caption__raw__norerank__*.csv 2>/dev/null | head -1); [ -n "$BM25_K10" ] && break; sleep 30; done
if [ -n "$DENSE_K10" ] && [ -n "$BM25_K10" ]; then
  run rrf_fusion python eval/rrf_fusion.py --dense "$DENSE_K10" --sparse "$BM25_K10"
else
  echo "[FAILED] rrf_fusion inputs missing (dense=$DENSE_K10 bm25=$BM25_K10)"
fi

run aggregate python eval/aggregate_rag_results.py
echo "GPU_LANE_DONE"
