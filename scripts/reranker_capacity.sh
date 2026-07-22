#!/bin/bash
# Reranker-capacity ablation: does an 8B reranker fix "rerank hurts video"?
# Holds embedder/candidate-lists fixed, swaps reranker 2B -> 8B.
# Compare against existing 2B-reranker rerank runs. Log: logs/rerank8b.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

# wait for the 8B reranker download
for i in $(seq 1 120); do
  grep -q RERANKER8B_DL_DONE eval/results/rag/logs/dl_reranker8b.log 2>/dev/null && break
  sleep 30
done
grep -q RERANKER8B_DL_DONE eval/results/rag/logs/dl_reranker8b.log || { echo "[ABORT] reranker download not done"; exit 1; }

run() { local L=$1; shift; echo "== [$(date '+%H:%M:%S')] $L =="; timeout 5400 "$@" && echo "[OK] $L" || echo "[FAILED/OOM] $L"; }

# Primary comparison: 2B-embedder candidates + 8B reranker (safe VRAM ~20GB)
# vs existing 2B-embedder + 2B reranker (text-video 73.5, video-only 4.9, text-only ~49.8)
run 2bEmb_8bRerank_text_video  python -m eval.retrieval_eval --mode text-video --config config/config_2b_rerank8b.yaml
run 2bEmb_8bRerank_video_only  python -m eval.retrieval_eval --mode video-only --config config/config_2b_rerank8b.yaml
run 2bEmb_8bRerank_text_only   python -m eval.retrieval_eval --mode text-only  --config config/config_2b_rerank8b.yaml

# Stretch: 8B-embedder + 8B reranker on the biggest-drop cell (8B text-video 88.6->78.0 with 2B reranker).
# May OOM at ~32GB on the 5090; [FAILED/OOM] is an acceptable, logged outcome.
run 8bEmb_8bRerank_text_video  python -m eval.retrieval_eval --mode text-video --config config/config_8b_rerank8b.yaml

run aggregate python eval/aggregate_rag_results.py
echo "RERANK8B_DONE"
