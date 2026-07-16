#!/bin/bash
# Complete the Table-2 grid to mirror Table 1: rerank the text-only + video-only dense top-3 for all
# three VL embedders, with BOTH rerankers, full bf16, two-pass offline. (text-video already done.)
# Batch A = 2B reranker (config_gemini_rr.yaml -> Reranker-2B), Batch B = 8B reranker.
# Re-validation: 2B-emb video-only must reproduce the earlier live 4.9 (+2B-rr) / 6.1 (+8B-rr).
# Log: eval/results/rag/logs/rerank_grid_phase2.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a
export PYTHONPATH=.

R2B=eval/results/rag/runs/qwen3-vl-embedding-2b
R8B=eval/results/rag/runs/qwen3-vl-embedding-8b
GEM=eval/results/rag/runs/gemini-embedding-2

SOURCES=(
  "$R2B/text_only__caption__norerank__20260705.csv"
  "$R8B/text_only__caption__norerank__20260706_005714.csv"
  "$GEM/text_only__caption__raw__norerank__20260706_112118.csv"
  "$R2B/video_only__caption__norerank__20260705_235728.csv"
  "$R8B/video_only__caption__norerank__20260706_010004.csv"
  "$GEM/video_only__caption__raw__norerank__20260706_125309.csv"
)

echo "########## BATCH A: + Reranker-2B ##########  [$(date '+%H:%M:%S')]"
timeout 7200 python -m eval.rerank_from_run --config config/config_gemini_rr.yaml \
  --out-suffix 2brr --source-csv "${SOURCES[@]}"

echo
echo "########## BATCH B: + Reranker-8B ##########  [$(date '+%H:%M:%S')]"
timeout 10800 python -m eval.rerank_from_run --config config/config_8b_rerank8b.yaml \
  --out-suffix 8brr --source-csv "${SOURCES[@]}"

echo
echo "########## aggregate ##########"
python eval/aggregate_rag_results.py 2>&1 | grep -E "Wrote|runs\)"
echo "RERANK_GRID_DONE"
