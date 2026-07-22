#!/bin/bash
# Phase-2 offline rerank on the free GPU: full bf16, no OOM.
# STEP 1 (validation): rerank the 2B-emb saved dense top-3 with the 8B reranker ->
#   MUST reproduce the known live result (72.2 / 89.0 / 0.801). If it does, the method is proven.
# STEP 2 (new cell): rerank the 8B-emb saved dense top-3 with the 8B reranker -> fills Table 2's
#   last cell (8B embedder x 8B reranker) at full precision.
# Log: eval/results/rag/logs/rr8b_phase2.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a
export PYTHONPATH=.

RUNDIR2B=eval/results/rag/runs/qwen3-vl-embedding-2b
RUNDIR8B=eval/results/rag/runs/qwen3-vl-embedding-8b

echo "== [$(date '+%H:%M:%S')] STEP 1 VALIDATION: 2B-emb dense top-3 + 8B reranker (expect ~72.2/89.0/0.801) =="
timeout 3600 python -m eval.rerank_from_run \
  --source-csv "$RUNDIR2B/text_video__caption__norerank__20260705.csv" \
  --config config/config_2b_rerank8b.yaml \
  --out-csv "$RUNDIR2B/text_video__caption__raw__rerank__phase2_validation.csv" \
  --embedder-label Qwen3-VL-Embedding-2B

echo
echo "== [$(date '+%H:%M:%S')] STEP 2 NEW CELL: 8B-emb dense top-3 + 8B reranker =="
timeout 3600 python -m eval.rerank_from_run \
  --source-csv "$RUNDIR8B/text_video__caption__norerank__20260706_005734.csv" \
  --config config/config_8b_rerank8b.yaml \
  --out-csv "$RUNDIR8B/text_video__caption__raw__rerank__phase2.csv" \
  --embedder-label Qwen3-VL-Embedding-8B

echo
echo "== [$(date '+%H:%M:%S')] aggregate =="
timeout 1200 python eval/aggregate_rag_results.py
echo "RR8B_PHASE2_DONE"
