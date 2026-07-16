#!/bin/bash
# Full text-reranker symmetry: each text embedder's text-only dense top-3 (saved) re-scored by
# Qwen3-Reranker-0.6B/4B/8B via the two-pass method. Processes each reranker as soon as it downloads.
# Log: eval/results/rag/logs/text_reranker_grid.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a
export PYTHONPATH=.

BGE=eval/results/rag/runs/baai-bge-m3/text_only__caption__norerank__20260706_051959.csv
QE06=eval/results/rag/runs/.-models-qwen3-embedding-0.6b/text_only__caption__raw__norerank__20260706_122723.csv
QE4=eval/results/rag/runs/.-models-qwen3-embedding-4b/text_only__caption__raw__norerank__20260706_124001.csv
QE8=eval/results/rag/runs/.-models-qwen3-embedding-8b/text_only__caption__raw__norerank__20260706_125248.csv
SOURCES=("$BGE" "$QE06" "$QE4" "$QE8")

run_batch() {
  local SIZE=$1 SUF=$2
  local DIR=models/Qwen3-Reranker-$SIZE
  # wait up to ~40 min for this reranker to finish downloading
  echo "== [$(date '+%H:%M:%S')] waiting for $DIR ..."
  for i in $(seq 1 160); do
    [ "$(ls $DIR/*.safetensors 2>/dev/null | wc -l)" -ge 1 ] && [ -f "$DIR/config.json" ] && break
    grep -q TEXT_RERANKERS_DL_DONE eval/results/rag/logs/dl_text_rerankers.log 2>/dev/null && break
    sleep 15
  done
  if [ "$(ls $DIR/*.safetensors 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "== [SKIP] $DIR not present — text-reranker $SIZE skipped =="; return 1
  fi
  echo "########## Text Reranker $SIZE ##########  [$(date '+%H:%M:%S')]"
  timeout 7200 python -m eval.rerank_from_run --config config/config_2b_rerank8b.yaml \
    --reranker-kind text --reranker-path "$DIR" --out-suffix "$SUF" --source-csv "${SOURCES[@]}"
}

run_batch 0.6B txt06rr
run_batch 4B   txt4rr
run_batch 8B   txt8rr

echo "== aggregate =="
python eval/aggregate_rag_results.py 2>&1 | grep -E "Wrote|runs\)"
echo "TEXT_RERANKER_GRID_DONE"
