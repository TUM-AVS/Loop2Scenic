#!/bin/bash
# Re-run backbone dense cells at top_k=20 to populate Recall@5/@10 (top-1/@3 unchanged).
# Waits for the GPU-heavy lanes (rerank8b) to finish to avoid VRAM contention.
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

for i in $(seq 1 240); do
  grep -q RERANK8B_DONE eval/results/rag/logs/rerank8b.log 2>/dev/null && break
  sleep 30
done

run() { local L=$1 M=$2 CFG=$3; echo "== [$(date '+%H:%M:%S')] $L =="; \
  timeout 3600 python -m eval.retrieval_eval --mode "$M" --config "$CFG" --top-k 20 && echo "[OK] $L" || echo "[FAILED] $L"; }

# text-only column (symmetric indices) + text-video column (VL, text+video index)
run 2b_to   text-only  config/config_2b_textonly.yaml
run 2b_tv   text-video config/config_norerank.yaml
run 8b_to   text-only  config/config_8b_textonly.yaml
run 8b_tv   text-video config/config_qwen8b_norerank.yaml
run gem_to  text-only  config/config_gemini_textonly.yaml
run gem_tv  text-video config/config_gemini_norerank.yaml
run bgem3   text-only  config/config_bgem3_norerank.yaml
run qe06    text-only  config/config_qwenemb06.yaml
run qe4b    text-only  config/config_qwenemb4b.yaml
run qe8b    text-only  config/config_qwenemb8b.yaml

python eval/aggregate_rag_results.py
echo "RECALL_CURVE_DONE"
