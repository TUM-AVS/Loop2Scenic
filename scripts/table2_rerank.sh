#!/bin/bash
# Table 2 = Table 1 layout (embedder x modality) with rerank ON (Qwen3-VL-Reranker-2B),
# on the SAME symmetric indices as T1. Waits for the 8B-reranker lane to free the GPU.
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

for i in $(seq 1 300); do
  grep -q RERANK8B_DONE eval/results/rag/logs/rerank8b.log 2>/dev/null && break
  sleep 30
done

run() { local L=$1 M=$2 CFG=$3; echo "== [$(date '+%H:%M:%S')] $L =="; \
  timeout 5400 python -m eval.retrieval_eval --mode "$M" --config "$CFG" && echo "[OK] $L" || echo "[FAILED] $L"; }

# text-only column (symmetric index) + 2B reranker, all 7 embedders
run 2b_to_rr   text-only  config/config_2b_textonly_rr.yaml
run 8b_to_rr   text-only  config/config_8b_textonly_rr.yaml
run gem_to_rr  text-only  config/config_gemini_textonly_rr.yaml
run bgem3_rr   text-only  config/config_bgem3_rr.yaml
run qe06_rr    text-only  config/config_qwenemb06_rr.yaml
run qe4b_rr    text-only  config/config_qwenemb4b_rr.yaml
run qe8b_rr    text-only  config/config_qwenemb8b_rr.yaml

# text-video column + 2B reranker (2B=73.5, 8B=78.0 already exist; add gemini)
run gem_tv_rr  text-video config/config_gemini_rr.yaml

python eval/aggregate_rag_results.py
echo "TABLE2_DONE"
