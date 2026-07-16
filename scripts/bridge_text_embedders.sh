#!/bin/bash
# Caption-bridge: let each TEXT embedder fill all 3 modality conditions by captioning —
#   text    = text_query_description (caption)      [already have from T1 text-only]
#   video   = video_only_description (video caption)
#   text+video = combined (both captions, the "two texts" bridge)
# All against the existing text-only index (no new index needed). Fast (text, no video encode).
# Runs after the postmaster lane frees the GPU.
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

for i in $(seq 1 600); do
  grep -q POST_RERANK8B_DONE eval/results/rag/logs/post_rerank8b.log 2>/dev/null && break
  sleep 30
done

run() { local L=$1 SRC=$2 CFG=$3; echo "== [$(date '+%H:%M:%S')] $L =="; \
  timeout 3600 python -m eval.retrieval_eval --mode text-only --query-text-source "$SRC" --config "$CFG" \
  && echo "[OK] $L" || echo "[FAILED] $L"; }

for e in "bgem3 config_bgem3_norerank" "qe06 config_qwenemb06" "qe4b config_qwenemb4b" "qe8b config_qwenemb8b"; do
  set -- $e
  run "${1}_video"    video_only_caption "config/$2.yaml"   # V-only via bridge
  run "${1}_textvideo" combined_caption  "config/$2.yaml"   # T+V via bridge (two texts)
done

python eval/aggregate_rag_results.py
echo "BRIDGE_DONE"
