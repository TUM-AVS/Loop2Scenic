#!/bin/bash
# Self-contained: retries the 8B-reranker download until complete, then runs the eval.
# Runs entirely on the workstation in tmux → survives laptop shutdown / SSH disconnect.
# Log: eval/results/rag/logs/rr8b_selfcontained.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

MODEL=models/Qwen3-VL-Reranker-8B
have() { ls $MODEL/model-0000*.safetensors 2>/dev/null | wc -l; }

echo "== [$(date '+%H:%M:%S')] ensuring 8B reranker download (have $(have)/4 shards) =="
for attempt in $(seq 1 30); do
  [ "$(have)" -ge 4 ] && break
  echo "== [$(date '+%H:%M:%S')] download attempt $attempt (have $(have)/4) =="
  # alternate endpoints: even attempts use the mirror, odd use the default
  if [ $((attempt % 2)) -eq 0 ]; then export HF_ENDPOINT=https://hf-mirror.com; else unset HF_ENDPOINT; fi
  timeout 1200 hf download Qwen/Qwen3-VL-Reranker-8B --local-dir $MODEL 2>&1 | tail -2
  sleep 15
done

if [ "$(have)" -lt 4 ]; then
  echo "[ABORT] 8B reranker still incomplete ($(have)/4 shards) after 30 attempts — download blocked"
  echo "RR8B_EVAL_SKIPPED"
  exit 1
fi
echo "== [$(date '+%H:%M:%S')] 8B reranker download COMPLETE ($(have)/4 shards) =="

run() { local L=$1; shift; echo "== [$(date '+%H:%M:%S')] $L =="; timeout 5400 "$@" && echo "[OK] $L" || echo "[FAILED/OOM] $L"; }
# 2B embedder candidates + 8B reranker (safe VRAM) on the modes where 2B reranker hurt
run 2bE_8bR_text_video  python -m eval.retrieval_eval --mode text-video --config config/config_2b_rerank8b.yaml
run 2bE_8bR_video_only  python -m eval.retrieval_eval --mode video-only --config config/config_2b_rerank8b.yaml
run 2bE_8bR_text_only   python -m eval.retrieval_eval --mode text-only  --config config/config_2b_rerank8b.yaml
# 8B embedder + 8B reranker on the biggest-drop cell (may OOM on 32GB — logged, not fatal)
run 8bE_8bR_text_video  python -m eval.retrieval_eval --mode text-video --config config/config_8b_rerank8b.yaml
run aggregate python eval/aggregate_rag_results.py
echo "RR8B_EVAL_DONE"
