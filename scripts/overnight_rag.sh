#!/bin/bash
# Overnight RAG ablation orchestrator — fully self-contained (survives laptop disconnect).
# Runs sequentially on the workstation GPU; each step is guarded so one failure never
# kills the chain. Progress: eval/results/rag/logs/overnight.log (this script's tee target).
# Morning check:  grep "STEP\|FAILED\|OVERNIGHT" eval/results/rag/logs/overnight.log

cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loop2scenic
set -a; source .env; set +a

LOG_DIR=eval/results/rag/logs
mkdir -p "$LOG_DIR"

step() { echo ""; echo "===== [$(date '+%H:%M:%S')] STEP $1: $2 ====="; }
run_guarded() { # run_guarded <label> <timeout-sec> <cmd...>
  local label=$1 tmo=$2; shift 2
  if timeout "$tmo" "$@" > >(tee "$LOG_DIR/${label}.log") 2>&1; then
    echo "[OK] $label"
  else
    echo "[FAILED] $label (continuing)"
  fi
}

step 1 "Finish 8B model rsync (idempotent resume)"
run_guarded 8b_rsync 7200 rsync -az avsaw1@10.147.17.157:/home/avsaw1/chenli/ads-mrag/models/Qwen3-VL-Embedding-8B/ models/Qwen3-VL-Embedding-8B/

step 2 "Ingest scenarios_qwen8b (GPU)"
run_guarded 8b_ingest 7200 python scripts/ingest_local_scenarios.py --config config/config_qwen8b_norerank.yaml --reset

step 3 "8B cells: text-only / text-video / video-only (dense)"
for m in text-only text-video video-only; do
  run_guarded "8b_${m}" 3600 python -m eval.retrieval_eval --mode "$m" --config config/config_qwen8b_norerank.yaml
done

step 4 "2B A1 rerank-ON arms: text-only / text-video (funnel)"
for m in text-only text-video; do
  run_guarded "2b_rerank_${m}" 3600 python -m eval.retrieval_eval --mode "$m"
done

step 5 "Wait for pure-video captions (tmux vcaption)"
for i in $(seq 1 240); do
  grep -q "VCAPTION_DONE" logs/experiments/vcaption.log 2>/dev/null && break
  sleep 60
done
grep "VCAPTION_DONE" logs/experiments/vcaption.log || echo "[WARN] vcaption not confirmed done — bridge rows may be partial"

step 6 "2B bridge arm: video->caption->text-only query (vs native video-only 11.4%)"
run_guarded 2b_bridge_vcaption 3600 python -m eval.retrieval_eval --mode text-only --query-text-source video_only_caption --config config/config_norerank.yaml

step 7 "bge-m3: ingest text-only collection + text-query row + bridge row"
run_guarded bgem3_ingest 3600 python scripts/ingest_local_scenarios.py --config config/config_bgem3_norerank.yaml --reset
run_guarded bgem3_caption 1800 python -m eval.retrieval_eval --mode text-only --query-text-source mllm_caption --config config/config_bgem3_norerank.yaml
run_guarded bgem3_bridge 1800 python -m eval.retrieval_eval --mode text-only --query-text-source video_only_caption --config config/config_bgem3_norerank.yaml

step 8 "Wait for gemini ingestion (tmux gemini-ingest), then gemini cells"
for i in $(seq 1 360); do
  grep -qE "INGEST_DONE" logs/experiments/gemini_ingest.log 2>/dev/null && break
  sleep 60
done
if grep -q "INGEST_DONE ok=" logs/experiments/gemini_ingest.log 2>/dev/null; then
  for m in text-only text-video video-only; do
    run_guarded "gemini_${m}" 7200 python -m eval.retrieval_eval --mode "$m" --config config/config_gemini_norerank.yaml
  done
else
  echo "[FAILED] gemini ingestion not confirmed — skipping gemini cells"
fi

step 9 "Aggregate summary"
run_guarded aggregate 600 python eval/aggregate_rag_results.py --markdown

echo ""
echo "===== [$(date '+%H:%M:%S')] OVERNIGHT_DONE ====="
