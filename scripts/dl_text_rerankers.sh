#!/bin/bash
# Download the three text rerankers with retry + endpoint alternation (0.6B fast, 4B ~8GB, 8B ~16GB).
# Log: eval/results/rag/logs/dl_text_rerankers.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

pull() {
  local REPO=$1 DIR=$2 NSHARD=$3
  have() { ls $DIR/model*.safetensors 2>/dev/null | wc -l; }
  echo "== [$(date '+%H:%M:%S')] $REPO -> $DIR (want >=$NSHARD shards) =="
  for a in $(seq 1 20); do
    [ "$(have)" -ge "$NSHARD" ] && { echo "  [$REPO] complete ($(have) shards)"; return 0; }
    if [ $((a % 2)) -eq 0 ]; then export HF_ENDPOINT=https://hf-mirror.com; else unset HF_ENDPOINT; fi
    echo "  [$REPO] attempt $a (have $(have))"
    timeout 1200 hf download "$REPO" --local-dir "$DIR" 2>&1 | tail -1
    sleep 10
  done
  [ "$(have)" -ge "$NSHARD" ] && return 0 || { echo "  [$REPO] INCOMPLETE after 20 tries"; return 1; }
}

pull Qwen/Qwen3-Reranker-0.6B models/Qwen3-Reranker-0.6B 1
pull Qwen/Qwen3-Reranker-4B   models/Qwen3-Reranker-4B   1
pull Qwen/Qwen3-Reranker-8B   models/Qwen3-Reranker-8B   1
echo "TEXT_RERANKERS_DL_DONE"
