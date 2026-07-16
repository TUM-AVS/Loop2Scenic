#!/bin/bash
# Re-run 8B-embedder + 8B-reranker text-video eval on the now-free GPU, sampling VRAM.
# Answers: (a) per-model + combined VRAM, (b) was the earlier OOM just CARLA contention?
# Log: eval/results/rag/logs/vram_8b8b.log  ·  sampler: .../vram_8b8b_samples.csv
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

LOG=eval/results/rag/logs/vram_8b8b.log
SAMP=eval/results/rag/logs/vram_8b8b_samples.csv
echo "ts_s,mem_used_mib,util_pct" > "$SAMP"

# background VRAM sampler (0.5s cadence) — records the whole timeline so we can read
# the embedder-only plateau vs the embedder+reranker plateau, and the peak.
( t=0; while true; do
    line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "${t},${line}" >> "$SAMP"
    t=$((t+1)); sleep 0.5
  done ) &
SAMPLER=$!

echo "== [$(date '+%H:%M:%S')] free-GPU baseline: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) =="
echo "== [$(date '+%H:%M:%S')] running 8bE_8bR text-video (config_8b_rerank8b.yaml) =="
timeout 5400 python -m eval.retrieval_eval --mode text-video --config config/config_8b_rerank8b.yaml
RC=$?
kill "$SAMPLER" 2>/dev/null

echo "== [$(date '+%H:%M:%S')] eval exit code = $RC =="
echo "== PEAK VRAM during run =="
awk -F, 'NR>1 && $2+0>max{max=$2+0} END{printf "  peak_used = %d MiB (%.1f GiB)\n", max, max/1024}' "$SAMP"
if [ $RC -eq 0 ]; then echo "RESULT: 8B+8B FITS on the free 32GB card — earlier OOM was CARLA contention"; else echo "RESULT: 8B+8B still failed (rc=$RC) — genuine capacity limit"; fi
echo "VRAM_MEASURE_DONE"
