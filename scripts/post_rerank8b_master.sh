#!/bin/bash
# Sequential master lane for all remaining GPU work — runs AFTER rerank8b finishes,
# one step at a time (no VRAM contention): 8B modality-gap -> video-only column (T1 3rd col)
# -> Table 2 rerank cells -> aggregate.
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

echo "== waiting for RERANK8B_DONE =="
for i in $(seq 1 480); do
  grep -q RERANK8B_DONE eval/results/rag/logs/rerank8b.log 2>/dev/null && break
  sleep 30
done

run() { local L=$1; shift; echo "== [$(date '+%H:%M:%S')] $L =="; timeout 5400 "$@" && echo "[OK] $L" || echo "[FAILED] $L"; }

# 1) 8B modality-gap (was the modgap-8b waiter)
run modgap_8b python eval/modality_gap.py --config config/config_qwen8b_norerank.yaml

# 2) VIDEO-ONLY column for T1 (symmetric: video-only index + video-only query), VL embedders
run ing_2b_vo  python scripts/ingest_local_scenarios.py --config config/config_2b_videoonly.yaml     --doc-modality video_only --reset
run 2b_vo      python -m eval.retrieval_eval --mode video-only --config config/config_2b_videoonly.yaml
run ing_8b_vo  python scripts/ingest_local_scenarios.py --config config/config_8b_videoonly.yaml     --doc-modality video_only --reset
run 8b_vo      python -m eval.retrieval_eval --mode video-only --config config/config_8b_videoonly.yaml
run ing_gem_vo python scripts/ingest_local_scenarios.py --config config/config_gemini_videoonly.yaml --doc-modality video_only --reset
run gem_vo     python -m eval.retrieval_eval --mode video-only --config config/config_gemini_videoonly.yaml

# 3) Table 2 (rerank ON, 2B reranker, symmetric indices)
for cfg_mode in "config_2b_textonly_rr text-only" "config_8b_textonly_rr text-only" \
                "config_gemini_textonly_rr text-only" "config_bgem3_rr text-only" \
                "config_qwenemb06_rr text-only" "config_qwenemb4b_rr text-only" \
                "config_qwenemb8b_rr text-only" "config_gemini_rr text-video"; do
  set -- $cfg_mode
  run "t2_$1" python -m eval.retrieval_eval --mode "$2" --config "config/$1.yaml"
done

run aggregate python eval/aggregate_rag_results.py
echo "POST_RERANK8B_DONE"
