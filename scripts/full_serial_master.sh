#!/bin/bash
# ONE serial GPU lane, no deadlock. Phase A (no 8B reranker) runs immediately;
# Phase B (8B-reranker capacity) waits for the download AT THE END (download runs concurrently,
# network-only). Log: eval/results/rag/logs/full_master.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a
run() { local L=$1; shift; echo "== [$(date '+%H:%M:%S')] $L =="; timeout 5400 "$@" && echo "[OK] $L" || echo "[FAILED] $L"; }

echo "###### PHASE A (no 8B reranker) ######"

# A1: 8B modality-gap
run modgap_8b python eval/modality_gap.py --config config/config_qwen8b_norerank.yaml

# A2: caption-bridge FIRST (fast: text-only queries, no reranking) — text embedders fill
#     video + text-video via captioning (video captions = qwen3.6-plus, 245/245 ready)
for e in "bgem3 config_bgem3_norerank" "qe06 config_qwenemb06" "qe4b config_qwenemb4b" "qe8b config_qwenemb8b"; do
  set -- $e
  run "${1}_br_video"     python -m eval.retrieval_eval --mode text-only --query-text-source video_only_caption --config "config/$2.yaml"
  run "${1}_br_textvideo" python -m eval.retrieval_eval --mode text-only --query-text-source combined_caption    --config "config/$2.yaml"
done
run aggregate_bridge python eval/aggregate_rag_results.py
echo "BRIDGE_DONE"

# A3: VIDEO-ONLY column (symmetric: video-only index + video-only query), VL embedders
run ing_2b_vo  python scripts/ingest_local_scenarios.py --config config/config_2b_videoonly.yaml     --doc-modality video_only --reset
run 2b_vo      python -m eval.retrieval_eval --mode video-only --config config/config_2b_videoonly.yaml
run ing_8b_vo  python scripts/ingest_local_scenarios.py --config config/config_8b_videoonly.yaml     --doc-modality video_only --reset
run 8b_vo      python -m eval.retrieval_eval --mode video-only --config config/config_8b_videoonly.yaml
run ing_gem_vo python scripts/ingest_local_scenarios.py --config config/config_gemini_videoonly.yaml --doc-modality video_only --reset
run gem_vo     python -m eval.retrieval_eval --mode video-only --config config/config_gemini_videoonly.yaml

# A4: Table 2 (rerank ON, 2B reranker, symmetric indices) — slowest, run last in Phase A
for cm in "config_2b_textonly_rr text-only" "config_8b_textonly_rr text-only" \
          "config_gemini_textonly_rr text-only" "config_bgem3_rr text-only" \
          "config_qwenemb06_rr text-only" "config_qwenemb4b_rr text-only" \
          "config_qwenemb8b_rr text-only" "config_gemini_rr text-video"; do
  set -- $cm; run "t2_$1" python -m eval.retrieval_eval --mode "$2" --config "config/$1.yaml"
done

run aggregate_A python eval/aggregate_rag_results.py
echo "PHASE_A_DONE"

echo "###### PHASE B (8B reranker capacity) ######"
for i in $(seq 1 240); do grep -q RR8B_DL_DONE eval/results/rag/logs/dl_rr8b.log 2>/dev/null && break; sleep 30; done
if grep -q RR8B_DL_DONE eval/results/rag/logs/dl_rr8b.log 2>/dev/null; then
  run 2bE_8bR_tv  python -m eval.retrieval_eval --mode text-video --config config/config_2b_rerank8b.yaml
  run 2bE_8bR_vo  python -m eval.retrieval_eval --mode video-only --config config/config_2b_rerank8b.yaml
  run 2bE_8bR_to  python -m eval.retrieval_eval --mode text-only  --config config/config_2b_rerank8b.yaml
  run 8bE_8bR_tv  python -m eval.retrieval_eval --mode text-video --config config/config_8b_rerank8b.yaml
  run aggregate_B python eval/aggregate_rag_results.py
else
  echo "[SKIP] 8B reranker download not done — Phase B skipped"
fi
echo "FULL_MASTER_DONE"
