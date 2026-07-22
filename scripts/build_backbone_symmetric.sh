#!/bin/bash
# Build the P-SYM backbone table T1: each embedder queried text-only against a matched
# text-only document index. text-video column (VL) already done under P-FIX (symmetric for
# that cell). Sequential to avoid VRAM contention. Log: eval/results/rag/logs/backbone.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a

run() { local L=$1; shift; echo "== [$(date '+%H:%M:%S')] $L =="; timeout 3600 "$@" && echo "[OK] $L" || echo "[FAILED] $L"; }

# VL embedders, TEXT-ONLY index (symmetric text-only cell)
run ing_8b_txt   python scripts/ingest_local_scenarios.py --config config/config_8b_textonly.yaml     --doc-modality text_only --reset
run 8b_txt       python -m eval.retrieval_eval --mode text-only --config config/config_8b_textonly.yaml
run ing_gem_txt  python scripts/ingest_local_scenarios.py --config config/config_gemini_textonly.yaml --doc-modality text_only --reset
run gem_txt      python -m eval.retrieval_eval --mode text-only --config config/config_gemini_textonly.yaml

# Text-only embedders (inherently text-only index)
for tag in qwenemb06 qwenemb4b qwenemb8b; do
  run ing_$tag python scripts/ingest_local_scenarios.py --config config/config_$tag.yaml --doc-modality text_only --reset
  run $tag     python -m eval.retrieval_eval --mode text-only --config config/config_$tag.yaml
done

# Re-run gemini video-only (previous run truncated at 94 rows)
run gemini_video_only_rerun python -m eval.retrieval_eval --mode video-only --config config/config_gemini_norerank.yaml

run aggregate python eval/aggregate_rag_results.py
echo "BACKBONE_DONE"
