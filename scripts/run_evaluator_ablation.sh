#!/usr/bin/env bash
# Run the 5 VLM-evaluator ablation groups (G1–G5) via e2e_eval.py.
# Continues to the next group even if a run fails.
#
# Usage (from repo root, or anywhere):
#   bash scripts/run_evaluator_ablation.sh
#   bash scripts/run_evaluator_ablation.sh --benchmark-root data/benchmark --limit 2
#   bash scripts/run_evaluator_ablation.sh --categories text-only
#
# Extra args after the script name are forwarded to every e2e_eval.py invocation.

set -u
set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ads-mrag 2>/dev/null || conda activate ads_mrag 2>/dev/null || true

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR="$REPO_ROOT/config/config_evaluator_ablation"
LOG_DIR="$REPO_ROOT/eval/results/evaluator_ablation_logs"
mkdir -p "$LOG_DIR"

# Prompt structure (G1–G3) then modality (G4–G5).
CONFIGS=(
  "$CONFIG_DIR/config_g1_vanilla.yaml"       # vanilla prompt, BEV only
  "$CONFIG_DIR/config_g2_contextual.yaml"    # + contextual prompting, BEV only
  "$CONFIG_DIR/config_g3_full.yaml"          # + CoT (full evaluator), BEV only
  "$CONFIG_DIR/config_g4_code_only.yaml"     # Scenic code only (no BEV)
  "$CONFIG_DIR/config_g5_code_video.yaml"    # Scenic code + BEV video
)

EXTRA_ARGS=("$@")
MASTER_LOG="$LOG_DIR/run_$(date '+%Y%m%d_%H%M%S').log"
PASSED=0
FAILED=0
FAILED_NAMES=()

echo "===== Evaluator ablation sweep start $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$MASTER_LOG"
echo "Repo: $REPO_ROOT" | tee -a "$MASTER_LOG"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  run_log="$LOG_DIR/${name}_$(date '+%Y%m%d_%H%M%S').log"

  if [[ ! -f "$cfg" ]]; then
    echo "[SKIP] missing config: $cfg" | tee -a "$MASTER_LOG"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("$name (missing)")
    continue
  fi

  echo "" | tee -a "$MASTER_LOG"
  echo "===== [$(date '+%H:%M:%S')] START $name =====" | tee -a "$MASTER_LOG"
  echo "Config: $cfg" | tee -a "$MASTER_LOG"
  echo "Run log: $run_log" | tee -a "$MASTER_LOG"

  python eval/e2e_eval.py --config-path "$cfg" "${EXTRA_ARGS[@]}" \
    > >(tee "$run_log") 2>&1
  status=$?

  if [[ $status -eq 0 ]]; then
    echo "[OK] $name" | tee -a "$MASTER_LOG"
    PASSED=$((PASSED + 1))
  else
    echo "[FAILED] $name (exit=$status); continuing" | tee -a "$MASTER_LOG"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("$name")
  fi
done

echo "" | tee -a "$MASTER_LOG"
echo "===== Evaluator ablation sweep done $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$MASTER_LOG"
echo "Passed: $PASSED  Failed: $FAILED" | tee -a "$MASTER_LOG"
if (( FAILED > 0 )); then
  echo "Failed configs: ${FAILED_NAMES[*]}" | tee -a "$MASTER_LOG"
fi
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"

# Exit 0 so a launcher/tmux session is not treated as aborted when some groups fail.
exit 0
