#!/usr/bin/env bash
# Run e2e_eval.py once per config under config/config_ollama/.
# Continues to the next config even if a run fails.
#
# Usage (from repo root, or anywhere):
#   bash scripts/run_ollama_e2e_configs.sh
#   bash scripts/run_ollama_e2e_configs.sh --benchmark-root data/benchmark --limit 2
#   bash scripts/run_ollama_e2e_configs.sh --categories text-only
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

CONFIG_DIR="$REPO_ROOT/config/config_ollama"
LOG_DIR="$REPO_ROOT/eval/results/ollama_e2e_logs"
mkdir -p "$LOG_DIR"

# Smallest → largest (Ollama on-disk Q4 sizes / param scale).
CONFIGS=(
  "$CONFIG_DIR/config_devstral_small2.yaml"   # ~15GB, 24B
  "$CONFIG_DIR/config_gemma4_31b.yaml"        # ~20GB, 31B
  "$CONFIG_DIR/config_qwen3vl32b.yaml"        # ~21GB, 32B
  "$CONFIG_DIR/config_qwen36_35b.yaml"        # ~24GB, 35B
  "$CONFIG_DIR/config_nemotron3_33b.yaml"     # ~28GB, 33B (largest artifact)
)

EXTRA_ARGS=("$@")
MASTER_LOG="$LOG_DIR/run_$(date '+%Y%m%d_%H%M%S').log"
PASSED=0
FAILED=0
FAILED_NAMES=()

echo "===== Ollama e2e sweep start $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$MASTER_LOG"
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
echo "===== Ollama e2e sweep done $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$MASTER_LOG"
echo "Passed: $PASSED  Failed: $FAILED" | tee -a "$MASTER_LOG"
if (( FAILED > 0 )); then
  echo "Failed configs: ${FAILED_NAMES[*]}" | tee -a "$MASTER_LOG"
fi
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"

# Exit 0 so a launcher/tmux session is not treated as aborted when some models fail.
exit 0
