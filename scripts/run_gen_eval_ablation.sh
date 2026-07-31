#!/usr/bin/env bash
# Run the 7 codegen / gen_eval ablation groups (G1–G7) via e2e_eval.py.
# Continues to the next group on ordinary failures, but ABORTS the whole sweep
# immediately on CUDA out-of-memory (e2e_eval exit code 99, or OOM text in the log).
#
# Results land under:
#   eval/results/gen_eval_ablation/run_<timestamp>/<config_name>/
# Logs:
#   eval/results/gen_eval_ablation/logs/
#
# Usage (from repo root, or anywhere):
#   bash scripts/run_gen_eval_ablation.sh
#   bash scripts/run_gen_eval_ablation.sh --benchmark-root data/benchmark --limit 2
#   bash scripts/run_gen_eval_ablation.sh --categories text-only
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
conda activate ads-mrag 2>/dev/null || conda activate chenli 2>/dev/null || conda activate ads_mrag 2>/dev/null || true

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR="$REPO_ROOT/config/config_generator_ablation"
RESULTS_BASE="$REPO_ROOT/eval/results/gen_eval_ablation"
LOG_DIR="$RESULTS_BASE/logs"
SWEEP_TS="$(date '+%Y%m%d_%H%M%S')"
SWEEP_DIR="$RESULTS_BASE/run_${SWEEP_TS}"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

# G1 zeroshot → G7 CP+CoT+snippets (see config/gen_eval_ablation.md).
CONFIGS=(
  "$CONFIG_DIR/config_g1_zeroshot.yaml"          # vanilla (no CP / CoT / ICL / snippets)
  "$CONFIG_DIR/config_g2_cp.yaml"                # + CP
  "$CONFIG_DIR/config_g3_cp_cot.yaml"            # CP + CoT
  "$CONFIG_DIR/config_g4_cp_icl.yaml"            # CP + ICL
  "$CONFIG_DIR/config_g5_cp_icl_cot.yaml"        # CP + ICL + CoT
  "$CONFIG_DIR/config_g6_cp_snippets.yaml"       # CP + snippets
  "$CONFIG_DIR/config_g7_cp_cot_snippets.yaml"   # CP + CoT + snippets
)

EXTRA_ARGS=("$@")
MASTER_LOG="$LOG_DIR/run_${SWEEP_TS}.log"
PASSED=0
FAILED=0
FAILED_NAMES=()

echo "===== Gen-eval ablation sweep start $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$MASTER_LOG"
echo "Repo: $REPO_ROOT" | tee -a "$MASTER_LOG"
echo "Sweep results: $SWEEP_DIR" | tee -a "$MASTER_LOG"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  group_results="$SWEEP_DIR/$name"
  run_log="$LOG_DIR/${name}_${SWEEP_TS}.log"
  mkdir -p "$group_results"

  if [[ ! -f "$cfg" ]]; then
    echo "[SKIP] missing config: $cfg" | tee -a "$MASTER_LOG"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("$name (missing)")
    continue
  fi

  echo "" | tee -a "$MASTER_LOG"
  echo "===== [$(date '+%H:%M:%S')] START $name =====" | tee -a "$MASTER_LOG"
  echo "Config: $cfg" | tee -a "$MASTER_LOG"
  echo "Results: $group_results" | tee -a "$MASTER_LOG"
  echo "Run log: $run_log" | tee -a "$MASTER_LOG"

  python eval/e2e_eval.py --config-path "$cfg" --results-root "$group_results" "${EXTRA_ARGS[@]}" \
    > >(tee "$run_log") 2>&1
  status=$?

  # Detect CUDA OOM via exit code (99) or log content (covers killed/init failures).
  oom_hit=0
  if [[ $status -eq 99 ]]; then
    oom_hit=1
  elif [[ -f "$run_log" ]] && grep -qiE \
    'CUDA out of memory|CUDA_OOM|OutOfMemoryError|torch\.cuda\.OutOfMemoryError|hip out of memory|\[ABORT\].*CUDA' \
    "$run_log"; then
    oom_hit=1
  fi

  if [[ $oom_hit -eq 1 ]]; then
    echo "[ABORT] CUDA OOM in $name (exit=$status); stopping remaining ablation groups" | tee -a "$MASTER_LOG"
    echo "Partial results: $SWEEP_DIR" | tee -a "$MASTER_LOG"
    echo "Run log: $run_log" | tee -a "$MASTER_LOG"
    echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
    exit 99
  fi

  if [[ $status -eq 0 ]]; then
    echo "[OK] $name -> $group_results" | tee -a "$MASTER_LOG"
    PASSED=$((PASSED + 1))
  else
    echo "[FAILED] $name (exit=$status); continuing" | tee -a "$MASTER_LOG"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("$name")
  fi
done

echo "" | tee -a "$MASTER_LOG"
echo "===== Gen-eval ablation sweep done $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$MASTER_LOG"
echo "Passed: $PASSED  Failed: $FAILED" | tee -a "$MASTER_LOG"
echo "Results root: $SWEEP_DIR" | tee -a "$MASTER_LOG"
if (( FAILED > 0 )); then
  echo "Failed configs: ${FAILED_NAMES[*]}" | tee -a "$MASTER_LOG"
fi
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"

# Exit 0 so a launcher/tmux session is not treated as aborted when some (non-OOM) groups fail.
exit 0
