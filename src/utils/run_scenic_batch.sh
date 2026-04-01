#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RECORDER_PY="${RECORDER_PY:-$ROOT_DIR/recorder_scenic.py}"
SCENIC_CMD="${SCENIC_CMD:-scenic}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/scenic_batch_logs}"
OUTDIR="${OUTDIR:-}"
DURATION="${DURATION:-25}"
SCENIC_TIME="${SCENIC_TIME:-25}"
SCENIC_TIME_STEPS="${SCENIC_TIME_STEPS:-}"
SCENIC_TIMESTEP="${SCENIC_TIMESTEP:-0.1}"
SCENIC_TIMESTEP_OVERRIDE="${SCENIC_TIMESTEP_OVERRIDE:-0}"
SCENIC_WORKDIR="${SCENIC_WORKDIR:-/home/dellpro2/chenli/ads-mrag/ads-mrag/Scenic}"
SCENIC_SEED="${SCENIC_SEED:-}"
SCENIC_SHOW_PARAMS="${SCENIC_SHOW_PARAMS:-0}"
SCENIC_RENDER="${SCENIC_RENDER:-1}"
SCENIC_VERSION="${SCENIC_VERSION:-3}"
SCENIC3_VENV_ACTIVATE="${SCENIC3_VENV_ACTIVATE:-$ROOT_DIR/carla/caiwang-venv/bin/activate}"
SCENIC2_CONDA_ENV="${SCENIC2_CONDA_ENV:-scenic2.0}"
EGO_ALIVE_THRESHOLD="${EGO_ALIVE_THRESHOLD:-0.5}"
AUTO_MAP="${AUTO_MAP:-1}"
MAP_ROOT="${MAP_ROOT:-$ROOT_DIR/Scenic/assets/maps/CARLA}"
USE_2D_FLAG="${USE_2D_FLAG:-1}"
SCENIC_COUNT="${SCENIC_COUNT:-1}"
SCENIC_START_TIMEOUT="${SCENIC_START_TIMEOUT:-30}"
RECORDER_GRACE="${RECORDER_GRACE:-10}"
RECORDER_START_DELAY="${RECORDER_START_DELAY:-1}"
RECORDER_READY_TIMEOUT="${RECORDER_READY_TIMEOUT:-8}"
DISABLE_RECORDER="${DISABLE_RECORDER:-0}"
CHECK_CARLA="${CHECK_CARLA:-1}"
AUTO_START_CARLA="${AUTO_START_CARLA:-1}"
CARLA_RESTART_EVERY="${CARLA_RESTART_EVERY:-25}"
CARLA_COOLDOWN_SEC="${CARLA_COOLDOWN_SEC:-5}"
CARLA_FORCE_RESTART="${CARLA_FORCE_RESTART:-0}"
CARLA_LOG_DIR="${CARLA_LOG_DIR:-$ROOT_DIR/scenic_batch_logs}"
CARLA_BINARY_DIR="${CARLA_BINARY_DIR:-/home/dellpro2/caiwang/carla/Dist/CARLA_Shipping_294096eb1/LinuxNoEditor}"
CARLA_ENV_ACTIVATE="${CARLA_ENV_ACTIVATE:-$ROOT_DIR/carla/caiwang-venv/bin/activate}"
CARLA_CMD_STR="${CARLA_CMD_STR:-./CarlaUE4.sh -quality-level=High -nosound -RenderOffScreen -carla-rpc-port=2000}"
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_RPC_PORT="${CARLA_RPC_PORT:-$CARLA_PORT}"
STOP_GRACE="${STOP_GRACE:-10}"
STOP_TERM_GRACE="${STOP_TERM_GRACE:-5}"
START_TIMEOUT="${START_TIMEOUT:-60}"
EGO_ROLENAME="${EGO_ROLENAME:-}"
EGO_ROLE_SCRIPT="${EGO_ROLE_SCRIPT:-$ROOT_DIR/tools/patch_scenic_ego_rolename.py}"
KEEP_EGO_TMP="${KEEP_EGO_TMP:-0}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_scenic_batch.sh [options] <scenic_file_or_dir>...

Options (env vars override):
  --outdir DIR            Output directory for recorder videos
  --logdir DIR            Log directory (default: ./scenic_batch_logs)
  --duration SEC          Recorder duration seconds (default: 25)
  --time SEC              Scenic simulation time seconds (default: 25)
  --time-steps N          Scenic max steps (overrides --time)
  --timestep SEC          Scenic timestep seconds (default: 0.1)
  --scenic-workdir DIR     Working dir for Scenic (default: ./Scenic)
  --seed SEED              Scenic random seed (default: none)
  --show-params 0|1        Log Scenic params (default: 0)
  --render 0|1             Scenic render (pygame window) (default: 1)
  --scenic-version 2|3     Scenic version switch (default: 3)
  --scenic3-venv PATH      Scenic3 venv activate path (default: ./carla/caiwang-venv/bin/activate)
  --scenic2-env NAME       Scenic2 conda env name (default: scenic2.0)
  --ego-alive-threshold SEC Minimum seconds ego must persist (default: 0.5)
  --auto-map 0|1          Auto override map param using Town (default: 1)
  --map-root DIR          Map root dir (default: ./Scenic/assets/maps/CARLA)
  --use-2d 0|1            Add --2d to Scenic (default: 1)
  --count N               Scenic --count (default: 1)
  --start-timeout SEC     Wait for Scenic sim start before recorder (default: 30)
  --recorder PATH         recorder_scenic.py path
  --scenic-cmd CMD         Scenic command (default: scenic)
  --recorder-ready SEC     Wait for recorder ready before scenic (default: 8)
  --no-recorder            Run Scenic only (skip recorder)
  --no-check-carla         Skip CARLA connection pre-check
  --auto-start-carla 0|1   Auto start CARLA if not running (default: 1)
  --carla-restart N        Restart CARLA every N scenarios (default: 25)
  --carla-cooldown SEC     Cooldown seconds between restarts (default: 5)
  --carla-start-timeout SEC Wait for CARLA to be ready (default: 60)
  --carla-force-restart 0|1 Stop CARLA even if not started by script (default: 0)
  --carla-dir DIR          CARLA binary dir (default: /home/dellpro2/caiwang/carla/Dist/CARLA_Shipping_294096eb1/LinuxNoEditor)
  --carla-cmd CMD          CARLA command string (default: ./CarlaUE4.sh -quality-level=High -nosound -RenderOffScreen -carla-rpc-port=2000)
  --carla-logdir DIR       CARLA log directory (default: ./scenic_batch_logs)
  --carla-env PATH          CARLA venv activate path (default: ./carla/caiwang-venv/bin/activate)
  --carla-rpc-port PORT    CARLA RPC port (default: 2000)
  --stop-grace SEC         Grace seconds after SIGINT (default: 10)
  --stop-term-grace SEC    Grace seconds after SIGTERM (default: 5)
  --host HOST              CARLA host for recorder (default: localhost)
  --port PORT              CARLA port for recorder (default: 2000)
  --ego-rolename NAME       Insert rolename for ego (default: none)
  --keep-ego-tmp            Keep temporary Scenic files (default: off)
  --help                   Show help
  --                       Pass remaining args to Scenic (e.g. --param foo bar)

Examples:
  ./run_scenic_batch.sh Scenic/examples/carla/Chatscene
  ./run_scenic_batch.sh --outdir ./recordings --time 25 -- Scenic/examples/carla/Chatscene/dynamic_0.scenic
USAGE
}

SCENIC_ARGS=()
INPUT_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="$2"; shift 2 ;;
    --logdir)
      LOG_DIR="$2"; shift 2 ;;
    --duration)
      DURATION="$2"; shift 2 ;;
    --time)
      SCENIC_TIME="$2"; shift 2 ;;
    --time-steps)
      SCENIC_TIME_STEPS="$2"; shift 2 ;;
    --timestep)
      SCENIC_TIMESTEP="$2"; SCENIC_TIMESTEP_OVERRIDE="1"; shift 2 ;;
    --scenic-workdir)
      SCENIC_WORKDIR="$2"; shift 2 ;;
    --seed)
      SCENIC_SEED="$2"; shift 2 ;;
    --show-params)
      if [[ -n "${2-}" && "${2}" != -* ]]; then
        SCENIC_SHOW_PARAMS="$2"; shift 2
      else
        SCENIC_SHOW_PARAMS="1"; shift
      fi ;;
    --render)
      SCENIC_RENDER="$2"; shift 2 ;;
    --scenic-version)
      SCENIC_VERSION="$2"; shift 2 ;;
    --scenic3-venv)
      SCENIC3_VENV_ACTIVATE="$2"; shift 2 ;;
    --scenic2-env)
      SCENIC2_CONDA_ENV="$2"; shift 2 ;;
    --ego-alive-threshold)
      EGO_ALIVE_THRESHOLD="$2"; shift 2 ;;
    --auto-map)
      AUTO_MAP="$2"; shift 2 ;;
    --map-root)
      MAP_ROOT="$2"; shift 2 ;;
    --use-2d)
      USE_2D_FLAG="$2"; shift 2 ;;
    --count)
      SCENIC_COUNT="$2"; shift 2 ;;
    --start-timeout)
      SCENIC_START_TIMEOUT="$2"; shift 2 ;;
    --recorder)
      RECORDER_PY="$2"; shift 2 ;;
    --scenic-cmd)
      SCENIC_CMD="$2"; shift 2 ;;
    --recorder-ready)
      RECORDER_READY_TIMEOUT="$2"; shift 2 ;;
    --no-recorder)
      DISABLE_RECORDER="1"; shift ;;
    --no-check-carla)
      CHECK_CARLA="0"; shift ;;
    --auto-start-carla)
      AUTO_START_CARLA="$2"; shift 2 ;;
    --carla-restart)
      CARLA_RESTART_EVERY="$2"; shift 2 ;;
    --carla-cooldown)
      CARLA_COOLDOWN_SEC="$2"; shift 2 ;;
    --carla-start-timeout)
      START_TIMEOUT="$2"; shift 2 ;;
    --carla-force-restart)
      CARLA_FORCE_RESTART="$2"; shift 2 ;;
    --carla-dir)
      CARLA_BINARY_DIR="$2"; shift 2 ;;
    --carla-cmd)
      CARLA_CMD_STR="$2"; shift 2 ;;
    --carla-logdir)
      CARLA_LOG_DIR="$2"; shift 2 ;;
    --carla-env)
      CARLA_ENV_ACTIVATE="$2"; shift 2 ;;
    --carla-rpc-port)
      CARLA_RPC_PORT="$2"; shift 2 ;;
    --stop-grace)
      STOP_GRACE="$2"; shift 2 ;;
    --stop-term-grace)
      STOP_TERM_GRACE="$2"; shift 2 ;;
    --host)
      CARLA_HOST="$2"; shift 2 ;;
    --port)
      CARLA_PORT="$2"; shift 2 ;;
    --ego-rolename)
      EGO_ROLENAME="$2"; shift 2 ;;
    --keep-ego-tmp)
      KEEP_EGO_TMP="1"; shift ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      SCENIC_ARGS=("$@")
      break ;;
    -*)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 2 ;;
    *)
      INPUT_PATHS+=("$1"); shift ;;
  esac
done

if [[ ${#INPUT_PATHS[@]} -eq 0 ]]; then
  echo "[ERROR] No input paths provided." >&2
  usage
  exit 2
fi

if [[ ! -f "$RECORDER_PY" ]]; then
  echo "[ERROR] recorder_scenic.py not found: $RECORDER_PY" >&2
  exit 2
fi
if [[ -n "$EGO_ROLENAME" && ! -f "$EGO_ROLE_SCRIPT" ]]; then
  echo "[ERROR] ego rolename script not found: $EGO_ROLE_SCRIPT" >&2
  exit 2
fi

if [[ "${SCENIC_CMD}" == "scenic" ]]; then
  if [[ "$SCENIC_VERSION" == "3" ]]; then
    if [[ -x "${SCENIC3_VENV_ACTIVATE%/bin/activate}/bin/scenic" ]]; then
      SCENIC_CMD="${SCENIC3_VENV_ACTIVATE%/bin/activate}/bin/scenic"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/scenic" ]]; then
      SCENIC_CMD="$VIRTUAL_ENV/bin/scenic"
    fi
  elif [[ "$SCENIC_VERSION" == "2" ]]; then
    SCENIC_CMD="conda run -n ${SCENIC2_CONDA_ENV} scenic"
  fi
fi

if [[ "$CARLA_PORT" == "2000" && "$CARLA_RPC_PORT" != "$CARLA_PORT" ]]; then
  CARLA_PORT="$CARLA_RPC_PORT"
fi

mkdir -p "$LOG_DIR" "$CARLA_LOG_DIR"

batch_ts="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/batch_${batch_ts}.log"
MASTER_CSV="$LOG_DIR/batch_${batch_ts}.csv"

log_master() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
}

init_csv() {
  python3 - "$MASTER_CSV" <<'PY'
import csv
import sys
path = sys.argv[1]
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow([
        "grandparent_dir",
        "parent_dir",
        "scenario_file",
        "map_name",
        "simulation_completed",
        "ran_simulation_seconds",
        "video_recorded",
        "overall_status",
        "ego_spawned",
        "extra_2",
        "extra_3",
    ])
PY
}

append_csv_row() {
  local scenic_file="$1"
  local grandparent_dir="$2"
  local parent_dir="$3"
  local base_file="$4"
  local scenic_rc="$5"
  local recorder_started="$6"
  local recorder_killed="$7"
  local rec_rc="$8"
  local run_log="$9"
  local ego_threshold="${10}"

  python3 - "$MASTER_CSV" "$scenic_file" "$grandparent_dir" "$parent_dir" "$base_file" "$scenic_rc" "$recorder_started" "$recorder_killed" "$rec_rc" "$run_log" "$ego_threshold" <<'PY'
import csv
import re
import sys

csv_path = sys.argv[1]
grandparent_dir = sys.argv[3]
parent_dir = sys.argv[4]
base_file = sys.argv[5]
scenic_rc = int(sys.argv[6])
recorder_started = sys.argv[7] == "1"
recorder_killed = sys.argv[8] == "1"
rec_rc = int(sys.argv[9]) if sys.argv[9].isdigit() else -1
run_log = sys.argv[10]
ego_threshold = float(sys.argv[11])

map_name = ""
sim_time = ""
ego_detected = False
ego_alive_seconds = None

try:
    with open(run_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("map_name:"):
                val = line.split("map_name:", 1)[1].strip()
                map_name = val.split("/")[-1] if val else map_name
            m = re.search(r"Ran simulation in ([0-9.]+) seconds", line)
            if m:
                sim_time = m.group(1)
            if "first ego detected" in line:
                ego_detected = True
            m = re.search(r"ego_alive_seconds:\\s*([0-9.]+)", line)
            if m:
                ego_alive_seconds = float(m.group(1))
except Exception:
    pass

simulation_completed = "yes" if scenic_rc == 0 else "no"
video_recorded = "recorded" if (recorder_started and not recorder_killed and rec_rc == 0) else "unrecorded"
overall_status = "success" if (scenic_rc == 0 and not recorder_killed) else "fail"

with open(csv_path, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    ego_status = "not_generated"
    if ego_alive_seconds is not None:
        if ego_alive_seconds >= ego_threshold:
            ego_status = "ego_spawned"
    elif ego_detected:
        ego_status = "ego_spawned"
    w.writerow([
        grandparent_dir,
        parent_dir,
        base_file,
        map_name,
        simulation_completed,
        sim_time,
        video_recorded,
        overall_status,
        ego_status,
        "",
        "",
    ])
PY
}

check_carla() {
  python3 - <<PY
import carla
client = carla.Client("${CARLA_HOST}", int("${CARLA_RPC_PORT}"))
client.set_timeout(2.0)
client.get_world()
print("CARLA OK")
PY
}

CARLA_PID=""
CARLA_PGID=""
CARLA_OWNED=0

# Find the real CARLA server by the RPC port listener (more reliable than $!).
find_carla_pid() {
  local port="$1"
  local pid=""
  # Prefer listening PID on the RPC port (most reliable).
  if command -v ss >/dev/null 2>&1; then
    pid="$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {for (i=1;i<=NF;i++) if ($i ~ /pid=/) {gsub(/.*pid=|,.*/, "", $i); print $i; exit}}')"
  elif command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $2; exit}')"
  fi
  if [[ -z "${pid:-}" ]]; then
    pid="$(pgrep -f "CarlaUE4.*carla-rpc-port[= ]${port}" | head -n1)"
  fi
  if [[ -z "${pid:-}" ]]; then
    pid="$(pgrep -f "CarlaUE4-Linux-Shipping.*carla-rpc-port[= ]${port}" | head -n1)"
  fi
  printf '%s' "${pid:-}"
}

get_pgid() {
  local pid="$1"
  ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' '
}

port_listening() {
  local port="$1"
  [[ -n "$(find_carla_pid "$port")" ]]
}

start_carla() {
  local ts carla_log listen_pid cmdline wait_start
  ts="$(date +%Y%m%d_%H%M%S)"
  carla_log="$CARLA_LOG_DIR/carla_${ts}.log"

  # Check port occupancy before starting; avoid conflicting instances.
  listen_pid="$(find_carla_pid "$CARLA_RPC_PORT")"
  if [[ -n "${listen_pid:-}" ]]; then
    cmdline="$(ps -p "$listen_pid" -o args= 2>/dev/null)"
    if [[ "$CARLA_FORCE_RESTART" == "1" ]]; then
      log_master "CARLA port ${CARLA_RPC_PORT} busy (pid=$listen_pid). Forcing stop before start."
      if ! stop_carla; then
        echo "[ERROR] CARLA port ${CARLA_RPC_PORT} still busy after stop attempt." >&2
        exit 3
      fi
    else
      echo "[ERROR] CARLA port ${CARLA_RPC_PORT} already in use (pid=$listen_pid cmdline='$cmdline')." >&2
      exit 3
    fi
  fi

  log_master "Starting CARLA: (cd \"$CARLA_BINARY_DIR\" && ${CARLA_CMD_STR})"
  (
    cd "$CARLA_BINARY_DIR" && \
    __NV_PRIME_RENDER_OFFLOAD=1 \
    __GLX_VENDOR_LIBRARY_NAME=nvidia
    if command -v setsid >/dev/null 2>&1; then
      setsid bash -c "exec $CARLA_CMD_STR"
    else
      bash -c "exec $CARLA_CMD_STR"
    fi
  ) >> "$carla_log" 2>&1 &
  CARLA_PID=$!
  CARLA_PGID="$(get_pgid "$CARLA_PID")"
  CARLA_OWNED=1

  # Wait for the real CARLA server to listen on RPC port and pass health check.
  wait_start=$SECONDS
  while (( SECONDS - wait_start < START_TIMEOUT )); do
    listen_pid="$(find_carla_pid "$CARLA_RPC_PORT")"
    if [[ -n "${listen_pid:-}" ]]; then
      if check_carla >/dev/null 2>&1; then
        log_master "CARLA ready (listen pid=$listen_pid, pgid=$(get_pgid "$listen_pid"), log: $carla_log)"
        return 0
      fi
    fi
    sleep 1
  done
  echo "[ERROR] CARLA did not become ready within ${START_TIMEOUT}s." >&2
  return 1
}

stop_carla() {
  local port="${CARLA_RPC_PORT}"
  local start_ts end_ts elapsed
  local listen_pid pgid target_pgid
  local log_prefix="CARLA stop"

  start_ts=$(date +%s)
  listen_pid="$(find_carla_pid "$port")"
  if [[ -z "${listen_pid:-}" ]]; then
    log_master "${log_prefix}: no listener on port ${port}"
    return 0
  fi

  pgid="$(get_pgid "$listen_pid")"
  target_pgid="$pgid"
  if [[ -n "${CARLA_PGID:-}" && "$CARLA_PGID" != "$pgid" ]]; then
    log_master "${log_prefix}: PGID mismatch (recorded=${CARLA_PGID}, current=${pgid}); using current"
  fi

  log_master "${log_prefix}: stopping pid=${listen_pid} pgid=${target_pgid:-unknown} on port=${port}"

  # Use process group so UE4 and its children exit together; fallback to PID if PGID is unknown.
  if [[ -n "${target_pgid:-}" ]]; then
    kill -2 "-${target_pgid}" >/dev/null 2>&1 || true
  else
    kill -2 "$listen_pid" >/dev/null 2>&1 || true
  fi
  sleep "$STOP_GRACE"
  if port_listening "$port"; then
    log_master "${log_prefix}: still listening after SIGINT, sending SIGTERM"
    if [[ -n "${target_pgid:-}" ]]; then
      kill -15 "-${target_pgid}" >/dev/null 2>&1 || true
    else
      kill -15 "$listen_pid" >/dev/null 2>&1 || true
    fi
    sleep "$STOP_TERM_GRACE"
  fi
  if port_listening "$port"; then
    log_master "${log_prefix}: still listening after SIGTERM, sending SIGKILL"
    if [[ -n "${target_pgid:-}" ]]; then
      kill -9 "-${target_pgid}" >/dev/null 2>&1 || true
    else
      kill -9 "$listen_pid" >/dev/null 2>&1 || true
    fi
    sleep 1
  fi

  if port_listening "$port"; then
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    log_master "${log_prefix}: FAILED to release port after ${elapsed}s"
    return 1
  fi

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  log_master "${log_prefix}: released port in ${elapsed}s"
  CARLA_PID=""
  CARLA_PGID=""
  CARLA_OWNED=0
  return 0
}

ensure_carla() {
  if check_carla >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$AUTO_START_CARLA" != "1" ]]; then
    echo "[ERROR] Cannot connect to CARLA at ${CARLA_HOST}:${CARLA_RPC_PORT}. Start CARLA or use --auto-start-carla 1." >&2
    exit 3
  fi
  if ! start_carla; then
    exit 3
  fi
}

if [[ "$CHECK_CARLA" == "1" ]]; then
  ensure_carla
fi

collect_scenic_files() {
  local path="$1"
  local files=()
  if [[ -f "$path" && "$path" == *.scenic ]]; then
    files+=("$path")
  elif [[ -d "$path" ]]; then
    while IFS= read -r -d '' f; do
      files+=("$f")
    done < <(find "$path" -type f -name "*.scenic" -print0 | sort -z)
  fi
  printf '%s\n' "${files[@]}"
}

RECORDER_ARGS=(--host "$CARLA_HOST" --port "$CARLA_PORT" --duration "$DURATION")
RECORDER_ARGS+=(--ego-alive-threshold "$EGO_ALIVE_THRESHOLD")
if [[ -n "$OUTDIR" ]]; then
  RECORDER_ARGS+=(--outdir "$OUTDIR")
fi

log_master "Recorder: $RECORDER_PY"
log_master "Scenic cmd: $SCENIC_CMD"
log_master "Recorder duration: ${DURATION}s | Scenic time: ${SCENIC_TIME}s | Scenic timestep: ${SCENIC_TIMESTEP}s | Scenic max steps: ${SCENIC_TIME_STEPS:-auto}"
log_master "Scenic workdir: ${SCENIC_WORKDIR}"
log_master "Scenic seed: ${SCENIC_SEED:-none} | Show params: ${SCENIC_SHOW_PARAMS} | Render: ${SCENIC_RENDER}"
log_master "Recorder disabled: ${DISABLE_RECORDER}"
log_master "Auto map override: ${AUTO_MAP} | Map root: ${MAP_ROOT}"
log_master "--2d flag: ${USE_2D_FLAG}"
log_master "Scenic --count: ${SCENIC_COUNT}"
log_master "Scenic start wait: ${SCENIC_START_TIMEOUT}s"
log_master "Recorder ready wait: ${RECORDER_READY_TIMEOUT}s"
log_master "Ego rolename: ${EGO_ROLENAME:-none}"
log_master "Auto start CARLA: ${AUTO_START_CARLA} | Restart every: ${CARLA_RESTART_EVERY} | Cooldown: ${CARLA_COOLDOWN_SEC}s"
log_master "CARLA dir: ${CARLA_BINARY_DIR} | Force restart: ${CARLA_FORCE_RESTART} | Start timeout: ${START_TIMEOUT}s"
log_master "CARLA rpc port: ${CARLA_RPC_PORT} | Stop grace: ${STOP_GRACE}s | Stop term grace: ${STOP_TERM_GRACE}s"
log_master "Scenic version: ${SCENIC_VERSION} | Scenic cmd: ${SCENIC_CMD}"
log_master "CARLA env activate: ${CARLA_ENV_ACTIVATE} | Scenic3 venv: ${SCENIC3_VENV_ACTIVATE} | Scenic2 env: ${SCENIC2_CONDA_ENV}"
log_master "Ego alive threshold: ${EGO_ALIVE_THRESHOLD}s"
log_master "CSV summary: ${MASTER_CSV}"

init_csv

EGO_TMP_DIR=""
cleanup_ego_tmp() {
  if [[ -n "$EGO_TMP_DIR" && -d "$EGO_TMP_DIR" ]]; then
    if [[ "$KEEP_EGO_TMP" == "1" ]]; then
      echo "[INFO] Keeping ego temp dir: $EGO_TMP_DIR"
    else
      rm -rf "$EGO_TMP_DIR"
    fi
  fi
}
trap cleanup_ego_tmp EXIT

total_count=0
success_count=0
fail_count=0
failed_list=()
executed_count=0
for p in "${INPUT_PATHS[@]}"; do
  mapfile -t SCENIC_FILES < <(collect_scenic_files "$p")
  if [[ ${#SCENIC_FILES[@]} -eq 0 ]]; then
    log_master "⚠️ SKIP no .scenic found in: $p"
    continue
  fi
  total_count=$((total_count + ${#SCENIC_FILES[@]}))
  log_master "▶️ PATH START $p (${#SCENIC_FILES[@]} scenario(s))"

  for scenic_file in "${SCENIC_FILES[@]}"; do
  base="$(basename "$scenic_file")"
  parent_dir="$(basename "$(dirname "$scenic_file")")"
  grandparent_dir="$(basename "$(dirname "$(dirname "$scenic_file")")")"
  prefix="${grandparent_dir}__${parent_dir}__${base%.scenic}"
  run_ts="$(date +%Y%m%d_%H%M%S)"
  run_id="${prefix}__${run_ts}"
  run_log="$LOG_DIR/${run_id}.log"

  log_master "▶️ START $scenic_file (log: $run_log)"

  if [[ "$CHECK_CARLA" == "1" ]]; then
    ensure_carla
  fi

  scenic_run_file="$scenic_file"
  if [[ -n "$EGO_ROLENAME" ]]; then
    if [[ -z "$EGO_TMP_DIR" ]]; then
      EGO_TMP_DIR="$(mktemp -d "$ROOT_DIR/.scenic_ego_tmp.XXXXXX")"
    fi
    rel_path="${scenic_file#$ROOT_DIR/}"
    if [[ "$rel_path" == "$scenic_file" ]]; then
      rel_path="$(basename "$scenic_file")"
    fi
    tmp_file="$EGO_TMP_DIR/$rel_path"
    mkdir -p "$(dirname "$tmp_file")"
    if ! python3 "$EGO_ROLE_SCRIPT" "$scenic_file" "$tmp_file" --rolename "$EGO_ROLENAME"; then
      echo "[ERROR] Failed to inject ego rolename for $scenic_file" >&2
      exit 4
    fi
    scenic_run_file="$tmp_file"
  fi

  scenic_file_abs="$(python3 - "$scenic_run_file" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"

  {
    echo "=== CARLA SETTINGS (pre-scenic) ==="
    python3 - <<PY || true
import carla
client = carla.Client("${CARLA_HOST}", int("${CARLA_PORT}"))
client.set_timeout(2.0)
world = client.get_world()
settings = world.get_settings()
print("map_name:", world.get_map().name)
print("fixed_delta_seconds:", settings.fixed_delta_seconds)
print("synchronous_mode:", settings.synchronous_mode)
print("no_rendering_mode:", settings.no_rendering_mode)
print("max_substep_delta_time:", settings.max_substep_delta_time)
print("max_substeps:", settings.max_substeps)
print("snapshot.frame:", world.get_snapshot().frame)
print("snapshot.timestamp.elapsed_seconds:", world.get_snapshot().timestamp.elapsed_seconds)
PY
  } >> "$run_log" 2>&1

  pre_sim_info="$(python3 - <<PY
import carla
client = carla.Client("${CARLA_HOST}", int("${CARLA_PORT}"))
client.set_timeout(2.0)
world = client.get_world()
snap = world.get_snapshot()
print(f"{snap.frame} {snap.timestamp.elapsed_seconds}")
PY
)"

  MAP_ARGS=()
  if [[ "$USE_2D_FLAG" == "1" ]]; then
    MAP_ARGS+=(--2d)
  fi

  SCENIC_EXTRA_ARGS=()
  if [[ -n "${SCENIC_SEED:-}" ]]; then
    SCENIC_EXTRA_ARGS+=(--seed "$SCENIC_SEED")
  fi
  if [[ "$SCENIC_SHOW_PARAMS" == "1" ]]; then
    SCENIC_EXTRA_ARGS+=(--show-params)
  fi
  if [[ -n "${SCENIC_RENDER:-}" ]]; then
    SCENIC_EXTRA_ARGS+=(--param render "$SCENIC_RENDER")
  fi
  if [[ "$SCENIC_TIMESTEP_OVERRIDE" == "1" ]]; then
    SCENIC_EXTRA_ARGS+=(--param timestep "$SCENIC_TIMESTEP")
  fi

  SCENIC_TIME_ARG=()
  if [[ -n "${SCENIC_TIME_STEPS:-}" ]]; then
    SCENIC_TIME_ARG=(--time "$SCENIC_TIME_STEPS")
  elif [[ -n "${SCENIC_TIME:-}" && "$SCENIC_TIME" != "0" ]]; then
    steps="$(python3 - "$SCENIC_TIME" "$SCENIC_TIMESTEP" <<'PY'
import math
import sys
sec = float(sys.argv[1])
ts = float(sys.argv[2])
if ts <= 0:
    raise SystemExit(1)
print(int(math.ceil(sec / ts)))
PY
)"
    SCENIC_TIME_ARG=(--time "$steps")
  fi
  MAP_NOTE="none"
  if [[ "$AUTO_MAP" == "1" ]]; then
    town="$(python3 - "$scenic_file" <<'PY'
import re
import sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"\\s*Town\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", line)
            if m:
                print(m.group(1))
                sys.exit(0)
except Exception:
    pass
PY
)"
    if [[ -n "${town:-}" ]]; then
      map_path="$MAP_ROOT/${town}.xodr"
      if [[ -f "$map_path" ]]; then
        if grep -Eq "^[[:space:]]*param[[:space:]]+map" "$scenic_file"; then
          MAP_ARGS+=(--param map "$map_path")
          MAP_NOTE="map=$map_path"
        fi
        if grep -Eq "^[[:space:]]*param[[:space:]]+carla_map" "$scenic_file"; then
          MAP_ARGS+=(--param carla_map "$town")
          if [[ "$MAP_NOTE" == "none" ]]; then
            MAP_NOTE="carla_map=$town"
          else
            MAP_NOTE="$MAP_NOTE, carla_map=$town"
          fi
        fi
      else
        MAP_NOTE="missing $map_path"
      fi
    fi
  fi

  {
    echo "=== RUN START $(date '+%F %T') ==="
    echo "Scenic file: $scenic_file"
    echo "Scenic run file: $scenic_run_file"
    echo "Scenic run file abs: $scenic_file_abs"
    echo "Prefix: $prefix"
    echo "Recorder: $RECORDER_PY ${RECORDER_ARGS[*]} --prefix $prefix"
    echo "Recorder disabled: ${DISABLE_RECORDER}"
    echo "Auto map: $MAP_NOTE"
    echo "Scenic: (cd \"$SCENIC_WORKDIR\" && $SCENIC_CMD \"$scenic_file_abs\" --simulate --model scenic.simulators.carla.model ${SCENIC_TIME_ARG[*]} --count $SCENIC_COUNT ${MAP_ARGS[*]} ${SCENIC_EXTRA_ARGS[*]} ${SCENIC_ARGS[*]})"
    echo "=== RECORDING START ==="
  } >> "$run_log"

  log_master "CMD recorder: python3 \"$RECORDER_PY\" --prefix \"$prefix\" ${RECORDER_ARGS[*]}"
  log_master "CMD scenic: (cd \"$SCENIC_WORKDIR\" && $SCENIC_CMD \"$scenic_file_abs\" --simulate --model scenic.simulators.carla.model ${SCENIC_TIME_ARG[*]} --count $SCENIC_COUNT ${MAP_ARGS[*]} ${SCENIC_EXTRA_ARGS[*]} ${SCENIC_ARGS[*]})"

  rec_pid=""
  recorder_started=0
  if [[ "$DISABLE_RECORDER" != "1" ]]; then
    python3 "$RECORDER_PY" --prefix "$prefix" "${RECORDER_ARGS[@]}" >> "$run_log" 2>&1 &
    rec_pid=$!
    recorder_started=1
    sleep "$RECORDER_START_DELAY"

    if [[ "$RECORDER_READY_TIMEOUT" -gt 0 ]]; then
      wait_start=$SECONDS
      while ! grep -q "recording started" "$run_log" 2>/dev/null; do
        if (( SECONDS - wait_start >= RECORDER_READY_TIMEOUT )); then
          echo "⚠️ Recorder ready wait timed out after ${RECORDER_READY_TIMEOUT}s." >> "$run_log"
          break
        fi
        sleep 0.2
      done
    fi
  else
    echo "ℹ️ Recorder disabled; running Scenic only." >> "$run_log"
  fi

  (
    cd "$SCENIC_WORKDIR" && \
    $SCENIC_CMD "$scenic_file_abs" --simulate --model scenic.simulators.carla.model "${SCENIC_TIME_ARG[@]}" --count "$SCENIC_COUNT" "${MAP_ARGS[@]}" "${SCENIC_EXTRA_ARGS[@]}" "${SCENIC_ARGS[@]}"
  ) >> "$run_log" 2>&1
  scenic_rc=$?

  {
    echo "=== CARLA SETTINGS (post-scenic) ==="
    python3 - <<PY || true
import carla
client = carla.Client("${CARLA_HOST}", int("${CARLA_PORT}"))
client.set_timeout(2.0)
world = client.get_world()
settings = world.get_settings()
print("map_name:", world.get_map().name)
print("fixed_delta_seconds:", settings.fixed_delta_seconds)
print("synchronous_mode:", settings.synchronous_mode)
print("no_rendering_mode:", settings.no_rendering_mode)
print("max_substep_delta_time:", settings.max_substep_delta_time)
print("max_substeps:", settings.max_substeps)
print("snapshot.frame:", world.get_snapshot().frame)
print("snapshot.timestamp.elapsed_seconds:", world.get_snapshot().timestamp.elapsed_seconds)
PY
  } >> "$run_log" 2>&1

  python3 - "$pre_sim_info" "$CARLA_HOST" "$CARLA_PORT" <<'PY' >> "$run_log" 2>&1 || true
import carla
import sys
pre = sys.argv[1].split()
if len(pre) == 2:
    pre_frame = int(pre[0])
    pre_time = float(pre[1])
    host = sys.argv[2]
    port = int(sys.argv[3])
    client = carla.Client(host, port)
    client.set_timeout(2.0)
    world = client.get_world()
    snap = world.get_snapshot()
    post_frame = snap.frame
    post_time = snap.timestamp.elapsed_seconds
    print("carla_frame_delta:", post_frame - pre_frame)
    print("carla_elapsed_seconds_delta:", post_time - pre_time)
PY

  recorder_killed=0
  rec_rc=0
  if [[ "$recorder_started" == "1" ]]; then
    wait_start=$SECONDS
    while kill -0 "$rec_pid" >/dev/null 2>&1; do
      if (( SECONDS - wait_start > RECORDER_GRACE )); then
        echo "⚠️ Recorder still running after scenic exit; terminating." >> "$run_log"
        kill "$rec_pid" >/dev/null 2>&1 || true
        sleep 1
        kill -9 "$rec_pid" >/dev/null 2>&1 || true
        recorder_killed=1
        break
      fi
      sleep 1
    done
    wait "$rec_pid" >/dev/null 2>&1
    rec_rc=$?
  fi

  {
    echo "=== RUN END $(date '+%F %T') ==="
    echo "Scenic exit code: $scenic_rc"
    echo "Recorder exit code: $rec_rc"
    echo "Recorder killed: $recorder_killed"
  } >> "$run_log"

  if [[ $scenic_rc -eq 0 && $recorder_killed -eq 0 ]]; then
    log_master "✅ DONE $scenic_file (ok)"
    success_count=$((success_count + 1))
  else
    log_master "❌ DONE $scenic_file (scenic_rc=$scenic_rc recorder_killed=$recorder_killed)"
    fail_count=$((fail_count + 1))
    failed_list+=("$scenic_file (scenic_rc=$scenic_rc recorder_killed=$recorder_killed)")
  fi

  append_csv_row "$scenic_file" "$grandparent_dir" "$parent_dir" "$base" "$scenic_rc" "$recorder_started" "$recorder_killed" "$rec_rc" "$run_log" "$EGO_ALIVE_THRESHOLD"

  executed_count=$((executed_count + 1))
  if [[ "$AUTO_START_CARLA" == "1" && "$CARLA_RESTART_EVERY" -gt 0 && $((executed_count % CARLA_RESTART_EVERY)) -eq 0 ]]; then
    log_master "🔄 Restarting CARLA after ${executed_count} scenario(s)"
    if [[ "$CARLA_OWNED" != "1" && "$CARLA_FORCE_RESTART" != "1" ]]; then
      log_master "⚠️ Skip restart: CARLA not started by script and force restart disabled"
    else
      if ! stop_carla; then
        log_master "❌ CARLA did not stop cleanly; aborting restart to avoid conflict"
        exit 3
      fi
      sleep "$CARLA_COOLDOWN_SEC"
      if ! start_carla; then
        echo "[ERROR] CARLA did not become ready within ${START_TIMEOUT}s after restart." >&2
        exit 3
      fi
    fi
  fi
done

  log_master "✅ PATH END $p"
done

log_master "✅ Batch complete (total: ${total_count}, success: ${success_count}, fail: ${fail_count})"
if [[ ${#failed_list[@]} -gt 0 ]]; then
  log_master "❌ Failed scenarios:"
  for f in "${failed_list[@]}"; do
    log_master "  - $f"
  done
fi

if [[ "$AUTO_START_CARLA" == "1" ]]; then
  log_master "🛑 Stopping CARLA after batch complete"
  if [[ "$CARLA_OWNED" != "1" && "$CARLA_FORCE_RESTART" != "1" ]]; then
    log_master "⚠️ Skip stop: CARLA not started by script and force restart disabled"
  else
    if ! stop_carla; then
      log_master "❌ CARLA did not stop cleanly after batch complete"
    fi
  fi
fi
