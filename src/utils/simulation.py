"""
Build CLI arguments and paths for run_scenic_batch.sh from config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.config import Config


def repo_root() -> Path:
    """Repository root (ads-mrag/)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve a config path relative to the repository root."""
    if path is None or str(path).strip() == "":
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((repo_root() / candidate).resolve())


def _append_flag(cmd: list[str], flag: str, value: Optional[str | int | float]) -> None:
    if value is None or value == "":
        return
    cmd.extend([flag, str(value)])


def _append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.extend([flag, "1" if value else "0"])


def build_run_scenic_batch_command(
    config: "Config",
    scenic_input: str | Path,
    *,
    outdir: str | Path,
    logdir: str | Path,
    recorder_py: Optional[str | Path] = None,
    extra_scenic_args: Optional[list[str]] = None,
) -> list[str]:
    """
    Build argv for run_scenic_batch.sh from ``config.simulation``.

    Positional ``scenic_input`` is appended last (file or directory).
    """
    sim = config.simulation
    script = resolve_path(sim.batch_script) or str(
        repo_root() / "src" / "utils" / "run_scenic_batch.sh"
    )

    cmd: list[str] = [script]
    recorder = recorder_py or sim.recorder_script
    if recorder:
        resolved_recorder = resolve_path(str(recorder)) if isinstance(recorder, str) else str(recorder)
        _append_flag(cmd, "--recorder", resolved_recorder)

    _append_flag(cmd, "--outdir", str(outdir))
    _append_flag(cmd, "--logdir", str(logdir))

    carla = sim.carla
    if resolve_path(carla.binary_dir):
        _append_flag(cmd, "--carla-dir", resolve_path(carla.binary_dir))
    _append_flag(cmd, "--carla-cmd", carla.cmd or None)
    _append_flag(cmd, "--carla-logdir", resolve_path(carla.log_dir))
    _append_flag(cmd, "--host", carla.host)
    _append_flag(cmd, "--port", carla.port)
    rpc_port = carla.rpc_port if carla.rpc_port is not None else carla.port
    _append_flag(cmd, "--carla-rpc-port", rpc_port)
    _append_bool_flag(cmd, "--auto-start-carla", carla.auto_start)
    if not carla.check_connection:
        cmd.append("--no-check-carla")
    _append_flag(cmd, "--carla-restart", carla.restart_every)
    _append_flag(cmd, "--carla-cooldown", carla.cooldown_sec)
    _append_flag(cmd, "--carla-start-timeout", carla.start_timeout_sec)
    _append_bool_flag(cmd, "--carla-force-restart", carla.force_restart)
    _append_flag(cmd, "--stop-grace", carla.stop_grace_sec)
    _append_flag(cmd, "--stop-term-grace", carla.stop_term_grace_sec)

    scenic = sim.scenic
    _append_flag(cmd, "--scenic-workdir", resolve_path(scenic.workdir))
    _append_flag(cmd, "--map-root", resolve_path(scenic.map_root))
    _append_flag(cmd, "--scenic-version", scenic.version)
    if scenic.conda_env is not None:
        _append_flag(cmd, "--conda-env", scenic.conda_env)
    _append_flag(cmd, "--scenic2-env", scenic.scenic2_conda_env)
    if resolve_path(scenic.scenic3_venv_activate):
        _append_flag(cmd, "--scenic3-venv", resolve_path(scenic.scenic3_venv_activate))
    _append_flag(cmd, "--duration", scenic.duration_sec)
    _append_flag(cmd, "--time", scenic.time_sec)
    if scenic.time_steps is not None:
        _append_flag(cmd, "--time-steps", scenic.time_steps)
    _append_flag(cmd, "--timestep", scenic.timestep)
    _append_flag(cmd, "--count", scenic.count)
    _append_bool_flag(cmd, "--use-2d", scenic.use_2d)
    _append_bool_flag(cmd, "--render", scenic.render)
    _append_bool_flag(cmd, "--show-params", scenic.show_params)
    _append_bool_flag(cmd, "--auto-map", scenic.auto_map)
    if scenic.seed is not None:
        _append_flag(cmd, "--seed", scenic.seed)

    recorder_cfg = sim.recorder
    _append_flag(cmd, "--recorder-ready", recorder_cfg.ready_timeout_sec)
    _append_flag(cmd, "--ego-alive-threshold", recorder_cfg.ego_alive_threshold_sec)
    if recorder_cfg.disabled:
        cmd.append("--no-recorder")

    if extra_scenic_args:
        cmd.append("--")
        cmd.extend(extra_scenic_args)

    cmd.append(str(scenic_input))
    return cmd
