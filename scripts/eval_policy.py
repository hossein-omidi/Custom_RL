"""Evaluate trained PPO policies and save face-milling trajectories.

This script is evaluation-only. It does not train, tune, or modify the plant.
It runs a trained policy in ``CustomODEPlate-v0`` and stores trajectories with
both normalized actions and physical machining quantities.

Current face-milling convention
-------------------------------
Default normalized action:
    u = [u_omega, u_ap] in [-1, 1]^2

Physical action after plant scaling:
    [omega_rad_s, ap_mm]

The radial immersion ``ae`` is a fixed process parameter unless the plant was
created with ``control_ae=True``. The environment observation contains physical
sensor displacement/velocity scaled for RL, plus normalized cutter coordinates
in the current plant:
    [w/w_scale, wdot/wdot_scale, cutter_x/L1, cutter_y/L2]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
import sys
import hashlib
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from custom_rl import DEFAULT_MODEL_DIR, DEFAULT_TRAJ_DIR, register_envs
from custom_rl.eval.monte_carlo import face_milling_process_from_info
from custom_rl.eval.pipeline import (
    discover_model_seeds,
    plate_env_kwargs,
    plant_plot_metadata,
    resolve_ppo_model_path,
)
from custom_rl.plants.plate import omega_to_rpm


ENV_ID = "CustomODEPlate-v0"
ENV_ID_FINISHING = "CustomODEPlateFinish-v0"
PASS_COMPLETED_REASONS = {"pass_completed_90percent", "pass_completed"}

# Standalone evaluation defaults must match the uploaded training script.
# Training uses RK4 dt=1e-4 with 10 RK4 substeps per environment/control step,
# giving a 1 ms control interval while still resolving the 4000-rpm tooth delay.
EVAL_DEFAULT_DT = 1.0e-4
EVAL_DEFAULT_N_SUBSTEPS = 10
# Legacy-checkpoint fallback only. Unified-run checkpoints load this cap from
# their saved training_config.json, avoiding silent train/eval truncation mismatch.
EVAL_DEFAULT_MAX_EPISODE_STEPS = 250000
EVAL_DEFAULT_DYNAMICS_UNCERTAINTY_STD = 0.001
EVAL_DEFAULT_RANDOMIZE_Y0 = True

PROCESS_KEYS = (
    "omega_rad_s",
    "omega_rpm",
    "ap_mm",
    "ae_mm",
    "cutter_x",
    "cutter_y",
    "feed_progress",
    "feed_distance_m",
    "spindle_phase_rad",
    "mean_chip_mm",
    "max_chip_mm",
    "max_abs_w_m",
    "Fx_N",
    "Fy_N",
    "Fz_N",
    "F_mag_N",
)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _to_jsonable(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON-safe values."""
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    return value


def _get_metadata(env: gym.Env, max_episode_steps: int, reward_id: str) -> dict[str, Any]:
    """Collect useful metadata for plotting and post-processing."""
    metadata = plant_plot_metadata(env)
    metadata["max_episode_steps"] = int(max_episode_steps)
    metadata["reward_id"] = reward_id
    return metadata


def _physical_action(env: gym.Env, action: np.ndarray) -> list[float]:
    """Convert normalized policy action to physical [omega_rad_s, ap_mm, optional ae_mm]."""
    plant = env.unwrapped.plant
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)

    if hasattr(plant, "_scale_action"):
        return np.asarray(plant._scale_action(action_arr), dtype=np.float64).reshape(-1).tolist()

    if hasattr(plant, "physical_action_bounds"):
        low, high = plant.physical_action_bounds()
        low = np.asarray(low, dtype=np.float64).reshape(-1)
        high = np.asarray(high, dtype=np.float64).reshape(-1)
        action_arr = np.clip(action_arr[: low.size], -1.0, 1.0)
        return (low + 0.5 * (action_arr + 1.0) * (high - low)).tolist()

    return action_arr.tolist()


def _physical_signal_from_info_or_obs(
    info: dict[str, Any],
    obs: np.ndarray,
    metadata: dict[str, Any],
) -> list[float]:
    """
    Return unscaled physical sensor signal:
        [w_sensor_1, ..., w_sensor_n, wdot_sensor_1, ..., wdot_sensor_n]

    Preferred source is info["w_sensor"] / info["wdot_sensor"]. The fallback
    correctly inverts the plant observation convention:
        obs[:n] = w_sensor / w_obs_scale
        obs[n:2n] = wdot_sensor / wdot_obs_scale
    Extra observation entries, such as cutter_x/L1 and cutter_y/L2, are ignored.
    """
    if "w_sensor" in info and "wdot_sensor" in info:
        w_sensor = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        wdot_sensor = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
        return np.concatenate([w_sensor, wdot_sensor]).tolist()

    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    n_sensors = int(metadata.get("n_sensors", 0))
    if n_sensors <= 0:
        # Current plant observations append two normalized cutter coordinates.
        # Do not count those coordinates as extra sensor channels.
        n_sensors = max((obs_arr.size - 2) // 2, 0)

    needed = 2 * n_sensors
    if n_sensors <= 0 or obs_arr.size < needed:
        return []

    w_obs_scale = _safe_float(metadata.get("w_obs_scale", metadata.get("w_obs_scale_m", 1.0)), 1.0)
    wdot_obs_scale = _safe_float(metadata.get("wdot_obs_scale", metadata.get("wdot_obs_scale_m_s", 1.0)), 1.0)
    if abs(w_obs_scale) < 1e-12:
        w_obs_scale = 1.0
    if abs(wdot_obs_scale) < 1e-12:
        wdot_obs_scale = 1.0

    w_sensor = obs_arr[:n_sensors] * w_obs_scale
    wdot_sensor = obs_arr[n_sensors:needed] * wdot_obs_scale
    return np.concatenate([w_sensor, wdot_sensor]).tolist()


def _load_ppo_model(model_path: Path, *, retries: int = 3) -> PPO:
    """Load PPO checkpoint; retry briefly if training is writing the file."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return PPO.load(str(model_path), device="cpu")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                import time

                time.sleep(0.5)
    assert last_error is not None
    raise last_error


def _resolve_seeds(model_dir: Path, seeds: list[int] | None) -> list[int]:
    """Use explicit seeds or auto-discover checkpoints under model_dir."""
    if seeds:
        return list(seeds)
    return discover_model_seeds(model_dir)


def _training_config_path(model_dir: Path, seed: int) -> Path:
    """Locate the unified trainer's per-seed config beside its models directory."""
    return model_dir.parent / "logs" / f"seed_{seed}" / "training_config.json"


def _load_training_config(model_dir: Path, seed: int) -> tuple[dict[str, Any] | None, Path | None]:
    path = _training_config_path(model_dir, seed)
    if not path.is_file():
        return None, None
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a JSON object: {path}")
    return config, path


def _checkpoint_uses_vecnormalize(model_dir: Path, seed: int, config: dict[str, Any] | None) -> bool:
    """Detect normalized-observation checkpoints that this raw-observation evaluator cannot load safely."""
    ppo_config = (config or {}).get("ppo", {})
    if isinstance(ppo_config, dict) and (
        bool(ppo_config.get("normalize_observations", False))
        or bool(ppo_config.get("normalize_rewards", False))
    ):
        return True
    legacy_stats = (
        model_dir / f"best_{seed}" / "vecnormalize.pkl",
        model_dir / f"vecnormalize_{seed}.pkl",
    )
    return any(path.is_file() for path in legacy_stats)


def _resolve_eval_settings(args: argparse.Namespace, config: dict[str, Any] | None) -> dict[str, Any]:
    """Use saved training settings unless the user explicitly overrides them."""
    env_cfg = (config or {}).get("environment", {})
    if not isinstance(env_cfg, dict):
        env_cfg = {}
    saved_kwargs = env_cfg.get("kwargs", {})
    if not isinstance(saved_kwargs, dict):
        saved_kwargs = {}

    env_id = args.env_id or env_cfg.get("env_id") or ENV_ID
    finishing = env_id == ENV_ID_FINISHING
    reward = args.reward or env_cfg.get("reward_id") or ("dense" if finishing else "productive")

    def pick(cli_name: str, saved_value: Any, fallback: Any) -> Any:
        cli_value = getattr(args, cli_name)
        return cli_value if cli_value is not None else (saved_value if saved_value is not None else fallback)

    settings = {
        "env_id": env_id,
        "reward": reward,
        "dt": float(pick("dt", env_cfg.get("dt_s"), EVAL_DEFAULT_DT)),
        "n_substeps": int(pick("n_substeps", env_cfg.get("n_substeps"), EVAL_DEFAULT_N_SUBSTEPS)),
        "max_episode_steps": int(pick("max_episode_steps", env_cfg.get("max_episode_steps"), EVAL_DEFAULT_MAX_EPISODE_STEPS)),
        "dynamics_uncertainty_std": float(pick("dynamics_uncertainty_std", env_cfg.get("dynamics_uncertainty_std"), EVAL_DEFAULT_DYNAMICS_UNCERTAINTY_STD)),
        "randomize_y0": bool(pick("randomize_y0", env_cfg.get("randomize_y0"), EVAL_DEFAULT_RANDOMIZE_Y0)),
        "ap_max": args.ap_max if args.ap_max is not None else saved_kwargs.get("ap_max", 1.0 if finishing else None),
        "training_env_kwargs": dict(saved_kwargs),
    }
    if finishing:
        settings["ap_max"] = float(settings["ap_max"] if settings["ap_max"] is not None else 1.0)
        if settings["ap_max"] <= 0.0:
            raise ValueError("Finishing ap_max must be positive.")
    elif args.ap_max is not None:
        raise ValueError("--ap-max is only valid with --env-id CustomODEPlateFinish-v0.")

    numeric_values = (settings["dt"], settings["dynamics_uncertainty_std"], settings["ap_max"])
    if any(value is not None and not np.isfinite(float(value)) for value in numeric_values):
        raise ValueError("Resolved environment settings must be finite numbers.")
    if settings["dt"] <= 0.0 or settings["n_substeps"] <= 0 or settings["max_episode_steps"] <= 0:
        raise ValueError("Resolved dt, n_substeps and max_episode_steps must be positive.")
    if settings["dynamics_uncertainty_std"] < 0.0:
        raise ValueError("Resolved dynamics_uncertainty_std cannot be negative.")
    return settings


def _assert_model_env_compatible(model: PPO, env: gym.Env, model_path: Path) -> None:
    """Fail early if a checkpoint and evaluation environment have different spaces."""
    for name in ("observation_space", "action_space"):
        model_space = getattr(model, name, None)
        env_space = getattr(env, name, None)
        if model_space is None or env_space is None:
            raise ValueError(f"Cannot validate {name} for checkpoint {model_path}.")
        if type(model_space) is not type(env_space) or tuple(model_space.shape) != tuple(env_space.shape):
            raise ValueError(
                f"Checkpoint/environment {name} mismatch for {model_path}: "
                f"model={model_space}, evaluation={env_space}. Check the control mode."
            )
        if hasattr(model_space, "low") and hasattr(env_space, "low"):
            if not (np.allclose(model_space.low, env_space.low) and np.allclose(model_space.high, env_space.high)):
                raise ValueError(
                    f"Checkpoint/environment {name} bounds differ for {model_path}. "
                    "The policy cannot be evaluated under different action/observation bounds."
                )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_ap(env: gym.Env, args: argparse.Namespace, episode: int) -> float | None:
    """Choose one finishing depth per test episode and reuse it across MC replicas."""
    if args.env_id != ENV_ID_FINISHING:
        return None
    plant = env.unwrapped.plant
    ap_min = float(getattr(plant, "ap_min"))
    ap_max = float(getattr(plant, "ap_max"))
    if args.ap is not None:
        ap = float(args.ap)
    else:
        rng = np.random.default_rng(int(args.evaluation_seed + 700_000 + episode))
        ap = float(rng.uniform(ap_min, ap_max))
    if not (ap_min <= ap <= ap_max):
        raise ValueError(f"Finishing ap={ap:g} mm is outside plant bounds [{ap_min:g}, {ap_max:g}] mm.")
    return ap


def _aggregate_scalar_mc_runs(
    values_runs: list[np.ndarray],
    times_runs: list[np.ndarray],
    default_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align scalar MC traces on their common time interval and return mean/sample SD."""
    valid = []
    for values, times in zip(values_runs, times_runs):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        n = min(values.size, times.size)
        if n < 2:
            continue
        values, times = values[:n], times[:n]
        finite = np.isfinite(values) & np.isfinite(times)
        values, times = values[finite], times[finite]
        if values.size >= 2 and np.all(np.diff(times) > 0.0):
            valid.append((values, times))
    if not valid:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dt = float(default_dt) if default_dt > 0 else float(np.median([np.median(np.diff(t)) for _, t in valid]))
    start = max(float(t[0]) for _, t in valid)
    end = min(float(t[-1]) for _, t in valid)
    if end <= start:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    grid = np.arange(start, end + 0.5 * dt, dt, dtype=np.float64)
    aligned = np.vstack([np.interp(grid, t, v) for v, t in valid])
    mean = np.mean(aligned, axis=0)
    std = np.std(aligned, axis=0, ddof=1) if aligned.shape[0] > 1 else np.zeros_like(mean)
    return grid, mean, std


def make_registered_plate_env(_env_id: str = ENV_ID, **env_kwargs: Any) -> gym.Env:
    """Create the registered plate environment using the same workflow as training.

    ``_env_id`` selects the control mode:
        "CustomODEPlate-v0"        first-mode / roughing (action = [omega, ap])
        "CustomODEPlateFinish-v0"  second-mode / finishing (action = [omega])
    """
    register_envs()
    return gym.make(_env_id, **env_kwargs)


def _episode_y_line(
    env: gym.Env,
    args: argparse.Namespace,
    *,
    randomize_y0: bool,
    episode: int,
) -> float | None:
    """Choose one reproducible milling line per test episode, shared by MC replicas."""
    plant = env.unwrapped.plant
    y_min = float(np.clip(_safe_float(getattr(plant, "y0_min", 0.0), 0.0), 0.0, float(plant.L2)))
    y_max = float(np.clip(_safe_float(getattr(plant, "y0_max", plant.L2), float(plant.L2)), 0.0, float(plant.L2)))
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    if args.y0 is not None:
        y_value = float(args.y0)
    elif args.y_position:
        y_value = float(args.y_position[episode % len(args.y_position)])
    elif args.n_mc > 1 and randomize_y0:
        rng = np.random.default_rng(int(args.evaluation_seed + 500_000 + episode))
        y_value = float(rng.uniform(y_min, y_max))
    else:
        return None

    if not (y_min <= y_value <= y_max):
        raise ValueError(f"Requested milling line y0={y_value:g} m is outside [{y_min:g}, {y_max:g}] m.")
    return y_value


def _process_snapshot(info: dict[str, Any], env: gym.Env) -> dict[str, float]:
    """Extract one time-step of process diagnostics as finite floats or NaN."""
    plant = env.unwrapped.plant
    process = face_milling_process_from_info(info, plant)

    for key in ("cutter_x", "cutter_y", "feed_progress", "feed_distance_m", "spindle_phase_rad"):
        if key in info:
            process[key] = _safe_float(info[key])

    if "w_sensor" in info:
        w_sensor = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        process["max_abs_w_m"] = float(np.max(np.abs(w_sensor))) if w_sensor.size else float("nan")

    if "F_total_N" in info:
        f_total = np.asarray(info["F_total_N"], dtype=np.float64).reshape(-1)
        if f_total.size >= 3:
            process["Fx_N"] = _safe_float(f_total[0])
            process["Fy_N"] = _safe_float(f_total[1])
            process["Fz_N"] = _safe_float(f_total[2])
            process["F_mag_N"] = _safe_float(float(np.linalg.norm(f_total)))

    return {key: _safe_float(process.get(key, np.nan)) for key in PROCESS_KEYS}


def _append_process(process_hist: dict[str, list[float]], process: dict[str, float]) -> None:
    for key in PROCESS_KEYS:
        process_hist.setdefault(key, []).append(_safe_float(process.get(key, np.nan)))


def _rollout_summary(run: dict[str, Any]) -> dict[str, Any]:
    """Return compact scalar diagnostics for one saved rollout."""
    w_arr = np.asarray(run.get("w_sensor", []), dtype=np.float64)
    if w_arr.size:
        max_abs_w = float(np.nanmax(np.abs(w_arr)))
        rms_w = float(np.sqrt(np.nanmean(w_arr**2)))
    else:
        max_abs_w = float("nan")
        rms_w = float("nan")

    process = run.get("process", {}) or {}

    def _last(key: str) -> float:
        values = process.get(key, [])
        if values:
            return _safe_float(values[-1])
        return float("nan")

    reason = run.get("termination_reason")
    return {
        "return": _safe_float(run.get("return", np.nan)),
        "length": int(run.get("length", 0)),
        "duration_s": _safe_float((run.get("times") or [float("nan")])[-1]),
        "max_abs_w_m": max_abs_w,
        "rms_w_m": rms_w,
        "pass_completed": bool(run.get("pass_completed", reason in PASS_COMPLETED_REASONS)),
        "termination_reason": reason,
        "final_cutter_x_m": _last("cutter_x"),
        "final_cutter_y_m": _last("cutter_y"),
        "final_feed_progress": _last("feed_progress"),
        "final_omega_rpm": _last("omega_rpm"),
        "final_ap_mm": _last("ap_mm"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate policy and save trajectories")

    parser.add_argument(
        "--model-dir",
        "--save-dir",
        dest="model_dir",
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory with trained checkpoints "
            f"(default: {DEFAULT_MODEL_DIR}, same as train_sb3_ppo.py --save-dir)"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds to evaluate (default: auto-discover from model-dir)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=2,
        help="Outer test scenarios (default 2; recommended final evaluation: 5).",
    )
    parser.add_argument("--out-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument(
        "--record-every", type=int, default=1,
        help=("Save each Nth environment-step sample and always the final sample; "
              "reward/length/RMS/peak summaries still use EVERY physical step. "
              "Default 1 is full-rate. N>1 may alias vibration and rapid actions; "
              "use only for large robustness studies, not spectral figures."),
    )
    parser.add_argument(
        "--env-id",
        default=None,
        choices=["CustomODEPlate-v0", "CustomODEPlateFinish-v0"],
        help=(
            "Control mode; must match training. CustomODEPlate-v0: first-mode "
            "roughing (action [omega, ap]). CustomODEPlateFinish-v0: second-mode "
            "finishing (action [omega], ap fixed/randomized per episode)."
        ),
    )

    parser.add_argument("--dt", type=float, default=None, help="Override saved training RK4 substep [s]")
    parser.add_argument("--n-substeps", type=int, default=None, help="Override saved training RK4 substeps")
    parser.add_argument("--policy-hold-env-steps", type=int, default=1,
                        help="Hold one RL policy action for this many environment steps; physical safety is checked every environment step.")
    parser.add_argument("--evaluation-seed", "--reset-seed-offset", dest="evaluation_seed", type=int, default=1000,
                        help="Base held-out test seed, shared across all training seeds (default: 1000).")
    parser.add_argument("--plant-override", default=None,
                        help="JSON object limited to initial_eta_std, initial_etad_std, process_noise_std, obs_noise_std.")

    parser.add_argument(
        "--reward",
        default=None,
        choices=["dense", "productive", "quadratic", "sparse"],
        help="Defaults to the saved training reward (or mode default for legacy checkpoints).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override saved training episode-step cap; otherwise use the saved config or legacy default.",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=1,
        help=(
            "Monte Carlo replicas per outer scenario (default 1; recommended final "
            "evaluation: 5). Replicas share the same y0 and finishing ap within an episode."
        ),
    )
    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=None,
        help="Override saved Gaussian modal-acceleration disturbance [0=deterministic].",
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override saved y0 randomization setting; default uses the training config.",
    )
    parser.add_argument(
        "--y0",
        type=float,
        default=None,
        help="Evaluate all episodes on one fixed milling line y=a [m]. Default None matches randomized training.",
    )
    parser.add_argument(
        "--y-position",
        type=float,
        nargs="*",
        default=None,
        help="Cycle through explicit milling lines y=a [m] across episodes",
    )
    parser.add_argument(
        "--ap",
        type=float,
        default=None,
        help=(
            "Second-mode / finishing only: pin the fixed axial depth of cut ap "
            "[mm] for every evaluation episode. Valid only in finishing mode "
            "(roughing controls ap as an action). Default None keeps the per-episode "
            "randomized ap of CustomODEPlateFinish-v0."
        ),
    )
    parser.add_argument(
        "--ap-max",
        type=float,
        default=None,
        help="Finishing only: override the training ap_max [mm]; otherwise load it from training_config.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing trajectories/manifest already present in --out-dir.",
    )
    parser.add_argument(
        "--prefer-final",
        action="store_true",
        help="Load final_<seed>.zip instead of best eval checkpoint (default: best_model.zip)",
    )

    args = parser.parse_args()

    if args.n_episodes <= 0:
        raise SystemExit("--n-episodes must be positive.")
    if args.n_mc <= 0:
        raise SystemExit("--n-mc must be positive.")
    if args.record_every <= 0:
        raise SystemExit("--record-every must be positive.")
    if args.policy_hold_env_steps <= 0:
        raise SystemExit("--policy-hold-env-steps must be positive.")
    if args.y0 is not None and args.y_position:
        raise SystemExit("Use either --y0 or --y-position, not both.")
    if args.evaluation_seed < 0:
        raise SystemExit("--evaluation-seed must be non-negative.")
    if args.ap is not None and (not np.isfinite(args.ap) or args.ap <= 0.0):
        raise SystemExit("--ap must be finite and positive.")
    if args.y0 is not None and not np.isfinite(args.y0):
        raise SystemExit("--y0 must be finite.")
    if args.y_position and any(not np.isfinite(v) for v in args.y_position):
        raise SystemExit("--y-position values must be finite.")

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not model_dir.is_dir():
        print(f"Error: model directory does not exist: {model_dir}")
        return 1

    seeds = _resolve_seeds(model_dir, args.seeds)
    print(f"Model directory : {model_dir}")
    print(f"Trajectory output: {out_dir}")

    if not seeds:
        print(
            "Error: no checkpoints found. Expected e.g.\n"
            f"  {model_dir / 'best_0' / 'best_model.zip'}\n"
            f"  {model_dir / 'final_0.zip'}\n"
            "Train first: python scripts/train_sb3_ppo.py --seeds 0"
        )
        return 1
    if any(seed < 0 for seed in seeds):
        raise SystemExit("Training seeds must be non-negative integers.")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Duplicate training seeds were supplied; each seed must be evaluated once.")

    configs: dict[int, dict[str, Any] | None] = {}
    config_paths: dict[int, Path | None] = {}
    for seed in seeds:
        configs[seed], config_paths[seed] = _load_training_config(model_dir, seed)
        if _checkpoint_uses_vecnormalize(model_dir, seed, configs[seed]):
            raise SystemExit(
                f"Checkpoint seed {seed} appears to use VecNormalize statistics. "
                "This evaluator uses raw observations and does not restore the matching "
                "normalization state, so evaluating it would be invalid. Use a checkpoint "
                "trained without VecNormalize or provide a normalization-aware evaluator."
            )

    configured_env_ids: set[str] = set()
    for config in configs.values():
        if config is None:
            continue
        environment_config = config.get("environment", {})
        if isinstance(environment_config, dict) and environment_config.get("env_id"):
            configured_env_ids.add(str(environment_config["env_id"]))
    if args.env_id is None:
        if len(configured_env_ids) > 1:
            raise SystemExit(f"Selected seeds have different training modes: {sorted(configured_env_ids)}")
        args.env_id = next(iter(configured_env_ids), ENV_ID)
    elif configured_env_ids and configured_env_ids != {args.env_id}:
        raise SystemExit(
            f"Requested --env-id {args.env_id} does not match the checkpoint training mode(s) "
            f"{sorted(configured_env_ids)}. Use the matching environment; do not evaluate across modes."
        )

    if args.ap is not None and args.env_id != ENV_ID_FINISHING:
        raise SystemExit("--ap is only valid with --env-id CustomODEPlateFinish-v0.")
    if args.ap_max is not None and args.env_id != ENV_ID_FINISHING:
        raise SystemExit("--ap-max is only valid with --env-id CustomODEPlateFinish-v0.")

    existing_outputs = list(out_dir.glob("trajectories_seed*.json")) + ([out_dir / "evaluation_manifest.json"] if (out_dir / "evaluation_manifest.json").exists() else [])
    if existing_outputs and not args.overwrite:
        raise SystemExit(
            f"Evaluation outputs already exist in {out_dir}. Choose a new --out-dir or pass --overwrite."
        )
    # A previous evaluation is never removed before new rollouts finish.
    # Stage artifacts in a temporary directory and replace only on success.
    settings_by_seed = {seed: _resolve_eval_settings(args, configs[seed]) for seed in seeds}
    conditions = {
        seed: json.dumps(settings_by_seed[seed], sort_keys=True, default=str)
        for seed in seeds
    }
    if len(set(conditions.values())) > 1:
        raise SystemExit(
            "Selected training seeds have different saved evaluation/plant settings. "
            "Do not pool them in one between-seed scientific figure. Evaluate "
            "distinct experimental conditions in separate output directories."
        )

    register_envs()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".ppo_eval_staging_", dir=out_dir))
    print(f"Seeds to evaluate: {seeds}")
    print(f"Trajectory sampling: every {args.record_every} environment step(s); full-step metrics retained")

    evaluated = 0
    manifest_models: list[dict[str, Any]] = []

    for seed in seeds:
        model_path = resolve_ppo_model_path(
            model_dir,
            seed,
            prefer_final=args.prefer_final,
        )

        if model_path is None:
            raise FileNotFoundError(f"Requested seed {seed} has no checkpoint under {model_dir}.")

        print(f"Loading seed {seed}: {model_path}")

        try:
            model = _load_ppo_model(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load seed {seed}: {model_path}") from exc

        training_config = configs.get(seed)
        if training_config is None or not isinstance(training_config.get("environment"), dict):
            print(
                f"WARNING: seed {seed} has no compatible unified environment config. "
                "Using legacy evaluator defaults; verify dt, substeps, reward, episode cap, "
                "disturbance and finishing ap_max against the original trainer."
            )
        settings = settings_by_seed[seed]
        if settings["env_id"] != args.env_id:
            raise ValueError(f"Seed {seed} config mode {settings['env_id']} differs from selected mode {args.env_id}.")

        env_kwargs = dict(settings["training_env_kwargs"])
        env_kwargs.update(plate_env_kwargs(
            reward_id=settings["reward"],
            dt=settings["dt"],
            n_substeps=settings["n_substeps"],
            max_episode_steps=settings["max_episode_steps"],
            dynamics_uncertainty_std=settings["dynamics_uncertainty_std"],
            randomize_y0=settings["randomize_y0"],
        ))
        if args.env_id == ENV_ID_FINISHING:
            env_kwargs["ap_max"] = float(settings["ap_max"])
        if args.plant_override:
            overrides = json.loads(args.plant_override)
            allowed = {"initial_eta_std", "initial_etad_std", "process_noise_std", "obs_noise_std"}
            if not isinstance(overrides, dict) or not set(overrides).issubset(allowed):
                raise ValueError(f"Fair evaluation --plant-override accepts only {sorted(allowed)}")
            for key, value in overrides.items():
                if not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"Plant override {key} must be a finite non-negative number.")
            env_kwargs.update(overrides)

        env = make_registered_plate_env(_env_id=args.env_id, **env_kwargs)
        _assert_model_env_compatible(model, env, model_path)

        actual_max_steps = getattr(env.unwrapped, "max_episode_steps", None)
        if actual_max_steps is not None and int(actual_max_steps) != int(settings["max_episode_steps"]):
            env.close()
            raise ValueError(
                f"Evaluation environment max_episode_steps={actual_max_steps} differs from "
                f"resolved training/evaluation setting {settings['max_episode_steps']}."
            )
        max_steps = int(settings["max_episode_steps"] or actual_max_steps)
        metadata = _get_metadata(env, max_steps, settings["reward"])
        metadata["env_id"] = args.env_id
        metadata["training_seed"] = int(seed)
        metadata["checkpoint_path"] = str(model_path)
        metadata["checkpoint_sha256"] = _file_sha256(model_path)
        metadata["training_config_path"] = str(config_paths.get(seed)) if config_paths.get(seed) else None
        metadata["training_config_available"] = training_config is not None
        metadata["training_env_kwargs"] = settings["training_env_kwargs"]
        metadata["evaluation_env_kwargs"] = env_kwargs
        metadata["evaluation_seed_base"] = int(args.evaluation_seed)
        metadata["evaluation_settings"] = {
            "reward": settings["reward"],
            "dt": settings["dt"],
            "n_substeps": settings["n_substeps"],
            "max_episode_steps": settings["max_episode_steps"],
            "dynamics_uncertainty_std": settings["dynamics_uncertainty_std"],
            "randomize_y0": settings["randomize_y0"],
            "ap_max": settings["ap_max"],
            "ap_fixed_mm": args.ap,
            "plant_override": json.loads(args.plant_override) if args.plant_override else {},
        }
        training_env = (training_config or {}).get("environment", {})
        if not isinstance(training_env, dict):
            training_env = {}
        training_kwargs = training_env.get("kwargs", {})
        if not isinstance(training_kwargs, dict):
            training_kwargs = {}
        comparison_pairs = [
            ("reward", training_env.get("reward_id"), settings["reward"]),
            ("dt", training_env.get("dt_s"), settings["dt"]),
            ("n_substeps", training_env.get("n_substeps"), settings["n_substeps"]),
            ("max_episode_steps", training_env.get("max_episode_steps"), settings["max_episode_steps"]),
            ("dynamics_uncertainty_std", training_env.get("dynamics_uncertainty_std"), settings["dynamics_uncertainty_std"]),
            ("randomize_y0", training_env.get("randomize_y0"), settings["randomize_y0"]),
            ("ap_max", training_kwargs.get("ap_max"), env_kwargs.get("ap_max")),
            ("initial_eta_std", training_kwargs.get("initial_eta_std"), env_kwargs.get("initial_eta_std")),
            ("initial_etad_std", training_kwargs.get("initial_etad_std"), env_kwargs.get("initial_etad_std")),
            ("process_noise_std", training_kwargs.get("process_noise_std"), env_kwargs.get("process_noise_std")),
            ("obs_noise_std", training_kwargs.get("obs_noise_std"), env_kwargs.get("obs_noise_std")),
        ]
        config_mismatches = []
        for key, trained_value, eval_value in comparison_pairs:
            if trained_value is None or eval_value is None:
                continue
            equal = (trained_value == eval_value) if isinstance(trained_value, bool) else (
                np.isclose(float(trained_value), float(eval_value))
                if isinstance(trained_value, (int, float)) and isinstance(eval_value, (int, float))
                else trained_value == eval_value
            )
            if not equal:
                config_mismatches.append({"parameter": key, "training": trained_value, "evaluation": eval_value})
        metadata["training_evaluation_config_mismatches"] = config_mismatches
        metadata["training_evaluation_config_match"] = not config_mismatches
        if config_mismatches:
            print(f"WARNING: seed {seed} evaluation overrides differ from training config: {config_mismatches}")

        plant = env.unwrapped.plant
        if args.env_id == ENV_ID_FINISHING and not np.isclose(float(plant.ap_max), float(settings["ap_max"])):
            env.close()
            raise ValueError(
                f"Finishing environment ap_max={plant.ap_max} does not match requested/trained "
                f"ap_max={settings['ap_max']}. Refusing an inconsistent evaluation."
            )
        # Record the complete stochastic/initialization protocol for paired audit.
        metadata["initial_eta_std"] = float(plant.initial_eta_std)
        metadata["initial_etad_std"] = float(plant.initial_etad_std)
        metadata["process_noise_std"] = float(env.unwrapped.process_noise_std)
        metadata["obs_noise_std"] = float(env.unwrapped.obs_noise_std)
        tau_min = 2.0 * np.pi / (max(int(getattr(plant, "N", 1)), 1) * max(float(plant.omega_max), 1e-12))
        if float(settings["dt"]) >= tau_min:
            env.close()
            raise ValueError(
                f"RK4 dt={settings['dt']:g} s is not smaller than the minimum regenerative "
                f"tooth delay tau_min={tau_min:g} s at omega_max={omega_to_rpm(plant.omega_max):.0f} rpm. "
                "Use --dt 0.0001 --n-substeps 10 for the current 4000-rpm setup."
            )

        control_dt = float(settings["dt"]) * int(settings["n_substeps"])
        metadata["eval_dt"] = float(settings["dt"])
        metadata["eval_n_substeps"] = int(settings["n_substeps"])
        metadata["eval_control_step_s"] = float(control_dt * args.policy_hold_env_steps)
        metadata["eval_environment_step_s"] = float(control_dt)
        metadata["eval_policy_hold_env_steps"] = int(args.policy_hold_env_steps)
        metadata["signal_timestamp_semantics"] = "post-step environment time"
        metadata["action_timestamp_semantics"] = "time at action application start"
        metadata["eval_seed_base"] = int(args.evaluation_seed)
        metadata["eval_dynamics_uncertainty_std"] = float(settings["dynamics_uncertainty_std"])
        metadata["eval_randomize_y0"] = bool(settings["randomize_y0"])
        metadata["eval_tau_min_s"] = float(tau_min)

        print(
            f"Eval model seed {seed}: rpm={omega_to_rpm(plant.omega_min):.0f}-"
            f"{omega_to_rpm(plant.omega_max):.0f}, dt={settings['dt']:g}, "
            f"n_substeps={settings['n_substeps']}, control_step={control_dt * args.policy_hold_env_steps:g}, "
            f"max_steps={max_steps}, randomize_y0={settings['randomize_y0']}, "
            f"w_limit={plant.w_limit:g} m"
        )
        metadata["eval_n_episodes"] = int(args.n_episodes)
        metadata["eval_n_mc"] = int(args.n_mc)
        metadata["trajectory_record_every"] = int(args.record_every)
        metadata["full_step_metrics"] = True
        metadata["recording_warning"] = (
            "Subsampled trajectories are not suitable for spectral inference or transient peak estimation. "
            "RMS and maxima in summary are computed from all full-resolution simulation steps."
            if args.record_every > 1 else None
        )
        metadata["eval_y0_fixed_m"] = None if args.y0 is None else float(args.y0)
        metadata["eval_y_positions_m"] = None if not args.y_position else [float(v) for v in args.y_position]
        metadata["eval_ap_fixed_mm"] = None if args.ap is None else float(args.ap)
        metadata["eval_ap_max_mm"] = settings["ap_max"]

        trajectories: list[dict[str, Any]] = []

        for ep in range(args.n_episodes):
            mc_runs: list[dict[str, Any]] = []
            episode_y_line = _episode_y_line(
                env, args, randomize_y0=settings["randomize_y0"], episode=ep
            )
            episode_ap = _episode_ap(env, args, ep)
            reset_options: dict[str, Any] | None = (
                {"y0": episode_y_line} if episode_y_line is not None else None
            )
            # Keep the finishing depth fixed across MC replicas within this test
            # episode. Different outer episodes sample different ap values when
            # --ap is omitted, preserving coverage of the configured ap range.
            if episode_ap is not None:
                reset_options = dict(reset_options or {})
                reset_options["ap"] = float(episode_ap)

            for mc in range(args.n_mc):
                run_seed = args.evaluation_seed + ep * args.n_mc + mc
                obs, reset_info = env.reset(seed=run_seed, options=reset_options)
                if not np.all(np.isfinite(np.asarray(obs, dtype=np.float64))):
                    raise RuntimeError(f"Environment reset produced non-finite observation (episode {ep}, MC {mc}).")
                if episode_y_line is not None and not np.isclose(
                    float(reset_info.get("y_cutter", float("nan"))), episode_y_line,
                    rtol=0.0, atol=1e-9,
                ):
                    raise RuntimeError("Plant did not apply the requested fixed y0 test scenario.")
                if episode_ap is not None and not np.isclose(
                    float(reset_info.get("ap_fixed_mm", episode_ap)), episode_ap,
                    rtol=0.0, atol=1e-9,
                ):
                    raise RuntimeError("Plant did not apply the requested fixed finishing ap.")

                observations: list[list[float]] = []
                physical_signals: list[list[float]] = []
                x_modal: list[list[float]] = []
                actions: list[list[float]] = []
                physical_actions: list[list[float]] = []
                rewards: list[float] = []
                times: list[float] = []
                action_times: list[float] = []
                sim_time = _safe_float(reset_info.get("t", 0.0), 0.0)
                w_sensor_hist: list[list[float]] = []
                wdot_sensor_hist: list[list[float]] = []
                reward_terms_hist: list[dict[str, Any]] = []
                process_hist: dict[str, list[float]] = {key: [] for key in PROCESS_KEYS}

                termination_reason = None
                terminated_final = False
                truncated_final = False
                pass_completed_final = False

                # Policy decisions retain the training-time cadence; the plant is
                # stepped and safety-tested at every (possibly finer) env interval.
                fair_eval_step = 0
                full_reward_sum = 0.0
                full_w_sum_sq = 0.0
                full_w_count = 0
                full_w_max = 0.0
                full_force_sum_sq = 0.0
                full_force_max = 0.0
                full_force_count = 0
                policy_latency_ms: list[float] = []
                omega_total_variation_rpm = 0.0
                ap_total_variation_mm = 0.0
                previous_physical_command = None
                while fair_eval_step < max_steps:
                    if fair_eval_step % args.policy_hold_env_steps == 0:
                        t_predict = time.perf_counter()
                        action, _ = model.predict(obs, deterministic=True)
                        policy_latency_ms.append((time.perf_counter()-t_predict)*1000.0)
                    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
                    fair_eval_step += 1
                    if not np.all(np.isfinite(action_arr)):
                        raise RuntimeError(f"Policy produced non-finite action at episode {ep}, MC {mc}.")

                    physical_command = np.asarray(_physical_action(env, action_arr), dtype=np.float64)
                    if not np.all(np.isfinite(physical_command)):
                        raise RuntimeError("Non-finite physical actuator command.")
                    if previous_physical_command is not None:
                        omega_total_variation_rpm += float(abs(omega_to_rpm(
                            physical_command[0] - previous_physical_command[0])))
                        if args.env_id != ENV_ID_FINISHING and len(physical_command) > 1:
                            ap_total_variation_mm += float(abs(physical_command[1] - previous_physical_command[1]))
                    previous_physical_command = physical_command.copy()
                    actions.append(action_arr.tolist())
                    physical_actions.append(physical_command.tolist())
                    action_times.append(float(sim_time))

                    obs, reward, terminated, truncated, info = env.step(action_arr)
                    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
                    if not np.all(np.isfinite(obs_arr)):
                        raise RuntimeError(f"Environment produced non-finite observation at episode {ep}, MC {mc}.")

                    # The physics summaries use EVERY simulated environment step.
                    physical = _physical_signal_from_info_or_obs(info, obs_arr, metadata)
                    n_sensors = int(metadata.get("n_sensors", len(physical) // 2))
                    w_full = np.asarray(physical[:n_sensors], dtype=np.float64)
                    if w_full.size != n_sensors or not np.all(np.isfinite(w_full)):
                        raise RuntimeError(f"Invalid physical displacement sensor data at episode {ep}, MC {mc}.")
                    if not np.isfinite(float(reward)):
                        raise RuntimeError(f"Non-finite reward at episode {ep}, MC {mc}.")
                    if w_full.size:
                        full_w_sum_sq += float(np.dot(w_full, w_full))
                        full_w_count += int(w_full.size)
                        full_w_max = max(full_w_max, float(np.max(np.abs(w_full))))
                    full_reward_sum += float(reward)
                    if "F_total_N" in info:
                        force = np.asarray(info["F_total_N"], dtype=np.float64).reshape(-1)
                        if not np.all(np.isfinite(force)):
                            raise RuntimeError("Non-finite cutting force from plant.")
                        f_mag = float(np.linalg.norm(force))
                        full_force_count += 1
                        full_force_sum_sq += f_mag * f_mag
                        full_force_max = max(full_force_max, f_mag)
                    sim_time = float(info.get("t", sim_time + control_dt))
                    if not np.isfinite(sim_time) or sim_time <= action_times[-1]:
                        raise RuntimeError("Physical simulator emitted non-monotonic or non-finite time.")
                    if not np.isclose(sim_time - action_times[-1], control_dt,
                                      rtol=1e-4, atol=1e-8):
                        raise RuntimeError("Environment-step duration differs from the configured RK4 dt*n_substeps.")

                    # Preserve the last sample as well as regular sampled steps;
                    # post-step samples correspond to post-step physical times.
                    save_step = (
                        args.record_every == 1
                        or fair_eval_step == 1
                        or fair_eval_step % args.record_every == 0
                        or bool(terminated or truncated)
                    )
                    if save_step:
                        observations.append(obs_arr.tolist())
                        physical_signals.append(physical)
                        if "w_sensor" in info:
                            w_sensor_hist.append(np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1).tolist())
                        if "wdot_sensor" in info:
                            wdot_sensor_hist.append(np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1).tolist())
                        if "reward_terms" in info:
                            reward_terms_hist.append(dict(info["reward_terms"]))
                        if "x_modal" in info:
                            x_modal.append(np.asarray(info["x_modal"], dtype=np.float64).reshape(-1).tolist())
                        _append_process(process_hist, _process_snapshot(info, env))
                        rewards.append(float(reward))
                        times.append(sim_time)
                    else:
                        # Do not store unsampled actions. The stored action
                        # timestamp must always correspond to a stored action.
                        actions.pop()
                        physical_actions.pop()
                        action_times.pop()

                    if terminated or truncated:
                        terminated_final = bool(terminated)
                        truncated_final = bool(truncated)
                        termination_reason = info.get("termination_reason")
                        pass_completed_final = bool(info.get("pass_completed") or termination_reason in PASS_COMPLETED_REASONS)
                        break

                if not (terminated_final or truncated_final):
                    raise RuntimeError(
                        f"Episode {ep}, MC {mc} reached {max_steps} steps without "
                        "environment termination/truncation; refusing an incomplete result."
                    )

                run = {
                    "mc_index": int(mc),
                    "seed": int(run_seed),
                    "reset_options": reset_options,
                    "reset_info": _to_jsonable(dict(reset_info)),
                    "y_line_m": _safe_float(reset_info.get("y_cutter", episode_y_line)),
                    "ap_mm": _safe_float(reset_options.get("ap") if reset_options else None),
                    "x_start_m": _safe_float(reset_info.get("x_start", np.nan)),
                    "x_end_target_m": _safe_float(reset_info.get("x_end", np.nan)),
                    "path_direction": reset_info.get("path_direction"),
                    "times": times,
                    "action_times": action_times,
                    "observations": observations,
                    "physical_signals": physical_signals,
                    "w_sensor": w_sensor_hist,
                    "wdot_sensor": wdot_sensor_hist,
                    "reward_terms": reward_terms_hist,
                    "x_modal": x_modal,
                    "actions": actions,
                    "physical_actions": physical_actions,
                    "process": process_hist,
                    "rewards": rewards,
                    "return": float(full_reward_sum),
                    "length": int(fair_eval_step),
                    "recorded_length": int(len(rewards)),
                    "rewards_are_subsampled": bool(args.record_every > 1),
                    "terminated": bool(terminated_final),
                    "truncated": bool(truncated_final),
                    "pass_completed": bool(pass_completed_final),
                    "termination_reason": termination_reason,
                }
                run["summary"] = _rollout_summary(run)
                run["summary"]["duration_s"] = float(sim_time - _safe_float(reset_info.get("t", 0.0), 0.0))
                run["summary"]["rms_w_m"] = float(np.sqrt(full_w_sum_sq / full_w_count)) if full_w_count else float("nan")
                run["summary"]["max_abs_w_m"] = float(full_w_max) if full_w_count else float("nan")
                run["summary"]["rms_force_N"] = float(np.sqrt(full_force_sum_sq / full_force_count)) if full_force_count else float("nan")
                run["summary"]["peak_force_N"] = float(full_force_max) if full_force_count else float("nan")
                run["summary"]["force_samples_count"] = int(full_force_count)
                run["summary"]["spindle_speed_total_variation_rpm"] = float(omega_total_variation_rpm)
                run["summary"]["depth_total_variation_mm"] = float(ap_total_variation_mm) if args.env_id != ENV_ID_FINISHING else None
                run["summary"]["policy_decisions_count"] = len(policy_latency_ms)
                run["summary"]["inference_mean_wall_ms"] = float(np.mean(policy_latency_ms)) if policy_latency_ms else None
                run["summary"]["inference_p95_wall_ms"] = float(np.percentile(policy_latency_ms, 95)) if policy_latency_ms else None
                mc_runs.append(run)

            returns = np.asarray([run["return"] for run in mc_runs], dtype=np.float64)
            lengths = np.asarray([run["length"] for run in mc_runs], dtype=np.float64)
            pass_completed_count = int(sum(bool(run.get("pass_completed")) for run in mc_runs))

            mc_summary = {
                "n_mc": int(args.n_mc),
                "return_mean": float(np.mean(returns)) if returns.size else float("nan"),
                "return_std": float(np.std(returns, ddof=1)) if returns.size > 1 else None,
                "length_mean": float(np.mean(lengths)) if lengths.size else float("nan"),
                "pass_completed_count": pass_completed_count,
                "pass_completed_fraction": float(pass_completed_count / max(len(mc_runs), 1)),
            }

            if args.n_mc > 1 and mc_runs:
                w_series: list[np.ndarray] = []
                times_runs: list[np.ndarray] = []
                for run in mc_runs:
                    if run["w_sensor"]:
                        w_arr = np.asarray(run["w_sensor"], dtype=np.float64)
                        w_series.append(np.max(np.abs(w_arr), axis=1))
                        times_runs.append(np.asarray(run["times"], dtype=np.float64))
                if w_series:
                    times_agg, mean_w, std_w = _aggregate_scalar_mc_runs(
                        w_series,
                        times_runs,
                        default_dt=float(metadata["step_dt"]),
                    )
                    mc_summary.update(
                        {
                            "max_abs_w_mean": mean_w.tolist(),
                            "max_abs_w_std": std_w.tolist(),
                            "times": times_agg.tolist(),
                        }
                    )

            primary = mc_runs[0]
            trajectories.append(
                {
                    "metadata": metadata,
                    "episode": int(ep),
                    "seed": int(seed),
                    "y_line_m": primary.get("y_line_m"),
                    "ap_mm": primary.get("ap_mm"),
                    "x_start_m": primary.get("x_start_m"),
                    "x_end_target_m": primary.get("x_end_target_m"),
                    "n_mc": int(args.n_mc),
                    "mc_summary": mc_summary,
                    "runs": mc_runs,
                    "summary": primary.get("summary"),
                    "times": primary["times"],
                    "action_times": primary["action_times"],
                    "observations": primary["observations"],
                    "physical_signals": primary["physical_signals"],
                    "x_modal": primary["x_modal"],
                    "actions": primary["actions"],
                    "physical_actions": primary["physical_actions"],
                    "process": primary["process"],
                    "w_sensor": primary["w_sensor"],
                    "wdot_sensor": primary["wdot_sensor"],
                    "rewards": primary["rewards"],
                    "return": primary["return"],
                    "length": primary["length"],
                    "terminated": primary["terminated"],
                    "truncated": primary["truncated"],
                    "pass_completed": primary["pass_completed"],
                    "termination_reason": primary["termination_reason"],
                }
            )


        out_path = out_dir / f"trajectories_seed{seed}.json"
        staged_path = stage_dir / out_path.name
        with staged_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(trajectories), f, indent=None, allow_nan=False)

        manifest_models.append({
            "training_seed": int(seed),
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "training_config_path": metadata["training_config_path"],
            "trajectory_file": str(out_path),
            "env_id": args.env_id,
            "reward": settings["reward"],
            "evaluation_seed_base": int(args.evaluation_seed),
        })
        print(f"Staged {args.n_episodes} episodes for {out_path}")
        evaluated += 1
        env.close()

    if evaluated == 0:
        print("Error: no models were evaluated successfully.")
        return 1

    manifest = {
        "schema_version": 2,
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "model_dir": str(model_dir),
        "output_dir": str(out_dir),
        "env_id": args.env_id,
        "training_seeds": [int(seed) for seed in seeds],
        "evaluation_seed_base": int(args.evaluation_seed),
        "n_episodes": int(args.n_episodes),
        "n_mc": int(args.n_mc),
        "policy_deterministic": True,
        "checkpoint_selection": "best EvalCallback checkpoint unless --prefer-final is supplied",
        "recording_unit": "post-step simulated environment step",
        "arguments": vars(args),
        "models": manifest_models,
    }
    manifest["trajectory_record_every"] = int(args.record_every)
    manifest["metrics_use_full_simulation_rate"] = True
    with (stage_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(_to_jsonable(manifest), stream, indent=2, allow_nan=False)

    # Construct paper-auditable physical outcome tables from per-replica
    # summaries; replicas of a scenario are NOT independent scenarios.
    replica_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for seed in [item["training_seed"] for item in manifest_models]:
        path = stage_dir / f"trajectories_seed{seed}.json"
        episodes = json.loads(path.read_text(encoding="utf-8"))
        for episode in episodes:
            summaries = []
            for run in episode["runs"]:
                summ = run["summary"]
                summaries.append(summ)
                replica_rows.append({
                    "training_seed": seed,
                    "scenario_index": episode["episode"],
                    "mc_index": run["mc_index"],
                    "reset_seed": run["seed"],
                    "y_line_m": run.get("y_line_m"),
                    "ap_fixed_mm": run.get("ap_mm") if args.env_id == ENV_ID_FINISHING else None,
                    "return": summ["return"],
                    "duration_s": summ["duration_s"],
                    "rms_w_m": summ["rms_w_m"],
                    "max_abs_w_m": summ["max_abs_w_m"],
                    "rms_force_N": summ["rms_force_N"],
                    "peak_force_N": summ["peak_force_N"],
                    "spindle_speed_total_variation_rpm": summ["spindle_speed_total_variation_rpm"],
                    "depth_total_variation_mm": summ["depth_total_variation_mm"],
                    "inference_mean_wall_ms": summ["inference_mean_wall_ms"],
                    "inference_p95_wall_ms": summ["inference_p95_wall_ms"],
                    "feed_progress": summ["final_feed_progress"],
                    "pass_completed": summ["pass_completed"],
                    "terminated": run["terminated"],
                    "truncated": run["truncated"],
                    "termination_reason": run["termination_reason"],
                })
            row = {"training_seed": seed, "scenario_index": episode["episode"], "n_mc": len(summaries)}
            for name in ("return", "duration_s", "rms_w_m", "max_abs_w_m",
                         "rms_force_N", "peak_force_N", "spindle_speed_total_variation_rpm",
                         "depth_total_variation_mm", "inference_mean_wall_ms",
                         "final_feed_progress"):
                values = np.asarray([x.get(name) if x.get(name) is not None else np.nan for x in summaries], dtype=float)
                finite = values[np.isfinite(values)]
                row[name + "_mean"] = float(np.mean(finite)) if finite.size else None
                row[name + "_sample_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else None
            row["pass_completion_fraction"] = float(np.mean([x["pass_completed"] for x in summaries]))
            scenario_rows.append(row)

    for filename, rows in (
        ("physical_outcomes_replicas.csv", replica_rows),
        ("physical_outcomes_scenarios.csv", scenario_rows),
    ):
        if rows:
            with (stage_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(_to_jsonable(rows))

    for path in stage_dir.iterdir():
        if path.is_file():
            os.replace(path, out_dir / path.name)
    if args.overwrite:
        keep_names = {item["trajectory_file"].split(os.sep)[-1] for item in manifest_models}
        for stale in out_dir.glob("trajectories_seed*.json"):
            if stale.name not in keep_names:
                stale.unlink()
    shutil.rmtree(stage_dir)

    print(f"Evaluation complete ({evaluated} seed(s)).")
    print(f"Evaluation manifest: {out_dir / 'evaluation_manifest.json'}")
    print(f"Plot results: python scripts/plot_results.py --traj-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# FAIR_EVAL_2026_10
