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
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from custom_rl import DEFAULT_MODEL_DIR, DEFAULT_TRAJ_DIR, register_envs
from custom_rl.eval.monte_carlo import (
    aggregate_mc_sensor_runs,
    face_milling_process_from_info,
    plot_mc_sensor_bands,
)
from custom_rl.eval.pipeline import (
    discover_model_seeds,
    plate_env_kwargs,
    plant_plot_metadata,
    resolve_ppo_model_path,
)
from custom_rl.plants.plate import omega_to_rpm


ENV_ID = "CustomODEPlate-v0"
PASS_COMPLETED_REASONS = {"pass_completed_90percent", "pass_completed"}

# Standalone evaluation defaults must match the uploaded training script.
# Training uses RK4 dt=1e-4 with 10 RK4 substeps per environment/control step,
# giving a 1 ms control interval while still resolving the 40000-rpm tooth delay.
EVAL_DEFAULT_DT = 1.0e-4
EVAL_DEFAULT_N_SUBSTEPS = 10
EVAL_DEFAULT_MAX_EPISODE_STEPS = 50000
EVAL_DEFAULT_DYNAMICS_UNCERTAINTY_STD = 0.0
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
    """Convert NumPy-heavy nested values to JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
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
        n_sensors = max((obs_arr.size - 2) // 2, obs_arr.size // 2)

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


def make_registered_plate_env(_env_id: str = ENV_ID, **env_kwargs: Any) -> gym.Env:
    """Create the registered plate environment using the same workflow as training.

    ``_env_id`` selects the control mode:
        "CustomODEPlate-v0"        first-mode / roughing (action = [omega, ap])
        "CustomODEPlateFinish-v0"  second-mode / finishing (action = [omega])
    """
    register_envs()
    return gym.make(_env_id, **env_kwargs)


def _episode_y_line(env: gym.Env, args: argparse.Namespace, *, seed: int, episode: int) -> float | None:
    """Return a fixed reset y-line for this episode, or None to let env.reset decide.

    If MC is requested with randomize_y0=True, one y-line is sampled per episode
    and reused for all MC replicas. This prevents a Monte Carlo band from mixing
    different milling lines with stochastic plant uncertainty.
    """
    plant = env.unwrapped.plant

    if args.y0 is not None:
        return float(np.clip(args.y0, 0.0, float(plant.L2)))

    if args.y_position:
        y_value = float(args.y_position[episode % len(args.y_position)])
        return float(np.clip(y_value, 0.0, float(plant.L2)))

    if args.n_mc > 1 and args.randomize_y0:
        y_min = _safe_float(getattr(plant, "y0_min", 0.0), 0.0)
        y_max = _safe_float(getattr(plant, "y0_max", getattr(plant, "L2", 1.0)), float(plant.L2))
        y_min = float(np.clip(y_min, 0.0, float(plant.L2)))
        y_max = float(np.clip(y_max, 0.0, float(plant.L2)))
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        rng = np.random.default_rng(int(seed + 500_000 + episode))
        return float(rng.uniform(y_min, y_max))

    return None


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
        max_abs_w = float(np.max(np.abs(w_arr)))
        rms_w = float(np.sqrt(np.mean(w_arr**2)))
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


def _save_mc_displacement_plot(
    mc_runs: list[dict[str, Any]],
    *,
    out_dir: Path,
    seed: int,
    ep: int,
    metadata: dict[str, Any],
) -> None:
    paired_runs = [run for run in mc_runs if run.get("w_sensor") and run.get("times")]
    if not paired_runs:
        return

    w_runs = [np.asarray(run["w_sensor"], dtype=np.float64) for run in paired_runs]
    times_runs = [np.asarray(run["times"], dtype=np.float64) for run in paired_runs]

    out_dir.mkdir(parents=True, exist_ok=True)
    n_sensors = int(metadata.get("n_sensors", 1))
    w_labels = [f"w_sensor_{i + 1} (m)" for i in range(n_sensors)]
    w_times, w_mean, w_std, _ = aggregate_mc_sensor_runs(
        w_runs,
        times_runs,
        default_dt=float(metadata["step_dt"]),
    )
    plot_mc_sensor_bands(
        w_times,
        w_mean,
        w_std,
        labels=w_labels,
        ylabel="Displacement (m)",
        title=f"Eval MC displacement ep={ep} seed={seed}",
        out_path=out_dir / f"mc_displacement_ep{ep}.png",
        w_limit=metadata.get("w_limit"),
    )


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
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--out-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument(
        "--env-id",
        default=ENV_ID,
        choices=["CustomODEPlate-v0", "CustomODEPlateFinish-v0"],
        help=(
            "Control mode; must match training. CustomODEPlate-v0: first-mode "
            "roughing (action [omega, ap]). CustomODEPlateFinish-v0: second-mode "
            "finishing (action [omega], ap fixed/randomized per episode)."
        ),
    )

    parser.add_argument("--dt", type=float, default=EVAL_DEFAULT_DT, help="Must match training RK4 substep [s]")
    parser.add_argument("--n-substeps", type=int, default=EVAL_DEFAULT_N_SUBSTEPS)

    parser.add_argument(
        "--reward",
        default="productive",
        choices=["dense", "productive", "quadratic", "sparse"],
        help="Must match the reward used during training",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=EVAL_DEFAULT_MAX_EPISODE_STEPS,
        help="Max environment/control steps per episode; default matches train_sb3_ppo.py",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=1,
        help="Monte Carlo rollouts per episode (mean/std when > 1)",
    )
    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=EVAL_DEFAULT_DYNAMICS_UNCERTAINTY_STD,
        help="Gaussian disturbance on modal accelerations [0=deterministic]; default matches training",
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=EVAL_DEFAULT_RANDOMIZE_Y0,
        help="Sample milling line y0 on reset unless --y0/--y-position is supplied",
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
            "[mm] for every evaluation episode. Ignored in first-mode roughing "
            "(where ap is an action). Default None keeps the per-episode "
            "randomized ap of CustomODEPlateFinish-v0."
        ),
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
    if args.dt <= 0.0:
        raise SystemExit("--dt must be positive.")
    if args.n_substeps <= 0:
        raise SystemExit("--n-substeps must be positive.")
    if args.y0 is not None and args.y_position:
        raise SystemExit("Use either --y0 or --y-position, not both.")

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    register_envs()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seeds(model_dir, args.seeds)

    print(f"Model directory : {model_dir}")
    print(f"Trajectory output: {out_dir}")
    if not model_dir.is_dir():
        print(f"Error: model directory does not exist: {model_dir}")
        return 1

    if not seeds:
        print(
            "Error: no checkpoints found. Expected e.g.\n"
            f"  {model_dir / 'best_0' / 'best_model.zip'}\n"
            f"  {model_dir / 'final_0.zip'}\n"
            "Train first: python scripts/train_sb3_ppo.py --seeds 0"
        )
        return 1

    print(f"Seeds to evaluate: {seeds}")

    evaluated = 0

    for seed in seeds:
        model_path = resolve_ppo_model_path(
            model_dir,
            seed,
            prefer_final=args.prefer_final,
        )

        if model_path is None:
            print(f"Skip seed {seed}: no checkpoint under {model_dir}")
            continue

        print(f"Loading seed {seed}: {model_path}")

        try:
            model = _load_ppo_model(model_path)
        except Exception as exc:
            print(f"Skip seed {seed}: failed to load {model_path} ({exc})")
            continue

        env_kwargs = plate_env_kwargs(
            reward_id=args.reward,
            dt=args.dt,
            n_substeps=args.n_substeps,
            max_episode_steps=args.max_episode_steps,
            dynamics_uncertainty_std=args.dynamics_uncertainty_std,
            randomize_y0=args.randomize_y0,
        )

        env = make_registered_plate_env(_env_id=args.env_id, **env_kwargs)

        max_steps = int(args.max_episode_steps or env.unwrapped.max_episode_steps)
        metadata = _get_metadata(env, max_steps, args.reward)

        plant = env.unwrapped.plant
        tau_min = 2.0 * np.pi / (max(int(getattr(plant, "N", 1)), 1) * max(float(plant.omega_max), 1e-12))
        if float(args.dt) >= tau_min:
            env.close()
            raise ValueError(
                f"RK4 dt={args.dt:g} s is not smaller than the minimum regenerative "
                f"tooth delay tau_min={tau_min:g} s at omega_max={omega_to_rpm(plant.omega_max):.0f} rpm. "
                "Use --dt 0.0001 --n-substeps 10 for the current 40000-rpm setup."
            )

        control_dt = float(args.dt) * int(args.n_substeps)
        metadata["eval_dt"] = float(args.dt)
        metadata["eval_n_substeps"] = int(args.n_substeps)
        metadata["eval_control_step_s"] = float(control_dt)
        metadata["eval_dynamics_uncertainty_std"] = float(args.dynamics_uncertainty_std)
        metadata["eval_randomize_y0"] = bool(args.randomize_y0)
        metadata["eval_tau_min_s"] = float(tau_min)

        print(
            f"Eval env seed {seed}: rpm={omega_to_rpm(plant.omega_min):.0f}-"
            f"{omega_to_rpm(plant.omega_max):.0f}, dt={args.dt:g}, "
            f"n_substeps={args.n_substeps}, control_step={control_dt:g}, "
            f"max_steps={max_steps}, randomize_y0={args.randomize_y0}, "
            f"w_limit={plant.w_limit:g} m"
        )
        metadata["eval_n_episodes"] = int(args.n_episodes)
        metadata["eval_n_mc"] = int(args.n_mc)
        metadata["eval_y0_fixed_m"] = None if args.y0 is None else float(args.y0)
        metadata["eval_y_positions_m"] = None if not args.y_position else [float(v) for v in args.y_position]

        trajectories: list[dict[str, Any]] = []

        for ep in range(args.n_episodes):
            mc_runs: list[dict[str, Any]] = []
            episode_y_line = _episode_y_line(env, args, seed=seed, episode=ep)
            reset_options: dict[str, Any] | None = (
                {"y0": episode_y_line} if episode_y_line is not None else None
            )
            # Second-mode / finishing: optionally pin the fixed axial depth ap so
            # a finishing pass can be evaluated at a chosen constant depth. When
            # --ap is not given, the finishing env randomizes ap per episode.
            if args.ap is not None:
                reset_options = dict(reset_options or {})
                reset_options["ap"] = float(args.ap)

            for mc in range(args.n_mc):
                run_seed = seed + 1000 + ep * args.n_mc + mc
                obs, reset_info = env.reset(seed=run_seed, options=reset_options)

                observations: list[list[float]] = []
                physical_signals: list[list[float]] = []
                x_modal: list[list[float]] = []
                actions: list[list[float]] = []
                physical_actions: list[list[float]] = []
                rewards: list[float] = []
                times: list[float] = []
                w_sensor_hist: list[list[float]] = []
                wdot_sensor_hist: list[list[float]] = []
                reward_terms_hist: list[dict[str, Any]] = []
                process_hist: dict[str, list[float]] = {key: [] for key in PROCESS_KEYS}

                termination_reason = None
                terminated_final = False
                truncated_final = False
                pass_completed_final = False

                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
                    if not np.all(np.isfinite(action_arr)):
                        raise RuntimeError(f"Policy produced non-finite action at episode {ep}, MC {mc}.")

                    actions.append(action_arr.tolist())
                    physical_actions.append(_physical_action(env, action_arr))

                    obs, reward, terminated, truncated, info = env.step(action_arr)
                    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
                    if not np.all(np.isfinite(obs_arr)):
                        raise RuntimeError(f"Environment produced non-finite observation at episode {ep}, MC {mc}.")

                    observations.append(obs_arr.tolist())
                    physical_signals.append(_physical_signal_from_info_or_obs(info, obs_arr, metadata))

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
                    times.append(float(info.get("t", len(rewards) * metadata["step_dt"])))

                    if terminated or truncated:
                        terminated_final = bool(terminated)
                        truncated_final = bool(truncated)
                        termination_reason = info.get("termination_reason")
                        pass_completed_final = bool(info.get("pass_completed") or termination_reason in PASS_COMPLETED_REASONS)
                        break

                run = {
                    "mc_index": int(mc),
                    "seed": int(run_seed),
                    "reset_options": reset_options,
                    "reset_info": _to_jsonable(dict(reset_info)),
                    "y_line_m": _safe_float(reset_info.get("y_cutter", episode_y_line)),
                    "x_start_m": _safe_float(reset_info.get("x_start", np.nan)),
                    "x_end_target_m": _safe_float(reset_info.get("x_end", np.nan)),
                    "path_direction": reset_info.get("path_direction"),
                    "times": times,
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
                    "return": float(sum(rewards)),
                    "length": int(len(rewards)),
                    "terminated": bool(terminated_final),
                    "truncated": bool(truncated_final),
                    "pass_completed": bool(pass_completed_final),
                    "termination_reason": termination_reason,
                }
                run["summary"] = _rollout_summary(run)
                mc_runs.append(run)

            returns = np.asarray([run["return"] for run in mc_runs], dtype=np.float64)
            lengths = np.asarray([run["length"] for run in mc_runs], dtype=np.float64)
            pass_completed_count = int(sum(bool(run.get("pass_completed")) for run in mc_runs))

            mc_summary = {
                "n_mc": int(args.n_mc),
                "return_mean": float(np.mean(returns)) if returns.size else float("nan"),
                "return_std": float(np.std(returns)) if returns.size > 1 else 0.0,
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
                    times_agg, mean_w, std_w, _ = aggregate_mc_sensor_runs(
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
                    "x_start_m": primary.get("x_start_m"),
                    "x_end_target_m": primary.get("x_end_target_m"),
                    "n_mc": int(args.n_mc),
                    "mc_summary": mc_summary,
                    "runs": mc_runs,
                    "summary": primary.get("summary"),
                    "times": primary["times"],
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

            if args.n_mc > 1:
                _save_mc_displacement_plot(
                    mc_runs,
                    out_dir=out_dir / f"seed{seed}",
                    seed=seed,
                    ep=ep,
                    metadata=metadata,
                )

        out_path = out_dir / f"trajectories_seed{seed}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(trajectories), f, indent=2)

        print(f"Saved {args.n_episodes} episodes to {out_path}")
        evaluated += 1
        env.close()

    if evaluated == 0:
        print("Error: no models were evaluated successfully.")
        return 1

    print(f"Evaluation complete ({evaluated} seed(s)).")
    print(f"Plot results: python scripts/plot_results.py --traj-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
