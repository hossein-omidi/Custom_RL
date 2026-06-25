"""Evaluate trained policy and save trajectories for plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from custom_rl import DEFAULT_MODEL_DIR, DEFAULT_TRAJ_DIR, register_envs
from custom_rl.eval.monte_carlo import (
    aggregate_mc_sensor_runs,
    plot_mc_sensor_bands,
)


ENV_ID = "CustomODEPlate-v0"


def _get_metadata(env: gym.Env, max_episode_steps: int) -> dict:
    """Collect useful metadata for plotting and post-processing."""
    base_env = env.unwrapped
    plant = base_env.plant

    metadata = {
        "env_id": ENV_ID,
        "dt": float(base_env.dt),
        "n_substeps": int(base_env.n_substeps),
        "step_dt": float(base_env._step_dt),
        "max_episode_steps": int(max_episode_steps),

        # Kept for backward/debug compatibility.
        "eta_limit": float(getattr(plant, "eta_limit", np.nan)),
        "eta_obs_limit": float(getattr(plant, "eta_obs_limit", np.nan)),
        "eta_dot_obs_limit": float(getattr(plant, "eta_dot_obs_limit", np.nan)),

        # Current observation meaning.
        "observation_type": "scaled_physical_sensor_disp_vel",
        "n_sensors": int(getattr(plant, "n_sensors", 0)),
        "sensor_points": np.asarray(
            getattr(plant, "sensor_points", []),
            dtype=np.float64,
        ).tolist(),
        "w_limit": float(getattr(plant, "w_limit", np.nan)),
        "w_obs_scale": float(getattr(plant, "w_obs_scale", np.nan)),
        "wdot_obs_scale": float(getattr(plant, "wdot_obs_scale", np.nan)),
        "dynamics_uncertainty_std": float(
            getattr(plant, "dynamics_uncertainty_std", 0.0)
        ),
        "randomize_y0": bool(getattr(plant, "randomize_y0", False)),
    }

    if hasattr(plant, "u_phys_low") and hasattr(plant, "u_phys_high"):
        metadata["physical_action_low"] = np.asarray(
            plant.u_phys_low, dtype=np.float64
        ).tolist()
        metadata["physical_action_high"] = np.asarray(
            plant.u_phys_high, dtype=np.float64
        ).tolist()

    return metadata


def _physical_action(env: gym.Env, action: np.ndarray) -> list[float]:
    """Convert normalized policy action to physical [omega, ac]."""
    plant = env.unwrapped.plant

    if hasattr(plant, "_scale_action"):
        return np.asarray(plant._scale_action(action), dtype=np.float64).tolist()

    return np.asarray(action, dtype=np.float64).reshape(-1).tolist()


def _physical_signal_from_info_or_obs(
    info: dict,
    obs: np.ndarray,
    metadata: dict,
) -> list[float]:
    """
    Return unscaled physical sensor signal:
        [w_sensor_1, ..., w_sensor_n, wdot_sensor_1, ..., wdot_sensor_n]

    Prefer info["w_sensor"] and info["wdot_sensor"] because they come directly
    from the plant. If unavailable, fall back to unscaling the observation.
    """
    if "w_sensor" in info and "wdot_sensor" in info:
        w_sensor = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        wdot_sensor = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)

        return np.concatenate([w_sensor, wdot_sensor]).tolist()

    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    n_sensors = int(metadata.get("n_sensors", obs_arr.size // 2))

    if n_sensors <= 0 or obs_arr.size != 2 * n_sensors:
        return []

    w_obs_scale = float(metadata.get("w_obs_scale", 1.0))
    wdot_obs_scale = float(metadata.get("wdot_obs_scale", 1.0))

    if not np.isfinite(w_obs_scale) or abs(w_obs_scale) < 1e-12:
        w_obs_scale = 1.0

    if not np.isfinite(wdot_obs_scale) or abs(wdot_obs_scale) < 1e-12:
        wdot_obs_scale = 1.0

    physical_signal = obs_arr.copy()
    physical_signal[:n_sensors] /= w_obs_scale
    physical_signal[n_sensors:] /= wdot_obs_scale

    return physical_signal.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy and save trajectories")

    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--out-dir", default=DEFAULT_TRAJ_DIR)

    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--n-substeps", type=int, default=1)

    parser.add_argument(
        "--reward",
        default="productive",
        choices=["dense", "productive", "quadratic", "sparse"],
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: auto from pass duration)",
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
        default=0.0,
        help="Gaussian disturbance on modal accelerations [0=deterministic]",
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample milling start y0 on each reset",
    )

    args = parser.parse_args()

    register_envs()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        model_path = Path(args.model_dir) / f"best_{seed}" / "best_model.zip"

        if not model_path.exists():
            model_path = Path(args.model_dir) / f"final_{seed}.zip"

        if not model_path.exists():
            print(f"Skip seed {seed}: no model found for this seed.")
            continue

        model = PPO.load(str(model_path), device="cpu")

        env_kwargs: dict = {
            "reward_id": args.reward,
            "dt": args.dt,
            "n_substeps": args.n_substeps,
            "dynamics_uncertainty_std": args.dynamics_uncertainty_std,
            "randomize_y0": args.randomize_y0,
        }
        if args.max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = args.max_episode_steps

        env = gym.make(ENV_ID, **env_kwargs)

        max_steps = args.max_episode_steps or env.unwrapped.max_episode_steps
        metadata = _get_metadata(env, max_steps)
        trajectories = []

        for ep in range(args.n_episodes):
            mc_runs: list[dict] = []

            for mc in range(args.n_mc):
                obs, reset_info = env.reset(seed=seed + 1000 + ep * args.n_mc + mc)

                observations: list[list[float]] = []
                physical_signals: list[list[float]] = []
                x_modal: list[list[float]] = []
                actions: list[list[float]] = []
                physical_actions: list[list[float]] = []
                rewards: list[float] = []
                times: list[float] = []
                w_sensor_hist: list[list[float]] = []
                wdot_sensor_hist: list[list[float]] = []
                reward_terms_hist: list[dict] = []

                termination_reason = None
                terminated_final = False
                truncated_final = False

                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)

                    actions.append(action_arr.tolist())
                    physical_actions.append(_physical_action(env, action_arr))

                    obs, reward, terminated, truncated, info = env.step(action_arr)

                    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
                    observations.append(obs_arr.tolist())

                    physical_signals.append(
                        _physical_signal_from_info_or_obs(info, obs_arr, metadata)
                    )

                    if "w_sensor" in info:
                        w_sensor_hist.append(
                            np.asarray(info["w_sensor"], dtype=np.float64).tolist()
                        )
                    if "wdot_sensor" in info:
                        wdot_sensor_hist.append(
                            np.asarray(info["wdot_sensor"], dtype=np.float64).tolist()
                        )
                    if "reward_terms" in info:
                        reward_terms_hist.append(dict(info["reward_terms"]))

                    if "x_modal" in info:
                        x_modal.append(
                            np.asarray(info["x_modal"], dtype=np.float64)
                            .reshape(-1)
                            .tolist()
                        )

                    rewards.append(float(reward))
                    times.append(float(info.get("t", len(rewards) * metadata["step_dt"])))

                    if terminated or truncated:
                        terminated_final = bool(terminated)
                        truncated_final = bool(truncated)
                        termination_reason = info.get("termination_reason", None)
                        break

                mc_runs.append(
                    {
                        "mc_index": mc,
                        "seed": seed + 1000 + ep * args.n_mc + mc,
                        "y0": reset_info.get("y0"),
                        "times": times,
                        "observations": observations,
                        "physical_signals": physical_signals,
                        "w_sensor": w_sensor_hist,
                        "wdot_sensor": wdot_sensor_hist,
                        "reward_terms": reward_terms_hist,
                        "x_modal": x_modal,
                        "actions": actions,
                        "physical_actions": physical_actions,
                        "rewards": rewards,
                        "return": float(sum(rewards)),
                        "length": len(rewards),
                        "terminated": terminated_final,
                        "truncated": truncated_final,
                        "termination_reason": termination_reason,
                    }
                )

            # Monte Carlo aggregate (physical displacement, sensor 1 max abs as summary)
            mc_summary = None
            if args.n_mc > 1 and mc_runs:
                w_series = []
                times_runs = []
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
                    mc_summary = {
                        "n_mc": args.n_mc,
                        "max_abs_w_mean": mean_w.tolist(),
                        "max_abs_w_std": std_w.tolist(),
                        "times": times_agg.tolist(),
                    }

            trajectories.append(
                {
                    "metadata": metadata,
                    "episode": int(ep),
                    "seed": int(seed),
                    "n_mc": args.n_mc,
                    "mc_summary": mc_summary,
                    "runs": mc_runs,
                    # Backward-compatible fields from first MC run
                    "times": mc_runs[0]["times"],
                    "observations": mc_runs[0]["observations"],
                    "physical_signals": mc_runs[0]["physical_signals"],
                    "x_modal": mc_runs[0]["x_modal"],
                    "actions": mc_runs[0]["actions"],
                    "physical_actions": mc_runs[0]["physical_actions"],
                    "rewards": mc_runs[0]["rewards"],
                    "return": mc_runs[0]["return"],
                    "length": mc_runs[0]["length"],
                    "terminated": mc_runs[0]["terminated"],
                    "truncated": mc_runs[0]["truncated"],
                    "termination_reason": mc_runs[0]["termination_reason"],
                }
            )

            if mc_summary is not None:
                out_plot_dir = Path(args.out_dir) / f"seed{seed}"
                out_plot_dir.mkdir(parents=True, exist_ok=True)
                n_sensors = int(metadata.get("n_sensors", 1))
                w_labels = [f"w_sensor_{i + 1} (m)" for i in range(n_sensors)]
                w_runs = [
                    np.asarray(run["w_sensor"], dtype=np.float64)
                    for run in mc_runs
                    if run["w_sensor"]
                ]
                times_runs = [
                    np.asarray(run["times"], dtype=np.float64)
                    for run in mc_runs
                    if run["times"]
                ]
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
                    out_path=out_plot_dir / f"mc_displacement_ep{ep}.png",
                    w_limit=metadata.get("w_limit"),
                )

        out_path = Path(args.out_dir) / f"trajectories_seed{seed}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2)

        print(f"Saved {args.n_episodes} episodes to {out_path}")

        env.close()


if __name__ == "__main__":
    main()