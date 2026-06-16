"""Monte Carlo policy evaluation under structured environment uncertainty."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from custom_rl import DEFAULT_MODEL_DIR, register_envs


ENV_ID = "CustomODEPlate-v0"


def _resolve_model_path(model_dir: Path, seed: int) -> Path | None:
    for rel in (f"best_{seed}/best_model.zip", f"final_{seed}.zip"):
        path = model_dir / rel
        if path.exists():
            return path
    return None


def _episode_record(
    env: gym.Env,
    model: PPO | None,
    *,
    seed: int,
    deterministic: bool,
    max_steps: int,
) -> dict[str, Any]:
    """Run one episode and collect diagnostics for MC analysis."""
    obs, reset_info = env.reset(seed=seed)
    plant = env.unwrapped.plant
    step_dt = float(env.unwrapped._step_dt)

    times: list[float] = [0.0]
    rewards: list[float] = []
    sensor_w: list[list[float]] = []
    sensor_w_dot: list[list[float]] = []
    actions_phys: list[list[float]] = []
    termination_reason = "unknown"

    for step in range(max_steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        if hasattr(plant, "_scale_action"):
            actions_phys.append(
                np.asarray(plant._scale_action(action), dtype=np.float64).tolist()
            )

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        times.append(float(info.get("t", times[-1] + step_dt)))

        if "sensor_w" in info:
            sensor_w.append(np.asarray(info["sensor_w"], dtype=np.float64).tolist())
        if "sensor_w_dot" in info:
            sensor_w_dot.append(np.asarray(info["sensor_w_dot"], dtype=np.float64).tolist())

        if terminated or truncated:
            termination_reason = str(info.get("termination_reason", "unknown"))
            break

    w_arr = np.asarray(sensor_w, dtype=np.float64) if sensor_w else np.empty((0, plant.n_sensors))
    wd_arr = (
        np.asarray(sensor_w_dot, dtype=np.float64)
        if sensor_w_dot
        else np.empty((0, plant.n_sensors))
    )

    return {
        "seed": int(seed),
        "return": float(np.sum(rewards)),
        "length": int(len(rewards)),
        "termination_reason": termination_reason,
        "failed": termination_reason not in {"pass_complete", "unknown"},
        "pass_complete": termination_reason == "pass_complete",
        "episode_context": {
            k: reset_info[k]
            for k in (
                "L1",
                "L2",
                "h",
                "E",
                "rho",
                "delta_L1",
                "delta_L2",
                "delta_h",
                "pass_line_index",
                "pass_x",
                "pass_duration",
                "sensor_coords",
                "geometry_uncertainty_enabled",
                "sensor_uncertainty_enabled",
                "process_noise_enabled",
            )
            if k in reset_info
        },
        "times": times,
        "rewards": rewards,
        "sensor_w": sensor_w,
        "sensor_w_dot": sensor_w_dot,
        "physical_actions": actions_phys,
        "max_abs_sensor_w": float(np.max(np.abs(w_arr))) if w_arr.size else 0.0,
        "max_abs_sensor_w_dot": float(np.max(np.abs(wd_arr))) if wd_arr.size else 0.0,
        "rms_sensor_w": float(np.sqrt(np.mean(w_arr**2))) if w_arr.size else 0.0,
    }


def aggregate_mc_results(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std and outcome frequencies across Monte Carlo rollouts."""
    if not episodes:
        return {}

    returns = np.asarray([ep["return"] for ep in episodes], dtype=np.float64)
    lengths = np.asarray([ep["length"] for ep in episodes], dtype=np.float64)
    max_w = np.asarray([ep["max_abs_sensor_w"] for ep in episodes], dtype=np.float64)
    max_wd = np.asarray([ep["max_abs_sensor_w_dot"] for ep in episodes], dtype=np.float64)
    rms_w = np.asarray([ep["rms_sensor_w"] for ep in episodes], dtype=np.float64)

    reasons = Counter(ep["termination_reason"] for ep in episodes)
    n = len(episodes)

    pass_idx = [
        ep["episode_context"].get("pass_line_index")
        for ep in episodes
        if "pass_line_index" in ep.get("episode_context", {})
    ]
    pass_returns: dict[str, list[float]] = {}
    for ep in episodes:
        ctx = ep.get("episode_context", {})
        if "pass_line_index" not in ctx:
            continue
        key = str(int(ctx["pass_line_index"]))
        pass_returns.setdefault(key, []).append(float(ep["return"]))

    pass_summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for k, v in sorted(pass_returns.items(), key=lambda item: int(item[0]))
    }

    geom_keys = ("delta_L1", "delta_L2", "delta_h")
    geom = {
        key: {
            "mean": float(np.mean([ep["episode_context"].get(key, 0.0) for ep in episodes])),
            "std": float(np.std([ep["episode_context"].get(key, 0.0) for ep in episodes])),
        }
        for key in geom_keys
    }

    return {
        "n_episodes": n,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_median": float(np.median(returns)),
        "return_q05": float(np.quantile(returns, 0.05)),
        "return_q95": float(np.quantile(returns, 0.95)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "success_rate": float(np.mean([ep["pass_complete"] for ep in episodes])),
        "failure_rate": float(np.mean([ep["failed"] for ep in episodes])),
        "termination_reason_counts": dict(reasons),
        "max_abs_sensor_w_mean": float(np.mean(max_w)),
        "max_abs_sensor_w_std": float(np.std(max_w)),
        "max_abs_sensor_w_dot_mean": float(np.mean(max_wd)),
        "max_abs_sensor_w_dot_std": float(np.std(max_wd)),
        "rms_sensor_w_mean": float(np.mean(rms_w)),
        "rms_sensor_w_std": float(np.std(rms_w)),
        "pass_line_return_summary": pass_summary,
        "geometry_delta_summary": geom,
        "pass_line_indices": pass_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo evaluation of policy robustness under uncertainty"
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=0, help="Policy / env seed base")
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--out-dir", default="eval_mc")
    parser.add_argument("--reward", default="productive", choices=["dense", "productive", "quadratic", "sparse"])
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--random-policy", action="store_true", help="Baseline without trained model")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic-policy", action="store_true", help="Use stochastic policy actions")
    parser.add_argument(
        "--disable-uncertainty",
        action="store_true",
        help="Turn off geometry/sensor/process uncertainty for ablation",
    )
    args = parser.parse_args()

    register_envs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs: dict[str, Any] = {"reward_id": args.reward}
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    if args.disable_uncertainty:
        env_kwargs.update(
            {
                "enable_geometry_uncertainty": False,
                "enable_sensor_uncertainty": False,
                "enable_process_noise": False,
            }
        )

    env = gym.make(ENV_ID, **env_kwargs)
    plant = env.unwrapped.plant
    step_dt = float(env.unwrapped._step_dt)
    max_steps = int(env.unwrapped.max_episode_steps)

    model: PPO | None = None
    if not args.random_policy:
        model_path = _resolve_model_path(Path(args.model_dir), args.seed)
        if model_path is None:
            raise FileNotFoundError(f"No model for seed {args.seed} in {args.model_dir}")
        model = PPO.load(str(model_path), device="cpu")

    episodes = []
    for i in range(args.n_rollouts):
        ep_seed = args.seed * 10000 + i
        episodes.append(
            _episode_record(
                env,
                model,
                seed=ep_seed,
                deterministic=not args.stochastic_policy,
                max_steps=max_steps,
            )
        )

    summary = aggregate_mc_results(episodes)
    summary["env_id"] = ENV_ID
    summary["step_dt"] = step_dt
    summary["max_episode_steps"] = max_steps
    summary["policy"] = "random" if args.random_policy else f"seed_{args.seed}"
    summary["deterministic_actions"] = not args.stochastic_policy
    if hasattr(plant, "get_process_noise_info"):
        summary["process_noise"] = plant.get_process_noise_info(step_dt)

    summary_path = out_dir / f"mc_summary_seed{args.seed}.json"
    traj_path = out_dir / f"mc_trajectories_seed{args.seed}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2)

    print(f"Monte Carlo ({summary['n_episodes']} rollouts)")
    print(f"  return: {summary['return_mean']:.4f} ± {summary['return_std']:.4f}")
    print(f"  success rate (pass_complete): {summary['success_rate']:.1%}")
    print(f"  failure rate: {summary['failure_rate']:.1%}")
    print(f"  termination: {summary['termination_reason_counts']}")
    print(f"Saved {summary_path}")
    print(f"Saved {traj_path}")

    env.close()


if __name__ == "__main__":
    main()
