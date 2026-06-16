#!/usr/bin/env python3
"""
Config-based evaluation, Monte Carlo analysis, and plotting.

Usage:
    python result.py --config conf1
    python result.py --config conf2 --skip-mc

Uses verified checkpoint selection (best vs final hold-out rollouts) unless
config eval.checkpoint is set to "best" or "final".

Outputs per config under runs/<config>/:
    trajectories/   episode rollouts (JSON)
    eval_mc/        Monte Carlo summaries
    plots/          learning curves, sensor dynamics, MC figures
    reports/        checkpoint_selection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from configs import list_configs, load_config
from custom_rl import register_envs
from scripts.checkpoint_utils import save_checkpoint_report, select_checkpoint
from scripts.monte_carlo_eval import aggregate_mc_results
from scripts.plot_results import plot_learning_curve, plot_mc_research_suite, plot_trajectory_summary

ENV_ID = "CustomODEPlate-v0"


def _build_env_kwargs(cfg: dict) -> dict:
    env_kwargs = dict(cfg["env"])
    env_kwargs.update(cfg["reward"])
    return env_kwargs


def resolve_all_checkpoints(cfg: dict) -> list:
    """Select evaluation checkpoint per seed; write report."""
    eval_cfg = cfg["eval"]
    model_dir = Path(cfg["model_dir"])
    log_dir = Path(cfg["log_dir"])
    report_dir = Path(cfg["report_dir"])
    env_kwargs = _build_env_kwargs(cfg)

    policy = eval_cfg.get("checkpoint", "verified")
    n_verify = int(eval_cfg.get("n_verify_episodes", 8))
    deterministic = bool(eval_cfg.get("deterministic_policy", True))
    seeds = eval_cfg.get("policy_seeds", eval_cfg["seeds"])

    selections = []
    for seed in seeds:
        sel = select_checkpoint(
            model_dir,
            log_dir,
            seed,
            policy=policy,
            env_kwargs=env_kwargs,
            n_verify_episodes=n_verify,
            deterministic=deterministic,
        )
        if sel is None:
            print(f"[checkpoint] seed {seed}: NO CHECKPOINT FOUND")
            continue
        selections.append(sel)
        print(
            f"[checkpoint] seed {seed}: {sel.source} ← {sel.path}"
            + (
                f"  (verified return={sel.mean_return:.2f}±{sel.std_return:.2f})"
                if sel.mean_return is not None
                else ""
            )
        )

    if selections:
        save_checkpoint_report(selections, report_dir / "checkpoint_selection.json")
    return selections


def _get_metadata(env: gym.Env, cfg: dict, model_path: str, checkpoint_source: str) -> dict:
    base_env = env.unwrapped
    plant = base_env.plant
    max_episode_steps = int(base_env.max_episode_steps)
    metadata = {
        "config_name": cfg["config_name"],
        "env_id": ENV_ID,
        "checkpoint_source": checkpoint_source,
        "model_path": model_path,
        "dt": float(base_env.dt),
        "n_substeps": int(base_env.n_substeps),
        "step_dt": float(base_env._step_dt),
        "max_episode_steps": max_episode_steps,
        "sensor_coords": np.asarray(plant.sensor_coords, dtype=np.float64).tolist(),
        "disp_norm_scale": float(plant.disp_norm_scale),
        "vel_norm_scale": float(plant.vel_norm_scale),
        "obs_clip": plant.obs_clip,
        "displacement_failure_limit": float(plant.displacement_failure_limit),
        "velocity_failure_limit": float(plant.velocity_failure_limit),
        "n_pass_lines": int(plant.n_pass_lines),
        "feed_speed": float(plant.feed_speed),
        "pass_duration": float(plant.pass_duration),
    }
    if hasattr(plant, "get_interface_metadata"):
        metadata["interface"] = plant.get_interface_metadata()
    if hasattr(plant, "u_phys_low"):
        metadata["physical_action_low"] = plant.u_phys_low.tolist()
        metadata["physical_action_high"] = plant.u_phys_high.tolist()
    return metadata


def run_trajectory_eval(cfg: dict, selections: list) -> None:
    traj_dir = Path(cfg["traj_dir"])
    traj_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = _build_env_kwargs(cfg)
    eval_cfg = cfg["eval"]
    n_episodes = int(eval_cfg["n_episodes"])
    deterministic = bool(eval_cfg.get("deterministic_policy", True))

    register_envs()
    sel_by_seed = {s.seed: s for s in selections}

    for seed in eval_cfg.get("policy_seeds", eval_cfg["seeds"]):
        sel = sel_by_seed.get(seed)
        if sel is None:
            continue

        model_path = Path(sel.path)
        print(f"[trajectories] seed {seed} ← {model_path} ({sel.source})")
        model = PPO.load(str(model_path), device="cpu")
        env = gym.make(ENV_ID, **env_kwargs)
        metadata = _get_metadata(env, cfg, str(model_path), sel.source)

        trajectories = []
        for ep in range(n_episodes):
            obs, reset_info = env.reset(seed=seed + 1000 + ep)
            episode_context = dict(reset_info)

            states, actions, physical_actions = [], [], []
            sensor_w, sensor_w_dot, reward_components, rewards, times = [], [], [], [], []
            current_time = 0.0
            termination_reason = "unknown"

            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                plant = env.unwrapped.plant
                u_phys = plant.normalized_to_physical_action(action)

                states.append(obs.tolist())
                actions.append(np.asarray(action, dtype=np.float64).tolist())
                physical_actions.append(u_phys.tolist())
                times.append(float(current_time))

                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                if "sensor_w" in info:
                    sensor_w.append(np.asarray(info["sensor_w"]).tolist())
                if "sensor_w_dot" in info:
                    sensor_w_dot.append(np.asarray(info["sensor_w_dot"]).tolist())
                if "reward_components" in info:
                    reward_components.append(info["reward_components"])
                current_time = float(info.get("t", current_time + metadata["step_dt"]))

                if terminated or truncated:
                    termination_reason = str(info.get("termination_reason", "unknown"))
                    break

            trajectories.append(
                {
                    "metadata": metadata,
                    "episode_context": episode_context,
                    "termination_reason": termination_reason,
                    "times": times,
                    "states": states,
                    "actions": actions,
                    "physical_actions": physical_actions,
                    "sensor_w": sensor_w,
                    "sensor_w_dot": sensor_w_dot,
                    "reward_components": reward_components,
                    "rewards": rewards,
                    "return": float(sum(rewards)),
                    "length": len(rewards),
                }
            )

        out_path = traj_dir / f"trajectories_seed{seed}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2)
        print(f"[trajectories] Saved {n_episodes} episodes → {out_path}")
        env.close()


def _mc_episode(
    env,
    model,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> dict[str, Any]:
    obs, reset_info = env.reset(seed=seed)
    plant = env.unwrapped.plant
    step_dt = float(env.unwrapped._step_dt)

    rewards, sensor_w, sensor_w_dot, times = [], [], [], [0.0]
    termination_reason = "unknown"

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        times.append(float(info.get("t", times[-1] + step_dt)))
        if "sensor_w" in info:
            sensor_w.append(np.asarray(info["sensor_w"]).tolist())
        if "sensor_w_dot" in info:
            sensor_w_dot.append(np.asarray(info["sensor_w_dot"]).tolist())
        if terminated or truncated:
            termination_reason = str(info.get("termination_reason", "unknown"))
            break

    w_arr = np.asarray(sensor_w) if sensor_w else np.empty((0, plant.n_sensors))
    return {
        "seed": seed,
        "return": float(np.sum(rewards)),
        "length": len(rewards),
        "termination_reason": termination_reason,
        "failed": termination_reason not in {"pass_complete", "unknown"},
        "pass_complete": termination_reason == "pass_complete",
        "episode_context": dict(reset_info),
        "times": times,
        "rewards": rewards,
        "sensor_w": sensor_w,
        "sensor_w_dot": sensor_w_dot,
        "max_abs_sensor_w": float(np.max(np.abs(w_arr))) if w_arr.size else 0.0,
        "rms_sensor_w": float(np.sqrt(np.mean(w_arr**2))) if w_arr.size else 0.0,
    }


def run_monte_carlo(cfg: dict, selections: list) -> None:
    eval_cfg = cfg["eval"]
    mc_dir = Path(cfg["mc_dir"])
    mc_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = _build_env_kwargs(cfg)
    if eval_cfg.get("mc_disable_uncertainty"):
        env_kwargs.update(
            {
                "enable_geometry_uncertainty": False,
                "enable_sensor_uncertainty": False,
                "enable_process_noise": False,
            }
        )

    register_envs()
    n_rollouts = int(eval_cfg["mc_rollouts"])
    deterministic = bool(eval_cfg.get("deterministic_policy", True))
    sel_by_seed = {s.seed: s for s in selections}

    for seed in eval_cfg.get("policy_seeds", eval_cfg["seeds"]):
        sel = sel_by_seed.get(seed)
        if sel is None:
            continue

        model_path = Path(sel.path)
        print(f"[monte_carlo] seed {seed}, {n_rollouts} rollouts ← {model_path}")
        model = PPO.load(str(model_path), device="cpu")
        env = gym.make(ENV_ID, **env_kwargs)
        max_steps = int(env.unwrapped.max_episode_steps)

        episodes = [
            _mc_episode(env, model, seed * 10000 + i, max_steps, deterministic)
            for i in range(n_rollouts)
        ]

        summary = aggregate_mc_results(episodes)
        summary.update(
            {
                "config_name": cfg["config_name"],
                "checkpoint_source": sel.source,
                "model_path": str(model_path),
                "step_dt": float(env.unwrapped._step_dt),
            }
        )

        summary_path = mc_dir / f"mc_summary_seed{seed}.json"
        traj_path = mc_dir / f"mc_trajectories_seed{seed}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2)

        print(
            f"[monte_carlo] return={summary['return_mean']:.3f}±{summary['return_std']:.3f} "
            f"success={summary['success_rate']:.1%} → {summary_path}"
        )
        env.close()


def run_plots(cfg: dict) -> None:
    eval_cfg = cfg["eval"]
    plot_dir = Path(cfg["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    seeds = eval_cfg.get("policy_seeds", eval_cfg["seeds"])
    smooth = int(eval_cfg.get("plot_smooth", 10))
    step_dt = float(cfg["env"].get("dt", 0.001)) * int(cfg["env"].get("n_substeps", 1))

    plot_learning_curve(Path(cfg["log_dir"]), plot_dir, seeds, smooth=smooth)
    plot_trajectory_summary(Path(cfg["traj_dir"]), plot_dir, seeds)
    plot_mc_research_suite(Path(cfg["mc_dir"]), plot_dir, seeds, step_dt=step_dt)
    print(f"[plots] Saved → {plot_dir}")


def results_from_config(cfg: dict) -> None:
    print("=" * 60)
    print(f"Results   : {cfg['config_name']}")
    print(f"Description: {cfg['description']}")
    print(f"Run dir   : {cfg['run_dir']}")
    print(f"Checkpoint: {cfg['eval'].get('checkpoint', 'verified')}")
    print("=" * 60)

    selections = resolve_all_checkpoints(cfg)
    if not selections:
        raise FileNotFoundError(
            f"No checkpoints found under {cfg['model_dir']}. Run training.py first."
        )

    eval_cfg = cfg["eval"]
    if eval_cfg.get("run_trajectories", True):
        run_trajectory_eval(cfg, selections)
    if eval_cfg.get("run_mc", True):
        run_monte_carlo(cfg, selections)
    if eval_cfg.get("run_plots", True):
        run_plots(cfg)

    print()
    print("Results complete.")
    print(f"  Plots   → {cfg['plot_dir']}")
    print(f"  Reports → {cfg['report_dir']}")


def main() -> None:
    available = ", ".join(list_configs()) or "conf1"
    parser = argparse.ArgumentParser(description="Evaluate/plot from configs/confN.py")
    parser.add_argument("--config", default="conf1", help=f"Config name ({available})")
    parser.add_argument("--skip-trajectories", action="store_true")
    parser.add_argument("--skip-mc", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.skip_trajectories:
        cfg["eval"]["run_trajectories"] = False
    if args.skip_mc:
        cfg["eval"]["run_mc"] = False
    if args.skip_plots:
        cfg["eval"]["run_plots"] = False

    results_from_config(cfg)


if __name__ == "__main__":
    main()
