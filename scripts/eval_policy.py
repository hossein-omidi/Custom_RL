"""Evaluate trained policy and save trajectories for plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from custom_rl import DEFAULT_MODEL_DIR, DEFAULT_TRAJ_DIR, register_envs


ENV_ID = "CustomODEPlate-v0"


def _get_metadata(env: gym.Env, max_episode_steps: int) -> dict:
    """Collect useful metadata for plotting."""
    base_env = env.unwrapped
    plant = base_env.plant

    metadata = {
        "env_id": ENV_ID,
        "dt": float(base_env.dt),
        "n_substeps": int(base_env.n_substeps),
        "step_dt": float(base_env._step_dt),
        "max_episode_steps": int(max_episode_steps),
        # Sensor model metadata
        "sensor_coords": np.asarray(getattr(plant, "sensor_coords", np.empty((0, 2))), dtype=np.float64).tolist(),
        "disp_norm_scale": float(getattr(plant, "disp_norm_scale", np.nan)),
        "vel_norm_scale": float(getattr(plant, "vel_norm_scale", np.nan)),
        "obs_clip": getattr(plant, "obs_clip", None),
        "displacement_failure_limit": float(getattr(plant, "displacement_failure_limit", np.nan)),
        "velocity_failure_limit": float(getattr(plant, "velocity_failure_limit", np.nan)),
        "n_pass_lines": int(getattr(plant, "n_pass_lines", 0)),
        "feed_speed": float(getattr(plant, "feed_speed", np.nan)),
        "pass_duration": float(getattr(plant, "pass_duration", np.nan)),
        "observation_note": (
            "PPO observation = [w_s_norm..., w_dot_s_norm..., prev_u_omega, prev_u_ac]. "
            "Physical sensor signals are in sensor_w / sensor_w_dot trajectory fields."
        ),
    }

    if hasattr(plant, "get_interface_metadata"):
        metadata["interface"] = plant.get_interface_metadata()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy and save trajectories")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--out-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument(
        "--reward",
        default="dense",
        choices=["dense", "productive", "quadratic", "sparse"],
    )
    parser.add_argument("--max-episode-steps", type=int, default=3000)
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

        env = gym.make(
            ENV_ID,
            reward_id=args.reward,
            max_episode_steps=args.max_episode_steps,
        )

        metadata = _get_metadata(env, args.max_episode_steps)
        trajectories = []

        for ep in range(args.n_episodes):
            obs, reset_info = env.reset(seed=seed + 1000 + ep)

            states = []
            actions = []
            physical_actions = []
            sensor_w = []
            sensor_w_dot = []
            reward_components = []
            rewards = []
            times = []
            episode_context = {
                k: reset_info[k]
                for k in (
                    "L1", "L2", "h", "E", "rho",
                    "delta_L1", "delta_L2", "delta_h", "delta_E", "delta_rho",
                    "pass_line_index", "pass_x", "pass_duration",
                    "sensor_coords", "geometry_uncertainty_enabled",
                    "sensor_uncertainty_enabled",
                )
                if k in reset_info
            }

            current_time = 0.0
            termination_reason = "unknown"

            while True:
                action, _ = model.predict(obs, deterministic=True)

                states.append(obs.tolist())
                actions.append(np.asarray(action, dtype=np.float64).reshape(-1).tolist())
                physical_actions.append(_physical_action(env, action))
                times.append(float(current_time))

                obs, reward, terminated, truncated, info = env.step(action)

                rewards.append(float(reward))
                if "sensor_w" in info:
                    sensor_w.append(np.asarray(info["sensor_w"], dtype=np.float64).tolist())
                if "sensor_w_dot" in info:
                    sensor_w_dot.append(np.asarray(info["sensor_w_dot"], dtype=np.float64).tolist())
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

        out_path = Path(args.out_dir) / f"trajectories_seed{seed}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2)

        print(f"Saved {args.n_episodes} episodes to {out_path}")

        env.close()


if __name__ == "__main__":
    main()