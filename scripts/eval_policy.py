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
        "eta_limit": float(getattr(plant, "eta_limit", np.nan)),
        "eta_obs_limit": float(getattr(plant, "eta_obs_limit", np.nan)),
        "eta_dot_obs_limit": float(getattr(plant, "eta_dot_obs_limit", np.nan)),
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
            obs, _ = env.reset(seed=seed + 1000 + ep)

            states = []
            actions = []
            physical_actions = []
            rewards = []
            times = []

            current_time = 0.0

            while True:
                action, _ = model.predict(obs, deterministic=True)

                states.append(obs.tolist())
                actions.append(np.asarray(action, dtype=np.float64).reshape(-1).tolist())
                physical_actions.append(_physical_action(env, action))
                times.append(float(current_time))

                obs, reward, terminated, truncated, info = env.step(action)

                rewards.append(float(reward))
                current_time = float(info.get("t", current_time + metadata["step_dt"]))

                if terminated or truncated:
                    break

            trajectories.append(
                {
                    "metadata": metadata,
                    "times": times,
                    "states": states,
                    "actions": actions,
                    "physical_actions": physical_actions,
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