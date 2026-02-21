"""Evaluate trained policy and save trajectories for plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from custom_rl import DEFAULT_MODEL_DIR, DEFAULT_TRAJ_DIR, register_envs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy and save trajectories")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--out-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--reward", default="dense", choices=["dense", "sparse"])
    args = parser.parse_args()

    register_envs()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        model_path = Path(args.model_dir) / f"best_{seed}" / "best_model.zip"
        if not model_path.exists():
            model_path = Path(args.model_dir) / f"final_{seed}.zip"
        if not model_path.exists():
            print(f"Skip seed {seed}: no model at {model_path}")
            continue

        model = PPO.load(str(model_path))
        env = gym.make("CustomODECartPole-v0", reward_id=args.reward)

        trajectories = []
        for ep in range(args.n_episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)
            states, actions, rewards = [], [], []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                states.append(obs.tolist())
                actions.append(action.tolist())
                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            trajectories.append({
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "return": sum(rewards),
                "length": len(rewards),
            })

        out_path = Path(args.out_dir) / f"trajectories_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(trajectories, f, indent=2)
        print(f"Saved {args.n_episodes} episodes to {out_path}")

        env.close()


if __name__ == "__main__":
    main()
