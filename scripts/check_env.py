"""Verify custom RL environment with Gymnasium check_env and short rollout."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from custom_rl import register_envs

ENV_ID = "CustomODEPlate-v0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check custom RL env with Gymnasium.")
    parser.add_argument("--reward", default="dense", choices=["dense", "sparse", "quadratic"])
    parser.add_argument("--steps", type=int, default=50, help="Rollout steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    args = parser.parse_args()

    register_envs()
    env = gym.make(
        ENV_ID,
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=1,
        max_episode_steps=args.max_episode_steps,
    )
    raw_env = env.unwrapped

    print(f"Running Gymnasium check_env on {ENV_ID} (skip_render_check=True)...")
    check_env(raw_env, skip_render_check=True)
    print("check_env passed.")

    print("\nShort rollout with random actions (using wrapped env)...")
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if not np.isfinite(obs).all():
            print(f"Step {step}: NaN/Inf in obs")
            return 1
        if terminated or truncated:
            obs, info = env.reset(seed=args.seed)
    print(f"  Steps: {args.steps}, total_reward: {total_reward:.2f}")
    env.close()

    print("\nDeterminism check (same seed -> same first obs)...")
    env2 = gym.make(
        ENV_ID,
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=1,
        max_episode_steps=args.max_episode_steps,
    )
    o1, _ = env.reset(seed=args.seed)
    o2, _ = env2.reset(seed=args.seed)
    env2.close()
    if (o1 == o2).all():
        print("  Deterministic reset OK.")
    else:
        print("  WARNING: reset non-deterministic")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
