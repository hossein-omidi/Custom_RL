"""Verify CustomODEPlate env with Gymnasium check_env and short rollout."""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from custom_rl import register_envs

ENV_ID = "CustomODEPlate-v0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check plate RL env with Gymnasium.")
    parser.add_argument("--reward", default="productive", choices=["dense", "productive", "quadratic", "sparse"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    register_envs()
    env = gym.make(
        ENV_ID,
        reward_id=args.reward,
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        max_episode_steps=200,
    )
    raw_env = env.unwrapped

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space:      {env.action_space.shape}")

    print("Running Gymnasium check_env...")
    check_env(raw_env, skip_render_check=True)
    print("check_env passed.")

    obs, _ = env.reset(seed=args.seed)
    if obs.shape != env.observation_space.shape:
        print(f"FAIL: obs shape {obs.shape} != {env.observation_space.shape}")
        return 1

    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if not np.isfinite(obs).all():
            print(f"Step {step}: NaN/Inf in obs")
            return 1
        if "sensor_w" not in info or "reward_components" not in info:
            print(f"Step {step}: missing sensor_w or reward_components in info")
            return 1
        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed + step)

    print(f"Rollout OK: {args.steps} steps, total_reward={total_reward:.2f}")
    env.close()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
