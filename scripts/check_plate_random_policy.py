"""Check CustomODEPlate-v0 with a random policy.

Run from the project root:

    python scripts/check_plate_random_policy.py

This script does not train any agent. It only verifies that:
- environment registration works
- reset() works
- step() works
- observation/action shapes are compatible
- rewards are finite
- no NaN/Inf appears in observations
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from custom_rl import register_envs


def main() -> None:
    register_envs()

    env = gym.make(
        "CustomODEPlate-v0",
        reward_id="dense",
        dt=0.001,
        n_substeps=1,
        max_episode_steps=1000,
    )

    obs, info = env.reset(seed=42)

    print("Environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    plant = env.unwrapped.plant
    expected_obs_dim = 2 * plant.n_sensors + 2

    print("Initial observation shape:", obs.shape)
    print("Expected sensor-based observation dim:", expected_obs_dim)
    print("Internal modal state dim (not exposed):", plant.state_dim)
    print("Pass line index:", info.get("pass_line_index"))
    print("Pass x (m):", info.get("pass_x"))
    print("Pass duration (s):", info.get("pass_duration"))
    print("Auto max episode steps:", env.unwrapped.max_episode_steps)
    print("Initial info:", info)

    assert obs.shape == (expected_obs_dim,), (
        f"Observation must be sensor-based with prev action, got {obs.shape}."
    )
    assert obs.shape != (plant.state_dim,), "PPO observation must not expose raw modal state."

    assert env.observation_space.contains(obs), (
        "Initial observation is outside observation_space."
    )

    total_reward = 0.0
    max_steps = 1000

    for step in range(max_steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape, (
            f"Observation shape mismatch at step {step}: "
            f"got {obs.shape}, expected {env.observation_space.shape}"
        )

        assert np.all(np.isfinite(obs)), (
            f"Observation contains NaN or Inf at step {step}."
        )

        assert np.isfinite(reward), (
            f"Reward is NaN or Inf at step {step}: {reward}"
        )

        assert isinstance(terminated, bool), (
            f"terminated must be bool, got {type(terminated)}"
        )

        assert isinstance(truncated, bool), (
            f"truncated must be bool, got {type(truncated)}"
        )

        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}.")
            print("terminated:", terminated)
            print("truncated:", truncated)
            print("info:", info)
            break

    env.close()

    print("Random policy test finished successfully.")
    print("Total reward:", total_reward)
    print("Final observation shape:", obs.shape)


if __name__ == "__main__":
    main()