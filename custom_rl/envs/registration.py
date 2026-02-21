"""Gymnasium environment registration."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from custom_rl.envs.ode_control_env import ODEControlEnv
from custom_rl.plants.cartpole import CartPolePlant
from custom_rl.rewards.cartpole_rewards import get_cartpole_reward

# Default dirs for train/eval/plot (single source of truth)
DEFAULT_LOG_DIR = "logs/ppo_cartpole"
DEFAULT_MODEL_DIR = "models/ppo_cartpole"
DEFAULT_TRAJ_DIR = "eval_trajectories"
DEFAULT_PLOT_DIR = "plots"


def make_cartpole_env(**kwargs: Any) -> ODEControlEnv:
    """
    Factory for CustomODECartPole env. Used by gymnasium.make().

    Kwargs:
        reward_id: "dense" | "sparse"
        dt: integration step size
        n_substeps: RK4 substeps per env step
        max_episode_steps: truncation length
        process_noise_std: std of Gaussian noise added to state (0 = deterministic)
        obs_noise_std: std of Gaussian noise added to observations (0 = perfect)
        plant kwargs: mass_cart, mass_pole, length, gravity, force_max, etc.
    """
    kwargs = dict(kwargs)  # Copy to avoid mutating caller's dict
    reward_id = kwargs.pop("reward_id", "dense")
    dt = kwargs.pop("dt", 0.02)
    n_substeps = kwargs.pop("n_substeps", 1)
    max_episode_steps = kwargs.pop("max_episode_steps", 500)
    process_noise_std = kwargs.pop("process_noise_std", 0.0)
    obs_noise_std = kwargs.pop("obs_noise_std", 0.0)

    plant_kwargs = {k: v for k, v in kwargs.items() if k in {"mass_cart", "mass_pole", "length", "gravity", "force_max", "x_limit", "theta_limit_rad"}}
    plant = CartPolePlant(**plant_kwargs)
    reward_fn = get_cartpole_reward(reward_id)

    return ODEControlEnv(
        plant=plant,
        reward_fn=reward_fn,
        dt=dt,
        n_substeps=n_substeps,
        max_episode_steps=max_episode_steps,
        process_noise_std=process_noise_std,
        obs_noise_std=obs_noise_std,
    )


def register_envs() -> None:
    """Register all custom RL environments with Gymnasium. Call before gymnasium.make()."""
    gym.register(
        id="CustomODECartPole-v0",
        entry_point="custom_rl.envs.registration:make_cartpole_env",
        max_episode_steps=500,
        kwargs={"reward_id": "dense"},
    )
