"""Gymnasium environment registration."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from custom_rl.envs.ode_control_env import ODEControlEnv
from custom_rl.plants.plate import PlatePlant
from custom_rl.rewards.plate_rewards import get_plate_reward


# Default dirs for train/eval/plot
DEFAULT_LOG_DIR = "logs/ppo_plate"
DEFAULT_MODEL_DIR = "models/ppo_plate"
DEFAULT_TRAJ_DIR = "eval_trajectories"
DEFAULT_PLOT_DIR = "plots"


def make_plate_env(**kwargs: Any) -> ODEControlEnv:
    """
    Factory for CustomODEPlate env. Used by gymnasium.make().

    Kwargs:
        reward_id: "dense" | "quadratic" | "sparse"

        Environment kwargs:
            dt
            n_substeps
            max_episode_steps
            process_noise_std
            obs_noise_std

        Plant kwargs:
            N, L1, L2, h, E, nu, rho,
            m_max, n_max,
            omega_min, omega_max,
            ac_min, ac_max,
            sensor_points, w_limit, w_obs_scale, wdot_obs_scale,
            eta_limit

        Reward kwargs:
            w_weight, wdot_weight (aliases: eta_weight, eta_dot_weight)
            action_weight
            w_scale, wdot_scale (aliases: eta_scale, eta_dot_scale)
            alive_bonus
            termination_penalty
    """
    kwargs = dict(kwargs)

    reward_id = kwargs.pop("reward_id", "dense")

    dt = kwargs.pop("dt", 0.001)
    n_substeps = kwargs.pop("n_substeps", 1)
    max_episode_steps = kwargs.pop("max_episode_steps", 10000)
    process_noise_std = kwargs.pop("process_noise_std", 0.0)
    obs_noise_std = kwargs.pop("obs_noise_std", 0.0)

    plant_keys = {
        "N",
        "L1",
        "L2",
        "h",
        "E",
        "nu",
        "rho",
        "m_max",
        "n_max",
        "omega_min",
        "omega_max",
        "ac_min",
        "ac_max",
        "sensor_points",
        "w_limit",
        "w_obs_scale",
        "wdot_obs_scale",
        "wdot_limit",
        "eta_limit",
    }

    reward_keys = {
        "w_weight",
        "wdot_weight",
        "eta_weight",
        "eta_dot_weight",
        "action_weight",
        "w_scale",
        "wdot_scale",
        "w_clip",
        "wdot_clip",
        "eta_scale",
        "eta_dot_scale",
        "alive_bonus",
        "termination_penalty",
    }

    plant_kwargs = {k: v for k, v in kwargs.items() if k in plant_keys}
    reward_kwargs = {k: v for k, v in kwargs.items() if k in reward_keys}

    plant = PlatePlant(**plant_kwargs)

    # Keep reward action scaling aligned with the plant.
    for bound_key in ("omega_min", "omega_max", "ac_min", "ac_max"):
        reward_kwargs.setdefault(bound_key, getattr(plant, bound_key))
    reward_kwargs.setdefault("ac_productive_target", plant.ac_max / 2.0)
    reward_kwargs.setdefault("w_scale", plant.w_limit)
    reward_kwargs.setdefault("wdot_scale", plant.wdot_limit)
    reward_kwargs.setdefault("w_clip", plant.w_limit)
    reward_kwargs.setdefault("wdot_clip", plant.wdot_limit)

    reward_fn = get_plate_reward(reward_id, **reward_kwargs)

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
    """Register custom RL environment with Gymnasium. Call before gymnasium.make()."""
    env_id = "CustomODEPlate-v0"

    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point="custom_rl.envs.registration:make_plate_env",
            max_episode_steps=10000,
            kwargs={"reward_id": "dense"},
        )