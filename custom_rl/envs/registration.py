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
            eta_limit, eta_obs_limit, eta_dot_obs_limit

        Reward kwargs:
            # Sensor-level reward weights (preferred names)
            eta_weight_disp
            eta_dot_weight_vel
            action_smoothness_weight
            productivity_weight
            ac_productive_target
            # Backward-compatible aliases (old modal reward names)
            eta_weight
            eta_dot_weight
            action_weight
            eta_scale
            eta_dot_scale
            alive_bonus
            termination_penalty
    """
    kwargs = dict(kwargs)

    reward_id = kwargs.pop("reward_id", "dense")

    dt = kwargs.pop("dt", 0.001)
    n_substeps = kwargs.pop("n_substeps", 1)
    max_episode_steps = kwargs.pop("max_episode_steps", None)
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
        "eta_limit",
        "eta_obs_limit",
        "eta_dot_obs_limit",
        # Sensor model configuration (optional)
        "use_eta_internal_safety",
        "disp_norm_scale",
        "vel_norm_scale",
        "obs_clip",
        "displacement_failure_limit",
        "velocity_failure_limit",
        "sensor_coords",
        # Pass schedule (one straight line per episode)
        "n_pass_lines",
        "pass_margin",
        "feed_speed",
        "traj_dt",
        "pass_sampling",
        "pass_complete_tolerance",
        "eta_dot_limit",
        # Structured uncertainty
        "enable_geometry_uncertainty",
        "enable_sensor_uncertainty",
        "enable_process_noise",
        "sensor_coords_relative",
    }

    # Reward keys (preferred sensor-based names).
    reward_keys = {
        "eta_weight_disp",
        "eta_dot_weight_vel",
        "action_smoothness_weight",
        "productivity_weight",
        "ac_productive_target",
        "alive_bonus",
        "termination_penalty",
    }

    plant_kwargs = {k: v for k, v in kwargs.items() if k in plant_keys}

    # Backward-compatible mappings from old modal reward kwargs:
    # - eta_scale, eta_dot_scale become sensor normalization scales
    if "eta_scale" in kwargs and "disp_norm_scale" not in kwargs:
        plant_kwargs["disp_norm_scale"] = kwargs["eta_scale"]
    if "eta_dot_scale" in kwargs and "vel_norm_scale" not in kwargs:
        plant_kwargs["vel_norm_scale"] = kwargs["eta_dot_scale"]

    reward_kwargs = {}
    for k, v in kwargs.items():
        if k in reward_keys:
            reward_kwargs[k] = v

    # Old aliases -> new reward keys
    if "eta_weight" in kwargs:
        reward_kwargs["eta_weight_disp"] = kwargs["eta_weight"]
    if "eta_dot_weight" in kwargs:
        reward_kwargs["eta_dot_weight_vel"] = kwargs["eta_dot_weight"]
    if "action_weight" in kwargs:
        reward_kwargs["action_smoothness_weight"] = kwargs["action_weight"]
    if "productivity_weight" in kwargs:
        reward_kwargs["productivity_weight"] = kwargs["productivity_weight"]
    if "ac_productive_target" in kwargs:
        reward_kwargs["ac_productive_target"] = kwargs["ac_productive_target"]
    if "alive_bonus" in kwargs:
        reward_kwargs["alive_bonus"] = kwargs["alive_bonus"]
    if "termination_penalty" in kwargs:
        reward_kwargs["termination_penalty"] = kwargs["termination_penalty"]

    plant = PlatePlant(**plant_kwargs)
    reward_fn = get_plate_reward(reward_id, **reward_kwargs)

    step_dt = dt * n_substeps
    if max_episode_steps is None and hasattr(plant, "recommended_max_episode_steps"):
        max_episode_steps = plant.recommended_max_episode_steps(step_dt)
    if max_episode_steps is None:
        max_episode_steps = 10000

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