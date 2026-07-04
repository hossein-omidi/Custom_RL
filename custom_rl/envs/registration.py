"""Gymnasium environment registration for the face-milling plate RL plant."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from custom_rl.envs.ode_control_env import ODEControlEnv
from custom_rl.plants.plate import (
    PlatePlant,
    estimate_training_episode_steps,
    OMEGA_MAX_RAD_S,
    OMEGA_MIN_RAD_S,
)
from custom_rl.rewards.plate_rewards import get_plate_reward


# Default dirs for train/eval/plot
DEFAULT_LOG_DIR = "logs/ppo_plate"
DEFAULT_MODEL_DIR = "models/ppo_plate"
DEFAULT_TRAJ_DIR = "eval_trajectories"
DEFAULT_PLOT_DIR = "plots"

# Default environment/face-milling parameters used by registration/factory.
# Keep these in one place so Gym registration and direct make_plate_env() calls
# remain consistent.
DEFAULT_ENV_DT = 0.001
DEFAULT_FEED_PER_TOOTH_MM = 0.20

# Reward scaling for the current flexible AL7075 face-milling task.
# Displacement scale is a vibration-quality target, not the crash/termination limit.
# Velocity scale follows the plant observation scale and keeps velocity secondary
# to displacement while still penalizing high-frequency chatter.
DEFAULT_REWARD_W_SCALE = 1.0e-4       # 0.1 mm; matches PlatePlant.w_obs_scale
DEFAULT_REWARD_WDOT_SCALE = 1.0       # 1 m/s
DEFAULT_REWARD_PRODUCTIVITY_WEIGHT = 10.0
DEFAULT_REWARD_OMEGA_COST_WEIGHT = 3.0


def _translate_legacy_depth_kwargs(kwargs: dict[str, Any]) -> None:
    """Map old peripheral name ac_* to face-milling axial depth ap_* for plant."""
    if "ac_min" in kwargs and "ap_min" not in kwargs:
        kwargs["ap_min"] = kwargs.pop("ac_min")
    else:
        kwargs.pop("ac_min", None)

    if "ac_max" in kwargs and "ap_max" not in kwargs:
        kwargs["ap_max"] = kwargs.pop("ac_max")
    else:
        kwargs.pop("ac_max", None)


def make_plate_env(**kwargs: Any) -> ODEControlEnv:
    """Factory for CustomODEPlate env. Used by ``gymnasium.make()``.

    Main face-milling plant action:
        u = [u_omega, u_ap] in [-1, 1]^2

    Optional if ``control_ae=True``:
        u = [u_omega, u_ap, u_ae]

    Common plant kwargs:
        N, L1, L2, h, E, nu, rho, rho_type,
        m_max, n_max, mode_clamped_axis,
        omega_min, omega_max,
        ap_min, ap_max, ae_min, ae_max, ae_default, control_ae,
        D_mm, feed_per_tooth_mm, gamma_L_deg, gamma_r_deg, gamma_a_deg,
        Kt, Kr, Ka, Kte, Kre, Kae, milling_mode,
        use_process_damping, Ksp, mu, VB,
        sensor_points, w_limit, w_obs_scale, wdot_limit, wdot_obs_scale,
        y_cutter, x0_cutter, x_pass_end_tol, modal_damping_ratio.

    Backward-compatible input:
        ac_min/ac_max are accepted and translated to ap_min/ap_max for the plant.
    """
    kwargs = dict(kwargs)

    reward_id = kwargs.pop("reward_id", "dense")

    dt = kwargs.pop("dt", DEFAULT_ENV_DT)
    n_substeps = kwargs.pop("n_substeps", 1)
    max_episode_steps = kwargs.pop("max_episode_steps", None)
    process_noise_std = kwargs.pop("process_noise_std", 0.0)
    obs_noise_std = kwargs.pop("obs_noise_std", 0.0)
    history_module = kwargs.pop("history_module", None)
    auto_history_module = kwargs.pop("auto_history_module", True)

    _translate_legacy_depth_kwargs(kwargs)

    plant_keys = {
        # Structural/model parameters
        "N",
        "L1",
        "L2",
        "h",
        "E",
        "nu",
        "rho",
        "rho_type",
        "m_max",
        "n_max",
        "mode_clamped_axis",
        "stiffness_grid_points",
        "modal_damping_ratio",
        # Action/process bounds
        "omega_min",
        "omega_max",
        "ap_min",
        "ap_max",
        "ae_min",
        "ae_max",
        "ae_default",
        "control_ae",
        # Face-milling force parameters
        "D_mm",
        "feed_per_tooth_mm",
        "gamma_L_deg",
        "gamma_r_deg",
        "gamma_a_deg",
        "eta_c_deg",
        "Kt",
        "Kr",
        "Ka",
        "Kte",
        "Kre",
        "Kae",
        "milling_mode",
        "theta0",
        "use_process_damping",
        "Ksp",
        "mu",
        "VB",
        "lambda_L_deg",
        "force_projection_mode",
        # Observation/path/RL options
        "sensor_points",
        "w_limit",
        "w_obs_scale",
        "wdot_obs_scale",
        "wdot_limit",
        "eta_limit",
        "y_cutter",
        "x0_cutter",
        "x_pass_end_tol",
        "dynamics_uncertainty_std",
        "y0_min",
        "y0_max",
        "randomize_y0",
        "initial_eta_std",
        "initial_etad_std",
    }

    reward_keys = {
        "w_weight",
        "wdot_weight",
        "eta_weight",
        "eta_dot_weight",
        "action_weight",
        "productivity_weight",
        "negative_ap_weight",
        "negative_ac_weight",
        "ap_productive_target",
        "ac_productive_target",
        "omega_cost_weight",
        "ap_action_weight",
        "ac_action_weight",
        "include_ae_in_productivity",
        "w_scale",
        "wdot_scale",
        "w_clip",
        "wdot_clip",
        "require_physical_info",
        "eta_scale",
        "eta_dot_scale",
        "alive_bonus",
        "termination_penalty",
        "pass_completion_bonus",
        # Reward action bounds
        "omega_min",
        "omega_max",
        "ap_min",
        "ap_max",
        "ac_min",
        "ac_max",
        "ae_min",
        "ae_max",
        "ae_default",
    }

    plant_kwargs = {k: v for k, v in kwargs.items() if k in plant_keys}
    reward_kwargs = {k: v for k, v in kwargs.items() if k in reward_keys}

    # Factory-level default for the current AL7075 face-milling setup.
    # User-supplied feed_per_tooth_mm still overrides this.
    plant_kwargs.setdefault("feed_per_tooth_mm", DEFAULT_FEED_PER_TOOTH_MM)

    unknown_keys = sorted(set(kwargs) - plant_keys - reward_keys)
    if unknown_keys:
        raise TypeError(
            "Unknown make_plate_env kwargs: "
            f"{unknown_keys}. Add them to plant_keys or reward_keys if intended."
        )

    plant = PlatePlant(**plant_kwargs)

    step_dt = float(dt) * int(n_substeps)
    if max_episode_steps is None:
        max_episode_steps = estimate_training_episode_steps(
            plant.L1,
            plant.feed_per_tooth_mm,
            plant.N,
            step_dt,
        )

    # Keep reward action scaling aligned with the plant.
    reward_kwargs.setdefault("omega_min", plant.omega_min)
    reward_kwargs.setdefault("omega_max", plant.omega_max)

    # New face-milling names.
    reward_kwargs.setdefault("ap_min", plant.ap_min)
    reward_kwargs.setdefault("ap_max", plant.ap_max)
    reward_kwargs.setdefault("ap_productive_target", plant.ap_max / 2.0)

    # Backward aliases for reward modules/configs that still use ac_*.
    reward_kwargs.setdefault("ac_min", plant.ap_min)
    reward_kwargs.setdefault("ac_max", plant.ap_max)
    reward_kwargs.setdefault("ac_productive_target", plant.ap_max / 2.0)

    reward_kwargs.setdefault("ae_min", plant.ae_min)
    reward_kwargs.setdefault("ae_max", plant.ae_max)
    reward_kwargs.setdefault("ae_default", plant.ae_default)

    # Reward vibration scales are stricter than the plant termination limits.
    # Do not set default clipping here; DenseProductivePlateReward leaves
    # vibration costs unclipped unless the user explicitly supplies w_clip/wdot_clip.
    reward_kwargs.setdefault("w_scale", DEFAULT_REWARD_W_SCALE)
    reward_kwargs.setdefault("wdot_scale", DEFAULT_REWARD_WDOT_SCALE)
    reward_kwargs.setdefault("productivity_weight", DEFAULT_REWARD_PRODUCTIVITY_WEIGHT)
    reward_kwargs.setdefault("omega_cost_weight", DEFAULT_REWARD_OMEGA_COST_WEIGHT)
    # Strict by default: reward must use physical sensor signals supplied in info.
    reward_kwargs.setdefault("require_physical_info", True)

    reward_fn = get_plate_reward(reward_id, **reward_kwargs)

    return ODEControlEnv(
        plant=plant,
        reward_fn=reward_fn,
        dt=dt,
        n_substeps=n_substeps,
        max_episode_steps=max_episode_steps,
        process_noise_std=process_noise_std,
        obs_noise_std=obs_noise_std,
        history_module=history_module,
        auto_history_module=auto_history_module,
    )


def register_envs() -> None:
    """Register custom RL environment with Gymnasium.

    Do not set Gymnasium's external ``max_episode_steps`` here.
    ``make_plate_env`` computes the episode limit from the actual constructed
    plant parameters, including feed_per_tooth_mm, dt, and n_substeps, and then
    passes that limit into ODEControlEnv. This avoids a stale TimeLimit wrapper
    if the user overrides feed or dt in gym.make(...).
    """
    env_id = "CustomODEPlate-v0"

    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point="custom_rl.envs.registration:make_plate_env",
            kwargs={
                "reward_id": "dense",
                "dt": DEFAULT_ENV_DT,
                "feed_per_tooth_mm": DEFAULT_FEED_PER_TOOTH_MM,
                "omega_min": OMEGA_MIN_RAD_S,
                "omega_max": OMEGA_MAX_RAD_S,
            },
        )
