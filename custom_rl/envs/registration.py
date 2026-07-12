"""Gymnasium environment registration for the face-milling plate RL plant."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from custom_rl.envs.ode_control_env import ODEControlEnv
from custom_rl.plants.plate import (
    PlatePlant,
    estimate_training_episode_steps,
    rpm_to_omega,
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
DEFAULT_ENV_DT = 0.0001
DEFAULT_ENV_N_SUBSTEPS = 10
DEFAULT_FEED_PER_TOOTH_MM = 0.20

# Default process settings shared by training, evaluation, and checker scripts.
# Realistic face-milling spindle range for this machine/tool: 400--4000 rpm.
# This window also contains the interesting part of the stability lobes for
# the plate's dominant modes (17-41 Hz), so speed selection genuinely matters.
DEFAULT_ENV_RPM_MIN = 400.0
DEFAULT_ENV_RPM_MAX = 4000.0
DEFAULT_ENV_OMEGA_MIN = float(rpm_to_omega(DEFAULT_ENV_RPM_MIN))
DEFAULT_ENV_OMEGA_MAX = float(rpm_to_omega(DEFAULT_ENV_RPM_MAX))
DEFAULT_ENV_AP_MIN_MM = 0.0
DEFAULT_ENV_AP_MAX_MM = 18
DEFAULT_ENV_AE_DEFAULT_MM = 28.0
DEFAULT_ENV_W_LIMIT = 1.0e-3
DEFAULT_ENV_W_OBS_SCALE = 5.0e-4
DEFAULT_ENV_WDOT_LIMIT = 10.0
DEFAULT_ENV_WDOT_OBS_SCALE = 0.5
DEFAULT_ENV_RANDOMIZE_Y0 = True
DEFAULT_ENV_Y_CUTTER = 0.20
DEFAULT_ENV_INITIAL_ETA_STD = 1.0e-7
DEFAULT_ENV_INITIAL_ETAD_STD = 0.0
DEFAULT_ENV_MODAL_DAMPING_RATIO = 0.02

# Reward scaling for the current flexible AL7075 face-milling task.
# The hard displacement limit is 1 mm.  The dense productivity term is
# multiplied by a smooth vibration gate, so high material removal is rewarded
# only when the measured physical vibration remains controlled.
DEFAULT_REWARD_W_SCALE = 7.5e-4          # softer dense vibration cost; hard limit remains 1 mm
DEFAULT_REWARD_WDOT_SCALE = 1.0          # 1 m/s velocity scale
DEFAULT_REWARD_W_WEIGHT = 0.6
DEFAULT_REWARD_WDOT_WEIGHT = 0.02
# productivity_weight/omega_cost_weight/w_weight were rebalanced from
# (20.0, 1.0, 1.0) after numerical verification showed the dense per-step
# reward was negative even at a safe, non-chattering operating point (the
# vibration cost dominated the productivity term everywhere). The values
# below keep dense reward close to neutral at a safe operating point and
# mildly positive near the edge of the stable envelope, while still growing
# quadratically as vibration approaches the termination limit.
DEFAULT_REWARD_PRODUCTIVITY_WEIGHT = 30.0
DEFAULT_REWARD_OMEGA_COST_WEIGHT = 2.0
DEFAULT_REWARD_ACTION_RATE_WEIGHT = 2.0
DEFAULT_REWARD_TERMINATION_PENALTY = 500.0
DEFAULT_REWARD_TRUNCATION_PENALTY = 500.0
DEFAULT_REWARD_PASS_COMPLETION_BONUS = 10000.0
DEFAULT_REWARD_FAILURE_PROGRESS_PENALTY_WEIGHT = 1.0

# Generalized safety-gated productivity:
#   G = 1 / (1 + (w_rms/w_gate)^2 + beta*(wdot_rms/wdot_gate)^2)
#   productivity = productivity_weight * normalized_MRR * G
DEFAULT_REWARD_PRODUCTIVITY_GATE_ENABLED = True
DEFAULT_REWARD_PRODUCTIVITY_W_GATE = 5.0e-4
DEFAULT_REWARD_PRODUCTIVITY_WDOT_GATE = 1.0
DEFAULT_REWARD_PRODUCTIVITY_WDOT_GATE_WEIGHT = 0.05
DEFAULT_REWARD_PRODUCTIVITY_GATE_POWER = 2.0
DEFAULT_REWARD_PRODUCTIVITY_GATE_MIN = 0.0


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
    n_substeps = kwargs.pop("n_substeps", DEFAULT_ENV_N_SUBSTEPS)
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
        "control_ap",
        "randomize_ap",
        "ap_fixed",
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
        "action_rate_weight",
        "include_omega_in_productivity",
        "ap_action_weight",
        "ac_action_weight",
        "include_ae_in_productivity",
        "productivity_gate_enabled",
        "productivity_w_gate",
        "productivity_wdot_gate",
        "productivity_wdot_gate_weight",
        "productivity_gate_power",
        "productivity_gate_min",
        "w_scale",
        "wdot_scale",
        "w_clip",
        "wdot_clip",
        "require_physical_info",
        "eta_scale",
        "eta_dot_scale",
        "alive_bonus",
        "termination_penalty",
        "truncation_penalty",
        "pass_completion_bonus",
        "failure_progress_penalty_weight",
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

    # Factory-level defaults for the current AL7075 face-milling setup.
    # User-supplied kwargs still override every value here.
    plant_kwargs.setdefault("omega_min", DEFAULT_ENV_OMEGA_MIN)
    plant_kwargs.setdefault("omega_max", DEFAULT_ENV_OMEGA_MAX)
    plant_kwargs.setdefault("ap_min", DEFAULT_ENV_AP_MIN_MM)
    plant_kwargs.setdefault("ap_max", DEFAULT_ENV_AP_MAX_MM)
    plant_kwargs.setdefault("ae_default", DEFAULT_ENV_AE_DEFAULT_MM)
    plant_kwargs.setdefault("feed_per_tooth_mm", DEFAULT_FEED_PER_TOOTH_MM)
    plant_kwargs.setdefault("w_limit", DEFAULT_ENV_W_LIMIT)
    plant_kwargs.setdefault("w_obs_scale", DEFAULT_ENV_W_OBS_SCALE)
    plant_kwargs.setdefault("wdot_limit", DEFAULT_ENV_WDOT_LIMIT)
    plant_kwargs.setdefault("wdot_obs_scale", DEFAULT_ENV_WDOT_OBS_SCALE)
    plant_kwargs.setdefault("randomize_y0", DEFAULT_ENV_RANDOMIZE_Y0)
    plant_kwargs.setdefault("y_cutter", DEFAULT_ENV_Y_CUTTER)
    plant_kwargs.setdefault("initial_eta_std", DEFAULT_ENV_INITIAL_ETA_STD)
    plant_kwargs.setdefault("initial_etad_std", DEFAULT_ENV_INITIAL_ETAD_STD)
    plant_kwargs.setdefault("modal_damping_ratio", DEFAULT_ENV_MODAL_DAMPING_RATIO)

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
    reward_kwargs.setdefault("w_weight", DEFAULT_REWARD_W_WEIGHT)
    reward_kwargs.setdefault("wdot_weight", DEFAULT_REWARD_WDOT_WEIGHT)
    reward_kwargs.setdefault("productivity_weight", DEFAULT_REWARD_PRODUCTIVITY_WEIGHT)
    reward_kwargs.setdefault("omega_cost_weight", DEFAULT_REWARD_OMEGA_COST_WEIGHT)
    reward_kwargs.setdefault("action_rate_weight", DEFAULT_REWARD_ACTION_RATE_WEIGHT)
    reward_kwargs.setdefault("include_omega_in_productivity", False)
    reward_kwargs.setdefault("productivity_gate_enabled", DEFAULT_REWARD_PRODUCTIVITY_GATE_ENABLED)
    reward_kwargs.setdefault("productivity_w_gate", DEFAULT_REWARD_PRODUCTIVITY_W_GATE)
    reward_kwargs.setdefault("productivity_wdot_gate", DEFAULT_REWARD_PRODUCTIVITY_WDOT_GATE)
    reward_kwargs.setdefault(
        "productivity_wdot_gate_weight",
        DEFAULT_REWARD_PRODUCTIVITY_WDOT_GATE_WEIGHT,
    )
    reward_kwargs.setdefault("productivity_gate_power", DEFAULT_REWARD_PRODUCTIVITY_GATE_POWER)
    reward_kwargs.setdefault("productivity_gate_min", DEFAULT_REWARD_PRODUCTIVITY_GATE_MIN)
    reward_kwargs.setdefault("termination_penalty", DEFAULT_REWARD_TERMINATION_PENALTY)
    reward_kwargs.setdefault("truncation_penalty", DEFAULT_REWARD_TRUNCATION_PENALTY)
    reward_kwargs.setdefault("pass_completion_bonus", DEFAULT_REWARD_PASS_COMPLETION_BONUS)
    reward_kwargs.setdefault(
        "failure_progress_penalty_weight",
        DEFAULT_REWARD_FAILURE_PROGRESS_PENALTY_WEIGHT,
    )
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
    """Register the custom face-milling RL environments with Gymnasium.

    Two environment ids share the same plant, reward, and physics and differ
    only in the RL action space:

    ``CustomODEPlate-v0`` (first-mode / roughing control)
        Action = [omega, ap].  Both spindle speed and axial depth of cut are
        controlled to maximize material removal while suppressing chatter.

    ``CustomODEPlateFinish-v0`` (second-mode / finishing control)
        Action = [omega] only.  Axial depth of cut ap is a fixed process
        parameter, randomized per episode over [ap_min, ap_max] exactly like the
        milling line y0.  Spindle speed alone is modulated to suppress chatter at
        a constant depth, for a uniform finishing pass.

    Do not set Gymnasium's external ``max_episode_steps`` here.
    ``make_plate_env`` computes the episode limit from the actual constructed
    plant parameters, including feed_per_tooth_mm, dt, and n_substeps, and then
    passes that limit into ODEControlEnv. This avoids a stale TimeLimit wrapper
    if the user overrides feed or dt in gym.make(...).
    """
    common_kwargs = {
        "reward_id": "dense",
        "dt": DEFAULT_ENV_DT,
        "n_substeps": DEFAULT_ENV_N_SUBSTEPS,
        "omega_min": DEFAULT_ENV_OMEGA_MIN,
        "omega_max": DEFAULT_ENV_OMEGA_MAX,
        "ap_min": DEFAULT_ENV_AP_MIN_MM,
        "ap_max": DEFAULT_ENV_AP_MAX_MM,
        "ae_default": DEFAULT_ENV_AE_DEFAULT_MM,
        "feed_per_tooth_mm": DEFAULT_FEED_PER_TOOTH_MM,
        "w_limit": DEFAULT_ENV_W_LIMIT,
        "w_obs_scale": DEFAULT_ENV_W_OBS_SCALE,
        "wdot_limit": DEFAULT_ENV_WDOT_LIMIT,
        "wdot_obs_scale": DEFAULT_ENV_WDOT_OBS_SCALE,
        "randomize_y0": DEFAULT_ENV_RANDOMIZE_Y0,
        "y_cutter": DEFAULT_ENV_Y_CUTTER,
        "initial_eta_std": DEFAULT_ENV_INITIAL_ETA_STD,
        "initial_etad_std": DEFAULT_ENV_INITIAL_ETAD_STD,
        "modal_damping_ratio": DEFAULT_ENV_MODAL_DAMPING_RATIO,
    }

    # First-mode / roughing control: action = [omega, ap].
    if "CustomODEPlate-v0" not in gym.envs.registry:
        gym.register(
            id="CustomODEPlate-v0",
            entry_point="custom_rl.envs.registration:make_plate_env",
            kwargs={**common_kwargs, "control_ap": True},
        )

    # Second-mode / finishing control: action = [omega]; ap fixed per episode
    # and randomized over [ap_min, ap_max] like the milling line y0.
    if "CustomODEPlateFinish-v0" not in gym.envs.registry:
        gym.register(
            id="CustomODEPlateFinish-v0",
            entry_point="custom_rl.envs.registration:make_plate_env",
            kwargs={**common_kwargs, "control_ap": False, "randomize_ap": True},
        )
