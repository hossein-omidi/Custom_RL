"""
conf_fast — minimal config for quick debugging (not for publication results).

Usage:
    python training.py --config conf_fast
    python result.py --config conf_fast
    python scripts/stability_lobes.py --config conf_fast
"""

from configs._base import make_config

CONFIG = make_config(
    name="conf_fast",
    description="Fast debug PPO — short horizon, tiny budget, small stability grid",
    total_timesteps=10_000,
    seeds=[0],
    ppo={
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [64, 64]},
    },
    env={
        "reward_id": "productive",
        "dt": 0.002,
        "n_substeps": 1,
        "max_episode_steps": 80,
        "integrator": "dde_rk4",
        "delay_mode": "constant_tau",
        "obs_noise_std": 0.0,
        "process_noise_std": 0.0,
        "enable_geometry_uncertainty": False,
        "enable_sensor_uncertainty": False,
        "enable_process_noise": False,
        "pass_sampling": "random",
        "n_pass_lines": 5,
        "trajectory_mode": "middle_line",
        "displacement_model": "feed_normal_full",
        "feed_speed": 0.05,
        "displacement_failure_limit": 1e-3,
        "velocity_failure_limit": 0.5,
    },
    reward={
        "eta_weight_disp": 1.0,
        "eta_dot_weight_vel": 0.1,
        "productivity_weight": 5.0,
        "ac_productive_target": 5.0,
        "action_smoothness_weight": 0.01,
        "alive_bonus": 1.0,
        "termination_penalty": 50.0,
    },
    train={
        "n_envs": 1,
        "vec_env": "dummy",
        "n_eval_episodes": 2,
        "eval_freq": 2000,
    },
    eval_cfg={
        "checkpoint": "final",
        "n_verify_episodes": 2,
        "n_episodes": 3,
        "mc_rollouts": 3,
        "mc_disable_uncertainty": True,
        "run_mc": False,
    },
    stability_lobe={
        "omega_grid": [400.0, 600.0, 800.0, 1000.0, 1200.0],
        "ac_grid": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        "n_rollouts": 4,
        "seeds": [0, 1, 2, 3],
        "t_final_s": 3.0,
        "macro_dt_s": 0.002,
        "transient_fraction": 0.35,
        "unstable_threshold": 0.5,
        "growth_factor": 3.0,
        "controller_modes": ["uncontrolled"],
    },
)
