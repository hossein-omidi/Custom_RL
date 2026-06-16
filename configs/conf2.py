"""
conf2 — Stochastic robust training.

- Full geometry / sensor / process uncertainty enabled
- Longer training, slightly lower learning rate
- Monte Carlo evaluation with uncertainty ON
"""

from configs._base import make_config

CONFIG = make_config(
    name="conf2",
    description="Robust PPO under structured simulator uncertainty",
    total_timesteps=300_000,
    seeds=[0, 1, 2],
    ppo={
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    env={
        "reward_id": "productive",
        "dt": 0.001,
        "n_substeps": 1,
        "max_episode_steps": None,
        "obs_noise_std": 0.0,
        "process_noise_std": 0.0,
        "enable_geometry_uncertainty": True,
        "enable_sensor_uncertainty": True,
        "enable_process_noise": True,
        "pass_sampling": "random",
        "n_pass_lines": 100,
        "feed_speed": 0.05,
        "displacement_failure_limit": 1e-3,
        "velocity_failure_limit": 0.5,
    },
    reward={
        "eta_weight_disp": 1.2,
        "eta_dot_weight_vel": 0.15,
        "productivity_weight": 8.0,
        "ac_productive_target": 5.0,
        "action_smoothness_weight": 0.02,
        "alive_bonus": 1.0,
        "termination_penalty": 120.0,
    },
    train={
        "n_envs": 2,
        "vec_env": "dummy",
        "n_eval_episodes": 5,
        "eval_freq": 10000,
    },
    eval_cfg={
        "checkpoint": "verified",
        "n_verify_episodes": 8,
        "n_episodes": 15,
        "mc_rollouts": 60,
        "mc_disable_uncertainty": False,
    },
)
