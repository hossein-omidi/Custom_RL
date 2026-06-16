"""
conf1 — Baseline deterministic training.

- Nominal plate geometry (no uncertainty)
- Productive sensor reward, standard PPO hyperparameters
- Moderate training budget
"""

from configs._base import make_config

CONFIG = make_config(
    name="conf1",
    description="Baseline deterministic PPO — nominal simulator, productive reward",
    total_timesteps=150_000,
    seeds=[0, 1, 2],
    ppo={
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
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
        # Plant / simulator
        "enable_geometry_uncertainty": False,
        "enable_sensor_uncertainty": False,
        "enable_process_noise": False,
        "pass_sampling": "random",
        "n_pass_lines": 100,
        "feed_speed": 0.05,
        "displacement_failure_limit": 1e-3,
        "velocity_failure_limit": 0.5,
    },
    reward={
        "eta_weight_disp": 1.0,
        "eta_dot_weight_vel": 0.1,
        "productivity_weight": 10.0,
        "ac_productive_target": 5.0,
        "action_smoothness_weight": 0.01,
        "alive_bonus": 1.0,
        "termination_penalty": 100.0,
    },
    train={
        "n_envs": 1,
        "vec_env": "dummy",
        "n_eval_episodes": 3,
        "eval_freq": 5000,
    },
    eval_cfg={
        "checkpoint": "verified",
        "n_verify_episodes": 6,
        "n_episodes": 12,
        "mc_rollouts": 30,
        "mc_disable_uncertainty": True,
    },
)
