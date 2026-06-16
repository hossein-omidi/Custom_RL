"""
conf3 — High-productivity focus with stochastic simulator.

- Stronger productivity reward, lighter vibration penalty
- Larger batch / more epochs per PPO update
- Extended MC evaluation for paper-grade statistics
"""

from configs._base import make_config

CONFIG = make_config(
    name="conf3",
    description="Productivity-focused PPO with stochastic milling environment",
    total_timesteps=400_000,
    seeds=[0, 1, 2],
    ppo={
        "learning_rate": 2.5e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 15,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "clip_range": 0.15,
        "ent_coef": 0.005,
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
        "displacement_failure_limit": 1.2e-3,
        "velocity_failure_limit": 0.6,
    },
    reward={
        "eta_weight_disp": 0.8,
        "eta_dot_weight_vel": 0.08,
        "productivity_weight": 15.0,
        "ac_productive_target": 6.0,
        "action_smoothness_weight": 0.005,
        "alive_bonus": 1.0,
        "termination_penalty": 80.0,
    },
    train={
        "n_envs": 4,
        "vec_env": "dummy",
        "n_eval_episodes": 5,
        "eval_freq": 15000,
    },
    eval_cfg={
        "checkpoint": "verified",
        "n_verify_episodes": 10,
        "n_episodes": 20,
        "mc_rollouts": 100,
        "mc_disable_uncertainty": False,
        "plot_smooth": 15,
    },
)
