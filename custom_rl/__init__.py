"""Custom RL: Gymnasium-compatible ODE control environment framework."""

from custom_rl.envs.registration import (
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_PLOT_DIR,
    DEFAULT_TRAJ_DIR,
    register_envs,
)

__all__ = [
    "register_envs",
    "DEFAULT_LOG_DIR",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_PLOT_DIR",
    "DEFAULT_TRAJ_DIR",
]
