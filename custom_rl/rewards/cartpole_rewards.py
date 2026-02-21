"""CartPole reward functions: dense and sparse."""

from __future__ import annotations

from typing import Any

import numpy as np

from custom_rl.rewards.base import RewardFn


class DenseBalanceReward:
    """
    Dense reward encouraging upright balance.
    Reward each step for staying near theta=0 and x=0.
    """

    def __init__(
        self,
        theta_scale: float = 1.0,
        x_scale: float = 0.4,
        alive_bonus: float = 1.0,
    ):
        self.theta_scale = theta_scale
        self.x_scale = x_scale
        self.alive_bonus = alive_bonus

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> float:
        if terminated:
            return 0.0
        theta = x_next[2]
        x_pos = x_next[0]
        cost = self.theta_scale * theta**2 + self.x_scale * x_pos**2
        return self.alive_bonus - cost


class SparseUprightReward:
    """
    Sparse reward: +1 per step while not terminated, 0 on termination.
    """

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> float:
        return 0.0 if terminated else 1.0


# Registry for pluggable rewards; extend via register_cartpole_reward()
_CARTPOLE_REWARD_REGISTRY: dict[str, type] = {
    "dense": DenseBalanceReward,
    "sparse": SparseUprightReward,
}


def register_cartpole_reward(reward_id: str, reward_cls: type) -> None:
    """Register a custom CartPole reward. Use for pluggable rewards without modifying core."""
    _CARTPOLE_REWARD_REGISTRY[reward_id] = reward_cls


def get_cartpole_reward(reward_id: str, **kwargs: Any) -> RewardFn:
    """Return CartPole reward by id. Used by registration. Kwargs passed to reward constructor."""
    if reward_id not in _CARTPOLE_REWARD_REGISTRY:
        raise ValueError(f"Unknown reward_id: {reward_id}. Known: {list(_CARTPOLE_REWARD_REGISTRY)}")
    return _CARTPOLE_REWARD_REGISTRY[reward_id](**kwargs)
