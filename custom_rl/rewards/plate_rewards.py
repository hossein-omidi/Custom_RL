"""Plate vibration reward functions: dense quadratic and sparse rewards."""

from __future__ import annotations

from typing import Any

import numpy as np

from custom_rl.rewards.base import RewardFn


class DenseQuadraticPlateReward:
    """
    Dense continuous linear-quadratic reward for suppressing plate vibration.

    State:
        x = [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]

    Reward:
        reward = alive_bonus - cost

    Cost:
        cost = eta_weight     * mean((eta / eta_scale)^2)
             + eta_dot_weight * mean((eta_dot / eta_dot_scale)^2)
             + action_weight  * mean(u^2)

    Notes:
        - eta suppresses modal displacement.
        - eta_dot suppresses modal velocity.
        - u is the normalized PPO action in [-1, 1]^2.
        - action penalty prevents unnecessarily aggressive control.
    """

    def __init__(
        self,
        eta_weight: float = 1.0,
        eta_dot_weight: float = 0.1,
        action_weight: float = 0.01,
        eta_scale: float = 1e-3,
        eta_dot_scale: float = 1e-2,
        alive_bonus: float = 1.0,
        termination_penalty: float = 100.0,
    ):
        self.eta_weight = eta_weight
        self.eta_dot_weight = eta_dot_weight
        self.action_weight = action_weight
        self.eta_scale = eta_scale
        self.eta_dot_scale = eta_dot_scale
        self.alive_bonus = alive_bonus
        self.termination_penalty = termination_penalty

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
        x_next = np.asarray(x_next, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        if not np.all(np.isfinite(x_next)):
            return -float(self.termination_penalty)

        eta = x_next[0::2]
        eta_dot = x_next[1::2]

        eta_cost = np.mean((eta / self.eta_scale) ** 2)
        eta_dot_cost = np.mean((eta_dot / self.eta_dot_scale) ** 2)

        # u is already clipped by ODEControlEnv according to the action space.
        # The extra clipping here is only a safety guard.
        action_cost = np.mean(np.clip(u, -1.0, 1.0) ** 2)

        cost = (
            self.eta_weight * eta_cost
            + self.eta_dot_weight * eta_dot_cost
            + self.action_weight * action_cost
        )

        reward = self.alive_bonus - cost

        if terminated:
            reward -= self.termination_penalty

        return float(reward)


class SparseStablePlateReward:
    """
    Sparse reward for simple stability testing.

    Returns:
        +1 while the episode is not terminated
         0 if terminated
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


_PLATE_REWARD_REGISTRY: dict[str, type] = {
    "dense": DenseQuadraticPlateReward,
    "quadratic": DenseQuadraticPlateReward,
    "sparse": SparseStablePlateReward,
}


def register_plate_reward(reward_id: str, reward_cls: type) -> None:
    """Register a custom Plate reward."""
    _PLATE_REWARD_REGISTRY[reward_id] = reward_cls


def get_plate_reward(reward_id: str, **kwargs: Any) -> RewardFn:
    """
    Return Plate reward by id.

    Args:
        reward_id: reward name
        **kwargs: parameters passed to the reward constructor
    """
    if reward_id not in _PLATE_REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward_id: {reward_id}. "
            f"Known reward ids: {list(_PLATE_REWARD_REGISTRY)}"
        )

    return _PLATE_REWARD_REGISTRY[reward_id](**kwargs)