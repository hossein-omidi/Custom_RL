"""Plate vibration reward functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from custom_rl.rewards.base import RewardFn


class DenseProductivePlateReward:
    """
    Dense reward for productive vibration suppression.

    Goal:
        Suppress vibration while maintaining productive cutting.

    State:
        x = [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]

    Normalized action:
        u = [u_omega, u_ac] in [-1, 1]^2

    Physical action reconstructed inside reward:
        omega in [omega_min, omega_max]
        ac    in [ac_min, ac_max]

    Reward:
        reward = productivity_weight * productivity_score
               - vibration_cost
               - velocity_cost
               - negative_ac_cost
               - action_regularization

    Notes:
        - productivity_score is zero when ac=0 (no cutting), so idle motion earns no reward.
        - vibration penalties suppress chatter/plate vibration.
        - negative_ac_cost discourages nonphysical negative cutting intensity.
        - action_regularization is optional and should stay small.
    """

    def __init__(
        self,
        eta_weight: float = 1.0,
        eta_dot_weight: float = 0.1,
        action_weight: float = 0.0,
        productivity_weight: float = 10.0,
        negative_ac_weight: float = 2.0,
        eta_scale: float = 1e-3,
        eta_dot_scale: float = 1e-2,
        omega_min: float = 50.0,
        omega_max: float = 2000.0,
        ac_min: float = 0.0,
        ac_max: float = 20.0,
        ac_productive_target: float = 10.0,
        alive_bonus: float = 0.0,
        termination_penalty: float = 100.0,
    ):
        self.eta_weight = eta_weight
        self.eta_dot_weight = eta_dot_weight
        self.action_weight = action_weight
        self.productivity_weight = productivity_weight
        self.negative_ac_weight = negative_ac_weight

        self.eta_scale = eta_scale
        self.eta_dot_scale = eta_dot_scale

        self.omega_min = omega_min
        self.omega_max = omega_max
        self.ac_min = ac_min
        self.ac_max = ac_max
        self.ac_productive_target = ac_productive_target

        self.alive_bonus = alive_bonus
        self.termination_penalty = termination_penalty

        if self.omega_max <= self.omega_min:
            raise ValueError("omega_max must be greater than omega_min.")

        if self.ac_max <= self.ac_min:
            raise ValueError("ac_max must be greater than ac_min.")

        if self.ac_productive_target <= 0.0:
            raise ValueError("ac_productive_target must be positive.")

    def _scale_action(self, u: np.ndarray) -> tuple[float, float]:
        """
        Convert normalized action in [-1, 1]^2 to physical [omega, ac].
        """
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        if u.size < 2:
            u_safe = np.zeros(2, dtype=np.float64)
            u_safe[: u.size] = u
            u = u_safe

        u = np.clip(u[:2], -1.0, 1.0)

        low = np.array([self.omega_min, self.ac_min], dtype=np.float64)
        high = np.array([self.omega_max, self.ac_max], dtype=np.float64)

        u_phys = low + 0.5 * (u + 1.0) * (high - low)

        omega = float(u_phys[0])
        ac = float(u_phys[1])

        return omega, ac

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

        omega, ac = self._scale_action(u)

        omega_score = (omega - self.omega_min) / (self.omega_max - self.omega_min)
        omega_score = float(np.clip(omega_score, 0.0, 1.0))

        positive_ac = max(ac, 0.0)
        ac_score = positive_ac / self.ac_productive_target
        ac_score = float(np.clip(ac_score, 0.0, 1.0))

        productivity_score = omega_score * ac_score

        negative_ac = max(-ac, 0.0)
        negative_ac_cost = (negative_ac / max(self.ac_max, 1e-12)) ** 2

        action_cost = np.mean(np.clip(u, -1.0, 1.0) ** 2)

        vibration_cost = (
            self.eta_weight * eta_cost
            + self.eta_dot_weight * eta_dot_cost
        )

        reward = (
            self.alive_bonus
            + self.productivity_weight * productivity_score
            - vibration_cost
            - self.negative_ac_weight * negative_ac_cost
            - self.action_weight * action_cost
        )

        if terminated:
            reward -= self.termination_penalty

        return float(reward)


class DenseQuadraticPlateReward:
    """
    Original pure quadratic vibration-suppression reward.

    This is kept for comparison/debugging.
    It does not explicitly reward productive cutting.
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
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

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
    "dense": DenseProductivePlateReward,
    "productive": DenseProductivePlateReward,
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