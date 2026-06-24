"""Plate vibration reward functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from custom_rl.rewards.base import RewardFn


def _physical_vibration_cost(
    info: dict[str, Any],
    x_next: np.ndarray,
    w_weight: float,
    wdot_weight: float,
    w_scale: float,
    wdot_scale: float,
    w_clip: float | None = None,
    wdot_clip: float | None = None,
) -> tuple[float, float]:
    """
    Return (displacement_cost, velocity_cost) using physical sensor signals.

    Signals are clipped to physical limits before squaring to keep rewards finite.
    """
    if "w_sensor" in info and "wdot_sensor" in info:
        w_sensor = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        wdot_sensor = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
    else:
        eta = x_next[0::2]
        eta_dot = x_next[1::2]
        w_sensor = eta
        wdot_sensor = eta_dot

    if not (np.all(np.isfinite(w_sensor)) and np.all(np.isfinite(wdot_sensor))):
        return float("inf"), float("inf")

    w_scale = max(float(w_scale), 1e-12)
    wdot_scale = max(float(wdot_scale), 1e-12)

    if w_clip is not None:
        w_sensor = np.clip(w_sensor, -w_clip, w_clip)
    if wdot_clip is not None:
        wdot_sensor = np.clip(wdot_sensor, -wdot_clip, wdot_clip)

    w_norm = w_sensor / w_scale
    wdot_norm = wdot_sensor / wdot_scale

    w_cost = float(np.mean(w_norm ** 2))
    wdot_cost = float(np.mean(wdot_norm ** 2))

    return w_weight * w_cost, wdot_weight * wdot_cost


class DenseProductivePlateReward:
    """
    Dense reward for productive cutting with physical chatter suppression.

    Chatter cost uses physical sensor displacement/velocity from info:
        w_sensor, wdot_sensor

    Reward:
        productivity_weight * productivity_score
        - w_weight * sum(w_sensor^2)
        - wdot_weight * sum(wdot_sensor^2)
        - action_weight * sum(action^2)
    """

    def __init__(
        self,
        w_weight: float = 1.0,
        wdot_weight: float = 0.1,
        action_weight: float = 0.0,
        productivity_weight: float = 10.0,
        negative_ac_weight: float = 2.0,
        w_scale: float = 1e-3,
        wdot_scale: float = 1e-2,
        omega_min: float = 50.0,
        omega_max: float = 2000.0,
        ac_min: float = 0.0,
        ac_max: float = 20.0,
        ac_productive_target: float = 10.0,
        alive_bonus: float = 0.0,
        termination_penalty: float = 100.0,
        pass_completion_bonus: float = 500.0,
        w_clip: float | None = None,
        wdot_clip: float | None = None,
        # Backward-compatible aliases for registration kwargs.
        eta_weight: float | None = None,
        eta_dot_weight: float | None = None,
        eta_scale: float | None = None,
        eta_dot_scale: float | None = None,
    ):
        if eta_weight is not None:
            w_weight = eta_weight
        if eta_dot_weight is not None:
            wdot_weight = eta_dot_weight
        if eta_scale is not None:
            w_scale = eta_scale
        if eta_dot_scale is not None:
            wdot_scale = eta_dot_scale

        self.w_weight = w_weight
        self.wdot_weight = wdot_weight
        self.action_weight = action_weight
        self.productivity_weight = productivity_weight
        self.negative_ac_weight = negative_ac_weight

        self.w_scale = w_scale
        self.wdot_scale = wdot_scale
        self.w_clip = w_scale if w_clip is None else w_clip
        self.wdot_clip = wdot_scale if wdot_clip is None else wdot_clip

        self.omega_min = omega_min
        self.omega_max = omega_max
        self.ac_min = ac_min
        self.ac_max = ac_max
        self.ac_productive_target = ac_productive_target

        self.alive_bonus = alive_bonus
        self.termination_penalty = termination_penalty
        self.pass_completion_bonus = pass_completion_bonus

        if self.omega_max <= self.omega_min:
            raise ValueError("omega_max must be greater than omega_min.")

        if self.ac_max <= self.ac_min:
            raise ValueError("ac_max must be greater than ac_min.")

        if self.ac_productive_target <= 0.0:
            raise ValueError("ac_productive_target must be positive.")

    def _scale_action(self, u: np.ndarray) -> tuple[float, float]:
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        if u.size < 2:
            u_safe = np.zeros(2, dtype=np.float64)
            u_safe[: u.size] = u
            u = u_safe

        u = np.clip(u[:2], -1.0, 1.0)

        low = np.array([self.omega_min, self.ac_min], dtype=np.float64)
        high = np.array([self.omega_max, self.ac_max], dtype=np.float64)

        u_phys = low + 0.5 * (u + 1.0) * (high - low)

        return float(u_phys[0]), float(u_phys[1])

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

        w_cost, wdot_cost = _physical_vibration_cost(
            info,
            x_next,
            self.w_weight,
            self.wdot_weight,
            self.w_scale,
            self.wdot_scale,
            w_clip=self.w_clip,
            wdot_clip=self.wdot_clip,
        )

        if not np.isfinite(w_cost) or not np.isfinite(wdot_cost):
            return -float(self.termination_penalty)

        omega, ac = self._scale_action(u)

        omega_score = (omega - self.omega_min) / (self.omega_max - self.omega_min)
        omega_score = float(np.clip(omega_score, 0.0, 1.0))

        positive_ac = max(ac, 0.0)
        ac_score = positive_ac / self.ac_productive_target
        ac_score = float(np.clip(ac_score, 0.0, 1.0))

        productivity_score = omega_score * ac_score

        negative_ac = max(-ac, 0.0)
        negative_ac_cost = (negative_ac / max(self.ac_max, 1e-12)) ** 2

        action_cost = float(np.sum(np.clip(u, -1.0, 1.0) ** 2))

        reward = (
            self.alive_bonus
            + self.productivity_weight * productivity_score
            - w_cost
            - wdot_cost
            - self.negative_ac_weight * negative_ac_cost
            - self.action_weight * action_cost
        )

        if terminated:
            reason = str(info.get("termination_reason", ""))
            if reason == "pass_completed" or info.get("pass_completed"):
                reward += self.pass_completion_bonus
            else:
                reward -= self.termination_penalty

        if not np.isfinite(reward):
            reward = -float(self.termination_penalty)

        return float(np.clip(reward, -1e4, 1e4))


class DenseQuadraticPlateReward:
    """Quadratic physical-sensor vibration suppression reward."""

    def __init__(
        self,
        w_weight: float = 1.0,
        wdot_weight: float = 0.1,
        action_weight: float = 0.01,
        w_scale: float = 1e-3,
        wdot_scale: float = 1e-2,
        alive_bonus: float = 0.0,
        termination_penalty: float = 100.0,
        pass_completion_bonus: float = 500.0,
        w_clip: float | None = None,
        wdot_clip: float | None = None,
        eta_weight: float | None = None,
        eta_dot_weight: float | None = None,
        eta_scale: float | None = None,
        eta_dot_scale: float | None = None,
    ):
        if eta_weight is not None:
            w_weight = eta_weight
        if eta_dot_weight is not None:
            wdot_weight = eta_dot_weight
        if eta_scale is not None:
            w_scale = eta_scale
        if eta_dot_scale is not None:
            wdot_scale = eta_dot_scale

        self.w_weight = w_weight
        self.wdot_weight = wdot_weight
        self.action_weight = action_weight
        self.w_scale = w_scale
        self.wdot_scale = wdot_scale
        self.w_clip = w_scale if w_clip is None else w_clip
        self.wdot_clip = wdot_scale if wdot_clip is None else wdot_clip
        self.alive_bonus = alive_bonus
        self.termination_penalty = termination_penalty
        self.pass_completion_bonus = pass_completion_bonus

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

        w_cost, wdot_cost = _physical_vibration_cost(
            info,
            x_next,
            self.w_weight,
            self.wdot_weight,
            self.w_scale,
            self.wdot_scale,
            w_clip=self.w_clip,
            wdot_clip=self.wdot_clip,
        )

        if not np.isfinite(w_cost) or not np.isfinite(wdot_cost):
            return -float(self.termination_penalty)

        action_cost = float(np.sum(np.clip(u, -1.0, 1.0) ** 2))

        reward = self.alive_bonus - w_cost - wdot_cost - self.action_weight * action_cost

        if terminated:
            reason = str(info.get("termination_reason", ""))
            if reason == "pass_completed" or info.get("pass_completed"):
                reward += self.pass_completion_bonus
            else:
                reward -= self.termination_penalty

        if not np.isfinite(reward):
            reward = -float(self.termination_penalty)

        return float(np.clip(reward, -1e4, 1e4))


class SparseStablePlateReward:
    """Sparse reward for simple stability testing."""

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
    """Return Plate reward by id."""
    if reward_id not in _PLATE_REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward_id: {reward_id}. "
            f"Known reward ids: {list(_PLATE_REWARD_REGISTRY)}"
        )

    return _PLATE_REWARD_REGISTRY[reward_id](**kwargs)
