"""Plate vibration reward functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from custom_rl.rewards.base import RewardFn


def _split_sensor_obs_norm(sensor_obs_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    sensor_obs_norm layout (from env/plant):
        [w_sensors_norm..., w_dot_sensors_norm...]
    """
    sensor_obs_norm = np.asarray(sensor_obs_norm, dtype=np.float64).reshape(-1)
    if sensor_obs_norm.size % 2 != 0 or sensor_obs_norm.size == 0:
        raise ValueError(
            f"sensor_obs_norm must have even, nonzero length; got size {sensor_obs_norm.size}."
        )
    n = sensor_obs_norm.size // 2
    w_norm = sensor_obs_norm[:n]
    w_dot_norm = sensor_obs_norm[n:]
    return w_norm, w_dot_norm


class SensorProductivePlateReward:
    """
    Sensor-level dense reward for productive vibration suppression.

    This reward DOES NOT use modal coordinates directly.
    It relies on env-provided sensor values via `info`.
    """

    def __init__(
        self,
        eta_weight_disp: float = 1.0,
        eta_dot_weight_vel: float = 0.1,
        action_smoothness_weight: float = 0.0,
        productivity_weight: float = 10.0,
        ac_productive_target: float = 5.0,
        alive_bonus: float = 1.0,
        termination_penalty: float = 100.0,
    ):
        self.eta_weight_disp = float(eta_weight_disp)
        self.eta_dot_weight_vel = float(eta_dot_weight_vel)
        self.action_smoothness_weight = float(action_smoothness_weight)
        self.productivity_weight = float(productivity_weight)
        self.ac_productive_target = float(ac_productive_target)
        self.alive_bonus = float(alive_bonus)
        self.termination_penalty = float(termination_penalty)

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
        sensor_obs_norm = info.get("sensor_obs_norm", None)
        if sensor_obs_norm is None:
            # If sensor info is missing, fall back to a safe penalty.
            return -float(self.termination_penalty)

        w_norm, w_dot_norm = _split_sensor_obs_norm(sensor_obs_norm)
        if not np.all(np.isfinite(w_norm)) or not np.all(np.isfinite(w_dot_norm)):
            return -float(self.termination_penalty)

        disp_cost = float(np.mean(w_norm**2))
        vel_cost = float(np.mean(w_dot_norm**2))

        # Productivity from physical spindle speed and depth of cut.
        action_phys = info.get("action_phys", None)
        action_phys_low = info.get("action_phys_low", None)
        action_phys_high = info.get("action_phys_high", None)
        if action_phys is None or action_phys_low is None or action_phys_high is None:
            productivity_score = 0.0
            omega = np.nan
            ac = np.nan
        else:
            action_phys = np.asarray(action_phys, dtype=np.float64).reshape(-1)
            action_phys_low = np.asarray(action_phys_low, dtype=np.float64).reshape(-1)
            action_phys_high = np.asarray(action_phys_high, dtype=np.float64).reshape(-1)
            omega = float(action_phys[0])
            ac = float(action_phys[1])

            omega_min = float(action_phys_low[0])
            omega_max = float(action_phys_high[0])
            ac_min = float(action_phys_low[1])

            omega_score = (omega - omega_min) / max(omega_max - omega_min, 1e-12)
            omega_score = float(np.clip(omega_score, 0.0, 1.0))

            # Productive cutting assumes non-negative ac.
            positive_ac = max(ac, 0.0)
            ac_score = float(np.clip(positive_ac / max(self.ac_productive_target, 1e-12), 0.0, 1.0))
            productivity_score = float(omega_score * ac_score)

        # Optional action smoothness penalty (difference between successive normalized actions).
        action_norm = np.asarray(info.get("action_norm", u), dtype=np.float64).reshape(-1)
        prev_action_norm = np.asarray(info.get("prev_action_norm", np.zeros_like(action_norm)), dtype=np.float64).reshape(-1)
        if action_norm.size >= 2 and prev_action_norm.size >= 2:
            action_delta = action_norm - prev_action_norm
            action_smooth_cost = float(np.mean(action_delta**2))
        else:
            action_smooth_cost = 0.0

        vibration_cost = self.eta_weight_disp * disp_cost + self.eta_dot_weight_vel * vel_cost
        reward = (
            self.alive_bonus
            + self.productivity_weight * productivity_score
            - vibration_cost
            - self.action_smoothness_weight * action_smooth_cost
        )

        failure_penalty = 0.0
        if terminated:
            failure_penalty = float(self.termination_penalty)
            reward -= failure_penalty

        # Expose breakdown for plotting/evaluation.
        info["reward_components"] = {
            "productivity_score": float(productivity_score),
            "disp_cost": float(disp_cost),
            "vel_cost": float(vel_cost),
            "vibration_cost": float(vibration_cost),
            "action_smoothness_cost": float(action_smooth_cost),
            "failure_penalty": float(failure_penalty),
            "reward_total": float(reward),
            # Optional raw values for debugging.
            "omega": float(omega) if np.isfinite(omega) else None,
            "ac": float(ac) if np.isfinite(ac) else None,
        }

        if not np.isfinite(reward):
            return -float(self.termination_penalty)

        return float(reward)


class SensorVibrationPlateReward:
    """
    Sensor-level vibration suppression reward (no explicit productivity term).
    """

    def __init__(
        self,
        eta_weight_disp: float = 1.0,
        eta_dot_weight_vel: float = 0.1,
        action_smoothness_weight: float = 0.0,
        alive_bonus: float = 1.0,
        termination_penalty: float = 100.0,
    ):
        self.eta_weight_disp = float(eta_weight_disp)
        self.eta_dot_weight_vel = float(eta_dot_weight_vel)
        self.action_smoothness_weight = float(action_smoothness_weight)
        self.alive_bonus = float(alive_bonus)
        self.termination_penalty = float(termination_penalty)

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
        sensor_obs_norm = info.get("sensor_obs_norm", None)
        if sensor_obs_norm is None:
            return -float(self.termination_penalty)

        w_norm, w_dot_norm = _split_sensor_obs_norm(sensor_obs_norm)
        if not np.all(np.isfinite(w_norm)) or not np.all(np.isfinite(w_dot_norm)):
            return -float(self.termination_penalty)

        disp_cost = float(np.mean(w_norm**2))
        vel_cost = float(np.mean(w_dot_norm**2))
        vibration_cost = self.eta_weight_disp * disp_cost + self.eta_dot_weight_vel * vel_cost

        action_norm = np.asarray(info.get("action_norm", u), dtype=np.float64).reshape(-1)
        prev_action_norm = np.asarray(info.get("prev_action_norm", np.zeros_like(action_norm)), dtype=np.float64).reshape(-1)
        if action_norm.size >= 2 and prev_action_norm.size >= 2:
            action_delta = action_norm - prev_action_norm
            action_smooth_cost = float(np.mean(action_delta**2))
        else:
            action_smooth_cost = 0.0

        reward = self.alive_bonus - vibration_cost - self.action_smoothness_weight * action_smooth_cost
        failure_penalty = 0.0
        if terminated:
            failure_penalty = float(self.termination_penalty)
            reward -= failure_penalty

        info["reward_components"] = {
            "productivity_score": 0.0,
            "disp_cost": float(disp_cost),
            "vel_cost": float(vel_cost),
            "vibration_cost": float(vibration_cost),
            "action_smoothness_cost": float(action_smooth_cost),
            "failure_penalty": float(failure_penalty),
            "reward_total": float(reward),
        }

        if not np.isfinite(reward):
            return -float(self.termination_penalty)
        return float(reward)


class SensorSparsePlateReward:
    """
    Sparse sensor-based reward: +1 each step until termination, else 0.
    """

    def __init__(self, alive_reward: float = 1.0, termination_reward: float = 0.0):
        self.alive_reward = float(alive_reward)
        self.termination_reward = float(termination_reward)

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
        reward = self.termination_reward if terminated else self.alive_reward
        info["reward_components"] = {"reward_total": float(reward)}
        return float(reward)


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
        reward = alive_bonus
               + productivity_weight * productivity_score
               - vibration_cost
               - velocity_cost
               - negative_ac_cost
               - action_regularization

    Notes:
        - productivity_score encourages nonzero productive cutting.
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
        ac_min: float = -10.0,
        ac_max: float = 10.0,
        ac_productive_target: float = 5.0,
        alive_bonus: float = 1.0,
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
        negative_ac_cost = (negative_ac / max(abs(self.ac_min), 1e-12)) ** 2

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
    # Sensor-based reward IDs used by the RL scripts.
    "dense": SensorProductivePlateReward,
    "productive": SensorProductivePlateReward,
    "quadratic": SensorVibrationPlateReward,
    "sparse": SensorSparsePlateReward,
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