"""Generic Gymnasium environment for ODE control plants."""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from custom_rl.integration.rk4 import integrate
from custom_rl.plants.base import ODEPlant


def default_reward(
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    x_next: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> float:
    """Placeholder reward (e.g. 0 or 1 per step). Override via reward_fn."""
    return 0.0


class ODEControlEnv(gym.Env):
    """
    Gymnasium environment wrapping an ODE plant with pluggable reward.
    Compatible with register, make, wrappers, and SB3.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        plant: ODEPlant,
        reward_fn: Optional[
            Callable[
                [float, np.ndarray, np.ndarray, np.ndarray, bool, bool, dict],
                float,
            ]
        ] = None,
        dt: float = 0.02,
        n_substeps: int = 1,
        max_episode_steps: int = 500,
        process_noise_std: float = 0.0,
        obs_noise_std: float = 0.0,
    ):
        """
        Args:
            plant: ODE plant with dynamics, reset, termination, spaces
            reward_fn: callable(t, x, u, x_next, terminated, truncated, info) -> reward
            dt: integration step size
            n_substeps: number of RK4 steps per env step
            max_episode_steps: truncation after this many steps
            process_noise_std: std of Gaussian noise added to state after dynamics (0 = deterministic)
            obs_noise_std: std of Gaussian noise added to observations (0 = perfect observation)
        """
        super().__init__()
        self.plant = plant
        self.reward_fn = reward_fn if reward_fn is not None else default_reward
        self.dt = dt
        self.n_substeps = n_substeps
        self.max_episode_steps = max_episode_steps
        self._step_dt = dt * n_substeps
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std

        self.observation_space = plant.get_observation_space()
        self.action_space = plant.get_action_space()

        self._state: np.ndarray = np.zeros(0)
        self._t: float = 0.0
        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # Use env's np_random (set by super) for determinism with reset() vs reset(seed=X)
        rng = self.np_random
        self._state, info = self.plant.reset(rng)
        self._t = 0.0
        self._step_count = 0
        obs = self.plant.state_to_obs(self._state)
        obs = self._add_obs_noise(obs)
        obs = self._clip_obs(obs)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = self._clamp_action(action)

        x_prev = self._state.copy()
        self._state = integrate(
            self.plant.dynamics,
            self._t,
            self._state,
            action,
            self.dt,
            n_steps=self.n_substeps,
        )
        # Add process noise (stochastic dynamics)
        if self.process_noise_std > 0:
            self._state = self._state + self.process_noise_std * self.np_random.standard_normal(self._state.shape)
        self._t += self._step_dt
        self._step_count += 1

        terminated, truncated_term, term_info = self.plant.termination(
            self._t, self._state
        )
        truncated_time = self._step_count >= self.max_episode_steps
        truncated = truncated_term or truncated_time

        reward = self.reward_fn(
            self._t - self._step_dt,
            x_prev,
            action,
            self._state,
            terminated,
            truncated,
            term_info,
        )

        info: dict[str, Any] = {"t": self._t, "state": self._state.copy(), **term_info}
        if truncated_time:
            info["TimeLimit.truncated"] = True

        obs = self.plant.state_to_obs(self._state)
        obs = self._add_obs_noise(obs)
        obs = self._clip_obs(obs)
        return obs, float(reward), terminated, truncated, info

    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float64)
            high = np.asarray(self.action_space.high, dtype=np.float64)
            return np.clip(action, low, high)
        return action

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        """Clip observation to observation_space bounds (Gymnasium compliance)."""
        if isinstance(self.observation_space, spaces.Box):
            low = np.asarray(self.observation_space.low, dtype=np.float64)
            high = np.asarray(self.observation_space.high, dtype=np.float64)
            return np.clip(np.asarray(obs, dtype=np.float64), low, high)
        return np.asarray(obs, dtype=np.float64)

    def _add_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add observation noise (stochastic observations)."""
        if self.obs_noise_std > 0:
            return obs + self.obs_noise_std * self.np_random.standard_normal(obs.shape)
        return obs
