"""Generic Gymnasium environment for ODE control plants."""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from custom_rl.integration import get_integrator
from custom_rl.plants.base import ODEPlant
from custom_rl.plants import f_nonlinear2


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
        integrator: str = "dde_rk4",
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
            integrator: "dde_rk4" (default) or "rk4"
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
        self._integrate = get_integrator(integrator)

        # Plant provides the sensor part of the observation.
        self._sensor_observation_space = plant.get_observation_space()
        self._sensor_obs_dim = int(np.prod(self._sensor_observation_space.shape))

        self.action_space = plant.get_action_space()
        self._prev_action_norm = np.zeros(self.action_space.shape, dtype=np.float64)

        # Full PPO observation:
        #   [sensor displacement/velocity (normalized), prev_action_norm (omega/ac)]
        if isinstance(self._sensor_observation_space, spaces.Box) and isinstance(self.action_space, spaces.Box):
            low = np.concatenate(
                [
                    np.asarray(self._sensor_observation_space.low, dtype=np.float64).reshape(-1),
                    np.asarray(self.action_space.low, dtype=np.float64).reshape(-1),
                ]
            )
            high = np.concatenate(
                [
                    np.asarray(self._sensor_observation_space.high, dtype=np.float64).reshape(-1),
                    np.asarray(self.action_space.high, dtype=np.float64).reshape(-1),
                ]
            )
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        else:
            # Fallback: keep sensor space only (should not happen for the current plate env).
            self.observation_space = self._sensor_observation_space

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
        self._prev_action_norm[:] = 0.0

        if hasattr(self.plant, "record_modal_state"):
            omega0 = float(self.plant._scale_action(self._prev_action_norm)[0])
            self.plant.record_modal_state(self._t, self._state, omega=omega0)
        else:
            f_nonlinear2.record_modal_state(self._t, self._state)

        sensor_obs = self.plant.state_to_obs(self._state)
        sensor_obs = self._add_obs_noise_sensor(sensor_obs)
        sensor_obs = self._clip_sensor_obs(sensor_obs)

        obs = np.concatenate([sensor_obs, self._prev_action_norm]).astype(np.float64, copy=False)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = self._clamp_action(action)

        x_prev = self._state.copy()
        if hasattr(self.plant, "_modal_state_history"):
            f_nonlinear2.bind_modal_history(self.plant._modal_state_history)
        try:
            t_cur = float(self._t)
            x = self._state.copy()
            omega_phys = (
                float(self.plant._scale_action(action)[0])
                if hasattr(self.plant, "_scale_action")
                else None
            )
            for _ in range(self.n_substeps):
                x = self._integrate(
                    self.plant.dynamics,
                    t_cur,
                    x,
                    action,
                    self.dt,
                    n_steps=1,
                )
                t_cur += self.dt
                if hasattr(self.plant, "record_modal_state"):
                    self.plant.record_modal_state(
                        t_cur, x, omega=omega_phys if omega_phys is not None else None
                    )
                else:
                    f_nonlinear2.record_modal_state(t_cur, x)
            self._state = x
        finally:
            f_nonlinear2.unbind_modal_history()
        # Add process noise (stochastic dynamics)
        if hasattr(self.plant, "apply_process_noise"):
            self._state = self.plant.apply_process_noise(
                self._state,
                self.np_random,
                self._step_dt,
            )
        elif self.process_noise_std > 0:
            self._state = self._state + self.process_noise_std * self.np_random.standard_normal(
                self._state.shape
            )
        self._t += self._step_dt
        self._step_count += 1

        terminated, truncated_term, term_info = self.plant.termination(
            self._t, self._state
        )
        truncated_time = self._step_count >= self.max_episode_steps
        truncated = truncated_term or truncated_time

        # Sensor reconstruction for reward/diagnostics.
        # (The PPO observation will be constructed after updating prev_action_norm.)
        # Use unclipped normalized sensor signals for reward.
        if hasattr(self.plant, "state_to_sensor_obs_norm"):
            sensor_obs_norm = self.plant.state_to_sensor_obs_norm(
                self._state,
                clip_for_observation=False,
            )
        else:
            # Fallback (older plant behavior).
            sensor_obs_norm = self.plant.state_to_obs(self._state)
        sensor_w = None
        sensor_w_dot = None
        if hasattr(self.plant, "state_to_sensor_signals"):
            sensor_w, sensor_w_dot = self.plant.state_to_sensor_signals(self._state)

        # Physical action values (for productivity terms).
        action_phys = None
        if hasattr(self.plant, "_scale_action"):
            action_phys = np.asarray(self.plant._scale_action(action), dtype=np.float64)

        reward_info: dict[str, Any] = {
            **term_info,
            "sensor_obs_norm": np.asarray(sensor_obs_norm, dtype=np.float64),
        }
        if sensor_w is not None and sensor_w_dot is not None:
            reward_info["sensor_w"] = np.asarray(sensor_w, dtype=np.float64)
            reward_info["sensor_w_dot"] = np.asarray(sensor_w_dot, dtype=np.float64)

        reward_info["action_norm"] = np.asarray(action, dtype=np.float64)
        reward_info["prev_action_norm"] = np.asarray(self._prev_action_norm, dtype=np.float64)
        reward_info["action_phys"] = (
            np.asarray(action_phys, dtype=np.float64).reshape(-1) if action_phys is not None else None
        )
        if hasattr(self.plant, "u_phys_low") and hasattr(self.plant, "u_phys_high"):
            reward_info["action_phys_low"] = np.asarray(self.plant.u_phys_low, dtype=np.float64)
            reward_info["action_phys_high"] = np.asarray(self.plant.u_phys_high, dtype=np.float64)

        reward = self.reward_fn(
            self._t - self._step_dt,
            x_prev,
            action,
            self._state,
            terminated,
            truncated,
            reward_info,
        )

        # Update prev action AFTER reward: prev_action_norm(t+1) = action(t)
        self._prev_action_norm = np.asarray(action, dtype=np.float64).copy()

        info: dict[str, Any] = {"t": self._t, **term_info}
        # Expose sensor-level diagnostics for evaluation/plotting.
        info["sensor_obs_norm"] = np.asarray(sensor_obs_norm, dtype=np.float64).copy()
        if sensor_w is not None and sensor_w_dot is not None:
            info["sensor_w"] = np.asarray(sensor_w, dtype=np.float64).copy()
            info["sensor_w_dot"] = np.asarray(sensor_w_dot, dtype=np.float64).copy()
        if action_phys is not None:
            info["action_phys"] = np.asarray(action_phys, dtype=np.float64).reshape(-1).copy()
        if hasattr(self.plant, "get_interface_metadata"):
            info["interface"] = self.plant.get_interface_metadata()
        if hasattr(self.plant, "tool_position_at"):
            x_tool, y_tool = self.plant.tool_position_at(self._t)
            info["tool_position"] = np.array([x_tool, y_tool], dtype=np.float64)
        info["action_norm"] = np.asarray(action, dtype=np.float64).copy()
        info["prev_action_norm"] = np.asarray(self._prev_action_norm, dtype=np.float64).copy()
        # reward_fn may attach breakdown into reward_info["reward_components"]
        if "reward_components" in reward_info:
            info["reward_components"] = reward_info["reward_components"]
        if truncated_time:
            info["TimeLimit.truncated"] = True

        # PPO observation (sensor part + prev action).
        sensor_obs = self.plant.state_to_obs(self._state)
        sensor_obs = self._add_obs_noise_sensor(sensor_obs)
        sensor_obs = self._clip_sensor_obs(sensor_obs)
        obs = np.concatenate([sensor_obs, self._prev_action_norm]).astype(np.float64, copy=False)
        return obs, float(reward), terminated, truncated, info

    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float64)
            high = np.asarray(self.action_space.high, dtype=np.float64)
            return np.clip(action, low, high)
        return action

    def _clip_sensor_obs(self, sensor_obs: np.ndarray) -> np.ndarray:
        """Clip sensor observation to sensor observation_space bounds."""
        if isinstance(self._sensor_observation_space, spaces.Box):
            low = np.asarray(self._sensor_observation_space.low, dtype=np.float64).reshape(-1)
            high = np.asarray(self._sensor_observation_space.high, dtype=np.float64).reshape(-1)
            return np.clip(np.asarray(sensor_obs, dtype=np.float64).reshape(-1), low, high)
        return np.asarray(sensor_obs, dtype=np.float64).reshape(-1)

    def _add_obs_noise_sensor(self, sensor_obs: np.ndarray) -> np.ndarray:
        """Add observation noise to the sensor part only."""
        if self.obs_noise_std > 0:
            return np.asarray(sensor_obs, dtype=np.float64) + self.obs_noise_std * self.np_random.standard_normal(sensor_obs.shape)
        return np.asarray(sensor_obs, dtype=np.float64)
