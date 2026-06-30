"""Generic Gymnasium environment for ODE control plants.

This version is compatible with the face-milling regenerative force module.
The environment is still generic, but when the plant dynamics is backed by
``f_nonlinear2_face_milling`` it passes the force module to the RK4 integrator
as ``history_module``. This keeps the regenerative delay history clean: RK4
stage states are rolled back and only accepted states are committed.
"""

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
    """Placeholder reward. Override via reward_fn."""
    return 0.0


class ODEControlEnv(gym.Env):
    """Gymnasium environment wrapping an ODE plant with a pluggable reward."""

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
        *,
        history_module: Any | None = None,
        auto_history_module: bool = True,
    ):
        """
        Parameters
        ----------
        plant:
            ODE plant with dynamics, reset, termination, and Gym spaces.
        reward_fn:
            Callable ``reward_fn(t, x, u, x_next, terminated, truncated, info)``.
        dt:
            RK4 substep size [s].
        n_substeps:
            Number of RK4 substeps per environment step.
        max_episode_steps:
            Gymnasium time-limit truncation in environment steps.
        process_noise_std:
            Standard deviation of additive state noise after integration.
        obs_noise_std:
            Standard deviation of additive observation noise.
        history_module:
            Optional explicit regenerative-history module. For face milling this
            should be ``custom_rl.plants.f_nonlinear2_face_milling``. If omitted,
            the environment tries to detect it from ``plant.dynamics``.
        auto_history_module:
            If True, automatically detect the face-milling force module from the
            plant's dynamics method globals.
        """
        super().__init__()
        self.plant = plant
        self.reward_fn = reward_fn if reward_fn is not None else default_reward
        self.dt = float(dt)
        self.n_substeps = int(n_substeps)
        self.max_episode_steps = int(max_episode_steps)
        self._step_dt = self.dt * self.n_substeps
        self.process_noise_std = float(process_noise_std)
        self.obs_noise_std = float(obs_noise_std)
        self.history_module = history_module
        self.auto_history_module = bool(auto_history_module)

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.n_substeps <= 0:
            raise ValueError("n_substeps must be positive.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")

        self.observation_space = plant.get_observation_space()
        self.action_space = plant.get_action_space()

        self._state: np.ndarray = np.zeros(0, dtype=np.float64)
        self._t: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Regenerative-history support
    # ------------------------------------------------------------------
    def _get_history_module(self):
        """Return the regenerative-history module if the active plant uses one."""
        if self.history_module is not None:
            return self.history_module
        if not self.auto_history_module:
            return None

        # Preferred explicit plant attributes, if future plants define them.
        for attr in (
            "history_module",
            "dynamics_history_module",
            "force_module",
            "face_milling_force_module",
        ):
            module = getattr(self.plant, attr, None)
            if self._looks_like_face_milling_history_module(module):
                return module

        # Robust fallback: PlatePlant.dynamics imports the force module as the
        # global name ``f_nonlinear2``. Detect only the face-milling module.
        dynamics = getattr(self.plant, "dynamics", None)
        globals_dict = None
        if hasattr(dynamics, "__func__"):
            globals_dict = getattr(dynamics.__func__, "__globals__", None)
        elif hasattr(dynamics, "__globals__"):
            globals_dict = getattr(dynamics, "__globals__", None)

        if isinstance(globals_dict, dict):
            module = globals_dict.get("f_nonlinear2")
            if self._looks_like_face_milling_history_module(module):
                return module

        return None

    @staticmethod
    def _looks_like_face_milling_history_module(module: Any) -> bool:
        """Check for the specific face-milling regenerative-history interface."""
        return bool(
            module is not None
            and hasattr(module, "_state_history")
            and hasattr(module, "_append_state_history")
            and hasattr(module, "compute_face_milling_force")
        )

    @staticmethod
    def _replace_last_history_state(history_module: Any | None, t: float, x: np.ndarray) -> None:
        """Keep history consistent if process noise changes the accepted state."""
        if history_module is None or not hasattr(history_module, "_state_history"):
            return
        hist = getattr(history_module, "_state_history")
        if not isinstance(hist, list) or len(hist) == 0:
            return
        t_last = float(hist[-1][0])
        if abs(t_last - float(t)) < 1e-10:
            hist[-1] = (t_last, np.asarray(x, dtype=np.float64).reshape(-1).copy())

    # ------------------------------------------------------------------
    # Info / reset / step
    # ------------------------------------------------------------------
    def _build_step_info(
        self,
        term_info: dict[str, Any],
        truncated_time: bool,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "t": float(self._t),
            "x_modal": self._state.copy(),
            **term_info,
        }

        if hasattr(self.plant, "modal_to_physical"):
            w_sensor, wdot_sensor = self.plant.modal_to_physical(
                self._state, clip=True
            )
            info["w_sensor"] = w_sensor
            info["wdot_sensor"] = wdot_sensor

        if truncated_time:
            info["TimeLimit.truncated"] = True

        return info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        if hasattr(self.plant, "bind_rng"):
            self.plant.bind_rng(rng)

        reset_options = options or {}
        try:
            self._state, info = self.plant.reset(rng, options=reset_options)
        except TypeError:
            self._state, info = self.plant.reset(rng)

        self._state = np.asarray(self._state, dtype=np.float64).reshape(-1)
        self._t = 0.0
        self._step_count = 0

        obs = self.plant.state_to_obs(self._state)
        obs = self._add_obs_noise(obs)
        obs = self._clip_obs(obs)

        info = dict(info)
        info["x_modal"] = self._state.copy()
        if hasattr(self.plant, "modal_to_physical"):
            w_sensor, wdot_sensor = self.plant.modal_to_physical(
                self._state, clip=True
            )
            info["w_sensor"] = w_sensor
            info["wdot_sensor"] = wdot_sensor

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = self._clamp_action(action)

        if hasattr(self.plant, "bind_rng"):
            self.plant.bind_rng(self.np_random)

        history_module = self._get_history_module()

        x_prev = self._state.copy()
        t_prev = float(self._t)

        self._state = integrate(
            self.plant.dynamics,
            self._t,
            self._state,
            action,
            self.dt,
            n_steps=self.n_substeps,
            history_module=history_module,
        )

        self._t += self._step_dt

        # Optional process noise. If used, synchronize the last accepted history
        # state so the regenerative model sees the same state as the environment.
        if self.process_noise_std > 0.0:
            self._state = (
                self._state
                + self.process_noise_std
                * self.np_random.standard_normal(self._state.shape)
            )
            self._replace_last_history_state(history_module, self._t, self._state)

        self._step_count += 1

        terminated, truncated_term, term_info = self.plant.termination(
            self._t, self._state
        )
        truncated_time = self._step_count >= self.max_episode_steps
        truncated = bool(truncated_term or truncated_time)

        info = self._build_step_info(term_info, truncated_time)

        reward = self.reward_fn(
            t_prev,
            x_prev,
            action,
            self._state,
            terminated,
            truncated,
            info,
        )

        if hasattr(self.reward_fn, "last_reward_terms"):
            info["reward_terms"] = dict(self.reward_fn.last_reward_terms)

        obs = self.plant.state_to_obs(self._state)
        obs = self._add_obs_noise(obs)
        obs = self._clip_obs(obs)

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float64)
            high = np.asarray(self.action_space.high, dtype=np.float64)
            return np.clip(action, low, high)
        return action

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        """Clip observation to observation_space bounds for Gymnasium compliance."""
        obs = np.asarray(obs, dtype=np.float64)
        if isinstance(self.observation_space, spaces.Box):
            low = np.asarray(self.observation_space.low, dtype=np.float64)
            high = np.asarray(self.observation_space.high, dtype=np.float64)
            return np.clip(obs, low, high)
        return obs

    def _add_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add observation noise if requested."""
        obs = np.asarray(obs, dtype=np.float64)
        if self.obs_noise_std > 0.0:
            return obs + self.obs_noise_std * self.np_random.standard_normal(obs.shape)
        return obs
