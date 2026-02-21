"""ODE plant interface for generic control systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import spaces


class ODEPlant(ABC):
    """
    Abstract interface for an ODE plant (dynamic system).
    Implement this to define a new plant without changing env or training code.
    """

    @abstractmethod
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State derivative: dx/dt = dynamics(t, x, u).

        Args:
            t: time
            x: state vector
            u: control input

        Returns:
            x_dot: state derivative
        """
        pass

    @abstractmethod
    def reset(self, rng) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Sample initial state.

        Args:
            rng: random number generator (np.random.Generator or compatible with .uniform())

        Returns:
            x0: initial state
            info: optional reset info
        """
        pass

    @abstractmethod
    def termination(self, t: float, x: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        """
        Check if episode should terminate or truncate.

        Args:
            t: current time
            x: current state

        Returns:
            terminated: True if episode ends due to failure/goal
            truncated: True if episode ends due to time limit
            info: optional termination info
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Return Gymnasium observation space."""
        pass

    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """Return Gymnasium action space."""
        pass

    def state_to_obs(self, x: np.ndarray) -> np.ndarray:
        """
        Map internal state to observation. Override if obs != state.

        Default: observation is the state itself.
        """
        return np.asarray(x, dtype=np.float64)
