"""Continuous-force CartPole ODE plant."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from custom_rl.plants.base import ODEPlant


class CartPolePlant(ODEPlant):
    """
    Continuous CartPole: cart on track with pole attached.
    State: [x, x_dot, theta, theta_dot]
    - x: cart position
    - theta: pole angle from vertical (0 = upright)
    Action: force F in [-F_max, F_max]
    """

    def __init__(
        self,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.81,
        force_max: float = 10.0,
        x_limit: float = 4.8,
        theta_limit_rad: float = np.pi,
    ):
        """
        Args:
            mass_cart: cart mass
            mass_pole: pole mass
            length: half-length of pole
            gravity: gravitational constant
            force_max: max |force| on cart
            x_limit: |x| > this => terminated
            theta_limit_rad: |theta| > this => terminated
        """
        self.mc = mass_cart
        self.mp = mass_pole
        self.l = length
        self.g = gravity
        self.force_max = force_max
        self.x_limit = x_limit
        self.theta_limit_rad = theta_limit_rad

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_pos, x_dot, theta, theta_dot = x
        # u in [-1, 1]; scale to force
        u_val = float(np.asarray(u).flat[0]) if u.size else 0.0
        F = np.clip(u_val, -1.0, 1.0) * self.force_max

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        total_mass = self.mc + self.mp
        temp = (F + self.mp * self.l * theta_dot**2 * sin_th) / total_mass
        theta_acc = (self.g * sin_th - cos_th * temp) / (
            self.l * (4.0 / 3.0 - self.mp * cos_th**2 / total_mass)
        )
        x_acc = temp - self.mp * self.l * theta_acc * cos_th / total_mass

        return np.array([x_dot, x_acc, theta_dot, theta_acc], dtype=np.float64)

    def reset(self, rng) -> tuple[np.ndarray, dict[str, Any]]:
        # Start near upright with small random perturbation
        theta_0 = rng.uniform(-0.05, 0.05)
        theta_dot_0 = rng.uniform(-0.05, 0.05)
        x_0 = rng.uniform(-0.1, 0.1)
        x_dot_0 = rng.uniform(-0.1, 0.1)
        x0 = np.array([x_0, x_dot_0, theta_0, theta_dot_0], dtype=np.float64)
        return x0, {}

    def termination(self, t: float, x: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        x_pos, _, theta, _ = x
        terminated = (
            abs(x_pos) > self.x_limit or abs(theta) > self.theta_limit_rad
        )
        return terminated, False, {}

    def get_observation_space(self) -> spaces.Space:
        # Bounded obs for Gymnasium compatibility and SB3 tips
        return spaces.Box(
            low=np.array([-self.x_limit - 1.0, -100.0, -self.theta_limit_rad - 1.0, -100.0], dtype=np.float64),
            high=np.array([self.x_limit + 1.0, 100.0, self.theta_limit_rad + 1.0, 100.0], dtype=np.float64),
            shape=(4,),
            dtype=np.float64,
        )

    def get_action_space(self) -> spaces.Space:
        # Normalized [-1, 1] per SB3 best practice; env scales to force
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float64,
        )
