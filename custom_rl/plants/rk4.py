"""Fixed-step RK4 integrator for ODE systems."""

from __future__ import annotations

import numpy as np
from typing import Callable


def rk4_step(
    dynamics: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Single RK4 step: x_next = x + dt * f(t,x,u) via Runge-Kutta 4th order.

    Args:
        dynamics: f(t, x, u) -> x_dot, state derivative
        t: current time
        x: current state
        u: control input (held constant over step)
        dt: step size

    Returns:
        x_next: state after one step
    """
    k1 = dynamics(t, x, u)
    k2 = dynamics(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = dynamics(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = dynamics(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    dynamics: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t0: float,
    x0: np.ndarray,
    u: np.ndarray,
    dt: float,
    n_steps: int = 1,
) -> np.ndarray:
    """
    Integrate ODE from t0 for n_steps with control u held constant.

    Args:
        dynamics: f(t, x, u) -> x_dot
        t0: initial time
        x0: initial state
        u: control (constant over integration)
        dt: step size per RK4 step
        n_steps: number of RK4 steps

    Returns:
        Final state after n_steps.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    t = float(t0)
    for _ in range(n_steps):
        x = rk4_step(dynamics, t, x, u, dt)
        t += dt
    return x
