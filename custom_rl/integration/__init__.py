"""ODE / DDE integrators for plant dynamics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from custom_rl.integration.dde_rk4 import integrate_dde_rk4
from custom_rl.integration.rk4 import integrate as integrate_rk4


IntegratorFn = Callable[
    [
        Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        float,
        np.ndarray,
        np.ndarray,
        float,
        int,
    ],
    np.ndarray,
]


def get_integrator(name: str) -> IntegratorFn:
    """
    Return an integration function f(dynamics, t0, x0, u, dt, n_steps) -> x_final.

    - ``rk4``: standard fixed-step RK4 (delay sampled from accepted-step history only)
    - ``dde_rk4``: method-of-steps RK4; delayed state interpolated at every RK4 sub-stage
    """
    key = str(name).lower().strip()
    if key in {"rk4", "ode_rk4"}:
        return integrate_rk4
    if key in {"dde_rk4", "dde", "method_of_steps"}:
        return integrate_dde_rk4
    raise ValueError(f"Unknown integrator '{name}'. Use 'rk4' or 'dde_rk4'.")
