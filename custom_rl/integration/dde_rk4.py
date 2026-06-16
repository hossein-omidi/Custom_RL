"""Method-of-steps RK4 for delay differential equations (DDE).

During each RK4 sub-stage at time ``t_stage``, the dynamics RHS may depend on
state at ``t_stage - tau``.  Delayed values are obtained by linear interpolation
over:

1. Accepted modal history from previous env steps (via ``f_nonlinear2`` history)
2. A per-step scratch buffer of RK4 stage states within the current step
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from custom_rl.plants import f_nonlinear2


def rk4_step_dde(
    dynamics: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    scratch_history: list[tuple[float, np.ndarray]],
) -> np.ndarray:
    """Single RK4 step with DDE-aware history at each sub-stage."""
    f_nonlinear2.bind_modal_history(scratch_history)
    try:
        k1 = dynamics(t, x, u)
        t2, x2 = t + 0.5 * dt, x + 0.5 * dt * k1
        scratch_history.append((t2, np.asarray(x2, dtype=np.float64).copy()))
        k2 = dynamics(t2, x2, u)

        t3, x3 = t + 0.5 * dt, x + 0.5 * dt * k2
        scratch_history[-1] = (t3, np.asarray(x3, dtype=np.float64).copy())
        k3 = dynamics(t3, x3, u)

        t4, x4 = t + dt, x + dt * k3
        scratch_history.append((t4, np.asarray(x4, dtype=np.float64).copy()))
        k4 = dynamics(t4, x4, u)

        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    finally:
        f_nonlinear2.unbind_modal_history()


def integrate_dde_rk4(
    dynamics: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t0: float,
    x0: np.ndarray,
    u: np.ndarray,
    dt: float,
    n_steps: int = 1,
) -> np.ndarray:
    """
    Integrate a DDE plant for ``n_steps`` using method-of-steps RK4.

    Requires ``f_nonlinear2`` accepted-step history to be bound on the plant
    before calling; scratch states for the current step are appended temporarily.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    t = float(t0)
    base_history = f_nonlinear2._current_history()

    for _ in range(n_steps):
        scratch: list[tuple[float, np.ndarray]] = list(base_history)
        scratch.append((t, x.copy()))
        x = rk4_step_dde(dynamics, t, x, u, dt, scratch)
        t += dt

    return x
