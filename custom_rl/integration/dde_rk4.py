"""Method-of-steps RK4 for delay differential equations (DDE).

During each RK4 sub-stage at time ``t_stage``, the dynamics RHS may depend on
state at ``t_stage - tau``.  Delayed values are obtained by linear interpolation
over:

1. Accepted modal history from previous env steps (via ``f_nonlinear2`` history)
2. A per-step scratch buffer of RK4 stage states within the current step only

Accepted global history must never contain RK predictor states; only states
recorded after a completed step belong in the plant history.
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
    scratch: list[tuple[float, np.ndarray]],
) -> np.ndarray:
    """
    Single RK4 step with DDE-aware history at each sub-stage.

    ``scratch`` holds predictor states at ``t``, ``t + 0.5*dt``, and ``t + dt``
    for delay lookup during this step only.  Accepted plant history is read via
    ``f_nonlinear2._current_history()`` (must be bound by the caller).
    """
    scratch.clear()
    hist = f_nonlinear2._current_history()
    if not hist or abs(hist[-1][0] - t) > 1e-12:
        scratch.append((t, np.asarray(x, dtype=np.float64).copy()))

    k1 = dynamics(t, x, u)
    t2, x2 = t + 0.5 * dt, x + 0.5 * dt * k1
    scratch.append((t2, np.asarray(x2, dtype=np.float64).copy()))
    k2 = dynamics(t2, x2, u)

    t3, x3 = t + 0.5 * dt, x + 0.5 * dt * k2
    scratch[-1] = (t3, np.asarray(x3, dtype=np.float64).copy())
    k3 = dynamics(t3, x3, u)

    t4, x4 = t + dt, x + dt * k3
    scratch.append((t4, np.asarray(x4, dtype=np.float64).copy()))
    k4 = dynamics(t4, x4, u)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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

    Uses accepted history from ``f_nonlinear2._current_history()`` plus an
    ephemeral per-step scratch list.  Predictor states are never written to
    the plant's global history (caller records accepted states after each step).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    t = float(t0)
    scratch: list[tuple[float, np.ndarray]] = []
    f_nonlinear2.set_rk4_scratch(scratch)
    try:
        for _ in range(n_steps):
            x = rk4_step_dde(dynamics, t, x, u, dt, scratch)
            t += dt
    finally:
        f_nonlinear2.set_rk4_scratch(None)

    return x
