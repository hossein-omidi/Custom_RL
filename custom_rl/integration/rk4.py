"""Fixed-step RK4 integrator for ODE systems.

This module is not cutting-force-specific. The change from peripheral milling
to face milling does not require a different Runge-Kutta formula.

The only face-milling-related issue is regenerative-delay history. The new
face-milling force module stores accepted modal states for interpolation of
x(t - tau). A standard RK4 step evaluates dynamics at four trial states. Those
trial states must not be permanently added to the delay history.

Usage for the face-milling model:
    from custom_rl.plants import f_nonlinear2_face_milling

    x_next = rk4_step(
        dynamics,
        t,
        x,
        u,
        dt,
        history_module=f_nonlinear2_face_milling,
    )

If ``history_module`` is None, this behaves like an ordinary RK4 integrator.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


Array = np.ndarray
DynamicsFn = Callable[[float, Array, Array], Array]


def _validate_dt(dt: float) -> float:
    """Validate and return a positive finite time step."""
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite scalar, got {dt!r}.")
    return dt


def _as_state(x: Array, name: str = "x") -> Array:
    """Return a finite 1D float64 state vector."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_arr.size == 0:
        raise ValueError(f"{name} must be a non-empty state vector.")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return x_arr


def _as_control(u: Array) -> Array:
    """Return a finite 1D float64 control vector."""
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    if u_arr.size == 0:
        raise ValueError("u must be a non-empty control vector.")
    if not np.all(np.isfinite(u_arr)):
        raise ValueError("u contains NaN or infinite values.")
    return u_arr


def _history_list(history_module: Optional[Any]):
    """Return the module history list if available; otherwise None."""
    if history_module is None or not hasattr(history_module, "_state_history"):
        return None
    hist = getattr(history_module, "_state_history")
    if not isinstance(hist, list):
        return None
    return hist


def _history_mark(history_module: Optional[Any]) -> Optional[int]:
    """Return a cheap rollback marker for regenerative history."""
    hist = _history_list(history_module)
    if hist is None:
        return None
    return len(hist)


def _history_rollback(history_module: Optional[Any], mark: Optional[int]) -> None:
    """
    Roll back history to a previous length marker.

    This is intentionally O(number_of_added_stage_entries), not O(history_size).
    It avoids the expensive full-history copy used in the previous version.
    """
    if mark is None:
        return
    hist = _history_list(history_module)
    if hist is None:
        return
    if len(hist) > mark:
        del hist[mark:]


def _history_commit(history_module: Optional[Any], t: float, x: Array) -> None:
    """Commit an accepted state to the optional regenerative history."""
    if history_module is None:
        return

    x_copy = np.asarray(x, dtype=np.float64).reshape(-1).copy()

    if hasattr(history_module, "_append_state_history"):
        history_module._append_state_history(float(t), x_copy)
        return

    hist = _history_list(history_module)
    if hist is not None:
        hist.append((float(t), x_copy))


def _history_ensure_current_state(
    history_module: Optional[Any],
    t: float,
    x: Array,
    *,
    tol: float = 1e-14,
) -> None:
    """
    Ensure the accepted current state is available before RK stage evaluation.

    This helps the regenerative force model interpolate x(t - tau), especially
    near the beginning of an episode. It is a committed accepted state, not an
    RK trial state.
    """
    hist = _history_list(history_module)
    if hist is None:
        return

    t = float(t)
    if len(hist) == 0:
        _history_commit(history_module, t, x)
        return

    last_t = float(hist[-1][0])
    if last_t < t - tol:
        _history_commit(history_module, t, x)


def _call_dynamics_without_stage_history(
    dynamics: DynamicsFn,
    t: float,
    x: Array,
    u: Array,
    history_module: Optional[Any],
) -> Array:
    """
    Evaluate dynamics but remove any history entries created by this RK stage.

    The dynamics may still update diagnostics such as ``last_force_info``.
    Only the regenerative state history is rolled back.
    """
    mark = _history_mark(history_module)
    x_dot = np.asarray(dynamics(float(t), x, u), dtype=np.float64).reshape(-1)
    _history_rollback(history_module, mark)

    if x_dot.shape != x.shape:
        raise ValueError(
            "dynamics returned derivative with incompatible shape. "
            f"Expected {x.shape}, got {x_dot.shape}."
        )
    if not np.all(np.isfinite(x_dot)):
        raise FloatingPointError("dynamics returned NaN or infinite derivative values.")

    return x_dot


def rk4_step(
    dynamics: DynamicsFn,
    t: float,
    x: Array,
    u: Array,
    dt: float,
    *,
    history_module: Optional[Any] = "f_nonlinear2_face_milling",
    commit_history: bool = True,
) -> Array:
    """
    Single fixed-step fourth-order Runge-Kutta update.

    Parameters
    ----------
    dynamics:
        Callable f(t, x, u) -> x_dot.
    t:
        Current time [s].
    x:
        Current accepted state vector.
    u:
        Control input, held constant during the step.
    dt:
        Step size [s].
    history_module:
        Optional module/object that owns regenerative-delay history. For the
        face-milling force module, pass ``f_nonlinear2_face_milling``.
    commit_history:
        If True and ``history_module`` is supplied, commit only accepted states
        to the regenerative history.

    Returns
    -------
    x_next:
        Accepted state after one RK4 step.

    Notes
    -----
    With ``history_module=None`` this is a standard RK4 implementation.

    With a regenerative history module, RK intermediate trial states are
    evaluated but rolled back, and only the accepted state at ``t + dt`` is
    committed. This prevents delay-history pollution without copying the whole
    history at every RK stage.
    """
    dt = _validate_dt(dt)
    t = float(t)
    x = _as_state(x, "x")
    u = _as_control(u)

    if commit_history and history_module is not None:
        _history_ensure_current_state(history_module, t, x)

    k1 = _call_dynamics_without_stage_history(dynamics, t, x, u, history_module)
    k2 = _call_dynamics_without_stage_history(
        dynamics,
        t + 0.5 * dt,
        x + 0.5 * dt * k1,
        u,
        history_module,
    )
    k3 = _call_dynamics_without_stage_history(
        dynamics,
        t + 0.5 * dt,
        x + 0.5 * dt * k2,
        u,
        history_module,
    )
    k4 = _call_dynamics_without_stage_history(
        dynamics,
        t + dt,
        x + dt * k3,
        u,
        history_module,
    )

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = np.asarray(x_next, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(x_next)):
        raise FloatingPointError("rk4_step produced NaN or infinite state values.")

    if commit_history and history_module is not None:
        _history_commit(history_module, t + dt, x_next)

    return x_next


def integrate(
    dynamics: DynamicsFn,
    t0: float,
    x0: Array,
    u: Array,
    dt: float,
    n_steps: int = 1,
    *,
    history_module: Optional[Any] = None,
    commit_history: bool = True,
) -> Array:
    """
    Integrate an ODE for ``n_steps`` fixed RK4 steps with constant control.

    Pass ``history_module=f_nonlinear2_face_milling`` only when the dynamics
    uses regenerative-delay history. Otherwise leave it as None.
    """
    dt = _validate_dt(dt)
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps!r}.")

    x = _as_state(x0, "x0").copy()
    u = _as_control(u)
    t = float(t0)

    for _ in range(n_steps):
        x = rk4_step(
            dynamics,
            t,
            x,
            u,
            dt,
            history_module=history_module,
            commit_history=commit_history,
        )
        t += dt

    return x


__all__ = ["rk4_step", "integrate"]
