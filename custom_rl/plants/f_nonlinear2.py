"""Nonlinear modal-force dynamics for the plate plant (one- or two-field)."""

from __future__ import annotations

import bisect

import numpy as np

from custom_rl.plants.milling_config import MillingForceConfig
from custom_rl.plants.milling_force import (
    compute_directional_forces,
    compute_projection_vector_v,
    compute_projection_vector_w,
    one_revolution_diagnostics,
)
from custom_rl.plants.modal_state import pack_modal_derivative, split_modal_state
from custom_rl.plants.mode_ordering import iter_mode_indices
from custom_rl.plants.units import MAX_FORCE_SAFETY_N

MAX_FORCE = MAX_FORCE_SAFETY_N

# Clip |eta| before eta^3 only to avoid float overflow (>> sensor failure limit).
_ETA_CUBIC_CLIP_M = 0.05

# Globals set by PlatePlant
m_max = None
n_max = None
N = None
K = None

zeta_vec = None
lambda_vec = None
omega_vec = None
zeta_f_vec = None
lambda_f_vec = None
omega_f_vec = None

xi_base = None
delta_base = None
W_mn = None
V_mn = None
mode_basis = None
cf = None

t_original = None
x_traj = None
y_traj = None
feed_speed = None
z_contact = None

M_modal = None
M_modal_f = None
milling_cfg: MillingForceConfig = MillingForceConfig()

decimal_places = 1.0
tau_floored = None

DELAY_MODE = "constant_tau"

OMEGA_EPS = 1e-9
P_MAX = 5000
CACHE_OMEGA_DECIMALS = 1
CACHE_AC_DECIMALS = 3

_state_history: list[tuple[float, np.ndarray]] = []
_omega_history: list[tuple[float, float]] = []
_active_history: list[tuple[float, np.ndarray]] | None = None
_rk4_scratch: list[tuple[float, np.ndarray]] | None = None
_history_times_cache: np.ndarray | None = None
_history_times_key: tuple[int, float] | None = None
_omega_segment_starts_cache: list[float] | None = None
_MAX_HISTORY = 100_000

cache = {
    "initialized": False,
    "hash": None,
    "time_discrete": None,
    "b_n_series": None,
    "b_f_series": None,
    "b_vec_series": None,
}


def expected_state_dim() -> int:
    if K is None:
        raise RuntimeError("K not initialized.")
    if milling_cfg.is_feed_normal_full():
        return 4 * int(K)
    return 2 * int(K)


def bind_modal_history(history: list[tuple[float, np.ndarray]]) -> None:
    global _active_history
    _active_history = history


def unbind_modal_history() -> None:
    global _active_history
    _active_history = None


def set_rk4_scratch(scratch: list[tuple[float, np.ndarray]] | None) -> None:
    """Ephemeral RK4 stage states for delay lookup within one integrator step."""
    global _rk4_scratch
    _rk4_scratch = scratch


def _invalidate_history_times_cache() -> None:
    global _history_times_cache, _history_times_key
    _history_times_cache = None
    _history_times_key = None


def _invalidate_omega_segment_cache() -> None:
    global _omega_segment_starts_cache
    _omega_segment_starts_cache = None


def _omega_segment_starts() -> list[float]:
    global _omega_segment_starts_cache
    if _omega_segment_starts_cache is None:
        _omega_segment_starts_cache = [h[0] for h in _omega_history]
    return _omega_segment_starts_cache


def _modal_history_times(hist: list[tuple[float, np.ndarray]]) -> np.ndarray:
    global _history_times_cache, _history_times_key
    key = (len(hist), float(hist[-1][0]) if hist else 0.0)
    if _history_times_key != key or _history_times_cache is None:
        _history_times_cache = np.array([entry[0] for entry in hist], dtype=np.float64)
        _history_times_key = key
    return _history_times_cache


def _interp_history_entry(
    hist: list[tuple[float, np.ndarray]],
    times: np.ndarray,
    t_query: float,
) -> np.ndarray | None:
    dim = expected_state_dim()
    if t_query < times[0] - 1e-12:
        return None
    if abs(t_query - times[0]) <= 1e-12:
        x0 = hist[0][1]
        return x0 if x0.size == dim else None
    idx = int(np.searchsorted(times, t_query, side="right"))
    if idx <= 0:
        return None
    if idx >= len(times):
        x_last = hist[-1][1]
        return x_last if x_last.size == dim else None
    t1, x1 = hist[idx - 1]
    t2, x2 = hist[idx]
    if t2 - t1 < 1e-12:
        return x1 if x1.size == dim else None
    alpha = (t_query - t1) / (t2 - t1)
    x_interp = (1.0 - alpha) * x1 + alpha * x2
    return x_interp if x_interp.size == dim else None


def _current_history() -> list[tuple[float, np.ndarray]]:
    return _active_history if _active_history is not None else _state_history


def _current_omega_history() -> list[tuple[float, float]]:
    return _omega_history


def _two_field() -> bool:
    return milling_cfg.is_feed_normal_full()


def _split_delayed(x_delayed: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    eta_n, _, eta_f, _ = split_modal_state(x_delayed, int(K), two_field=_two_field())
    return eta_n, eta_f


def tooth_period(omega: float) -> float:
    omega = max(float(omega), OMEGA_EPS)
    return float(2.0 * np.pi / (float(N) * omega))


def theta_at(t: float) -> float:
    return _integrated_phase(0.0, float(t))


def _cubic_restoring(eta: np.ndarray, lambda_k: np.ndarray) -> np.ndarray:
    """lambda * eta^3 with argument clip to prevent overflow (numerical guard only)."""
    eta = np.asarray(eta, dtype=np.float64)
    eta_lim = np.clip(eta, -_ETA_CUBIC_CLIP_M, _ETA_CUBIC_CLIP_M)
    return np.asarray(lambda_k, dtype=np.float64) * eta_lim**3


def _get_modal_state_at_delay(t_current: float, delay: float) -> np.ndarray | None:
    if delay <= 0.0:
        return None
    t_delayed = float(t_current) - float(delay)
    if t_delayed < 0.0:
        return None
    history = _current_history()
    if history:
        times = _modal_history_times(history)
        x = _interp_history_entry(history, times, t_delayed)
        if x is not None:
            return x
        if t_delayed <= times[-1] + 1e-12:
            x_last = history[-1][1]
            return x_last if x_last.size == expected_state_dim() else None
    scratch = _rk4_scratch
    if scratch:
        s_times = np.array([entry[0] for entry in scratch], dtype=np.float64)
        if t_delayed >= s_times[0] - 1e-12:
            return _interp_history_entry(scratch, s_times, t_delayed)
    if history:
        x_last = history[-1][1]
        return x_last if x_last.size == expected_state_dim() else None
    return None


def _integrated_phase(t_from: float, t_to: float) -> float:
    if t_to <= t_from:
        return 0.0
    hist = _current_omega_history()
    if not hist:
        return 0.0
    segment_starts = _omega_segment_starts()
    phase = 0.0
    t_cursor = t_from
    while t_cursor < t_to - 1e-15:
        idx = bisect.bisect_right(segment_starts, t_cursor) - 1
        idx = max(0, min(idx, len(hist) - 1))
        omega_seg = max(float(hist[idx][1]), OMEGA_EPS)
        t_next = min(float(hist[idx + 1][0]), t_to) if idx + 1 < len(hist) else t_to
        phase += omega_seg * (t_next - t_cursor)
        t_cursor = t_next
    return phase


def delay_time(t_current: float, omega_current: float) -> float | None:
    if DELAY_MODE == "spindle_phase":
        target = 2.0 * np.pi / float(N)
        if t_current <= 0.0:
            return None
        if not _current_omega_history():
            return tooth_period(omega_current)
        t_low, t_high = 0.0, t_current
        for _ in range(60):
            t_mid = 0.5 * (t_low + t_high)
            if _integrated_phase(t_mid, t_current) > target:
                t_low = t_mid
            else:
                t_high = t_mid
        delay = t_current - t_high
        return delay if delay > 0.0 else None
    return tooth_period(omega_current)


def reset_episode_state(history: list[tuple[float, np.ndarray]] | None = None) -> None:
    global _state_history, cache, tau_floored, _omega_history
    if history is not None:
        history.clear()
        _invalidate_history_times_cache()
    else:
        _state_history = []
        _invalidate_history_times_cache()
    _omega_history = []
    _invalidate_omega_segment_cache()
    tau_floored = None
    cache["initialized"] = False
    cache["hash"] = None
    cache["time_discrete"] = None
    cache["b_n_series"] = None
    cache["b_f_series"] = None
    cache["b_vec_series"] = None


def record_omega(t: float, omega: float) -> None:
    """Record a spindle-speed segment boundary (only when omega or time knot changes)."""
    global _omega_history
    t = float(t)
    omega = max(float(omega), OMEGA_EPS)
    if not _omega_history:
        _omega_history.append((t, omega))
        _invalidate_omega_segment_cache()
        return
    last_t, last_omega = _omega_history[-1]
    if abs(last_t - t) < 1e-12:
        if abs(last_omega - omega) > 1e-12:
            _omega_history[-1] = (t, omega)
            _invalidate_omega_segment_cache()
        return
    if abs(last_omega - omega) < 1e-12 and t > last_t:
        # Constant omega between knots — open segment already covers [last_t, t_to).
        return
    _omega_history.append((t, omega))
    _invalidate_omega_segment_cache()
    if len(_omega_history) > _MAX_HISTORY:
        del _omega_history[: len(_omega_history) - _MAX_HISTORY]
        _invalidate_omega_segment_cache()


def record_modal_state(
    t: float,
    x: np.ndarray,
    history: list[tuple[float, np.ndarray]] | None = None,
) -> None:
    global _state_history
    if K is None:
        return
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    dim = expected_state_dim()
    if x.size != dim:
        return
    hist = history if history is not None else _current_history()
    t = float(t)
    for i, (ti, _) in enumerate(hist):
        if abs(ti - t) < 1e-12:
            hist[i] = (t, x.copy())
            break
    else:
        hist.append((t, x.copy()))
    _invalidate_history_times_cache()
    if len(hist) > _MAX_HISTORY:
        if history is not None:
            del hist[: len(hist) - _MAX_HISTORY]
        else:
            _state_history = hist[-_MAX_HISTORY:]
        _invalidate_history_times_cache()


def _require_initialized() -> None:
    required = {
        "m_max": m_max,
        "n_max": n_max,
        "N": N,
        "K": K,
        "zeta_vec": zeta_vec,
        "lambda_vec": lambda_vec,
        "omega_vec": omega_vec,
        "xi_base": xi_base,
        "delta_base": delta_base,
        "W_mn": W_mn,
        "cf": cf,
        "t_original": t_original,
        "x_traj": x_traj,
        "y_traj": y_traj,
        "feed_speed": feed_speed,
        "M_modal": M_modal,
        "z_contact": z_contact,
    }
    if _two_field():
        required.update(
            {
                "V_mn": V_mn,
                "omega_f_vec": omega_f_vec,
                "lambda_f_vec": lambda_f_vec,
                "zeta_f_vec": zeta_f_vec,
                "M_modal_f": M_modal_f,
            }
        )
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"f_nonlinear2 globals not initialized. Missing: {missing}")


def _safe_action(u: np.ndarray) -> tuple[float, float]:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if u.size < 2:
        raise ValueError(f"Expected physical action [omega, ac], got {u.shape}.")
    omega = max(float(u[0]) if np.isfinite(u[0]) else OMEGA_EPS, OMEGA_EPS)
    ac = float(u[1]) if np.isfinite(u[1]) else 0.0
    return omega, ac


def _make_cache_hash(omega: float, ac: float) -> tuple:
    return (
        round(float(omega), CACHE_OMEGA_DECIMALS),
        round(float(ac), CACHE_AC_DECIMALS),
        float(np.sum(x_traj)),
        float(np.sum(y_traj)),
        milling_cfg.milling_type,
        milling_cfg.phi_st,
        milling_cfg.phi_ex,
        milling_cfg.displacement_model,
    )


def _make_time_grid(t_start: float, t_end: float, dt_grid: float) -> np.ndarray:
    if dt_grid <= 0.0 or t_end <= 0.0 or t_start >= t_end:
        return np.array([max(t_end, 0.0)], dtype=np.float64)
    estimated = int(np.floor((t_end - t_start) / dt_grid)) + 1
    if estimated > P_MAX:
        return np.linspace(t_start, t_end, P_MAX, dtype=np.float64)
    td = np.arange(t_start, t_end + dt_grid, dt_grid, dtype=np.float64)
    return td[td <= t_end] if td.size else np.array([t_end], dtype=np.float64)


def _tool_position_at(t: float) -> tuple[float, float]:
    x_c = float(np.interp(float(t), t_original, x_traj, left=x_traj[0], right=x_traj[-1]))
    y_c = float(np.interp(float(t), t_original, y_traj, left=y_traj[0], right=y_traj[-1]))
    return x_c, y_c


def _update_cache(omega: float, ac: float, current_hash) -> None:
    global tau_floored, cache
    tau = tooth_period(omega)
    tau_floored = tau
    t_end = float(t_original[-1])
    time_discrete = _make_time_grid(tau_floored, t_end, tau_floored)
    P = int(time_discrete.size)
    z_c = float(z_contact)
    x_path = np.interp(
        time_discrete,
        t_original,
        x_traj,
        left=float(x_traj[0]),
        right=float(x_traj[-1]),
    )
    y_path = np.interp(
        time_discrete,
        t_original,
        y_traj,
        left=float(y_traj[0]),
        right=float(y_traj[-1]),
    )
    if mode_basis is not None:
        b_n_series = mode_basis.w_values_batch(x_path, y_path) / float(M_modal)
        b_f_series = (
            mode_basis.v_values_batch_z(z_c, y_path) / float(M_modal_f)
            if _two_field()
            else np.zeros((K, P), dtype=np.float64)
        )
    else:
        b_n_series = np.zeros((K, P), dtype=np.float64)
        b_f_series = np.zeros((K, P), dtype=np.float64)
        for j in range(P):
            x_c, y_c = float(x_path[j]), float(y_path[j])
            b_n_series[:, j] = compute_projection_vector_w(
                x_c, y_c, W_mn, int(m_max), int(n_max), float(M_modal)
            )
            if _two_field():
                b_f_series[:, j] = compute_projection_vector_v(
                    z_c, y_c, V_mn, int(m_max), int(n_max), float(M_modal_f)
                )
    cache["time_discrete"] = time_discrete
    cache["b_n_series"] = np.nan_to_num(b_n_series, nan=0.0)
    cache["b_f_series"] = np.nan_to_num(b_f_series, nan=0.0)
    cache["b_vec_series"] = cache["b_n_series"]
    cache["hash"] = current_hash
    cache["initialized"] = True


def _interp_b_vec_series(
    t_now: float,
    time_discrete: np.ndarray,
    b_vec_series: np.ndarray,
) -> np.ndarray:
    """Vectorized linear interpolation for all modal projection coeffs at once."""
    t_now = float(t_now)
    times = np.asarray(time_discrete, dtype=np.float64)
    series = np.asarray(b_vec_series, dtype=np.float64)
    if times.size == 0:
        return np.zeros(series.shape[0], dtype=np.float64)
    if t_now <= times[0]:
        return series[:, 0].copy()
    if t_now >= times[-1]:
        return series[:, -1].copy()
    idx = int(np.searchsorted(times, t_now, side="right"))
    t0, t1 = float(times[idx - 1]), float(times[idx])
    alpha = (t_now - t0) / (t1 - t0) if t1 > t0 else 0.0
    return (1.0 - alpha) * series[:, idx - 1] + alpha * series[:, idx]


def _b_vec_at(t_now: float, field: str = "n") -> np.ndarray:
    key = "b_n_series" if field == "n" else "b_f_series"
    time_discrete = cache["time_discrete"]
    b_vec_series = cache[key]
    if time_discrete is None or b_vec_series is None or len(time_discrete) <= 1:
        x_c, y_c = _tool_position_at(t_now)
        if field == "n":
            return compute_projection_vector_w(
                x_c,
                y_c,
                W_mn,
                int(m_max),
                int(n_max),
                float(M_modal),
                mode_basis=mode_basis,
            )
        return compute_projection_vector_v(
            float(z_contact),
            y_c,
            V_mn,
            int(m_max),
            int(n_max),
            float(M_modal_f),
            mode_basis=mode_basis,
        )
    return _interp_b_vec_series(t_now, time_discrete, b_vec_series)


def _directional_force_result(
    t_now: float,
    eta_n: np.ndarray,
    eta_f: np.ndarray | None,
    omega: float,
    ac: float,
) -> dict:
    return compute_directional_forces(
        t_now,
        omega,
        ac,
        milling_cfg,
        eta_n=eta_n,
        eta_f=eta_f,
        n_teeth=int(N),
        xi_base=np.asarray(xi_base, dtype=np.float64),
        delta_base=np.asarray(delta_base, dtype=np.float64),
        w_mn=W_mn,
        v_mn=V_mn,
        m_max=int(m_max),
        n_max=int(n_max),
        feed_speed=float(feed_speed),
        theta_at=theta_at,
        tool_position=_tool_position_at,
        delay_time_fn=delay_time,
        modal_state_at_delay=_get_modal_state_at_delay,
        z_contact=float(z_contact),
        M_n=float(M_modal),
        M_f=float(M_modal_f) if M_modal_f is not None else float(M_modal),
        split_delayed_state=_split_delayed,
        max_force=MAX_FORCE,
        cf=float(cf) if cf is not None else None,
        b_vec_at=_b_vec_at,
    )


def get_revolution_diagnostics(
    t_start: float,
    omega: float,
    ac: float,
    *,
    delta_f: float = 0.0,
    delta_n: float = 0.0,
    n_samples: int = 200,
) -> dict:
    _require_initialized()
    return one_revolution_diagnostics(
        t_start,
        omega,
        ac,
        milling_cfg,
        n_teeth=int(N),
        xi_base=np.asarray(xi_base, dtype=np.float64),
        delta_base=np.asarray(delta_base, dtype=np.float64),
        feed_speed=float(feed_speed),
        theta_at=theta_at,
        tool_position=_tool_position_at,
        n_samples=n_samples,
        delta_f=delta_f,
        delta_n=delta_n,
        cf=float(cf) if cf is not None else None,
    )


def _regenerative_delta_n(t: float, eta_n: np.ndarray, omega: float) -> float:
    delay = delay_time(t, omega)
    if delay is None or delay <= 0.0:
        return 0.0
    x_d = _get_modal_state_at_delay(t, delay)
    if x_d is None:
        return 0.0
    eta_n_d, _ = _split_delayed(x_d)
    t_d = t - delay
    b_n = _b_vec_at(t, "n")
    b_n_d = _b_vec_at(t_d, "n")
    q_n = float(M_modal * np.dot(b_n, eta_n))
    q_nd = float(M_modal * np.dot(b_n_d, eta_n_d))
    return q_n - q_nd


_regenerative_delta_q = _regenerative_delta_n
_regenerative_delta_w = _regenerative_delta_n


def _scalar_cutting_force_at(
    t_now: float,
    x: np.ndarray,
    omega: float,
    ac: float,
) -> float:
    eta_n, _, eta_f, _ = split_modal_state(x, int(K), two_field=_two_field())
    res = _directional_force_result(t_now, eta_n, eta_f, omega, ac)
    if _two_field():
        return float(res["F_normal_total"])
    return float(res["F_surface_normal_total"])


def f_nonlinear2(t, x, u):
    """Modal dynamics with directional regenerative cutting force."""
    _require_initialized()
    omega, ac = _safe_action(u)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    dim = expected_state_dim()
    if x.size != dim:
        raise ValueError(f"Expected state ({dim},), got {x.shape}.")
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    eta_n, etad_n, eta_f, etad_f = split_modal_state(x, int(K), two_field=_two_field())

    current_hash = _make_cache_hash(omega, ac)
    if not cache["initialized"] or cache["hash"] != current_hash:
        _update_cache(omega, ac, current_hash)

    t_now = float(t)
    force = _directional_force_result(t_now, eta_n, eta_f, omega, ac)

    if _two_field():
        F_n = float(force["F_normal_total"])
        F_f = float(force["F_feed_total"])
        b_n = force["b_n"]
        b_f = force["b_f"]
        Fk_n = np.nan_to_num(b_n * F_n, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)
        Fk_f = np.nan_to_num(b_f * F_f, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)
        zeta_omega_n = 2.0 * zeta_vec * omega_vec
        zeta_omega_f = 2.0 * zeta_f_vec * omega_f_vec
        ddeta_n = (
            -zeta_omega_n * etad_n
            - omega_vec**2 * eta_n
            - _cubic_restoring(eta_n, lambda_vec)
            + Fk_n
        )
        ddeta_f = (
            -zeta_omega_f * etad_f
            - omega_f_vec**2 * eta_f
            - _cubic_restoring(eta_f, lambda_f_vec)
            + Fk_f
        )
        dx = pack_modal_derivative(
            etad_n, ddeta_n, etad_f, ddeta_f, two_field=True
        )
    else:
        f_sn = float(force["F_surface_normal_total"])
        b_n = force["b_n"]
        Fk_n = np.nan_to_num(b_n * f_sn, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)
        zeta_omega = 2.0 * zeta_vec * omega_vec
        ddeta_n = (
            -zeta_omega * etad_n
            - omega_vec**2 * eta_n
            - _cubic_restoring(eta_n, lambda_vec)
            + Fk_n
        )
        dx = pack_modal_derivative(etad_n, ddeta_n, None, None, two_field=False)

    return np.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)
