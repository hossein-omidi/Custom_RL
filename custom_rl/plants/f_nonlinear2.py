"""Efficient nonlinear modal-force dynamics for the plate plant.

Public interface::

    f_nonlinear2(t, x, u)

where ``t`` is time, ``x`` is modal state ``[eta1, eta1_dot, ...]``, and
``u`` is physical action ``[omega, ac]``.

Structural dynamics (per mode k, **current** state only)::

    eta_ddot_k = -2*zeta_k*omega_k*eta_dot_k
                 - omega_k^2*eta_k
                 - lambda_k*eta_k^3
                 + F_k(t)

Regenerative chatter enters through the **cutting force**, not delayed
structural stiffness/damping. Tooth-period delay::

    tau = 2*pi / (N*omega)

Contact displacement from modal coordinates::

    w_c(t)     = sum_k W_k(x_c(t),     y_c(t))     * eta_k(t)
    w_c(t-tau) = sum_k W_k(x_c(t-tau), y_c(t-tau)) * eta_k(t-tau)
    Delta_w    = w_c(t) - w_c(t-tau)    (0 if t < tau or no history)

Cutting force (geometry cached; regeneration at runtime)::

    h_geom(t)     = path-based nominal chip thickness (cached)
    Delta_w_c     = w_c(t) - w_c(t_delay)
    h_total       = h_geom + Delta_w_c   (zero force if h_total <= 0 or ac <= 0)
    F_scalar      = polynomial(|Ft(h)|, |Fr(h)|) with ac-scaled coefficients
    F_k(t)        = W_k(x_c(t), y_c(t)) / M_modal * F_scalar

Startup convention: for ``t < tau`` or missing delayed history,
``Delta_w = 0`` (no regenerative feedback yet).
"""

from __future__ import annotations

import bisect

import numpy as np

from custom_rl.plants.mode_ordering import iter_mode_indices


# ============================================================
# Globals set by PlatePlant
# ============================================================

m_max = None
n_max = None
N = None
K = None

zeta_vec = None
lambda_vec = None
omega_vec = None

xi_base = None
delta_base = None
W_mn = None
cf = None

t_original = None
x_traj = None
y_traj = None

M_modal = None

decimal_places = 1.0
tau_floored = None

# Dimensionless scale for regenerative chip-thickness → force coupling.
REGENERATIVE_GAIN_SCALE = 1.0

# Delay model:
#   "constant_tau"  — tau = 2*pi/(N*omega_current)  [local constant-speed approx]
#   "spindle_phase" — find t_delay s.t. theta(t)-theta(t_delay)=2*pi/N
DELAY_MODE = "constant_tau"


# ============================================================
# RL-safety / efficiency constants
# ============================================================

OMEGA_EPS = 1e-9
P_MAX = 5000
CACHE_OMEGA_DECIMALS = 1
CACHE_AC_DECIMALS = 3
MAX_FORCE = 1e5


# ============================================================
# Modal state history (tooth-period delay for regeneration only)
# ============================================================

_state_history: list[tuple[float, np.ndarray]] = []
_omega_history: list[tuple[float, float]] = []
_active_history: list[tuple[float, np.ndarray]] | None = None
_MAX_HISTORY = 100_000


def bind_modal_history(history: list[tuple[float, np.ndarray]]) -> None:
    """Use per-plant history during dynamics (supports multiple envs in one process)."""
    global _active_history
    _active_history = history


def unbind_modal_history() -> None:
    """Stop routing history updates to a plant-local buffer."""
    global _active_history
    _active_history = None


def _current_history() -> list[tuple[float, np.ndarray]]:
    return _active_history if _active_history is not None else _state_history


def _current_omega_history() -> list[tuple[float, float]]:
    return _omega_history


def tooth_period(omega: float) -> float:
    """Regenerative delay tau = 2*pi / (N*omega) [s]."""
    omega = max(float(omega), OMEGA_EPS)
    return float(2.0 * np.pi / (float(N) * omega))


def _get_modal_state_at_delay(t_current: float, delay: float) -> np.ndarray | None:
    """
    Interpolate modal state at ``t_current - delay`` from episode history.

    Returns None if ``delay <= 0``, ``t_current < delay``, or history is empty.
    """
    if delay <= 0.0:
        return None

    t_delayed = t_current - delay
    if t_delayed < 0.0:
        return None

    history = _current_history()
    if len(history) == 0:
        return None

    times = [entry[0] for entry in history]
    idx = bisect.bisect_left(times, t_delayed)

    if idx == 0:
        return None
    if idx == len(times):
        return history[-1][1]

    t1, x1 = history[idx - 1]
    t2, x2 = history[idx]
    if t2 - t1 < 1e-12:
        return x1
    alpha = (t_delayed - t1) / (t2 - t1)
    return (1.0 - alpha) * x1 + alpha * x2


def _omega_at_time(t_query: float) -> float:
    """Piecewise-constant omega from recorded spindle-speed history."""
    hist = _current_omega_history()
    if not hist:
        return OMEGA_EPS
    times = [entry[0] for entry in hist]
    idx = bisect.bisect_right(times, t_query) - 1
    if idx < 0:
        return max(float(hist[0][1]), OMEGA_EPS)
    return max(float(hist[idx][1]), OMEGA_EPS)


def _integrated_phase(t_from: float, t_to: float) -> float:
    """Integral of omega dt from t_from to t_to (t_to >= t_from)."""
    if t_to <= t_from:
        return 0.0
    hist = _current_omega_history()
    if not hist:
        return 0.0

    phase = 0.0
    segment_starts = [h[0] for h in hist]
    t_cursor = t_from
    while t_cursor < t_to - 1e-15:
        idx = bisect.bisect_right(segment_starts, t_cursor) - 1
        idx = max(0, min(idx, len(hist) - 1))
        omega_seg = max(float(hist[idx][1]), OMEGA_EPS)
        if idx + 1 < len(hist):
            t_next = min(float(hist[idx + 1][0]), t_to)
        else:
            t_next = t_to
        phase += omega_seg * (t_next - t_cursor)
        t_cursor = t_next
    return phase


def delay_time(t_current: float, omega_current: float) -> float | None:
    """
    Regenerative delay interval [s] for lookup of delayed state at t_current - delay.

    constant_tau: delay = 2*pi/(N*omega_current)
    spindle_phase: find t_prev < t_current with theta(t_current)-theta(t_prev)=2*pi/N
    """
    if DELAY_MODE == "spindle_phase":
        target = 2.0 * np.pi / float(N)
        if t_current <= 0.0:
            return None
        hist = _current_omega_history()
        if not hist:
            return tooth_period(omega_current)

        t_low = 0.0
        t_high = t_current
        for _ in range(60):
            t_mid = 0.5 * (t_low + t_high)
            phase = _integrated_phase(t_mid, t_current)
            if phase > target:
                t_low = t_mid
            else:
                t_high = t_mid
        delay = t_current - t_high
        return delay if delay > 0.0 else None

    # Default: local constant-speed tooth period.
    return tooth_period(omega_current)


# ============================================================
# Cache (geometry / action dependent force magnitude only)
# ============================================================

cache = {
    "initialized": False,
    "hash": None,
    "time_discrete": None,
    "F_geom": None,
    "b_vec_series": None,
    "h_geom_series": None,
}


def reset_episode_state(
    history: list[tuple[float, np.ndarray]] | None = None,
) -> None:
    """Clear per-episode history and force cache."""
    global _state_history, cache, tau_floored, _omega_history

    if history is not None:
        history.clear()
    else:
        _state_history = []
    _omega_history = []
    tau_floored = None
    cache["initialized"] = False
    cache["hash"] = None
    cache["time_discrete"] = None
    cache["F_geom"] = None
    cache["b_vec_series"] = None
    cache["h_geom_series"] = None


def record_omega(t: float, omega: float) -> None:
    """Store spindle speed at accepted simulation time (piecewise-constant per step)."""
    global _omega_history
    t = float(t)
    omega = max(float(omega), OMEGA_EPS)
    if _omega_history and abs(_omega_history[-1][0] - t) < 1e-12:
        _omega_history[-1] = (t, omega)
    else:
        _omega_history.append((t, omega))
    if len(_omega_history) > _MAX_HISTORY:
        del _omega_history[: len(_omega_history) - _MAX_HISTORY]


def record_modal_state(
    t: float,
    x: np.ndarray,
    history: list[tuple[float, np.ndarray]] | None = None,
) -> None:
    """
    Store accepted modal state at simulation time t for regenerative delay.

    Must be called once per environment integration step (after RK4), NOT inside
    intermediate RK4 stages. ODEControlEnv handles this automatically.
    """
    global _state_history

    if K is None:
        return

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != 2 * K:
        return

    hist = history if history is not None else _current_history()
    t = float(t)
    if hist and abs(hist[-1][0] - t) < 1e-12:
        hist[-1] = (t, x.copy())
    else:
        hist.append((t, x.copy()))

    if len(hist) > _MAX_HISTORY:
        if history is not None:
            del hist[: len(hist) - _MAX_HISTORY]
        else:
            _state_history = hist[-_MAX_HISTORY:]


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
        "M_modal": M_modal,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "f_nonlinear2 globals are not initialized. "
            f"Missing: {missing}. "
            "Make sure PlatePlant.__init__ sets them before dynamics is called."
        )


def _safe_action(u: np.ndarray) -> tuple[float, float]:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if u.size < 2:
        raise ValueError(
            f"f_nonlinear2 expects physical action [omega, ac], got shape {u.shape}."
        )

    omega = float(u[0])
    ac = float(u[1])
    if not np.isfinite(omega):
        omega = OMEGA_EPS
    if not np.isfinite(ac):
        ac = 0.0
    omega = max(omega, OMEGA_EPS)
    return omega, ac


def _make_cache_hash(omega: float, ac: float) -> tuple[float, float, float, float, float, float]:
    omega_key = round(float(omega), CACHE_OMEGA_DECIMALS)
    ac_key = round(float(ac), CACHE_AC_DECIMALS)
    return (
        omega_key,
        ac_key,
        float(np.sum(x_traj)),
        float(np.sum(y_traj)),
        float(np.sum(xi_base)),
        float(np.sum(delta_base)),
    )


def _make_time_grid(t_start: float, t_end: float, dt_grid: float) -> np.ndarray:
    if not np.isfinite(t_start) or not np.isfinite(t_end) or not np.isfinite(dt_grid):
        return np.array([0.0], dtype=np.float64)
    if dt_grid <= 0.0 or t_end <= 0.0:
        return np.array([0.0], dtype=np.float64)
    if t_start >= t_end:
        return np.array([t_end], dtype=np.float64)

    estimated_points = int(np.floor((t_end - t_start) / dt_grid)) + 1
    if estimated_points <= 1:
        return np.array([t_start], dtype=np.float64)
    if estimated_points > P_MAX:
        return np.linspace(t_start, t_end, P_MAX, dtype=np.float64)

    time_discrete = np.arange(t_start, t_end + dt_grid, dt_grid, dtype=np.float64)
    time_discrete = time_discrete[time_discrete <= t_end]
    return time_discrete if time_discrete.size else np.array([t_end], dtype=np.float64)


def _compute_b_vec_at(x_c: float, y_c: float) -> np.ndarray:
    """Modal projection vector: F_k = b_k * F_scalar, b_k = W_k(x_c,y_c)/M_modal."""
    b_vec = np.zeros(K, dtype=np.float64)
    for m, n, k in iter_mode_indices(m_max, n_max):
        b_vec[k] = float(W_mn[m][n](x_c, y_c)) / M_modal
    return b_vec


def _tool_position_at(t: float) -> tuple[float, float]:
    x_c = float(np.interp(float(t), t_original, x_traj, left=x_traj[0], right=x_traj[-1]))
    y_c = float(np.interp(float(t), t_original, y_traj, left=y_traj[0], right=y_traj[-1]))
    return x_c, y_c


def _contact_displacement(eta: np.ndarray, t: float) -> float:
    """Physical out-of-plane displacement at the cutter contact point."""
    eta = np.asarray(eta, dtype=np.float64).reshape(-1)
    x_c, y_c = _tool_position_at(t)
    w_c = 0.0
    for m, n, k in iter_mode_indices(m_max, n_max):
        w_c += float(W_mn[m][n](x_c, y_c)) * float(eta[k])
    return w_c


def _regenerative_delta_w(t: float, eta: np.ndarray, omega: float) -> float:
    """
    Regenerative chip-thickness / surface-displacement difference.

    Delta_w = w_c(t) - w_c(t_delay). Zero during startup or if history unavailable.
    """
    delay = delay_time(t, omega)
    if delay is None or delay <= 0.0:
        return 0.0

    x_delayed = _get_modal_state_at_delay(t, delay)
    if x_delayed is None:
        return 0.0

    eta_delayed = x_delayed[0::2]
    t_delay = t - delay
    w_now = _contact_displacement(eta, t)
    w_delayed = _contact_displacement(eta_delayed, t_delay)
    return w_now - w_delayed


def _cutting_force_scalar_from_h(h: float, ac: float) -> float:
    """
    Scalar cutting-force magnitude from chip thickness h and depth of cut ac.

    Ft = sum xi_i(ac) * h^i,  Fr = sum delta_i(ac) * h^i  (i=1..4 as cubic poly)
    Returns 0 if ac <= 0 or h <= 0 (loss of contact / no cutting).
    """
    if ac <= 0.0 or h <= 0.0:
        return 0.0

    h = float(h)
    ac_xi = np.asarray(xi_base, dtype=np.float64) * ac
    ac_delta = np.asarray(delta_base, dtype=np.float64) * ac
    h2, h3 = h * h, h * h * h

    ft = float(ac_xi[0] * h3 + ac_xi[1] * h2 + ac_xi[2] * h + ac_xi[3])
    fr = float(ac_delta[0] * h3 + ac_delta[1] * h2 + ac_delta[2] * h + ac_delta[3])
    f_mag = float(np.sqrt(ft * ft + fr * fr))
    return float(np.clip(f_mag, 0.0, MAX_FORCE))


def _regenerative_force_gain(ac: float) -> float:
    """
    Cutting-force sensitivity to chip-thickness variation [N/m].

    Proportional to depth of cut; zero when ac <= 0 (no cutting).
    """
    if ac <= 0.0:
        return 0.0
    ac_delta = np.asarray(delta_base, dtype=np.float64) * ac
    norm_delta = float(np.linalg.norm(ac_delta))
    return REGENERATIVE_GAIN_SCALE * float(cf) * norm_delta / max(float(M_modal), 1e-12)


def _update_cache(omega: float, ac: float, current_hash) -> None:
    """Precompute geometry/action-dependent cutting-force magnitude F_geom(t)."""
    global tau_floored, cache

    cf_m_per_rad = cf / 1000.0 / (2.0 * np.pi)
    tau = tooth_period(omega)
    tau_floored = tau

    max_iterations = 3
    iteration = 0
    max_displacement = np.inf
    local_decimal_places = 1.0
    t_end = float(t_original[-1])
    time_discrete = np.array([0.0], dtype=np.float64)
    threshold = cf_m_per_rad * N * omega * tau

    while max_displacement > threshold and iteration < max_iterations:
        scale = max(local_decimal_places, 1.0)
        tau_floored = tau_floored / scale
        if not np.isfinite(tau_floored) or tau_floored <= 0.0:
            tau_floored = max(t_end / P_MAX, OMEGA_EPS)

        time_discrete = _make_time_grid(
            t_start=tau_floored,
            t_end=t_end,
            dt_grid=tau_floored,
        )

        x_discrete = np.interp(time_discrete, t_original, x_traj)
        y_discrete = np.interp(time_discrete, t_original, y_traj)

        if x_discrete.size > 1:
            dx_tmp = np.diff(x_discrete)
            dy_tmp = np.diff(y_discrete)
            max_displacement = float(np.max(np.sqrt(dx_tmp * dx_tmp + dy_tmp * dy_tmp)))
        else:
            max_displacement = 0.0

        if max_displacement > threshold:
            local_decimal_places += 1.0
        else:
            break
        iteration += 1

    ac_xi = xi_base * ac
    ac_delta = delta_base * ac
    xi1, xi2, xi3, xi4 = ac_xi[0], ac_xi[1], ac_xi[2], ac_xi[3]
    d1, d2, d3, d4 = ac_delta[0], ac_delta[1], ac_delta[2], ac_delta[3]

    pi_34 = 2.356194490192345
    pi_12 = 1.570796326794897
    pi_14 = 0.785398163397448

    alpha1 = 0.25 * (xi1 + pi_34 * d1)
    beta1 = 0.25 * (d1 + pi_34 * xi1)
    alpha2 = (1.0 / 3.0) * (xi2 + 2.0 * d2)
    beta2 = (1.0 / 3.0) * (d2 + 2.0 * xi2)
    alpha3 = 0.5 * (xi3 + pi_12 * d3)
    beta3 = 0.5 * (d3 + pi_12 * xi3)

    gamma1 = 0.25 * (d1 + pi_14 * xi1)
    gamma2 = 0.25 * (xi1 + pi_14 * d1)
    gamma3 = (1.0 / 3.0) * (d2 + xi2)
    gamma4 = xi4 + d4

    alpha_p1 = 0.25 * (-d1 + pi_34 * xi1)
    beta_p1 = 0.25 * (xi1 - pi_34 * d1)
    alpha_p2 = (1.0 / 3.0) * (-d2 + 2.0 * xi2)
    beta_p2 = (1.0 / 3.0) * (xi2 - 2.0 * d2)
    alpha_p3 = 0.5 * (-d3 + pi_12 * xi3)
    beta_p3 = 0.5 * (xi3 - pi_12 * d3)

    gamma_p1 = 0.25 * (xi1 - pi_14 * d1)
    gamma_p2 = 0.25 * (-d1 + pi_14 * xi1)
    gamma_p3 = (1.0 / 3.0) * (xi2 - d2)
    gamma_p4 = -d4 + xi4

    P = int(time_discrete.size)
    Fx = np.zeros(P, dtype=np.float64)
    Fy = np.zeros(P, dtype=np.float64)

    if P > 1:
        x_interp = np.interp(time_discrete, t_original, x_traj)
        y_interp = np.interp(time_discrete, t_original, y_traj)

        dx = np.diff(x_interp, prepend=0.0)
        dy = np.diff(y_interp, prepend=0.0)
        dx2, dx3 = dx * dx, dx * dx * dx
        dy2, dy3 = dy * dy, dy * dy * dy
        dxy = dx * dy
        N_2pi = N / (2.0 * np.pi)

        Fx = -N_2pi * (
            alpha1 * dx3 + beta1 * dy3 + alpha2 * dx2 + beta2 * dy2
            + alpha3 * dx + beta3 * dy
            + 3.0 * gamma1 * dx2 * dy + 3.0 * gamma2 * dx * dy2
            + 2.0 * gamma3 * dxy + gamma4
        )
        Fy = +N_2pi * (
            alpha_p1 * dx3 + beta_p1 * dy3 + alpha_p2 * dx2 + beta_p2 * dy2
            + alpha_p3 * dx + beta_p3 * dy
            + 3.0 * gamma_p1 * dx2 * dy + 3.0 * gamma_p2 * dx * dy2
            + 2.0 * gamma_p3 * dxy + gamma_p4
        )
        Fx = np.sign(Fx) * np.minimum(np.abs(Fx), MAX_FORCE)
        Fy = np.sign(Fy) * np.minimum(np.abs(Fy), MAX_FORCE)

    F_geom = np.sqrt(Fx * Fx + Fy * Fy)
    F_geom = np.nan_to_num(F_geom, nan=0.0, posinf=MAX_FORCE, neginf=0.0)

    h_geom_series = np.zeros(P, dtype=np.float64)
    if P > 1:
        h_geom_series = np.sqrt(dx * dx + dy * dy)
        h_geom_series = np.maximum(h_geom_series, 0.0)

    b_vec_series = np.zeros((K, P), dtype=np.float64)
    for j in range(P):
        t_j = float(time_discrete[j])
        x_c, y_c = _tool_position_at(t_j)
        b_vec_series[:, j] = _compute_b_vec_at(x_c, y_c)

    b_vec_series = np.nan_to_num(b_vec_series, nan=0.0, posinf=0.0, neginf=0.0)

    cache["time_discrete"] = time_discrete
    cache["F_geom"] = F_geom
    cache["h_geom_series"] = h_geom_series
    cache["b_vec_series"] = b_vec_series
    cache["hash"] = current_hash
    cache["initialized"] = True


def _geometry_at(t_now: float) -> tuple[float, float, np.ndarray]:
    """Interpolate cached h_geom, F_geom (legacy), and b_vec at simulation time."""
    time_discrete = cache["time_discrete"]
    F_geom = cache["F_geom"]
    h_geom_series = cache["h_geom_series"]
    b_vec_series = cache["b_vec_series"]

    if (
        time_discrete is None
        or b_vec_series is None
        or len(time_discrete) <= 1
    ):
        return 0.0, 0.0, np.zeros(K, dtype=np.float64)

    h_geom = 0.0
    if h_geom_series is not None:
        h_geom = float(
            np.interp(t_now, time_discrete, h_geom_series, left=0.0, right=0.0)
        )

    f_geom = 0.0
    if F_geom is not None:
        f_geom = float(np.interp(t_now, time_discrete, F_geom, left=0.0, right=0.0))

    b_vec = np.array(
        [
            float(np.interp(t_now, time_discrete, b_vec_series[k], left=0.0, right=0.0))
            for k in range(K)
        ],
        dtype=np.float64,
    )
    return h_geom, f_geom, b_vec


def _geometry_force_at(t_now: float) -> tuple[float, np.ndarray]:
    """Legacy helper: geometry-only force magnitude and b_vec."""
    _, f_geom, b_vec = _geometry_at(t_now)
    return f_geom, b_vec


def _scalar_cutting_force_at(
    t_now: float,
    eta: np.ndarray,
    omega: float,
    ac: float,
) -> float:
    """
    Total scalar cutting force from geometry + regenerative chip thickness.

    h_total = h_geom(t) + Delta_w_c; force from polynomial in h_total.
    Dynamic regenerative part vanishes when eta(t) == eta(t_delay).
    """
    if ac <= 0.0:
        return 0.0

    h_geom, _, _ = _geometry_at(t_now)
    delta_w = _regenerative_delta_w(t_now, eta, omega)
    h_total = h_geom + delta_w
    return _cutting_force_scalar_from_h(h_total, ac)


def f_nonlinear2(t, x, u):
    """
    Nonlinear plate dynamics with regenerative cutting force.

    Structural terms use the **current** modal state. Regenerative delay
    tau = 2*pi/(N*omega) enters only through Delta_w in the cutting force.
    """
    _require_initialized()

    omega, ac = _safe_action(u)

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != 2 * K:
        raise ValueError(
            f"f_nonlinear2 expects state shape ({2 * K},), but received {x.shape}."
        )
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    eta = x[0::2]
    etad = x[1::2]

    # History is updated via record_modal_state() after each accepted env step.
    # Do NOT append here — RK4 calls dynamics at intermediate (t, x) pairs.

    current_hash = _make_cache_hash(omega, ac)
    if (
        not cache["initialized"]
        or cache["hash"] != current_hash
        or cache["time_discrete"] is None
        or cache["F_geom"] is None
        or cache["b_vec_series"] is None
        or cache["h_geom_series"] is None
    ):
        _update_cache(omega, ac, current_hash)

    t_now = float(t)
    _, b_vec = _geometry_force_at(t_now)
    f_scalar = _scalar_cutting_force_at(t_now, eta, omega, ac)
    f_scalar = float(np.clip(f_scalar, -MAX_FORCE, MAX_FORCE))

    Fk = b_vec * f_scalar
    Fk = np.nan_to_num(Fk, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)

    # Structural dynamics: current-state damping, stiffness, nonlinearity only.
    zeta_omega = 2.0 * zeta_vec * omega_vec
    omega2 = omega_vec ** 2
    ddeta = (
        -zeta_omega * etad
        - omega2 * eta
        - lambda_vec * (eta ** 3)
        + Fk
    )
    ddeta = np.nan_to_num(ddeta, nan=0.0, posinf=1e6, neginf=-1e6)

    dx = np.zeros(2 * K, dtype=np.float64)
    dx[0::2] = etad
    dx[1::2] = ddeta
    return np.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)
