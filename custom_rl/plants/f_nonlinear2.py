"""Efficient nonlinear modal-force dynamics for the plate plant.

Units (consistent with the original MATLAB model):
    t, STATE_DELAY          : seconds [s]
    L1, L2, h, coordinates  : meters [m]
    omega (spindle speed)   : rad/s
    ac (depth of cut)       : millimeters [mm]
    cf (chip load)          : mm/tooth (converted to m via /1000 in feed_rate)
    feed_rate               : m/s
    modal coords eta        : nondimensional (mode-shape projection)
    F_normal, Fk            : N (generalized modal force uses b_vec projection)

Public interface:

    f_nonlinear2(t, x, u)

where:
    t : current time
    x : state vector [eta1, eta1_dot, eta2, eta2_dot, ...]
    u : physical action [omega, ac]

The implementation is adapted for RL training:
- prevents division by zero when omega is too small
- avoids globally growing decimal_places
- limits the maximum number of force-discretization points
- avoids storing the full F_modal matrix
- caches force data efficiently
"""

from __future__ import annotations

import numpy as np
import bisect


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

# Straight-line milling geometry (set by PlatePlant).
L1 = None
x0_cutter = 0.0
y_cutter = 0.2
x_pass_end_tol = 0.01
USE_MOVING_FORCE_PROJECTION = True

# Kept for compatibility, but no longer allowed to grow globally.
decimal_places = 1
tau_floored = None


# ============================================================
# RL-safety / efficiency constants
# ============================================================

OMEGA_EPS = 1e-9

# Maximum number of time-discretization points used for force calculation.
# This prevents huge temporary arrays during PPO training.
P_MAX = 5000

# Cache-key rounding.
# This improves cache reuse for continuous PPO actions.
# omega is physical [rad/s], operating range 50-4000 rpm.
CACHE_OMEGA_DECIMALS = 1
CACHE_AC_DECIMALS = 3

MAX_FORCE = 1e5


# ============================================================
# State-delay configuration
# ============================================================

#STATE_DELAY = 0.0000075            # seconds; set this to desired delay (>0)
STATE_DELAY = 0.00001          # seconds; set this to desired delay (>0)
_state_history = []          # list of (t, x_copy) in increasing time order
_MAX_HISTORY = 100000        # prevent unbounded memory growth


def reset_state_history() -> None:
    """Clear delayed-state buffer. Call at the start of each RL episode."""
    global _state_history
    _state_history = []


def _get_delayed_state(t_current: float, delay: float):
    """
    Return state vector at time t_current - delay using linear interpolation
    from the stored history. If delay <= 0 or no history or t_delayed < 0,
    returns None (caller should use current state).
    """
    if delay <= 0.0:
        return None

    t_delayed = t_current - delay
    if t_delayed < 0.0:
        return None

    global _state_history
    if len(_state_history) == 0:
        return None

    # Binary search for the interval containing t_delayed
    times = [entry[0] for entry in _state_history]
    idx = bisect.bisect_left(times, t_delayed)

    if idx == 0:
        # Before first recorded time
        return None
    if idx == len(times):
        # After last recorded time – use last state
        return _state_history[-1][1]

    t1, x1 = _state_history[idx - 1]
    t2, x2 = _state_history[idx]

    # Linear interpolation
    if t2 - t1 < 1e-12:
        return x1
    alpha = (t_delayed - t1) / (t2 - t1)
    x_delayed = (1 - alpha) * x1 + alpha * x2
    return x_delayed


# ============================================================
# Cache
# ============================================================

cache = {
    "initialized": False,
    "hash": None,
    "time_discrete": None,
    "F_normal": None,
    "b_vec": None,
}


def _require_initialized() -> None:
    """Check that PlatePlant has initialized all required globals."""
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
        "L1": L1,
    }

    missing = [name for name, value in required.items() if value is None]

    if missing:
        raise RuntimeError(
            "f_nonlinear2 globals are not initialized. "
            f"Missing: {missing}. "
            "Make sure PlatePlant.__init__ sets them before dynamics is called."
        )


def _safe_action(u: np.ndarray) -> tuple[float, float]:
    """Return safe physical action values omega and ac."""
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
    """Create a stable cache key."""
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
    """
    Create bounded time grid.

    Avoids allocating huge arrays when dt_grid becomes very small.
    """
    if not np.isfinite(t_start) or not np.isfinite(t_end) or not np.isfinite(dt_grid):
        return np.array([0.0], dtype=np.float64)

    if dt_grid <= 0.0:
        return np.array([0.0], dtype=np.float64)

    if t_end <= 0.0:
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

    if time_discrete.size == 0:
        return np.array([t_end], dtype=np.float64)

    return time_discrete


def feed_rate(omega: float) -> float:
    """Feed speed along x [m/s] for spindle speed omega [rad/s]."""
    cf_m = float(cf) / 1000.0
    return cf_m * float(N) * max(float(omega), OMEGA_EPS) / (2.0 * np.pi)


def pass_duration(omega: float) -> float:
    """Time to traverse plate length L1 at constant omega [s]."""
    return float(L1) / max(feed_rate(omega), 1e-12)


def cutter_point(t: float, omega: float) -> tuple[float, float]:
    """
    Straight-line cutter position in the x-y plane.

    Starts at x = L1 (free end) and feeds toward x = 0.
    """
    feed_distance = feed_rate(omega) * float(t)
    xc = float(L1) - (feed_distance + float(x0_cutter))
    xc = float(np.clip(xc, 0.0, float(L1)))
    yc = float(y_cutter)
    return xc, yc


def _straight_tool_path(
    time_array: np.ndarray,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Straight pass coordinates at each time sample."""
    time_array = np.asarray(time_array, dtype=np.float64)
    feed = feed_rate(omega) * time_array
    x_path = np.clip(float(L1) - (feed + float(x0_cutter)), 0.0, float(L1))
    y_path = np.full_like(time_array, float(y_cutter), dtype=np.float64)
    return x_path, y_path


def _compute_b_vec_at(xc: float, yc: float) -> np.ndarray:
    """Modal force projection at cutter contact point (xc, yc)."""
    b_vec = np.zeros(K, dtype=np.float64)

    cnt = 0
    for m in range(m_max):
        for n in range(n_max):
            Wk = W_mn[m][n]
            b_vec[cnt] = Wk(float(xc), float(yc)) / M_modal
            cnt += 1

    return b_vec


def _tool_path_increments(
    time_discrete: np.ndarray,
    omega: float,
    dt_tooth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-tooth path increments [m] aligned with each force sample.

    The legacy ``np.diff(..., prepend=0)`` assumed the path started at the
    origin. For straight milling the cutter starts near x = L1, which produced
    a spurious first increment ~ L1 and a huge artificial force pulse.
    """
    time_discrete = np.asarray(time_discrete, dtype=np.float64)
    if time_discrete.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    if time_discrete.size == 1:
        step = feed_rate(omega) * max(float(dt_tooth), 0.0)
        return np.array([step], dtype=np.float64), np.array([0.0], dtype=np.float64)

    t_prev = np.concatenate(
        [[max(time_discrete[0] - float(dt_tooth), 0.0)], time_discrete[:-1]]
    )
    x_prev, y_prev = _straight_tool_path(t_prev, omega)
    x_curr, y_curr = _straight_tool_path(time_discrete, omega)
    return x_curr - x_prev, y_curr - y_prev


def _compute_b_vec() -> np.ndarray:
    """Fixed-point projection (legacy / fallback)."""
    xc, yc = cutter_point(0.0, 50.0)
    return _compute_b_vec_at(xc, yc)


def _update_cache(omega: float, ac: float, current_hash) -> None:
    """
    Recompute force time history and cache it.

    Important efficiency change:
        The old code stored F_modal with shape (K, P).
        Since F_modal[k, :] = b_vec[k] * F_normal,
        we only store F_normal with shape (P,) and b_vec with shape (K,).
    """
    global tau_floored, cache

    cf_m_per_rad = cf / 1000.0 / (2.0 * np.pi)

    tau = (2.0 * np.pi) / (N * omega)
    tau_floored = tau

    max_iterations = 3
    iteration = 0
    max_displacement = np.inf

    # Local only. Do not mutate the global decimal_places during RL.
    local_decimal_places = 1.0

    pass_time = pass_duration(omega)
    t_end = min(float(t_original[-1]), pass_time * 1.05 + tau)
    time_discrete = np.array([0.0], dtype=np.float64)

    threshold = cf_m_per_rad * N * omega * tau

    while max_displacement > threshold and iteration < max_iterations:
        scale = max(local_decimal_places, 1.0)
        tau_floored = tau_floored / scale

        if not np.isfinite(tau_floored) or tau_floored <= 0.0:
            tau_floored = max(t_end / P_MAX, OMEGA_EPS)

        t_start = tau_floored

        time_discrete = _make_time_grid(
            t_start=t_start,
            t_end=t_end,
            dt_grid=tau_floored,
        )

        x_discrete, y_discrete = _straight_tool_path(time_discrete, omega)

        if x_discrete.size > 1:
            dx_tmp = np.diff(x_discrete)
            dy_tmp = np.diff(y_discrete)
            disp = np.sqrt(dx_tmp * dx_tmp + dy_tmp * dy_tmp)
            max_displacement = float(np.max(disp))
        else:
            max_displacement = 0.0

        if max_displacement > threshold:
            local_decimal_places += 1.0
        else:
            break

        iteration += 1

    # =========================================================
    # Force coefficients
    # =========================================================
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

    # =========================================================
    # Fx, Fy
    # =========================================================
    P = int(time_discrete.size)

    Fx = np.zeros(P, dtype=np.float64)
    Fy = np.zeros(P, dtype=np.float64)

    if P > 1:
        dx, dy = _tool_path_increments(time_discrete, omega, tau_floored)

        dx2 = dx * dx
        dx3 = dx2 * dx

        dy2 = dy * dy
        dy3 = dy2 * dy

        dxy = dx * dy

        N_2pi = N / (2.0 * np.pi)

        Fx = -N_2pi * (
            alpha1 * dx3
            + beta1 * dy3
            + alpha2 * dx2
            + beta2 * dy2
            + alpha3 * dx
            + beta3 * dy
            + 3.0 * gamma1 * dx2 * dy
            + 3.0 * gamma2 * dx * dy2
            + 2.0 * gamma3 * dxy
            + gamma4
        )

        Fy = +N_2pi * (
            alpha_p1 * dx3
            + beta_p1 * dy3
            + alpha_p2 * dx2
            + beta_p2 * dy2
            + alpha_p3 * dx
            + beta_p3 * dy
            + 3.0 * gamma_p1 * dx2 * dy
            + 3.0 * gamma_p2 * dx * dy2
            + 2.0 * gamma_p3 * dxy
            + gamma_p4
        )

        Fx = np.sign(Fx) * np.minimum(np.abs(Fx), MAX_FORCE)
        Fy = np.sign(Fy) * np.minimum(np.abs(Fy), MAX_FORCE)

    F_normal = np.sqrt(Fx * Fx + Fy * Fy)
    F_normal = np.nan_to_num(F_normal, nan=0.0, posinf=MAX_FORCE, neginf=0.0)

    b_vec = _compute_b_vec()
    b_vec = np.nan_to_num(b_vec, nan=0.0, posinf=0.0, neginf=0.0)

    cache["time_discrete"] = time_discrete
    cache["F_normal"] = F_normal
    cache["b_vec"] = b_vec
    cache["hash"] = current_hash
    cache["initialized"] = True


def f_nonlinear2(t, x, u):
    """
    Nonlinear plate dynamics with optional state delay.

    The delay is controlled by the module‑level variable STATE_DELAY.
    If STATE_DELAY > 0, the damping, stiffness, and nonlinear terms
    use the state at time t - STATE_DELAY (interpolated from history).

    Args:
        t: current time
        x: state vector [eta1, eta1_dot, eta2, eta2_dot, ...]
        u: physical action [omega, ac]

    Returns:
        dx: state derivative with shape (2*K,)
    """
    _require_initialized()

    omega, ac = _safe_action(u)

    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if x.size != 2 * K:
        raise ValueError(
            f"f_nonlinear2 expects state shape ({2 * K},), but received {x.shape}."
        )

    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    # ---------- History management (only needed when STATE_DELAY > 0) ----------
    global _state_history
    if STATE_DELAY > 0.0:
        _state_history.append((t, x.copy()))
        if len(_state_history) > _MAX_HISTORY:
            _state_history = _state_history[-_MAX_HISTORY:]

    # ---------- Retrieve delayed state (if any) ----------
    x_delayed = _get_delayed_state(t, STATE_DELAY)
    if x_delayed is None:
        # No valid delayed state yet → use current state (no delay)
        x_delayed = x

    # ---------- Force calculation (unchanged) ----------
    current_hash = _make_cache_hash(omega, ac)

    key_changed = (
        not cache["initialized"]
        or cache["hash"] != current_hash
        or cache["time_discrete"] is None
        or cache["F_normal"] is None
        or cache["b_vec"] is None
    )

    if key_changed:
        _update_cache(omega, ac, current_hash)

    time_discrete = cache["time_discrete"]
    F_normal = cache["F_normal"]
    b_vec = cache["b_vec"]

    if (
        time_discrete is None
        or F_normal is None
        or b_vec is None
        or len(time_discrete) <= 1
    ):
        Fk = np.zeros(K, dtype=np.float64)
    else:
        F_normal_t = np.interp(
            float(t),
            time_discrete,
            F_normal,
            left=0.0,
            right=0.0,
        )
        if USE_MOVING_FORCE_PROJECTION:
            xc, yc = cutter_point(float(t), omega)
            b_vec_t = _compute_b_vec_at(xc, yc)
            Fk = b_vec_t * F_normal_t
        else:
            Fk = b_vec * F_normal_t

    Fk = np.nan_to_num(Fk, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)

    # ---------- Dynamics using DELAYED state for the acceleration ----------
    dx = np.zeros(2 * K, dtype=np.float64)

    # Current state (for the derivative of position)
    eta_current = x[0::2]
    etad_current = x[1::2]

    # Delayed state (for damping, stiffness, nonlinearity)
    eta_delayed = x_delayed[0::2]
    etad_delayed = x_delayed[1::2]

    zeta_omega = 2.0 * zeta_vec * omega_vec
    omega2 = omega_vec ** 2
    eta3_delayed = eta_delayed ** 3

    ddeta = (-zeta_omega * etad_delayed
             - omega2 * eta_delayed
             - lambda_vec * eta3_delayed
             + Fk)
    ddeta = np.nan_to_num(ddeta, nan=0.0, posinf=1e6, neginf=-1e6)

    dx[0::2] = etad_current      # d(eta)/dt = current velocity
    dx[1::2] = ddeta             # d(etad)/dt = acceleration from delayed state

    dx = np.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)

    return dx