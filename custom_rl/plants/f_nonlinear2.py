"""Efficient nonlinear modal-force dynamics for the plate plant.

This file keeps the original public interface:

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
# omega is physical, usually in [50, 2000].
CACHE_OMEGA_DECIMALS = 1
CACHE_AC_DECIMALS = 3

MAX_FORCE = 1e5


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


def _compute_b_vec() -> np.ndarray:
    """Compute modal force projection vector."""
    c1 = -0.0985
    c2 = -0.0941

    b_vec = np.zeros(K, dtype=np.float64)

    cnt = 0
    for m in range(m_max):
        for n in range(n_max):
            Wk = W_mn[m][n]
            b_vec[cnt] = Wk(c1, c2) / M_modal
            cnt += 1

    return b_vec


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

    t_end = float(t_original[-1])
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

        x_discrete = np.interp(time_discrete, t_original, x_traj)
        y_discrete = np.interp(time_discrete, t_original, y_traj)

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

    #if iteration >= max_iterations:
        # Keep the warning, but do not crash training.
     #   print("Warning: max iterations reached; displacement condition may not be satisfied.")

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
        x_interp = np.interp(time_discrete, t_original, x_traj)
        y_interp = np.interp(time_discrete, t_original, y_traj)

        dx = np.diff(x_interp, prepend=0.0)
        dy = np.diff(y_interp, prepend=0.0)

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
    Nonlinear plate dynamics.

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
        Fk = b_vec * F_normal_t

    Fk = np.nan_to_num(Fk, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)

    dx = np.zeros(2 * K, dtype=np.float64)

    eta = x[0::2]
    etad = x[1::2]

    zeta_omega = 2.0 * zeta_vec * omega_vec
    omega2 = omega_vec ** 2
    eta3 = eta ** 3

    ddeta = -zeta_omega * etad - omega2 * eta - lambda_vec * eta3 + Fk
    ddeta = np.nan_to_num(ddeta, nan=0.0, posinf=1e6, neginf=-1e6)

    dx[0::2] = etad
    dx[1::2] = ddeta

    dx = np.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)

    return dx