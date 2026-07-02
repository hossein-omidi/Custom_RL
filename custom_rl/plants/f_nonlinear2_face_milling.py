"""Face-milling regenerative cutting-force dynamics in modal coordinates.

This file is a replacement for the previous peripheral-milling force module.
It keeps the public plant interface:

    f_nonlinear2(t, x, u)

where
    t : time [s]
    x : modal state [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]
    u : physical action [omega, ap] or [omega, ap, ae]

Units used internally
---------------------
    time                         : s
    mode-shape coordinates        : m for W_mn(x, y) arguments
    modal displacement eta        : assumed to reconstruct physical displacement in m
    spindle speed omega           : rad/s by default
    feed per tooth cf             : mm/tooth
    D_mm, ap, ae, h_i             : mm
    cutting coefficients Kt,Kr,Ka : N/mm^2
    edge coefficients Kte,Kre,Kae : N/mm
    global forces Fx,Fy,Fz        : N

Implemented face-milling model
------------------------------
For tooth i:

    theta_i = theta_base(t) + theta0 + (i-1)*2*pi/N
    For constant spindle speed:
        tau = 2*pi/(N*omega)

    For variable spindle speed:
        find t_d from theta_base(t) - theta_base(t_d) = 2*pi/N
        tau_eff = t - t_d

    h_i = ft*sin(theta_i)
        + Delta_xr*sin(theta_i)*cos(gamma_L)
        + Delta_yr*cos(theta_i)*cos(gamma_L)
        - Delta_zr*sin(gamma_L)

where

    Delta_r = r_rel(t) - r_rel(t_d)
    r_rel   = r_cutter_dynamic - r_workpiece_dynamic

For each engaged tooth:

    Ft_i = Kt*ap*h_i + Kte*ap
    Fr_i = Kr*ap*h_i + Kre*ap
    Fa_i = Ka*ap*h_i + Kae*ap

and the CS1 -> CS0 transformation is:

    Fx_i = -Ft_i*cos(theta_i) - Fr_i*sin(theta_i)
    Fy_i =  Ft_i*sin(theta_i) - Fr_i*cos(theta_i)
    Fz_i =  Fa_i

The force is projected into the plate modal equation. If only the scalar plate
mode shape W_mn is available, it is treated as the transverse/axial mode shape
and only Fz is projected:

    Q_k = W_k(xc, yc) * Fz / M_k

If vector component mode shapes W_mn_x, W_mn_y, W_mn_z are supplied, set
FORCE_PROJECTION_MODE = "vector" to project all global force components:

    Q_k = (Wx_k*Fx + Wy_k*Fy + Wz_k*Fz) / M_k

Important difference from the old peripheral-milling code
--------------------------------------------------------
The old code precomputed a scalar F_normal = sqrt(Fx^2 + Fy^2) from a
peripheral-milling polynomial model. That is removed here. Face milling is
computed tooth-by-tooth, gives [Fx, Fy, Fz], and uses the regenerative delay
inside chip thickness, not as an artificial delay in the structural terms.
When spindle speed changes, the delayed state is selected by one-tooth phase
separation rather than by the instantaneous period alone.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ============================================================
# Globals set by PlatePlant / environment
# ============================================================

# Modal/plate configuration
m_max = None
n_max = None
N = None                         # number of teeth/inserts
K = None                         # number of retained modes = m_max*n_max

zeta_vec = None                  # damping ratios, shape (K,)
lambda_vec = None                # cubic stiffness coefficients, shape (K,)
omega_vec = None                 # natural circular frequencies [rad/s], shape (K,)

W_mn = None                      # scalar transverse mode shapes W_mn[m][n](x,y)
W_mn_x = None                    # optional vector mode-shape x component
W_mn_y = None                    # optional vector mode-shape y component
W_mn_z = None                    # optional vector mode-shape z component
M_modal = None                   # scalar modal mass or array of length K

# Geometry/path. Coordinates are meters because mode-shape functions use meters.
L1 = None                        # workpiece/plate length in feed direction [m]
L2 = None                        # optional width [m]
x0_cutter = 0.0                  # initial feed offset from x=L1 [m]
y_cutter = 0.2                   # nominal cutter center/contact y-coordinate [m]

# Feed and action convention
cf = None                        # feed per tooth ft [mm/tooth]
ACTION_OMEGA_UNIT = "rad/s"      # "rad/s" or "rpm"; previous code used rad/s

# Face-milling geometry and force coefficients.
# These must be set by PlatePlant for the actual tool/material.
D_mm = None                      # cutter diameter [mm]
ae_default_mm = None             # radial depth/immersion [mm]
gamma_L = None                   # lead/approach angle [rad]
gamma_r = None                   # radial rake angle [rad], optional
gamma_a = None                   # axial rake angle [rad], optional
eta_c = None                     # chip-flow angle [rad], optional

Kt = None                        # tangential cutting coefficient [N/mm^2]
Kr = None                        # radial cutting coefficient [N/mm^2]
Ka = None                        # axial cutting coefficient [N/mm^2]
Kte = 0.0                        # tangential edge coefficient [N/mm]
Kre = 0.0                        # radial edge coefficient [N/mm]
Kae = 0.0                        # axial edge coefficient [N/mm]

milling_mode = "up"              # "up" or "down"
theta0 = 0.0                     # initial tooth angle [rad]

# Process damping parameters. Disabled by default for first validation.
USE_PROCESS_DAMPING = False
Ksp = None                       # specific indentation coefficient [N/mm^3]
mu = 0.3                         # Coulomb friction coefficient [-]
VB = 0.0                         # flank wear land [mm]
lambda_L = None                  # axial/inclination angle used in bzz term [rad]

# Projection and displacement reconstruction settings.
FORCE_PROJECTION_MODE = "z"      # "z" or "vector"
MODAL_DISPLACEMENT_TO_MM = 1000.0 # reconstructed displacement [m] -> [mm]
USE_DELAYED_CUTTER_POSITION_FOR_REGEN = True
EDGE_FORCE_WHEN_ZERO_CHIP = False

# Numerical safety.
OMEGA_EPS = 1e-9
MAX_FORCE = 1e6
MAX_ACCEL = 1e8
MIN_IMMERSION = 0.0


# ============================================================
# History for regenerative chatter
# ============================================================

_state_history = []              # sorted list of (t, x_copy)
_MAX_HISTORY = 200000

# Accepted tool-path/spindle kinematics for variable-omega RL control.
# If disabled, the module falls back to the old constant-omega convention:
#   x_c(t) = L1 - feed_rate(omega_current)*t - x0_cutter
#   theta(t) = omega_current*t + theta0
# PlatePlant enables this flag and updates the accepted feed distance and
# spindle phase at the beginning/end of each environment step.
USE_ACCEPTED_PATH_KINEMATICS = False
_kinematic_history = []          # sorted list of (t, feed_distance_m, spindle_phase_rad)
_KINEMATIC_TOL = 1e-14
_kinematic_t0 = 0.0
_kinematic_feed0_m = 0.0
_kinematic_phase0_rad = 0.0


def reset_state_history() -> None:
    """Clear modal and accepted-kinematic histories at episode/pass reset."""
    global _state_history, last_force_info
    global _kinematic_history, _kinematic_t0, _kinematic_feed0_m, _kinematic_phase0_rad
    _state_history = []
    _kinematic_history = []
    _kinematic_t0 = 0.0
    _kinematic_feed0_m = 0.0
    _kinematic_phase0_rad = 0.0
    last_force_info = None


def _append_state_history(t: float, x: np.ndarray) -> None:
    """Append state history while preserving monotonic order as much as possible."""
    global _state_history
    t = float(t)
    x_copy = np.asarray(x, dtype=np.float64).copy()

    if len(_state_history) == 0 or t >= _state_history[-1][0]:
        _state_history.append((t, x_copy))
    else:
        # ODE solvers can occasionally evaluate at non-monotonic times.
        times = [entry[0] for entry in _state_history]
        idx = bisect.bisect_left(times, t)
        if idx < len(_state_history) and abs(_state_history[idx][0] - t) < 1e-14:
            _state_history[idx] = (t, x_copy)
        else:
            _state_history.insert(idx, (t, x_copy))

    if len(_state_history) > _MAX_HISTORY:
        _state_history = _state_history[-_MAX_HISTORY:]


def _get_state_at_time(t_query: float) -> Optional[np.ndarray]:
    """Return modal state at t_query by linear interpolation from history."""
    global _state_history
    if len(_state_history) == 0:
        return None

    t_query = float(t_query)
    times = [entry[0] for entry in _state_history]
    idx = bisect.bisect_left(times, t_query)

    if idx == 0:
        # If the requested delay time exactly matches the first accepted
        # state, return it instead of treating it as unavailable.
        if abs(times[0] - t_query) < 1e-14:
            return _state_history[0][1].copy()
        return None
    if idx == len(times):
        # Do not extrapolate a future delayed state from the last accepted state.
        # If this happens, the RK substep is too large for the current tooth delay
        # or a proper within-step DDE interpolant is required.
        if t_query <= times[-1] + 1e-14:
            return _state_history[-1][1].copy()
        return None

    t1, x1 = _state_history[idx - 1]
    t2, x2 = _state_history[idx]
    if abs(t2 - t1) < 1e-14:
        return x1.copy()

    a = (t_query - t1) / (t2 - t1)
    return (1.0 - a) * x1 + a * x2



# ============================================================
# Accepted tool-path / spindle kinematic history
# ============================================================


def _append_kinematic_history(t: float, feed_distance_m: float, spindle_phase_rad: float) -> None:
    """Append accepted feed distance and spindle phase for interpolation."""
    global _kinematic_history
    t = float(t)
    entry = (t, float(feed_distance_m), float(spindle_phase_rad))

    if len(_kinematic_history) == 0 or t >= _kinematic_history[-1][0]:
        if len(_kinematic_history) > 0 and abs(_kinematic_history[-1][0] - t) < _KINEMATIC_TOL:
            _kinematic_history[-1] = entry
        else:
            _kinematic_history.append(entry)
    else:
        times = [e[0] for e in _kinematic_history]
        idx = bisect.bisect_left(times, t)
        if idx < len(_kinematic_history) and abs(_kinematic_history[idx][0] - t) < _KINEMATIC_TOL:
            _kinematic_history[idx] = entry
        else:
            _kinematic_history.insert(idx, entry)

    if len(_kinematic_history) > _MAX_HISTORY:
        _kinematic_history = _kinematic_history[-_MAX_HISTORY:]


def set_accepted_kinematic_state(
    t: float,
    feed_distance_m: float,
    spindle_phase_rad: float,
    *,
    append: bool = True,
) -> None:
    """Set accepted tool-path/spindle kinematics used by cutter_point().

    Parameters
    ----------
    t:
        Accepted time [s].
    feed_distance_m:
        Accumulated feed distance from the free side toward the clamped side [m].
    spindle_phase_rad:
        Accumulated spindle phase angle [rad]. This may be unbounded.
    append:
        If True, store the accepted kinematic state for delayed interpolation.
    """
    global _kinematic_t0, _kinematic_feed0_m, _kinematic_phase0_rad
    _kinematic_t0 = float(t)
    _kinematic_feed0_m = max(float(feed_distance_m), 0.0)
    _kinematic_phase0_rad = float(spindle_phase_rad)
    if append:
        _append_kinematic_history(_kinematic_t0, _kinematic_feed0_m, _kinematic_phase0_rad)


def _get_kinematic_at_time(t_query: float):
    """Return (feed_distance_m, spindle_phase_rad) at t_query by interpolation."""
    if len(_kinematic_history) == 0:
        return None

    t_query = float(t_query)
    times = [entry[0] for entry in _kinematic_history]
    idx = bisect.bisect_left(times, t_query)

    if idx == 0:
        if abs(times[0] - t_query) < _KINEMATIC_TOL:
            return _kinematic_history[0][1], _kinematic_history[0][2]
        return None

    if idx == len(times):
        # For queries after the last accepted point, the caller should normally
        # propagate from the current accepted state using the current held omega.
        return _kinematic_history[-1][1], _kinematic_history[-1][2]

    t1, feed1, phase1 = _kinematic_history[idx - 1]
    t2, feed2, phase2 = _kinematic_history[idx]
    if abs(t2 - t1) < _KINEMATIC_TOL:
        return feed1, phase1

    a = (t_query - t1) / (t2 - t1)
    feed = (1.0 - a) * feed1 + a * feed2
    phase = (1.0 - a) * phase1 + a * phase2
    return float(feed), float(phase)


def feed_phase_at_time(t: float, omega_rad_s: float) -> Tuple[float, float]:
    """Return accumulated feed distance [m] and spindle phase [rad] at time t.

    With USE_ACCEPTED_PATH_KINEMATICS=True, past times are interpolated from
    accepted kinematic history and current RK-stage times are propagated from
    the last accepted state. This avoids artificial jumps when the RL action
    changes omega between environment steps.
    """
    t = float(t)
    omega_rad_s = max(float(omega_rad_s), OMEGA_EPS)

    if not USE_ACCEPTED_PATH_KINEMATICS:
        return feed_rate_m_s(omega_rad_s) * max(t, 0.0), omega_rad_s * max(t, 0.0)

    if t >= _kinematic_t0 - _KINEMATIC_TOL:
        dt_local = max(t - _kinematic_t0, 0.0)
        feed = _kinematic_feed0_m + feed_rate_m_s(omega_rad_s) * dt_local
        phase = _kinematic_phase0_rad + omega_rad_s * dt_local
        return float(feed), float(phase)

    past = _get_kinematic_at_time(max(t, 0.0))
    if past is not None:
        return past

    return 0.0, 0.0


def spindle_phase(t: float, omega_rad_s: float) -> float:
    """Return accumulated spindle phase [rad] at time t."""
    _, phase = feed_phase_at_time(t, omega_rad_s)
    return float(phase)


def tooth_pitch_phase() -> float:
    """Return one-tooth pitch angle [rad]."""
    return 2.0 * np.pi / float(N)


def _get_time_at_phase(phase_query: float) -> Optional[float]:
    """Return the time at which the accepted spindle phase reached phase_query.

    The kinematic history stores unwrapped spindle phase, so the inverse mapping
    phase -> time is well defined while the spindle speed is positive.
    """
    if len(_kinematic_history) == 0:
        return None

    phase_query = float(phase_query)
    phases = [entry[2] for entry in _kinematic_history]
    idx = bisect.bisect_left(phases, phase_query)

    if idx == 0:
        if abs(phases[0] - phase_query) < _KINEMATIC_TOL:
            return float(_kinematic_history[0][0])
        return None

    if idx == len(phases):
        if phase_query <= phases[-1] + _KINEMATIC_TOL:
            return float(_kinematic_history[-1][0])
        return None

    t1, _, phase1 = _kinematic_history[idx - 1]
    t2, _, phase2 = _kinematic_history[idx]
    if abs(phase2 - phase1) < _KINEMATIC_TOL:
        return float(t1)

    a = (phase_query - phase1) / (phase2 - phase1)
    return float((1.0 - a) * t1 + a * t2)


def regenerative_delay_time(t: float, omega_rad_s: float) -> Tuple[Optional[float], float]:
    """Return (t_delay, tau_eff) for one-tooth regenerative delay.

    For constant spindle speed, this reduces to t_delay = t - 2*pi/(N*omega).
    For variable spindle speed, the delayed time is defined by one tooth pitch
    of unwrapped spindle phase:

        phase(t) - phase(t_delay) = 2*pi/N

    If less than one tooth of accepted history exists, t_delay is None and the
    caller should use zero regenerative displacement.
    """
    t = float(t)
    omega_rad_s = max(float(omega_rad_s), OMEGA_EPS)
    tau_const = tooth_period(omega_rad_s)

    if not USE_ACCEPTED_PATH_KINEMATICS:
        return max(t - tau_const, 0.0), tau_const

    phase_now = spindle_phase(t, omega_rad_s)
    target_phase = phase_now - tooth_pitch_phase()

    if target_phase < -_KINEMATIC_TOL:
        return None, tau_const
    target_phase = max(target_phase, 0.0)

    # If the whole delay lies inside the current held-action segment, compute it
    # directly. The state history may not support this case unless the RK substep
    # is smaller than the tooth delay; f_nonlinear2 checks that before use.
    if t >= _kinematic_t0 - _KINEMATIC_TOL and target_phase >= _kinematic_phase0_rad - _KINEMATIC_TOL:
        t_delay = _kinematic_t0 + max(target_phase - _kinematic_phase0_rad, 0.0) / omega_rad_s
        t_delay = min(max(t_delay, 0.0), t)
        return float(t_delay), float(t - t_delay)

    t_delay = _get_time_at_phase(target_phase)
    if t_delay is None:
        return None, tau_const

    t_delay = min(max(float(t_delay), 0.0), t)
    return t_delay, float(t - t_delay)


def _latest_state_history_time() -> Optional[float]:
    """Return the latest accepted modal-history time, if available."""
    if len(_state_history) == 0:
        return None
    return float(_state_history[-1][0])

# ============================================================
# Debug information from the last force calculation
# ============================================================

last_force_info = None


@dataclass
class FaceMillingForceInfo:
    t: float
    omega_rad_s: float
    n_rpm: float
    tau: float
    ap_mm: float
    ae_mm: float
    ft_mm: float
    cutter_xy_m: Tuple[float, float]
    cutter_xy_delay_m: Tuple[float, float]
    rel_now_mm: np.ndarray
    rel_delay_mm: np.ndarray
    delta_rel_mm: np.ndarray
    rel_vel_now_mm_s: np.ndarray
    theta: np.ndarray
    engagement: np.ndarray
    chip_raw_mm: np.ndarray
    chip_eff_mm: np.ndarray
    tooth_forces_local_N: np.ndarray
    tooth_forces_global_N: np.ndarray
    F_cut_N: np.ndarray
    F_pd_N: np.ndarray
    F_total_N: np.ndarray
    F_modal_N_per_kg: np.ndarray


# ============================================================
# Initialization and parameter helpers
# ============================================================


def _require_initialized() -> None:
    """Check that the required globals are initialized."""
    required = {
        "m_max": m_max,
        "n_max": n_max,
        "N": N,
        "K": K,
        "zeta_vec": zeta_vec,
        "lambda_vec": lambda_vec,
        "omega_vec": omega_vec,
        "W_mn": W_mn,
        "M_modal": M_modal,
        "cf": cf,
        "L1": L1,
        "D_mm": D_mm,
        "ae_default_mm": ae_default_mm,
        "gamma_L": gamma_L,
        "Kt": Kt,
        "Kr": Kr,
        "Ka": Ka,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "Face-milling f_nonlinear2 globals are not initialized. "
            f"Missing: {missing}. Set these in PlatePlant.__init__ before calling dynamics."
        )

    if int(K) != int(m_max) * int(n_max):
        raise RuntimeError(
            f"K must equal m_max*n_max. Got K={K}, m_max*n_max={m_max*n_max}."
        )


def set_force_coefficients_from_rake(Kn: float, Kf: float,
                                     gamma_r_value: float,
                                     gamma_a_value: float,
                                     eta_c_value: float) -> Tuple[float, float, float]:
    """
    Compute Kt, Kr, Ka from rake-face normal/friction pressure coefficients.

    This function follows the CS2->CS1 matrix product:

        [Ft, Fr, Fa]^T = R_r(gamma_r) R_a(gamma_a)
                         [Fn, Ff*cos(eta_c), Ff*sin(eta_c)]^T

    Parameters are angles in rad and pressure coefficients in N/mm^2.
    """
    cr, sr = np.cos(gamma_r_value), np.sin(gamma_r_value)
    ca, sa = np.cos(gamma_a_value), np.sin(gamma_a_value)
    ce, se = np.cos(eta_c_value), np.sin(eta_c_value)

    kt = Kn * cr * ca + Kf * sr * ce + Kf * cr * sa * se
    kr = -Kn * sr * ca + Kf * cr * ce - Kf * sr * sa * se
    ka = -Kn * sa + Kf * ca * se
    return float(kt), float(kr), float(ka)


def _safe_action(u: np.ndarray) -> Tuple[float, float, float]:
    """
    Read physical action.

    Expected:
        u[0] = omega [rad/s] by default, or [rpm] if ACTION_OMEGA_UNIT="rpm"
        u[1] = ap    [mm]
        u[2] = ae    [mm], optional. If absent, ae_default_mm is used.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if u.size < 2:
        raise ValueError(f"Expected action [omega, ap] or [omega, ap, ae], got {u.shape}.")

    omega_in = float(u[0])
    ap_mm = float(u[1])
    ae_mm = float(u[2]) if u.size >= 3 else float(ae_default_mm)

    if not np.isfinite(omega_in):
        omega_in = OMEGA_EPS
    if ACTION_OMEGA_UNIT.lower() in {"rpm", "rev/min", "r/min"}:
        omega = 2.0 * np.pi * omega_in / 60.0
    else:
        omega = omega_in

    if not np.isfinite(ap_mm):
        ap_mm = 0.0
    if not np.isfinite(ae_mm):
        ae_mm = 0.0

    omega = max(float(omega), OMEGA_EPS)
    ap_mm = max(float(ap_mm), 0.0)
    ae_mm = max(float(ae_mm), 0.0)
    return omega, ap_mm, ae_mm


def _modal_mass_vec() -> np.ndarray:
    mass = np.asarray(M_modal, dtype=np.float64)
    if mass.size == 1:
        return np.full(int(K), float(mass), dtype=np.float64)
    mass = mass.reshape(-1)
    if mass.size != int(K):
        raise RuntimeError(f"M_modal must be scalar or length K={K}, got shape {mass.shape}.")
    return mass


def _eval_mode_vector(mode_family, xc: float, yc: float) -> np.ndarray:
    """Evaluate mode shapes at a physical point; return vector of length K."""
    phi = np.zeros(int(K), dtype=np.float64)
    if mode_family is None:
        return phi

    cnt = 0
    for m in range(int(m_max)):
        for n in range(int(n_max)):
            phi[cnt] = float(mode_family[m][n](float(xc), float(yc)))
            cnt += 1
    return np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)


def _phi_components(xc: float, yc: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mode-shape component vectors (phi_x, phi_y, phi_z)."""
    if W_mn_x is None and W_mn_y is None and W_mn_z is None:
        # Existing plate code usually provides only one bending mode-shape family.
        return (
            np.zeros(int(K), dtype=np.float64),
            np.zeros(int(K), dtype=np.float64),
            _eval_mode_vector(W_mn, xc, yc),
        )

    phi_x = _eval_mode_vector(W_mn_x, xc, yc)
    phi_y = _eval_mode_vector(W_mn_y, xc, yc)
    phi_z = _eval_mode_vector(W_mn_z if W_mn_z is not None else W_mn, xc, yc)
    return phi_x, phi_y, phi_z


# ============================================================
# Tool path, engagement, and face-milling kinematics
# ============================================================


def spindle_rpm(omega_rad_s: float) -> float:
    return float(omega_rad_s) * 60.0 / (2.0 * np.pi)


def tooth_period(omega_rad_s: float) -> float:
    return 2.0 * np.pi / (float(N) * max(float(omega_rad_s), OMEGA_EPS))


def feed_rate_m_s(omega_rad_s: float) -> float:
    """Straight tool-center feed speed [m/s] using cf as feed per tooth [mm/tooth]."""
    ft_mm = float(cf)
    teeth_per_second = float(N) * max(float(omega_rad_s), OMEGA_EPS) / (2.0 * np.pi)
    return (ft_mm / 1000.0) * teeth_per_second


def cutter_point(t: float, omega_rad_s: float) -> Tuple[float, float]:
    """
    Nominal tool-center/contact point for force projection [m].

    The path starts at the free side x=L1 and feeds toward the clamped side x=0.
    With accepted kinematics enabled, accumulated feed distance is continuous
    when omega changes between RL steps.
    """
    feed_dist, _ = feed_phase_at_time(t, omega_rad_s)
    xc = float(L1) - (float(feed_dist) + float(x0_cutter))
    xc = float(np.clip(xc, 0.0, float(L1)))
    yc = float(y_cutter)
    return xc, yc


def entry_exit_angles(ae_mm: float) -> Tuple[float, float]:
    """Return face-milling entry and exit angles [rad]."""
    immersion = float(ae_mm) / max(float(D_mm), 1e-12)
    immersion = float(np.clip(immersion, MIN_IMMERSION, 1.0))

    mode = str(milling_mode).lower()
    if mode in {"down", "climb", "climb_milling"}:
        theta_s = np.arccos(np.clip(2.0 * immersion - 1.0, -1.0, 1.0))
        theta_e = np.pi
    elif mode in {"up", "conventional", "conventional_milling"}:
        theta_s = 0.0
        theta_e = np.arccos(np.clip(1.0 - 2.0 * immersion, -1.0, 1.0))
    else:
        raise ValueError("milling_mode must be 'up'/'conventional' or 'down'/'climb'.")

    return float(theta_s), float(theta_e)


def _engaged(theta: float, theta_s: float, theta_e: float) -> bool:
    return bool(theta_s < theta < theta_e)


# ============================================================
# Relative displacement/velocity reconstruction at the cutter point
# ============================================================


def relative_displacement_mm(x_modal: np.ndarray, xc: float, yc: float) -> np.ndarray:
    """
    Relative dynamic displacement r = cutter - workpiece at (xc,yc), in mm.

    The cutter is assumed dynamically rigid here, so r = -workpiece displacement.
    If vector mode shapes are not provided, only z displacement is reconstructed.
    """
    x_modal = np.asarray(x_modal, dtype=np.float64).reshape(-1)
    eta = x_modal[0::2]

    phi_x, phi_y, phi_z = _phi_components(xc, yc)
    w_x_m = float(phi_x @ eta)
    w_y_m = float(phi_y @ eta)
    w_z_m = float(phi_z @ eta)

    workpiece_mm = MODAL_DISPLACEMENT_TO_MM * np.array([w_x_m, w_y_m, w_z_m], dtype=np.float64)
    cutter_dynamic_mm = np.zeros(3, dtype=np.float64)
    return cutter_dynamic_mm - workpiece_mm


def relative_velocity_mm_s(x_modal: np.ndarray, xc: float, yc: float) -> np.ndarray:
    """Relative dynamic velocity r_dot = cutter_dot - workpiece_dot at (xc,yc), in mm/s."""
    x_modal = np.asarray(x_modal, dtype=np.float64).reshape(-1)
    etad = x_modal[1::2]

    phi_x, phi_y, phi_z = _phi_components(xc, yc)
    wd_x_m_s = float(phi_x @ etad)
    wd_y_m_s = float(phi_y @ etad)
    wd_z_m_s = float(phi_z @ etad)

    workpiece_mm_s = MODAL_DISPLACEMENT_TO_MM * np.array(
        [wd_x_m_s, wd_y_m_s, wd_z_m_s], dtype=np.float64
    )
    cutter_dynamic_mm_s = np.zeros(3, dtype=np.float64)
    return cutter_dynamic_mm_s - workpiece_mm_s


# ============================================================
# Face-milling force model
# ============================================================


def process_damping_matrix(theta_values: np.ndarray, engagement: np.ndarray,
                           omega_rad_s: float, ap_mm: float) -> np.ndarray:
    """Return B(t) process-damping matrix from the face-milling model."""
    B = np.zeros((3, 3), dtype=np.float64)
    if not USE_PROCESS_DAMPING:
        return B
    if Ksp is None:
        raise RuntimeError("USE_PROCESS_DAMPING=True requires Ksp [N/mm^3].")

    vt_mm_s = max(float(D_mm) * max(float(omega_rad_s), OMEGA_EPS) / 2.0, 1e-12)
    cd = 0.25 * float(VB) ** 2
    ceq = float(Ksp) * float(ap_mm) * cd / vt_mm_s

    tan_gL = np.tan(float(gamma_L))
    tan_lL = np.tan(float(lambda_L if lambda_L is not None else gamma_L))
    mu_val = float(mu)

    for th, eng in zip(theta_values, engagement):
        if not eng:
            continue
        s = np.sin(th)
        c = np.cos(th)
        B[0, 0] += ceq * (mu_val * c + s) * s
        B[0, 1] += ceq * (mu_val * c + s) * c
        B[0, 2] += ceq * (mu_val * c + s) * tan_gL
        B[1, 0] += ceq * (-mu_val * s + c) * s
        B[1, 1] += ceq * (-mu_val * s + c) * c
        B[1, 2] += ceq * (-mu_val * s + c) * tan_gL
        B[2, 0] += ceq * (-s)
        B[2, 1] += ceq * (-c)
        B[2, 2] += ceq * (-tan_lL)

    return B


def compute_face_milling_force(
    t: float,
    x_modal: np.ndarray,
    x_delay_modal: np.ndarray,
    omega_rad_s: float,
    ap_mm: float,
    ae_mm: float,
    *,
    t_delay_s: Optional[float] = None,
    tau_s: Optional[float] = None,
    delay_available: bool = True,
):
    """
    Compute global face-milling force and diagnostic quantities.

    Returns:
        F_total_N : np.ndarray shape (3,), [Fx,Fy,Fz]
        F_modal   : generalized modal force vector shape (K,)
        info      : FaceMillingForceInfo
    """
    t = float(t)
    omega_rad_s = max(float(omega_rad_s), OMEGA_EPS)
    ap_mm = max(float(ap_mm), 0.0)
    ae_mm = max(float(ae_mm), 0.0)
    ft_mm = float(cf)

    tau = tooth_period(omega_rad_s) if tau_s is None else max(float(tau_s), 0.0)
    theta_s, theta_e = entry_exit_angles(ae_mm)

    xc, yc = cutter_point(t, omega_rad_s)
    if delay_available and USE_DELAYED_CUTTER_POSITION_FOR_REGEN:
        t_delay = max(t - tau, 0.0) if t_delay_s is None else max(float(t_delay_s), 0.0)
        xc_delay, yc_delay = cutter_point(t_delay, omega_rad_s)
    else:
        # Before one-tooth history exists, use zero regenerative displacement:
        # compare current state with itself at the same cutter location.
        xc_delay, yc_delay = xc, yc

    rel_now = relative_displacement_mm(x_modal, xc, yc)
    rel_delay = relative_displacement_mm(x_delay_modal, xc_delay, yc_delay)
    delta_rel = rel_now - rel_delay
    rel_vel_now = relative_velocity_mm_s(x_modal, xc, yc)

    phi_pitch = 2.0 * np.pi / float(N)
    theta_base = spindle_phase(t, omega_rad_s) + float(theta0)
    theta_values = np.zeros(int(N), dtype=np.float64)
    engagement = np.zeros(int(N), dtype=bool)
    chip_raw = np.zeros(int(N), dtype=np.float64)
    chip_eff = np.zeros(int(N), dtype=np.float64)
    local_forces = np.zeros((int(N), 3), dtype=np.float64)   # Ft, Fr, Fa
    global_forces = np.zeros((int(N), 3), dtype=np.float64)  # Fx, Fy, Fz

    cos_gL = np.cos(float(gamma_L))
    sin_gL = np.sin(float(gamma_L))

    for i in range(int(N)):
        theta = (theta_base + i * phi_pitch) % (2.0 * np.pi)
        theta_values[i] = theta

        is_engaged = _engaged(theta, theta_s, theta_e)
        engagement[i] = is_engaged
        if not is_engaged or ap_mm <= 0.0 or ae_mm <= 0.0:
            continue

        s = np.sin(theta)
        c = np.cos(theta)

        h = (
            ft_mm * s
            + delta_rel[0] * s * cos_gL
            + delta_rel[1] * c * cos_gL
            - delta_rel[2] * sin_gL
        )
        chip_raw[i] = h
        h_eff = max(float(h), 0.0)
        chip_eff[i] = h_eff

        if h_eff <= 0.0 and not EDGE_FORCE_WHEN_ZERO_CHIP:
            continue

        Ft = float(Kt) * ap_mm * h_eff + float(Kte) * ap_mm
        Fr = float(Kr) * ap_mm * h_eff + float(Kre) * ap_mm
        Fa = float(Ka) * ap_mm * h_eff + float(Kae) * ap_mm

        local_forces[i, :] = [Ft, Fr, Fa]

        Fx_i = -Ft * c - Fr * s
        Fy_i = Ft * s - Fr * c
        Fz_i = Fa
        global_forces[i, :] = [Fx_i, Fy_i, Fz_i]

    F_cut = np.sum(global_forces, axis=0)
    F_cut = np.nan_to_num(F_cut, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)
    F_cut = np.clip(F_cut, -MAX_FORCE, MAX_FORCE)

    B_pd = process_damping_matrix(theta_values, engagement, omega_rad_s, ap_mm)
    F_pd = B_pd @ rel_vel_now
    F_pd = np.nan_to_num(F_pd, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)
    F_pd = np.clip(F_pd, -MAX_FORCE, MAX_FORCE)

    F_total = F_cut + F_pd
    F_total = np.nan_to_num(F_total, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)
    F_total = np.clip(F_total, -MAX_FORCE, MAX_FORCE)

    F_modal = project_force_to_modal(F_total, xc, yc)

    info = FaceMillingForceInfo(
        t=t,
        omega_rad_s=omega_rad_s,
        n_rpm=spindle_rpm(omega_rad_s),
        tau=tau,
        ap_mm=ap_mm,
        ae_mm=ae_mm,
        ft_mm=ft_mm,
        cutter_xy_m=(xc, yc),
        cutter_xy_delay_m=(xc_delay, yc_delay),
        rel_now_mm=rel_now,
        rel_delay_mm=rel_delay,
        delta_rel_mm=delta_rel,
        rel_vel_now_mm_s=rel_vel_now,
        theta=theta_values,
        engagement=engagement,
        chip_raw_mm=chip_raw,
        chip_eff_mm=chip_eff,
        tooth_forces_local_N=local_forces,
        tooth_forces_global_N=global_forces,
        F_cut_N=F_cut,
        F_pd_N=F_pd,
        F_total_N=F_total,
        F_modal_N_per_kg=F_modal,
    )
    return F_total, F_modal, info


def project_force_to_modal(F_global_N: np.ndarray, xc: float, yc: float) -> np.ndarray:
    """Project [Fx,Fy,Fz] onto modal coordinates at cutter point."""
    F_global_N = np.asarray(F_global_N, dtype=np.float64).reshape(3)
    mass = _modal_mass_vec()
    mass = np.where(np.abs(mass) < 1e-18, 1e-18, mass)

    phi_x, phi_y, phi_z = _phi_components(xc, yc)

    mode = str(FORCE_PROJECTION_MODE).lower()
    if mode == "vector":
        q = (phi_x * F_global_N[0] + phi_y * F_global_N[1] + phi_z * F_global_N[2]) / mass
    elif mode in {"z", "axial", "transverse"}:
        q = (phi_z * F_global_N[2]) / mass
    else:
        raise ValueError("FORCE_PROJECTION_MODE must be 'z' or 'vector'.")

    return np.nan_to_num(q, nan=0.0, posinf=MAX_FORCE, neginf=-MAX_FORCE)


# ============================================================
# Main plant dynamics
# ============================================================


def f_nonlinear2(t, x, u):
    """
    Modal plate dynamics forced by the face-milling cutting force.

    The structural equation is intentionally not delayed. Regeneration appears
    only inside the chip-thickness term through x(t)-x(t_delay), where t_delay
    is selected by one-tooth phase separation when accepted kinematics are used.
    """
    _require_initialized()

    omega_rad_s, ap_mm, ae_mm = _safe_action(u)

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != 2 * int(K):
        raise ValueError(f"Expected state shape ({2*K},), got {x.shape}.")
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    t_delay, tau = regenerative_delay_time(float(t), omega_rad_s)
    delay_available = t_delay is not None
    if delay_available:
        latest_t = _latest_state_history_time()
        if latest_t is not None and t_delay > latest_t + 1e-12:
            raise RuntimeError(
                "Regenerative delay time falls inside the current RK substep. "
                "Reduce env dt or increase n_substeps so each RK substep is "
                "smaller than the minimum tooth-passing delay, or use a DDE "
                "integrator with within-step interpolation."
            )
        x_delay = _get_state_at_time(t_delay)
        if x_delay is None:
            delay_available = False
            x_delay = x.copy()
    else:
        # Before one-tooth history exists, use current state. This gives zero
        # regenerative displacement and keeps the static chip-load force active.
        x_delay = x.copy()

    global last_force_info
    _, Fk, info = compute_face_milling_force(
        t=float(t),
        x_modal=x,
        x_delay_modal=x_delay,
        omega_rad_s=omega_rad_s,
        ap_mm=ap_mm,
        ae_mm=ae_mm,
        t_delay_s=t_delay,
        tau_s=tau,
        delay_available=delay_available,
    )
    last_force_info = info

    # Append after force computation so t-tau interpolation uses past states.
    _append_state_history(float(t), x)

    eta = x[0::2]
    etad = x[1::2]

    zeta_omega = 2.0 * np.asarray(zeta_vec, dtype=np.float64) * np.asarray(omega_vec, dtype=np.float64)
    omega2 = np.asarray(omega_vec, dtype=np.float64) ** 2
    lam = np.asarray(lambda_vec, dtype=np.float64)

    ddeta = -zeta_omega * etad - omega2 * eta - lam * eta**3 + Fk
    ddeta = np.nan_to_num(ddeta, nan=0.0, posinf=MAX_ACCEL, neginf=-MAX_ACCEL)
    ddeta = np.clip(ddeta, -MAX_ACCEL, MAX_ACCEL)

    dx = np.zeros(2 * int(K), dtype=np.float64)
    dx[0::2] = etad
    dx[1::2] = ddeta
    return dx


# ============================================================
# Optional utilities for environment/debugging
# ============================================================


def modal_to_face_milling_observation(x_modal: np.ndarray, t: float, omega_rad_s: float) -> np.ndarray:
    """
    Return physical displacement and velocity at the cutter point.

    Output is [rx, ry, rz, rdx, rdy, rdz] in [mm, mm/s], where r is
    cutter-workpiece relative dynamic displacement.
    """
    xc, yc = cutter_point(t, omega_rad_s)
    r = relative_displacement_mm(x_modal, xc, yc)
    rd = relative_velocity_mm_s(x_modal, xc, yc)
    return np.concatenate([r, rd])


def get_last_force_info():
    """Return diagnostics from the most recent force calculation."""
    return last_force_info
