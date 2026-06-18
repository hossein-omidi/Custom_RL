"""Directional regenerative milling force (tooth angle, engagement, Ft/Fr)."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from custom_rl.plants.milling_config import MillingForceConfig
from custom_rl.plants.mode_ordering import iter_mode_indices
from custom_rl.plants.units import (
    MAX_FORCE_SAFETY_N,
    ac_to_meters,
    calibrated_cutting_coefficients,
    chip_thickness_for_force_polynomial,
)

TWO_PI = 2.0 * math.pi


def wrap_to_2pi(phi: float) -> float:
    x = float(phi) % TWO_PI
    return x if x >= 0.0 else x + TWO_PI


def engagement(phi: float, phi_st: float, phi_ex: float) -> float:
    """g(phi)=1 on open interval (phi_st, phi_ex), with wrap-around support."""
    p = wrap_to_2pi(phi)
    st = wrap_to_2pi(phi_st)
    ex = wrap_to_2pi(phi_ex)
    if st < ex:
        return 1.0 if st < p < ex else 0.0
    if st > ex:
        return 1.0 if (p > st or p < ex) else 0.0
    return 0.0


def tooth_immersion_angle(
    theta: float,
    tooth_index: int,
    n_teeth: int,
    z: float,
    cfg: MillingForceConfig,
) -> float:
    """phi_j(t,z) = theta + j*2*pi/N - 2*z*tan(beta)/D_c + phi_0."""
    phi_p = TWO_PI / float(n_teeth)
    phi = theta + tooth_index * phi_p + float(cfg.phi_0)
    if abs(cfg.helix_angle) > 1e-15:
        phi -= 2.0 * z * math.tan(cfg.helix_angle) / float(cfg.cutter_diameter)
    return phi


def feed_per_tooth(
    feed_speed: float,
    omega: float,
    n_teeth: int,
    *,
    cf: float | None = None,
    source: str = "from_feed_speed",
    cf_units: str = "m_per_tooth",
) -> float:
    """Nominal geometric feed chip per tooth f_t [m]."""
    if source == "from_cf":
        if cf is None:
            raise ValueError("cf required when feed_per_tooth_source is 'from_cf'.")
        val = float(cf)
        if cf_units == "mm_per_tooth":
            return val * 1e-3
        if cf_units == "m_per_tooth":
            return val
        raise ValueError(f"Unknown cf_units: {cf_units}")
    omega = max(float(omega), 1e-9)
    return TWO_PI * float(feed_speed) / (float(n_teeth) * omega)


def chip_thickness_surface_normal_reduced(
    phi: float,
    f_t: float,
    delta_q: float,
    cfg: MillingForceConfig,
) -> float:
    """h_j = [f_t*sin(phi) + Delta_q*cos(phi)] * g(phi) (single-field reduced)."""
    g = engagement(phi, cfg.phi_st, cfg.phi_ex)
    if g <= 0.0:
        return 0.0
    sp, cp = math.sin(phi), math.cos(phi)
    return float((f_t * sp + delta_q * cp) * g)


def chip_thickness_feed_normal_full(
    phi: float,
    f_t: float,
    delta_f: float,
    delta_n: float,
    cfg: MillingForceConfig,
) -> float:
    """
    Two-direction Nasiri/Moradi-style chip thickness.

    h_j = [f_t*sin(phi) + Delta_f*sin(phi) + Delta_n*cos(phi)] * g(phi)
    """
    g = engagement(phi, cfg.phi_st, cfg.phi_ex)
    if g <= 0.0:
        return 0.0
    sp, cp = math.sin(phi), math.cos(phi)
    return float((f_t * sp + delta_f * sp + delta_n * cp) * g)


def tangential_radial_increment(
    h: float,
    xi: np.ndarray,
    delta: np.ndarray,
    dz: float,
) -> tuple[float, float]:
    """
    Elemental tangential/radial force increment.

    Parameters
    ----------
    h : chip thickness [m] (geometric / regenerative convention)
    xi, delta : cutting coefficients in mm–m hybrid form (see units.py)
    dz : axial slice thickness [m]
    """
    if h <= 0.0:
        return 0.0, 0.0
    h_poly = chip_thickness_for_force_polynomial(h)
    h2, h3 = h_poly * h_poly, h_poly * h_poly * h_poly
    dft = float(xi[0] * h3 + xi[1] * h2 + xi[2] * h_poly + xi[3]) * float(dz)
    dfr = float(delta[0] * h3 + delta[1] * h2 + delta[2] * h_poly + delta[3]) * float(dz)
    return dft, dfr


def transform_to_feed_normal(dft: float, dfr: float, phi: float) -> tuple[float, float]:
    """
    In-plane feed / normal-to-feed force components.

    dF_feed   = -dF_t*cos(phi) - dF_r*sin(phi)
    dF_normal =  dF_t*sin(phi) - dF_r*cos(phi)
    """
    cp, sp = math.cos(phi), math.sin(phi)
    d_feed = -dft * cp - dfr * sp
    d_normal = dft * sp - dfr * cp
    return float(d_feed), float(d_normal)


# Backward-compatible alias
transform_to_feed_plane = transform_to_feed_normal


def surface_normal_force_component(dft: float, dfr: float, phi: float) -> float:
    """Thrust (+z) for reduced model: dF_t*sin(phi) + dF_r*cos(phi)."""
    sp, cp = math.sin(phi), math.cos(phi)
    return float(dft * sp + dfr * cp)


def reconstruct_q_normal(
    eta_n: np.ndarray,
    x_c: float,
    y_c: float,
    w_mn,
    m_max: int,
    n_max: int,
    *,
    b_vec: np.ndarray | None = None,
    M_n: float | None = None,
) -> float:
    """q_n = sum_k W_k(x_c,y_c) * eta_n,k.  Fast path: M_n * dot(b_vec, eta_n)."""
    eta_n = np.asarray(eta_n, dtype=np.float64).reshape(-1)
    if b_vec is not None and M_n is not None:
        return float(M_n * np.dot(b_vec, eta_n))
    q = 0.0
    for m, n, k in iter_mode_indices(m_max, n_max):
        q += float(w_mn[m][n](x_c, y_c)) * float(eta_n[k])
    return q


def reconstruct_q_feed(
    eta_f: np.ndarray,
    z_c: float,
    y_c: float,
    v_mn,
    m_max: int,
    n_max: int,
    *,
    b_vec: np.ndarray | None = None,
    M_f: float | None = None,
) -> float:
    """q_f = sum_k V_k(z_c,y_c) * eta_f,k.  Fast path: M_f * dot(b_vec, eta_f)."""
    eta_f = np.asarray(eta_f, dtype=np.float64).reshape(-1)
    if b_vec is not None and M_f is not None:
        return float(M_f * np.dot(b_vec, eta_f))
    q = 0.0
    for m, n, k in iter_mode_indices(m_max, n_max):
        q += float(v_mn[m][n](z_c, y_c)) * float(eta_f[k])
    return q


reconstruct_q_surface_normal = reconstruct_q_normal
reconstruct_q_out_of_plane = reconstruct_q_normal


def compute_projection_vector_w(
    x_c: float,
    y_c: float,
    w_mn,
    m_max: int,
    n_max: int,
    M_n: float,
    *,
    mode_basis=None,
) -> np.ndarray:
    if mode_basis is not None:
        return mode_basis.projection_w(x_c, y_c, M_n)
    K = m_max * n_max
    b = np.zeros(K, dtype=np.float64)
    for m, n, k in iter_mode_indices(m_max, n_max):
        b[k] = float(w_mn[m][n](x_c, y_c)) / float(M_n)
    return b


def compute_projection_vector_v(
    z_c: float,
    y_c: float,
    v_mn,
    m_max: int,
    n_max: int,
    M_f: float,
    *,
    mode_basis=None,
) -> np.ndarray:
    if mode_basis is not None:
        return mode_basis.projection_v(z_c, y_c, M_f)
    K = m_max * n_max
    b = np.zeros(K, dtype=np.float64)
    for m, n, k in iter_mode_indices(m_max, n_max):
        b[k] = float(v_mn[m][n](z_c, y_c)) / float(M_f)
    return b


def compute_directional_forces(
    t: float,
    omega: float,
    ac: float,
    cfg: MillingForceConfig,
    *,
    eta_n: np.ndarray,
    eta_f: np.ndarray | None,
    n_teeth: int,
    xi_base: np.ndarray,
    delta_base: np.ndarray,
    w_mn,
    v_mn,
    m_max: int,
    n_max: int,
    feed_speed: float,
    theta_at: Callable[[float], float],
    tool_position: Callable[[float], tuple[float, float]],
    delay_time_fn: Callable[[float, float], float | None],
    modal_state_at_delay: Callable[[float, float], np.ndarray | None],
    z_contact: float,
    M_n: float,
    M_f: float,
    split_delayed_state: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]],
    max_force: float = MAX_FORCE_SAFETY_N,
    cf: float | None = None,
    b_vec_at: Callable[[float, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    if ac <= 0.0:
        return _zero_force_result(cfg, m_max * n_max)

    ac_m = ac_to_meters(ac, cfg.ac_units)
    xi = np.asarray(xi_base, dtype=np.float64)
    delta = np.asarray(delta_base, dtype=np.float64)
    if not cfg.ac_via_axial_integration:
        xi = xi * ac_m
        delta = delta * ac_m
        z_vals = np.array([0.5 * ac_m], dtype=np.float64)
        dz = 1.0
    else:
        n_z = int(cfg.axial_quadrature_points)
        z_vals = np.linspace(0.0, ac_m, n_z, dtype=np.float64)
        dz = ac_m / n_z if n_z > 0 else ac_m

    eta_n = np.asarray(eta_n, dtype=np.float64).reshape(-1)
    theta = float(theta_at(t))
    x_c, y_c = tool_position(t)
    z_c = float(z_contact)

    use_b_cache = b_vec_at is not None
    b_n_now = b_vec_at(t, "n") if use_b_cache else None
    b_f_now = b_vec_at(t, "f") if use_b_cache and cfg.is_feed_normal_full() else None

    delay = delay_time_fn(t, omega)
    eta_n_d = None
    eta_f_d = None
    x_cd, y_cd = x_c, y_c
    b_n_d = None
    b_f_d = None
    if delay is not None and delay > 0.0:
        x_delayed = modal_state_at_delay(t, delay)
        if x_delayed is not None:
            eta_n_d, eta_f_d = split_delayed_state(x_delayed)
            t_d = t - delay
            x_cd, y_cd = tool_position(t_d)
            if use_b_cache:
                b_n_d = b_vec_at(t_d, "n")
                if cfg.is_feed_normal_full():
                    b_f_d = b_vec_at(t_d, "f")

    if use_b_cache:
        q_n = reconstruct_q_normal(
            eta_n, x_c, y_c, w_mn, m_max, n_max, b_vec=b_n_now, M_n=M_n
        )
        q_n_d = (
            reconstruct_q_normal(
                eta_n_d, x_cd, y_cd, w_mn, m_max, n_max, b_vec=b_n_d, M_n=M_n
            )
            if eta_n_d is not None and b_n_d is not None
            else q_n
        )
    else:
        q_n = reconstruct_q_normal(eta_n, x_c, y_c, w_mn, m_max, n_max)
        q_n_d = (
            reconstruct_q_normal(eta_n_d, x_cd, y_cd, w_mn, m_max, n_max)
            if eta_n_d is not None
            else q_n
        )
    delta_n = q_n - q_n_d

    if cfg.is_feed_normal_full():
        if eta_f is None or v_mn is None:
            raise ValueError("eta_f and v_mn required for feed_normal_full.")
        eta_f = np.asarray(eta_f, dtype=np.float64).reshape(-1)
        if use_b_cache:
            q_f = reconstruct_q_feed(
                eta_f, z_c, y_c, v_mn, m_max, n_max, b_vec=b_f_now, M_f=M_f
            )
            q_f_d = (
                reconstruct_q_feed(
                    eta_f_d, z_c, y_cd, v_mn, m_max, n_max, b_vec=b_f_d, M_f=M_f
                )
                if eta_f_d is not None and b_f_d is not None
                else q_f
            )
        else:
            q_f = reconstruct_q_feed(eta_f, z_c, y_c, v_mn, m_max, n_max)
            q_f_d = (
                reconstruct_q_feed(eta_f_d, z_c, y_cd, v_mn, m_max, n_max)
                if eta_f_d is not None
                else q_f
            )
        delta_f = q_f - q_f_d
    else:
        delta_f = 0.0
        q_f = 0.0

    f_t = feed_per_tooth(
        feed_speed,
        omega,
        n_teeth,
        cf=cf,
        source=cfg.feed_per_tooth_source,
        cf_units=cfg.cf_units,
    )

    f_feed_total = 0.0
    f_normal_total = 0.0
    f_surface_normal = 0.0
    h_list: list[float] = []
    phi_list: list[float] = []
    g_list: list[float] = []
    ft_list: list[float] = []
    fr_list: list[float] = []

    for j in range(int(n_teeth)):
        for z in z_vals:
            phi = tooth_immersion_angle(theta, j, n_teeth, float(z), cfg)
            g = engagement(phi, cfg.phi_st, cfg.phi_ex)
            if cfg.is_feed_normal_full():
                h = chip_thickness_feed_normal_full(phi, f_t, delta_f, delta_n, cfg)
            else:
                h = chip_thickness_surface_normal_reduced(phi, f_t, delta_n, cfg)
            dft, dfr = tangential_radial_increment(h, xi, delta, dz)
            df, dn = transform_to_feed_normal(dft, dfr, phi)
            d_sn = surface_normal_force_component(dft, dfr, phi)
            f_feed_total += df
            f_normal_total += dn
            f_surface_normal += d_sn
            h_list.append(h)
            phi_list.append(phi)
            g_list.append(g)
            ft_list.append(dft)
            fr_list.append(dfr)

    f_feed_raw = f_feed_total
    f_normal_raw = f_normal_total
    f_feed_total = float(np.clip(f_feed_total, -max_force, max_force))
    f_normal_total = float(np.clip(f_normal_total, -max_force, max_force))
    f_surface_normal = float(np.clip(f_surface_normal, -max_force, max_force))
    clipped = (
        abs(f_feed_raw) > max_force
        or abs(f_normal_raw) > max_force
        or abs(f_surface_normal) > max_force
    )

    b_n = b_n_now if use_b_cache else compute_projection_vector_w(x_c, y_c, w_mn, m_max, n_max, M_n)
    if cfg.is_feed_normal_full():
        b_f = (
            b_f_now
            if use_b_cache
            else compute_projection_vector_v(z_c, y_c, v_mn, m_max, n_max, M_f)
        )
        f_projected = f_normal_total
    else:
        b_f = np.zeros_like(b_n)
        f_projected = f_surface_normal

    return {
        "F_normal_total": f_normal_total,
        "F_feed_total": f_feed_total,
        "F_normal_raw": f_normal_raw,
        "F_feed_raw": f_feed_raw,
        "force_clipped": clipped,
        "F_projected": f_projected,
        "F_surface_normal_total": f_surface_normal,
        "F_feed_in_plane_total": f_feed_total,
        "F_normal_in_plane_total": f_normal_total,
        "Delta_n": delta_n,
        "Delta_f": delta_f,
        "Delta_q": delta_n,
        "q_n": q_n,
        "q_f": q_f,
        "f_t": f_t,
        "b_n": b_n,
        "b_f": b_f,
        "displacement_model": cfg.displacement_model,
        "is_surface_normal_reduced": cfg.is_surface_normal_reduced(),
        "is_feed_normal_full": cfg.is_feed_normal_full(),
        "h_list": h_list,
        "phi_list": phi_list,
        "g_list": g_list,
        "Ft_list": ft_list,
        "Fr_list": fr_list,
    }


def _zero_force_result(cfg: MillingForceConfig, K: int) -> dict[str, Any]:
    z = np.zeros(K, dtype=np.float64)
    return {
        "F_normal_total": 0.0,
        "F_feed_total": 0.0,
        "F_projected": 0.0,
        "F_surface_normal_total": 0.0,
        "F_feed_in_plane_total": 0.0,
        "F_normal_in_plane_total": 0.0,
        "Delta_n": 0.0,
        "Delta_f": 0.0,
        "Delta_q": 0.0,
        "q_n": 0.0,
        "q_f": 0.0,
        "f_t": 0.0,
        "b_n": z.copy(),
        "b_f": z.copy(),
        "displacement_model": cfg.displacement_model,
        "is_surface_normal_reduced": cfg.is_surface_normal_reduced(),
        "is_feed_normal_full": cfg.is_feed_normal_full(),
        "h_list": [],
        "phi_list": [],
        "g_list": [],
        "Ft_list": [],
        "Fr_list": [],
    }


def one_revolution_diagnostics(
    t_start: float,
    omega: float,
    ac: float,
    cfg: MillingForceConfig,
    *,
    n_teeth: int,
    xi_base: np.ndarray,
    delta_base: np.ndarray,
    feed_speed: float,
    theta_at: Callable[[float], float],
    tool_position: Callable[[float], tuple[float, float]],
    n_samples: int = 200,
    delta_f: float = 0.0,
    delta_n: float = 0.0,
    cf: float | None = None,
) -> dict[str, np.ndarray]:
    period = TWO_PI / max(omega, 1e-9)
    times = np.linspace(t_start, t_start + period, n_samples, dtype=np.float64)
    f_t = feed_per_tooth(
        feed_speed,
        omega,
        n_teeth,
        cf=cf,
        source=cfg.feed_per_tooth_source,
        cf_units=cfg.cf_units,
    )
    xi = np.asarray(xi_base, dtype=np.float64)
    delta = np.asarray(delta_base, dtype=np.float64)
    ac_m = ac_to_meters(ac, cfg.ac_units)
    if not cfg.ac_via_axial_integration:
        xi, delta = xi * ac_m, delta * ac_m
    z = 0.5 * ac_m
    dz = ac_m / max(cfg.axial_quadrature_points, 1) if cfg.ac_via_axial_integration else 1.0

    phi_arr = np.zeros(n_samples)
    g_arr = np.zeros(n_samples)
    h_arr = np.zeros(n_samples)
    ft_arr = np.zeros(n_samples)
    fr_arr = np.zeros(n_samples)
    ff_arr = np.zeros(n_samples)
    fn_arr = np.zeros(n_samples)

    for i, t in enumerate(times):
        theta = theta_at(t)
        phi = tooth_immersion_angle(theta, 0, n_teeth, z, cfg)
        g = engagement(phi, cfg.phi_st, cfg.phi_ex)
        if cfg.is_feed_normal_full():
            h = chip_thickness_feed_normal_full(phi, f_t, delta_f, delta_n, cfg)
        else:
            h = chip_thickness_surface_normal_reduced(phi, f_t, delta_n, cfg)
        dft, dfr = tangential_radial_increment(h, xi, delta, dz)
        df, dn = transform_to_feed_normal(dft, dfr, phi)
        phi_arr[i] = phi
        g_arr[i] = g
        h_arr[i] = h
        ft_arr[i] = dft
        fr_arr[i] = dfr
        ff_arr[i] = df
        fn_arr[i] = dn

    return {
        "t": times,
        "theta": np.array([theta_at(t) for t in times]),
        "phi": phi_arr,
        "g": g_arr,
        "h": h_arr,
        "Ft": ft_arr,
        "Fr": fr_arr,
        "F_feed": ff_arr,
        "F_normal": fn_arr,
        "Delta_f": np.full(n_samples, delta_f),
        "Delta_n": np.full(n_samples, delta_n),
    }
