#!/usr/bin/env python3
"""
Phase-domain internal-model fidelity validation for NMPC-MHE face milling.

The full-order time-domain Gymnasium milling plant is used as the reference
simulator. The reduced phase-domain model is propagated independently with
exactly the same phase-cell zero-order-held input commands.

The script validates the reduced phase-domain model at three complementary
levels without mixing the associated approximation errors:

1) Regenerative-delay approximation
   Compares the finite Chebyshev delay realization G_M(s) with the exact
   phase delay exp(-s p) over the phase-frequency band of the retained modes.

2) Differentiable force approximation
   At matched reduced states, compares the internal-model phase-cell force
   (finite epsilon + analytic engagement intervals + midpoint quadrature)
   with a dense phase average of the corresponding hard-switch reduced force
   law. This isolates force regularization/quadrature from trajectory error.

3) Complete open-loop trajectory fidelity
   Drives the full time-domain Gym plant and the reduced phase-domain model
   with the same piecewise-constant commands and compares their responses at
   the same phase locations theta_k. For presentation, those synchronized
   samples are plotted against their corresponding physical times. Primary
   outputs are the physical sensor displacement and cutter-point displacement. The full-plant axial force is
   also phase-averaged as an overall diagnostic.

The code uses the same mode selection, delay order, phase-cell geometry and
force regularization as the NMPC-MHE internal model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import gymnasium as gym

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from mpc_mhe_dompc import (
    PhasePlant,
    PhaseSchedule,
    auto_delay_order,
    build_model,
    cheb_lobatto_D,
)

try:
    from do_mpc.simulator import Simulator
except ImportError:
    print("do-mpc required. Install with: pip install do-mpc casadi")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _deinterleave_modal(x_modal: Any, K_full: int) -> tuple[np.ndarray, np.ndarray]:
    """Return full-order modal displacement and physical-time velocity arrays."""
    x = np.asarray(x_modal, dtype=np.float64).reshape(-1)
    if x.size != 2 * K_full:
        raise ValueError(f"Expected x_modal length {2*K_full}, got {x.size}.")
    return x[0::2].copy(), x[1::2].copy()


def _extract_plant_force(info: dict[str, Any]) -> float:
    """Extract physical axial force Fz [N] from Gym info."""
    F = np.asarray(info.get("F_total_N", np.full(3, np.nan)), dtype=np.float64).reshape(-1)
    return float(F[2]) if F.size >= 3 else float("nan")


def _to_env_action(plant, omega: float, ap: float) -> np.ndarray:
    """Map physical commands to the normalized Gym action vector."""
    low = np.asarray(plant._ctrl_low, dtype=np.float64).reshape(-1)
    high = np.asarray(plant._ctrl_high, dtype=np.float64).reshape(-1)
    physical = np.array([omega, ap], dtype=np.float64)[: low.size]
    a = 2.0 * (physical - low) / np.maximum(high - low, 1e-15) - 1.0
    return np.clip(a, -1.0, 1.0)


def _unwrap_increment(raw: float, prev_raw: float, prev_unwrapped: float) -> float:
    """Incrementally unwrap a modulo-2pi spindle phase."""
    d = float(raw - prev_raw)
    if d < -np.pi:
        d += 2.0 * np.pi
    elif d > np.pi:
        d -= 2.0 * np.pi
    return float(prev_unwrapped + d)


def _interp_scalar(theta: np.ndarray, value: np.ndarray, target: float) -> float:
    """Linear interpolation of a scalar signal on a monotone phase grid."""
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    val = np.asarray(value, dtype=np.float64).reshape(-1)
    if target < th[0] - 1e-12 or target > th[-1] + 1e-12:
        raise ValueError(f"Target phase {target:.6g} outside [{th[0]:.6g}, {th[-1]:.6g}].")
    return float(np.interp(target, th, val))


def _interp_vector(theta: np.ndarray, value: np.ndarray, target: float) -> np.ndarray:
    """Linear interpolation of a vector signal on a monotone phase grid."""
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    val = np.asarray(value, dtype=np.float64)
    if val.ndim != 2 or val.shape[0] != th.size:
        raise ValueError("Vector interpolation expects shape (n_samples, n_components).")
    return np.array([np.interp(target, th, val[:, j]) for j in range(val.shape[1])],
                    dtype=np.float64)


def _phase_average(theta: np.ndarray, value: np.ndarray, a: float, b: float) -> float:
    """
    Phase-weighted average (1/(b-a)) integral_a^b value(theta) dtheta.

    Boundary values are interpolated, so the result is not biased by phase-cell
    crossings that occur between plant sampling instants.
    """
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    y = np.asarray(value, dtype=np.float64).reshape(-1)
    if b <= a:
        return float("nan")
    if a < th[0] - 1e-12 or b > th[-1] + 1e-12:
        return float("nan")

    mask = (th > a) & (th < b) & np.isfinite(y)
    th_i = th[mask]
    y_i = y[mask]

    finite = np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    th_f = th[finite]
    y_f = y[finite]
    ya = np.interp(a, th_f, y_f)
    yb = np.interp(b, th_f, y_f)

    th_seg = np.concatenate(([a], th_i, [b]))
    y_seg = np.concatenate(([ya], y_i, [yb]))
    order = np.argsort(th_seg)
    th_seg = th_seg[order]
    y_seg = y_seg[order]

    # np.trapezoid is unavailable in older NumPy releases.
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(y_seg, th_seg) / (b - a))


def _relative_rms(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Relative RMS/L2 error; NaN if the reference energy is numerically zero."""
    r = np.asarray(reference, dtype=np.float64)
    a = np.asarray(approximation, dtype=np.float64)
    mask = np.isfinite(r) & np.isfinite(a)
    if not np.any(mask):
        return float("nan")
    r = r[mask]
    a = a[mask]
    den = float(np.sum(r * r))
    if den <= 1e-30:
        return float("nan")
    return float(np.sqrt(np.sum((a - r) ** 2) / den))


def _model_state_vector(sim: Simulator) -> np.ndarray:
    """Return simulator state as a flat NumPy vector."""
    x0 = sim.x0
    if hasattr(x0, "cat"):
        return np.asarray(x0.cat, dtype=np.float64).reshape(-1)
    return np.asarray(x0, dtype=np.float64).reshape(-1)


# -----------------------------------------------------------------------------
# Independent approximation checks
# -----------------------------------------------------------------------------

def _delay_error_data(pp: PhasePlant, M: int, n_omega: int = 80) -> dict[str, Any]:
    """
    Evaluate |G_M(j nu)-exp(-j nu p)| on the phase-frequency bands nu=wn/omega
    generated by the retained modes over the admissible spindle-speed interval.
    """
    _, D = cheb_lobatto_D(M)
    A_q = -D[1:, 1:] / pp.p
    B_q = -D[1:, 0] / pp.p
    C_q = np.zeros(M, dtype=np.float64)
    C_q[-1] = 1.0

    omega_grid = np.linspace(pp.omega_min, pp.omega_max, max(5, int(n_omega)))
    rows: list[tuple[int, float, float]] = []
    I = np.eye(M)

    for mode_idx, wn in enumerate(np.asarray(pp.wn, dtype=np.float64)):
        for omega in omega_grid:
            nu = float(wn / omega)
            Gm = C_q @ np.linalg.solve(1j * nu * I - A_q, B_q)
            Gd = np.exp(-1j * nu * pp.p)
            rows.append((mode_idx, nu, float(abs(Gm - Gd))))

    rows.sort(key=lambda r: r[1])
    mode = np.array([r[0] for r in rows], dtype=int)
    nu = np.array([r[1] for r in rows], dtype=np.float64)
    err = np.array([r[2] for r in rows], dtype=np.float64)
    return {
        "mode": mode,
        "nu": nu,
        "error": err,
        "max_error": float(np.max(err)) if err.size else float("nan"),
    }


def _exact_reduced_force_point(
    pp: PhasePlant,
    theta: float,
    xc: float,
    yc: float,
    eta: np.ndarray,
    wd: float,
    ap: float,
) -> float:
    """Hard-switch reduced force law at one phase, with no smoothing."""
    g, s = pp.tooth_terms(theta, sharp=0.0)
    phi = pp.phi_at(xc, yc)
    wc = float(phi @ np.asarray(eta, dtype=np.float64).reshape(-1))
    h = pp.ft * s + pp.kr * (wc - wd)
    return float(ap * np.sum(g * (pp.Ka * np.maximum(h, 0.0) + pp.Kae * (h > 0.0))))


def _dense_exact_cell_force(
    pp: PhasePlant,
    theta0: float,
    dtheta: float,
    xc: float,
    yc: float,
    eta: np.ndarray,
    wd: float,
    ap: float,
    n_ref: int,
) -> float:
    """
    Dense midpoint phase average of the hard-switch reduced force law.

    The structural/regenerative state is held local to the cell, matching the
    state-local force representation used by the internal model. Only geometry
    varies over the integration phase.
    """
    n_ref = max(64, int(n_ref))
    phases = theta0 + (np.arange(n_ref) + 0.5) * dtheta / n_ref
    vals = np.array([
        _exact_reduced_force_point(pp, th, xc, yc, eta, wd, ap) for th in phases
    ], dtype=np.float64)
    return float(np.mean(vals))


def _internal_cell_force(
    pp: PhasePlant,
    theta0: float,
    dtheta: float,
    xc: float,
    yc: float,
    eta: np.ndarray,
    wd: float,
    ap: float,
    chip_eps: float,
    n_quad: int,
) -> float:
    """Internal-model force: exact engagement intersections + midpoint quadrature."""
    weights, s = pp.tooth_grid(theta0, dtheta, n_quad)
    phi = pp.phi_at(xc, yc)
    wc = float(phi @ np.asarray(eta, dtype=np.float64).reshape(-1))
    h = pp.ft * s + pp.kr * (wc - wd)
    eps = float(chip_eps)
    root = np.sqrt(h * h + eps * eps)
    psi = 0.5 * (h + root)
    chi = 0.5 * (1.0 + h / root)
    return float(ap * np.sum(weights * (pp.Ka * psi + pp.Kae * chi)))


# -----------------------------------------------------------------------------
# Command profiles: bounded, persistently exciting, and phase-cell ZOH
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandPoint:
    omega: float
    ap: float
    label: str


def _bounded_profile_limits(args: argparse.Namespace, pp: PhasePlant) -> tuple[float, float, float, float]:
    """Return conservative validation ranges (omega_lo, omega_hi, ap_lo, ap_hi)."""
    om_span = float(pp.omega_max - pp.omega_min)
    ap_span = float(pp.ap_max - pp.ap_min)

    margin = float(np.clip(args.omega_margin_frac, 0.0, 0.49))
    omega_lo = float(pp.omega_min + margin * om_span)
    omega_hi = float(pp.omega_max - margin * om_span)

    ap_lo_frac = float(np.clip(args.ap_low_frac, 0.0, 1.0))
    ap_hi_frac = float(np.clip(args.ap_high_frac, ap_lo_frac, 1.0))
    ap_lo = float(pp.ap_min + ap_lo_frac * ap_span)
    ap_hi = float(pp.ap_min + ap_hi_frac * ap_span)

    return omega_lo, omega_hi, ap_lo, ap_hi


def _command_for_cell(
    args: argparse.Namespace,
    pp: PhasePlant,
    cell_idx: int,
    n_cells: int,
    ap_fixed: float | None,
) -> CommandPoint:
    """
    Return the physical command for phase cell ``cell_idx``.

    The command is sampled once at the cell boundary and held exactly constant
    over that phase cell (zero-order hold).  The default ``research`` profile
    is deliberately bounded: spindle speed explores almost the full admissible
    range, while depth excitation is restricted to a user-selected conservative
    fraction of the admissible depth range to reduce the risk of driving the
    reference plant into divergent chatter.

    The research profile contains six consecutive excitation regimes:
      1) low-depth constant baseline,
      2) low-depth spindle-speed staircase,
      3) low-depth full-span spindle-speed sinusoid,
      4) axial-depth staircase at moderate speed,
      5) axial-depth sinusoid at fixed speed,
      6) mixed two-input sinusoidal excitation.

    This gives constant, step and smooth periodic input shapes while preserving
    an unambiguous phase-cell ZOH input for both the full plant and the internal
    model.
    """
    n_cells = max(1, int(n_cells))
    k = int(np.clip(cell_idx, 0, n_cells - 1))
    s = (k + 0.5) / n_cells

    omega_lo, omega_hi, ap_lo, ap_hi = _bounded_profile_limits(args, pp)
    omega_mid = 0.5 * (omega_lo + omega_hi)
    omega_span = omega_hi - omega_lo
    ap_mid = 0.5 * (ap_lo + ap_hi)
    ap_span = ap_hi - ap_lo

    profile = args.command_profile

    if profile == "constant":
        omega = float(np.clip(args.omega_constant, omega_lo, omega_hi))
        ap = float(np.clip(args.ap_constant, ap_lo, ap_hi))
        label = "constant"

    elif profile == "multistep":
        omega_levels = np.array([0.05, 0.35, 0.65, 0.95, 0.50])
        ap_levels = np.array([0.00, 0.25, 0.55, 1.00, 0.35])
        block = max(1, n_cells // max(len(omega_levels), len(ap_levels)))
        j = min(k // block, len(omega_levels) - 1)
        omega = omega_lo + omega_levels[j] * omega_span
        ap = ap_lo + ap_levels[j] * ap_span
        label = "multistep"

    elif profile == "sine":
        # Cell-index parametrization ensures the requested range is exercised
        # independent of the nonuniform physical update time.
        q = 2.0 * np.pi * args.profile_cycles * s
        omega = omega_mid + 0.5 * omega_span * np.sin(q)
        ap = ap_mid + 0.5 * ap_span * np.sin(args.ap_frequency_ratio * q + args.ap_phase)
        label = "sine"

    else:  # research
        if s < 0.12:
            # 1) Small-depth constant baseline.
            omega = omega_lo + 0.35 * omega_span
            ap = ap_lo
            label = "constant baseline"

        elif s < 0.30:
            # 2) Spindle-speed staircase at the smallest validation depth.
            local = (s - 0.12) / (0.30 - 0.12)
            levels = np.array([0.05, 0.35, 0.65, 0.95])
            j = min(int(local * len(levels)), len(levels) - 1)
            omega = omega_lo + levels[j] * omega_span
            ap = ap_lo
            label = "speed staircase"

        elif s < 0.48:
            # 3) Smooth speed sweep covering almost the full speed range.
            local = (s - 0.30) / (0.48 - 0.30)
            q = 2.0 * np.pi * args.speed_sine_cycles * local - 0.5 * np.pi
            omega = omega_mid + 0.5 * omega_span * np.sin(q)
            ap = ap_lo
            label = "speed sinusoid"

        elif s < 0.63:
            # 4) Depth staircase at a moderate spindle speed.
            local = (s - 0.48) / (0.63 - 0.48)
            levels = np.array([0.0, 0.35, 0.70, 1.0])
            j = min(int(local * len(levels)), len(levels) - 1)
            omega = omega_lo + 0.60 * omega_span
            ap = ap_lo + levels[j] * ap_span
            label = "depth staircase"

        elif s < 0.78:
            # 5) Smooth depth modulation at fixed speed.
            local = (s - 0.63) / (0.78 - 0.63)
            q = 2.0 * np.pi * args.depth_sine_cycles * local - 0.5 * np.pi
            omega = omega_lo + 0.68 * omega_span
            ap = ap_mid + 0.5 * ap_span * np.sin(q)
            label = "depth sinusoid"

        else:
            # 6) Mixed excitation.  Different frequencies/phases avoid a simple
            # one-dimensional path in the (omega, ap) plane.
            local = (s - 0.78) / (1.0 - 0.78)
            q = 2.0 * np.pi * args.mixed_speed_cycles * local
            omega = omega_mid + 0.5 * omega_span * np.sin(q)
            ap = ap_mid + 0.5 * ap_span * np.sin(
                args.mixed_depth_ratio * q + args.ap_phase
            )
            label = "mixed excitation"

    omega = float(np.clip(omega, pp.omega_min, pp.omega_max))
    if ap_fixed is None:
        ap = float(np.clip(ap, pp.ap_min, pp.ap_max))
    else:
        ap = float(ap_fixed)

    return CommandPoint(omega=omega, ap=ap, label=label)


def _profile_change_indices(labels: list[str]) -> list[int]:
    """Indices at which the named excitation regime changes."""
    if not labels:
        return []
    out = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            out.append(i)
    return out


# -----------------------------------------------------------------------------
# Main validation
# -----------------------------------------------------------------------------

def run_validation(args: argparse.Namespace) -> None:
    register_envs()

    env_kwargs = plate_env_kwargs(
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=args.n_substeps,
        max_episode_steps=args.max_episode_steps,
        dynamics_uncertainty_std=0.0,
        randomize_y0=False,
    )
    env = gym.make(args.env_id, **env_kwargs)
    plant = env.unwrapped.plant

    pp = PhasePlant(env, n_modes=args.n_modes)
    K = pp.K
    p = pp.p
    dtheta = args.dtheta_frac * p
    env_dt = float(args.dt) * int(args.n_substeps)

    M = args.n_delay if args.n_delay is not None else auto_delay_order(pp, tol=args.delay_tol)
    delay_check = _delay_error_data(pp, M, args.delay_omega_grid)

    # Reference sampling adequacy: phase-cell duration is shortest at omega_max.
    nominal_min_samples = dtheta / max(pp.omega_max * env_dt, 1e-30)
    if nominal_min_samples < args.min_samples_per_cell:
        print(
            "WARNING: the full-model reference may be too coarse: "
            f"only ~{nominal_min_samples:.1f} samples/cell at omega_max. "
            f"Target >= {args.min_samples_per_cell}. Reduce dt or n_substeps."
        )

    # Build the same phase-domain model used by NMPC-MHE.
    model, _ = build_model(pp, M, args.chip_eps, args.n_quad)
    sim = Simulator(model)
    sim.settings.t_step = dtheta
    sim.settings.integration_tool = "idas"

    sched = PhaseSchedule(pp, dtheta, 0.0, args.n_quad)
    tvp_template = sim.get_tvp_template()

    def tvp_fun(_t_now):
        phi, g, s = sched.at(0)
        tvp_template["phi"] = phi.reshape(-1, 1)
        tvp_template["gvec"] = g.reshape(-1, 1)
        tvp_template["svec"] = s.reshape(-1, 1)
        return tvp_template

    sim.set_tvp_fun(tvp_fun)
    sim.setup()

    # Paper model has 2K+M states. For compatibility with an older implementation
    # that still contains a nominal dF state, allow one extra state and hold it at 0.
    n_model = int(getattr(model, "n_x", model.x.shape[0]))
    n_paper = 2 * K + M
    if n_model not in (n_paper, n_paper + 1):
        raise RuntimeError(
            f"Unexpected internal-model dimension n_x={n_model}; expected {n_paper} "
            f"(paper model) or {n_paper+1} (legacy model with dF)."
        )
    legacy_dF = n_model == n_paper + 1
    if legacy_dF:
        print("NOTE: build_model still contains legacy dF; it is initialized at zero and remains nominal in this validation.")

    print("===== Internal-model validation =====")
    print(f"env dt                : {env_dt:.3e} s")
    print(f"phase step Delta theta: {dtheta:.6f} rad = {args.dtheta_frac:.3f} p")
    print(f"retained modes K      : {K}")
    print(f"delay order M         : {M}")
    print(f"delay max error       : {delay_check['max_error']:.3e}")
    print(f"chip smoothing eps    : {args.chip_eps:.3e} mm")
    print(f"quadrature nodes      : {args.n_quad} per active segment")

    # ------------------------------ Warm-up ----------------------------------
    obs, info = env.reset(seed=args.seed)
    control_ap = bool(getattr(pp, "control_ap", True))
    ap_fixed = None
    if not control_ap:
        ap_fixed = float(info.get("ap_fixed_mm", getattr(plant, "_episode_ap", args.ap_constant)))

    warmup_omega = float(np.clip(args.warmup_omega, pp.omega_min, pp.omega_max))
    warmup_ap = float(ap_fixed if ap_fixed is not None else np.clip(args.warmup_ap, pp.ap_min, pp.ap_max))
    warmup_action = _to_env_action(plant, warmup_omega, warmup_ap)

    # Ensure at least several tooth pitches are available for delay initialization.
    min_warmup_steps = int(np.ceil(args.warmup_pitches * p / max(warmup_omega * env_dt, 1e-30))) + 10
    warmup_steps = max(int(args.warmup_steps), min_warmup_steps)
    print(f"warm-up               : {warmup_steps} plant samples")

    warm_theta_raw: list[float] = []
    warm_wc_ret: list[float] = []

    for _ in range(warmup_steps):
        obs, _, term, trunc, info = env.step(warmup_action)
        if term or trunc:
            raise RuntimeError(f"Warm-up terminated: {info.get('termination_reason')}")

        theta_raw = float(info.get("spindle_phase_rad", 0.0))
        eta_full, _ = _deinterleave_modal(info.get("x_modal", np.zeros(2 * pp.K_full)), pp.K_full)
        eta_ret = eta_full[pp.sel]
        xc = float(info.get("cutter_x", plant.x0_cutter))
        yc = float(info.get("cutter_y", plant.y_cutter))
        wc_ret = float(pp.phi_at(xc, yc) @ eta_ret)

        warm_theta_raw.append(theta_raw)
        warm_wc_ret.append(wc_ret)

    warm_theta = np.unwrap(np.asarray(warm_theta_raw, dtype=np.float64))
    warm_wc_ret_arr = np.asarray(warm_wc_ret, dtype=np.float64)
    if warm_theta[-1] - warm_theta[0] < p:
        raise RuntimeError("Warm-up history spans less than one tooth pitch; increase --warmup-steps.")

    # Remove duplicate phases, if any, before spline interpolation.
    warm_theta_u, unique_idx = np.unique(warm_theta, return_index=True)
    warm_wc_u = warm_wc_ret_arr[unique_idx]
    if warm_theta_u.size < 4:
        raise RuntimeError("Insufficient unique warm-up phase samples for delay initialization.")

    eta_full_cur, v_full_cur = _deinterleave_modal(
        info.get("x_modal", np.zeros(2 * pp.K_full)), pp.K_full
    )
    eta_cur = eta_full_cur[pp.sel]
    v_cur = v_full_cur[pp.sel]
    theta_now = float(warm_theta[-1])
    theta_raw_prev = float(warm_theta_raw[-1])
    xc_now = float(info.get("cutter_x", plant.x0_cutter))
    yc_now = float(info.get("cutter_y", plant.y_cutter))

    xi, _ = cheb_lobatto_D(M)
    target_history_phase = theta_now - p * xi[1:]
    if np.min(target_history_phase) < warm_theta_u[0] - 1e-12:
        raise RuntimeError("Warm-up does not cover the full one-tooth history required by q_h.")

    history_spline = CubicSpline(warm_theta_u, warm_wc_u, extrapolate=False)
    qh_init = np.asarray(history_spline(target_history_phase), dtype=np.float64)
    if not np.all(np.isfinite(qh_init)):
        raise RuntimeError("Failed to initialize the pseudospectral delay state from warm-up history.")

    z0 = np.concatenate([eta_cur, v_cur, qh_init])
    if legacy_dF:
        z0 = np.concatenate([z0, [0.0]])
    z0 = z0.reshape(-1, 1)
    sim.x0 = z0.copy()
    sim.set_initial_guess()

    # --------------------------- Plant data store -----------------------------
    initial_w = np.asarray(info.get("w_sensor", np.zeros(pp.nS)), dtype=np.float64).reshape(-1)
    initial_Fz = _extract_plant_force(info)
    initial_wc_full = float(pp._phi_full(xc_now, yc_now) @ eta_full_cur)
    initial_wc_ret = float(pp.phi_at(xc_now, yc_now) @ eta_cur)

    plant_data: dict[str, list[Any]] = {
        "time": [0.0],
        "theta": [theta_now],
        "w_sensor": [initial_w],
        "eta_ret": [eta_cur.copy()],
        "wc_full": [initial_wc_full],
        "wc_ret": [initial_wc_ret],
        "Fz": [initial_Fz],
        "xc": [xc_now],
        "yc": [yc_now],
    }

    # Preserve warm-up history for evaluating w(theta-p) at early validation cells.
    history_theta = list(warm_theta_u)
    history_wc_ret = list(warm_wc_u)

    cell_data: list[dict[str, Any]] = []
    model_state = z0.copy()
    cell_theta = theta_now
    cell_time = 0.0
    plant_steps = 0
    max_env_steps = int(np.ceil(args.max_duration / env_dt)) + 2
    min_actual_samples = 10**9

    term = trunc = False
    safety_abort = False
    safety_reason = None

    while plant_steps < max_env_steps and len(cell_data) < args.n_cells:
        k = len(cell_data)
        cmd = _command_for_cell(args, pp, k, args.n_cells, ap_fixed)
        omega_cmd, ap_cmd = cmd.omega, cmd.ap
        action = _to_env_action(plant, omega_cmd, ap_cmd)
        target_theta = cell_theta + dtheta

        # Exact plant quantities at the beginning of the phase cell.
        th_arr = np.asarray(plant_data["theta"], dtype=np.float64)
        eta_arr = np.vstack(plant_data["eta_ret"])
        xc_arr = np.asarray(plant_data["xc"], dtype=np.float64)
        yc_arr = np.asarray(plant_data["yc"], dtype=np.float64)
        w_arr = np.vstack(plant_data["w_sensor"])
        wc_full_arr = np.asarray(plant_data["wc_full"], dtype=np.float64)

        eta_plant_start = _interp_vector(th_arr, eta_arr, cell_theta)
        xc_start = _interp_scalar(th_arr, xc_arr, cell_theta)
        yc_start = _interp_scalar(th_arr, yc_arr, cell_theta)
        w_plant_start = _interp_vector(th_arr, w_arr, cell_theta)
        wc_full_start = _interp_scalar(th_arr, wc_full_arr, cell_theta)

        # Retained delayed displacement from actual preceding history.
        hist_th = np.asarray(history_theta + plant_data["theta"], dtype=np.float64)
        hist_wc = np.asarray(history_wc_ret + plant_data["wc_ret"], dtype=np.float64)
        order = np.argsort(hist_th)
        hist_th = hist_th[order]
        hist_wc = hist_wc[order]
        hist_th, unique_hist_idx = np.unique(hist_th, return_index=True)
        hist_wc = hist_wc[unique_hist_idx]
        wd_plant_ret = _interp_scalar(hist_th, hist_wc, cell_theta - p)

        # Isolated force-approximation test at the same local reduced state.
        Fz_force_ref = _dense_exact_cell_force(
            pp, cell_theta, dtheta, xc_start, yc_start,
            eta_plant_start, wd_plant_ret, ap_cmd, args.force_ref_samples,
        )
        Fz_force_int = _internal_cell_force(
            pp, cell_theta, dtheta, xc_start, yc_start,
            eta_plant_start, wd_plant_ret, ap_cmd,
            args.chip_eps, args.n_quad,
        )

        # Advance the full plant until it crosses the target phase.
        samples_before = len(plant_data["theta"])
        crossed = False
        while plant_steps < max_env_steps:
            obs, _, term, trunc, info = env.step(action)
            plant_steps += 1
            t = plant_steps * env_dt

            theta_raw = float(info.get("spindle_phase_rad", 0.0))
            theta_unwrapped = _unwrap_increment(
                theta_raw,
                theta_raw_prev,
                float(plant_data["theta"][-1]),
            )
            theta_raw_prev = theta_raw

            eta_full, _ = _deinterleave_modal(
                info.get("x_modal", np.zeros(2 * pp.K_full)), pp.K_full
            )
            eta_ret = eta_full[pp.sel]
            xc = float(info.get("cutter_x", plant.x0_cutter))
            yc = float(info.get("cutter_y", plant.y_cutter))
            wc_ret = float(pp.phi_at(xc, yc) @ eta_ret)
            wc_full = float(pp._phi_full(xc, yc) @ eta_full)

            w_sensor_now = np.asarray(
                info.get("w_sensor", np.zeros(pp.nS)), dtype=np.float64
            ).reshape(-1)

            plant_data["time"].append(t)
            plant_data["theta"].append(theta_unwrapped)
            plant_data["w_sensor"].append(w_sensor_now)
            plant_data["eta_ret"].append(eta_ret.copy())
            plant_data["wc_full"].append(wc_full)
            plant_data["wc_ret"].append(wc_ret)
            plant_data["Fz"].append(_extract_plant_force(info))
            plant_data["xc"].append(xc)
            plant_data["yc"].append(yc)

            # Conservative guard against a divergent reference trajectory.
            # It is not part of either model; it only prevents an intentionally
            # broad validation profile from running numerically into infinity.
            if args.abort_w_frac > 0.0:
                w_abs = float(np.max(np.abs(w_sensor_now))) if w_sensor_now.size else 0.0
                if w_abs >= args.abort_w_frac * pp.w_limit:
                    safety_abort = True
                    safety_reason = (
                        f"|w|={w_abs:.3e} m reached "
                        f"{args.abort_w_frac:.2f} w_limit"
                    )

            crossed = theta_unwrapped >= target_theta
            if crossed or term or trunc or safety_abort:
                break

        if not crossed:
            why = safety_reason if safety_abort else info.get("termination_reason")
            print(f"Validation stopped before completing cell {k}: {why}")
            break

        samples_this_cell = len(plant_data["theta"]) - samples_before
        min_actual_samples = min(min_actual_samples, samples_this_cell)

        # Exact full-plant quantities at the phase-cell endpoint.
        th_arr = np.asarray(plant_data["theta"], dtype=np.float64)
        time_arr = np.asarray(plant_data["time"], dtype=np.float64)
        w_arr = np.vstack(plant_data["w_sensor"])
        eta_arr = np.vstack(plant_data["eta_ret"])
        wc_full_arr = np.asarray(plant_data["wc_full"], dtype=np.float64)
        Fz_arr = np.asarray(plant_data["Fz"], dtype=np.float64)
        xc_arr = np.asarray(plant_data["xc"], dtype=np.float64)
        yc_arr = np.asarray(plant_data["yc"], dtype=np.float64)

        time_end = _interp_scalar(th_arr, time_arr, target_theta)
        w_plant_end = _interp_vector(th_arr, w_arr, target_theta)
        eta_plant_end = _interp_vector(th_arr, eta_arr, target_theta)
        wc_full_end = _interp_scalar(th_arr, wc_full_arr, target_theta)
        xc_end = _interp_scalar(th_arr, xc_arr, target_theta)
        yc_end = _interp_scalar(th_arr, yc_arr, target_theta)
        Fz_plant_phase_avg = _phase_average(th_arr, Fz_arr, cell_theta, target_theta)

        # Advance the internal model over the same phase cell.
        sched.set_now(cell_theta, xc_start, yc_start)
        sim.x0 = model_state.copy()
        sim.set_initial_guess()
        sim.make_step(np.array([[omega_cmd], [ap_cmd]], dtype=np.float64))
        z_next = _model_state_vector(sim)

        eta_model_start = model_state[:K, 0].copy()
        eta_model_end = z_next[:K].copy()
        w_model_start = pp.Phi @ eta_model_start
        w_model_end = pp.Phi @ eta_model_end
        wc_model_start = float(pp.phi_at(xc_start, yc_start) @ eta_model_start)
        wc_model_end = float(pp.phi_at(xc_end, yc_end) @ eta_model_end)

        cell_data.append({
            "cell": k,
            "theta_start": float(cell_theta),
            "theta_end": float(target_theta),
            "time_start": float(cell_time),
            "time_end": float(time_end),
            "omega": float(omega_cmd),
            "ap": float(ap_cmd),
            "profile": cmd.label,
            "samples": int(samples_this_cell),
            "phase_overshoot": float(th_arr[-1] - target_theta),
            "w_plant_start": w_plant_start,
            "w_plant_end": w_plant_end,
            "w_model_start": w_model_start,
            "w_model_end": w_model_end,
            "wc_plant_start": float(wc_full_start),
            "wc_plant_end": float(wc_full_end),
            "wc_model_start": float(wc_model_start),
            "wc_model_end": float(wc_model_end),
            "eta_plant_start": eta_plant_start,
            "eta_plant_end": eta_plant_end,
            "eta_model_start": eta_model_start,
            "eta_model_end": eta_model_end,
            "Fz_force_ref": float(Fz_force_ref),
            "Fz_force_int": float(Fz_force_int),
            "Fz_plant_phase_avg": float(Fz_plant_phase_avg),
        })

        model_state = z_next.reshape(-1, 1)
        cell_theta = target_theta
        cell_time = time_end

        if safety_abort:
            print(f"Safety guard triggered after completed cell {k}: {safety_reason}")
            break
        if term or trunc:
            print(f"Plant terminated after completed cell {k}: {info.get('termination_reason')}")
            break

    env.close()

    if not cell_data:
        raise RuntimeError("No complete validation cells were obtained.")

    # ------------------------------ Metrics ----------------------------------
    phase_nodes = np.array(
        [cell_data[0]["theta_start"]] + [d["theta_end"] for d in cell_data],
        dtype=np.float64,
    )
    phase_rel = phase_nodes - phase_nodes[0]

    w_plant_nodes = np.vstack(
        [cell_data[0]["w_plant_start"]] + [d["w_plant_end"] for d in cell_data]
    )
    w_model_nodes = np.vstack(
        [cell_data[0]["w_model_start"]] + [d["w_model_end"] for d in cell_data]
    )
    wc_plant_nodes = np.array(
        [cell_data[0]["wc_plant_start"]] + [d["wc_plant_end"] for d in cell_data],
        dtype=np.float64,
    )
    wc_model_nodes = np.array(
        [cell_data[0]["wc_model_start"]] + [d["wc_model_end"] for d in cell_data],
        dtype=np.float64,
    )

    force_ref = np.array([d["Fz_force_ref"] for d in cell_data], dtype=np.float64)
    force_int = np.array([d["Fz_force_int"] for d in cell_data], dtype=np.float64)
    force_plant = np.array([d["Fz_plant_phase_avg"] for d in cell_data], dtype=np.float64)

    sensor_rel_rms = _relative_rms(w_plant_nodes, w_model_nodes)
    cutter_rel_rms = _relative_rms(wc_plant_nodes, wc_model_nodes)
    force_map_rel_rms = _relative_rms(force_ref, force_int)
    force_overall_rel_rms = _relative_rms(force_plant, force_int)

    metrics = {
        "n_cells": len(cell_data),
        "K": int(K),
        "M": int(M),
        "dtheta_rad": float(dtheta),
        "dtheta_fraction_pitch": float(args.dtheta_frac),
        "chip_eps_mm": float(args.chip_eps),
        "n_quad": int(args.n_quad),
        "delay_max_abs_error": float(delay_check["max_error"]),
        "force_map_relative_rms": float(force_map_rel_rms),
        "sensor_displacement_relative_rms": float(sensor_rel_rms),
        "cutter_displacement_relative_rms": float(cutter_rel_rms),
        "overall_force_relative_rms_diagnostic": float(force_overall_rel_rms),
        "min_actual_plant_samples_per_cell": int(min_actual_samples),
        "max_phase_overshoot_rad": float(max(d["phase_overshoot"] for d in cell_data)),
        "legacy_dF_present": bool(legacy_dF),
        "command_profile": str(args.command_profile),
        "requested_cells": int(args.n_cells),
        "profile_completed": bool(len(cell_data) >= args.n_cells),
        "safety_abort": bool(safety_abort),
        "safety_reason": safety_reason,
        "omega_command_min_rad_s": float(min(d["omega"] for d in cell_data)),
        "omega_command_max_rad_s": float(max(d["omega"] for d in cell_data)),
        "ap_command_min_mm": float(min(d["ap"] for d in cell_data)),
        "ap_command_max_mm": float(max(d["ap"] for d in cell_data)),
    }

    print("\n===== Validation metrics =====")
    print(f"Delay max |G_M-exp(-sp)|        : {metrics['delay_max_abs_error']:.3e}")
    print(f"Force-map relative RMS          : {100*force_map_rel_rms:.3f}%")
    print(f"Sensor displacement relative RMS: {100*sensor_rel_rms:.3f}%")
    print(f"Cutter displacement relative RMS: {100*cutter_rel_rms:.3f}%")
    print(f"Overall force diagnostic RMS    : {100*force_overall_rel_rms:.3f}%")
    print(f"Min plant samples/cell          : {min_actual_samples}")

    # ------------------------------- Output ----------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "validation_cells.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cell", "theta_start", "theta_end", "time_start_s", "time_end_s",
            "omega_rad_s", "ap_mm", "profile", "samples", "phase_overshoot_rad",
            "Fz_exact_reduced_N", "Fz_internal_N", "Fz_full_plant_phase_avg_N",
        ])
        for d in cell_data:
            writer.writerow([
                d["cell"], d["theta_start"], d["theta_end"], d["time_start"], d["time_end"],
                d["omega"], d["ap"], d["profile"], d["samples"], d["phase_overshoot"],
                d["Fz_force_ref"], d["Fz_force_int"], d["Fz_plant_phase_avg"],
            ])

    # ------------------------------------------------------------------
    # Publication-oriented comparison in PHYSICAL TIME.
    # Trajectories, phase-cell forces, and the zero-order-held inputs are
    # shown on one shared physical-time axis for direct interpretation.
    # The numerical comparison itself remains phase-synchronous: each model
    # node corresponds to the same phase crossing, with its measured physical
    # time inherited from the full reference simulator.
    # ------------------------------------------------------------------
    time_nodes = np.array(
        [cell_data[0]["time_start"]] + [d["time_end"] for d in cell_data],
        dtype=np.float64,
    )
    cell_centers_time = 0.5 * np.array(
        [d["time_start"] + d["time_end"] for d in cell_data], dtype=np.float64
    )

    omega_cells = np.array([d["omega"] for d in cell_data], dtype=np.float64)
    ap_cells = np.array([d["ap"] for d in cell_data], dtype=np.float64)
    labels = [str(d["profile"]) for d in cell_data]

    # ZOH input traces use cell edges so the applied hold is explicit.
    time_edges = time_nodes.copy()
    omega_zoh = np.r_[omega_cells, omega_cells[-1]]
    ap_zoh = np.r_[ap_cells, ap_cells[-1]]

    # Cell-level internal force approximation: one representative phase-averaged
    # value is available for each phase cell.  For visualization in physical time,
    # extend each value across the corresponding cell as a zero-order-held trace.
    # This is a plotting representation of the cellwise force approximation; it
    # does not alter the continuous internal-model propagation within the cell.
    force_int_zoh = np.r_[force_int, force_int[-1]]
    change_idx = _profile_change_indices(labels)

    fig, axes = plt.subplots(5, 1, figsize=(9.2, 11.8), sharex=True)

    sensor_idx = int(np.clip(args.sensor_index, 0, pp.nS - 1))
    axes[0].plot(time_nodes, 1e6 * w_plant_nodes[:, sensor_idx], label="Full model")
    axes[0].plot(time_nodes, 1e6 * w_model_nodes[:, sensor_idx], "--", label="Internal model")
    axes[0].set_ylabel("Sensor displacement [µm]")
    axes[0].set_title(f"Sensor {sensor_idx} response")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_nodes, 1e6 * wc_plant_nodes, label="Full model")
    axes[1].plot(time_nodes, 1e6 * wc_model_nodes, "--", label="Internal model")
    axes[1].set_ylabel("Cutter displacement [µm]")
    axes[1].set_title("Cutter-point response")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(cell_centers_time, force_ref, "o", label="Exact reduced force map")
    axes[2].step(
        time_edges, force_int_zoh, where="post",
        label="Internal force approximation (cellwise ZOH)",
    )
    if np.any(np.isfinite(force_plant)):
        axes[2].plot(cell_centers_time, force_plant, ".", label="Full plant phase average")
    axes[2].set_ylabel("Axial force [N]")
    axes[2].set_title("Phase-cell force comparison in physical time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].step(time_edges, omega_zoh, where="post", label=r"$\omega_k$ (ZOH)")
    axes[3].axhline(pp.omega_min, linestyle=":", linewidth=0.9, label="admissible bounds")
    axes[3].axhline(pp.omega_max, linestyle=":", linewidth=0.9)
    axes[3].set_ylabel(r"$\omega$ [rad/s]")
    axes[3].set_title("Applied spindle-speed command")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.25)

    axes[4].step(time_edges, ap_zoh, where="post", label=r"$a_{p,k}$ (ZOH)")
    axes[4].axhline(pp.ap_min, linestyle=":", linewidth=0.9, label="admissible bounds")
    axes[4].axhline(pp.ap_max, linestyle=":", linewidth=0.9)
    axes[4].set_xlabel("Physical time [s]")
    axes[4].set_ylabel(r"$a_p$ [mm]")
    axes[4].set_title("Applied axial-depth command")
    axes[4].legend(loc="best")
    axes[4].grid(True, alpha=0.25)

    # Mark transitions between designed excitation regimes on the input panels.
    for idx in change_idx[1:]:
        if idx < len(time_edges):
            x = time_edges[idx]
            axes[3].axvline(x, linestyle="--", linewidth=0.7, alpha=0.55)
            axes[4].axvline(x, linestyle="--", linewidth=0.7, alpha=0.55)
    for idx in change_idx:
        if idx < len(labels) and idx < len(time_edges):
            axes[3].text(
                time_edges[idx], 1.02, labels[idx],
                transform=axes[3].get_xaxis_transform(),
                fontsize=7, va="bottom", ha="left",
            )

    fig.tight_layout()
    fig.savefig(out_dir / "internal_model_validation_time.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_dir / "internal_model_validation_time.pdf", bbox_inches="tight")
    plt.close(fig)

    # Standalone physical-time input figure for manuscript use when the inputs
    # need to be shown separately from the response comparison.
    fig_u, ax_u = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True)
    ax_u[0].step(time_edges, omega_zoh, where="post")
    ax_u[0].axhline(pp.omega_min, linestyle=":", linewidth=0.9, label="admissible bounds")
    ax_u[0].axhline(pp.omega_max, linestyle=":", linewidth=0.9)
    ax_u[0].set_ylabel(r"Spindle speed $\omega$ [rad/s]")
    ax_u[0].set_title("Applied validation inputs (phase-cell zero-order hold)")
    ax_u[0].grid(True, alpha=0.25)
    ax_u[0].legend(loc="best")

    ax_u[1].step(time_edges, ap_zoh, where="post")
    ax_u[1].axhline(pp.ap_min, linestyle=":", linewidth=0.9, label="admissible bounds")
    ax_u[1].axhline(pp.ap_max, linestyle=":", linewidth=0.9)
    ax_u[1].set_xlabel("Physical time [s]")
    ax_u[1].set_ylabel(r"Axial depth $a_p$ [mm]")
    ax_u[1].grid(True, alpha=0.25)
    ax_u[1].legend(loc="best")

    for idx in change_idx[1:]:
        if idx < len(time_edges):
            x = time_edges[idx]
            for a in ax_u:
                a.axvline(x, linestyle="--", linewidth=0.7, alpha=0.55)

    fig_u.tight_layout()
    fig_u.savefig(out_dir / "validation_inputs_time.png", dpi=args.dpi, bbox_inches="tight")
    fig_u.savefig(out_dir / "validation_inputs_time.pdf", bbox_inches="tight")
    plt.close(fig_u)

    # Independent delay-fidelity figure.
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.0))
    ax2.semilogy(delay_check["nu"], delay_check["error"], ".", markersize=3)
    ax2.axhline(args.delay_tol, linestyle="--", label="delay-order tolerance")
    ax2.set_xlabel(r"Phase frequency $\nu$ [rad$^{-1}$]")
    ax2.set_ylabel(r"$|G_M(j\nu)-e^{-j\nu p}|$")
    ax2.set_title("Pseudospectral delay approximation")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "delay_fidelity.png", dpi=args.dpi, bbox_inches="tight")
    fig2.savefig(out_dir / "delay_fidelity.pdf", bbox_inches="tight")
    plt.close(fig2)

    print(f"\nValidation complete. Results saved in: {out_dir.resolve()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the phase-domain NMPC-MHE internal model against the true "
            "full-order Gymnasium milling simulator."
        )
    )
    parser.add_argument("--env-id", default="CustomODEPlate-v0")
    parser.add_argument("--reward", default="dense")

    # Fine full-order reference integration.  Keep n_substeps=1 unless the
    # environment definition requires otherwise; phase-cell force averages need
    # several full-plant samples per Delta-theta interval.
    parser.add_argument("--dt", type=float, default=2.5e-5)
    parser.add_argument("--n-substeps", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=400000)
    parser.add_argument("--max-duration", type=float, default=3.0,
                        help="Safety ceiling on physical validation duration [s].")
    parser.add_argument("--seed", type=int, default=0)

    # Designed input excitation.  ``research`` is recommended for the paper:
    # baseline -> speed steps -> speed sine -> depth steps -> depth sine -> mixed.
    parser.add_argument(
        "--command-profile",
        choices=["research", "constant", "multistep", "sine"],
        default="research",
    )
    parser.add_argument("--n-cells", type=int, default=160,
                        help="Number of phase cells in the designed excitation profile.")
    parser.add_argument("--omega-margin-frac", type=float, default=0.03,
                        help="Fraction of omega span excluded at each hard bound.")
    parser.add_argument("--ap-low-frac", type=float, default=0.03,
                        help="Lowest test depth as a fraction of the admissible ap span.")
    parser.add_argument("--ap-high-frac", type=float, default=0.30,
                        help="Highest default test depth as a fraction of the ap span. "
                             "Increase only after confirming bounded response.")
    parser.add_argument("--omega-constant", type=float, default=100.0)
    parser.add_argument("--ap-constant", type=float, default=0.05)
    parser.add_argument("--profile-cycles", type=float, default=2.0,
                        help="Cycles for --command-profile sine.")
    parser.add_argument("--ap-frequency-ratio", type=float, default=0.55,
                        help="ap/omega frequency ratio for the generic sine profile.")
    parser.add_argument("--ap-phase", type=float, default=np.pi / 3.0)
    parser.add_argument("--speed-sine-cycles", type=float, default=1.5)
    parser.add_argument("--depth-sine-cycles", type=float, default=1.25)
    parser.add_argument("--mixed-speed-cycles", type=float, default=1.5)
    parser.add_argument("--mixed-depth-ratio", type=float, default=0.55)

    # Runtime guard only; it does not modify either model.  The default research
    # profile is intentionally conservative in ap, but this prevents a divergent
    # full-order reference trajectory from contaminating the validation.
    parser.add_argument("--abort-w-frac", type=float, default=0.90,
                        help="Abort if max sensor |w| reaches this fraction of w_limit; "
                             "set <=0 to disable.")

    # Internal-model resolution: set these equal to the NMPC-MHE study values.
    parser.add_argument("--n-modes", type=int, default=3)
    parser.add_argument("--n-delay", type=int, default=None)
    parser.add_argument("--delay-tol", type=float, default=1e-6)
    parser.add_argument("--delay-omega-grid", type=int, default=80)
    parser.add_argument("--dtheta-frac", type=float, default=0.10)
    parser.add_argument("--chip-eps", type=float, default=2e-3)
    parser.add_argument("--n-quad", type=int, default=2)
    parser.add_argument("--force-ref-samples", type=int, default=512,
                        help="Dense midpoint samples for the hard-switch reduced-force reference.")

    # Delay-history initialization and reference-quality checks.
    parser.add_argument("--warmup-omega", type=float, default=100.0)
    parser.add_argument("--warmup-ap", type=float, default=0.05)
    parser.add_argument("--warmup-pitches", type=float, default=3.0)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--min-samples-per-cell", type=float, default=10.0)

    # Output.
    parser.add_argument("--sensor-index", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--output-dir", default="internal_model_validation")

    args = parser.parse_args()
    if args.n_cells < 8:
        parser.error("--n-cells should be at least 8; use >=80 for the research profile.")
    if not (0.0 <= args.ap_low_frac <= args.ap_high_frac <= 1.0):
        parser.error("Require 0 <= ap-low-frac <= ap-high-frac <= 1.")

    run_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
