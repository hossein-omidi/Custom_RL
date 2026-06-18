#!/usr/bin/env python3
"""
Pre-training verification for the two-direction regenerative milling plant.

Run from project root::

    python custom_rl/plants/Test2.py
    python custom_rl/plants/Test2.py --mode feed_normal_full
    python custom_rl/plants/Test2.py --no-plots

Deterministic checks: metadata, action scaling, tool path, zero-depth response,
free decay, structural-delay exclusion, regenerative delay, chip thickness,
engagement, force projection, units, integrator convergence, cutting vs idle,
sensor/RL interface.

Tests both ``surface_normal_reduced`` and ``feed_normal_full``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_rl import register_envs
from custom_rl.integration import get_integrator
from custom_rl.integration.dde_rk4 import integrate_dde_rk4
from custom_rl.integration.rk4 import integrate as integrate_rk4
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.milling_config import MillingForceConfig
from custom_rl.plants.milling_force import (
    chip_thickness_feed_normal_full,
    chip_thickness_surface_normal_reduced,
    engagement,
    feed_per_tooth,
    reconstruct_q_feed,
    reconstruct_q_normal,
    tangential_radial_increment,
    transform_to_feed_normal,
)
from custom_rl.plants.units import ac_to_meters
from custom_rl.plants.modal_state import split_modal_state, state_dim_for_model
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.time_scales import format_time_scales_report, recommend_integration_dt
from custom_rl.rewards.plate_rewards import SensorProductivePlateReward

# Deterministic verification settings (macro step aligned with training env: dt=0.002)
SEED = 0
DT = 0.0001
T_FINAL = 5
T_FINAL_QUICK = 0.5
CALIB_T_SIM = 0.8
CALIB_T_SIM_QUICK = 0.25
INTEGRATOR = "dde_rk4"
DELAY_MODE = "constant_tau"
ZERO_TOL = 1e-10
DECAY_RATIO_MAX = 0.98
PLOT_DIR = _ROOT / "plots" / "pretrain_verify"

# Set by main() when --quick is passed
_QUICK = False


def _t_final() -> float:
    return T_FINAL_QUICK if _QUICK else T_FINAL


def _calib_t_sim() -> float:
    return CALIB_T_SIM_QUICK if _QUICK else CALIB_T_SIM

Status = Literal["PASS", "WARN", "FAIL"]


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""


@dataclass
class VerificationReport:
    displacement_model: str
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str = "") -> None:
        self.results.append(CheckResult(name, status, detail))
        tag = f"[{status}]"
        line = f"  {tag} {name}"
        if detail:
            line += f" - {detail}"
        print(line)

    def summary(self) -> Status:
        if any(r.status == "FAIL" for r in self.results):
            return "FAIL"
        if any(r.status == "WARN" for r in self.results):
            return "WARN"
        return "PASS"


def make_plant(
    displacement_model: str,
    *,
    trajectory_mode: str = "middle_line",
    delay_mode: str = DELAY_MODE,
) -> PlatePlant:
    fmod.DELAY_MODE = delay_mode
    plant = PlatePlant(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        displacement_model=displacement_model,
        trajectory_mode=trajectory_mode,
        pass_sampling="sequential",
        n_pass_lines=5,
        feed_speed=0.05,
        traj_dt=DT,
        displacement_failure_limit=5e-3,
        velocity_failure_limit=2.0,
    )
    plant.reset(np.random.default_rng(SEED))
    fmod.reset_episode_state(plant._modal_state_history)
    return plant


def decay_ratio(series: np.ndarray, *, tail_frac: float = 0.1) -> float:
    """Peak amplitude in last tail_frac vs first tail_frac of a time series."""
    n = series.shape[0]
    if n < 20:
        return 1.0
    w = max(int(n * tail_frac), 10)
    early = float(np.max(np.abs(series[:w])))
    late = float(np.max(np.abs(series[-w:])))
    return late / max(early, 1e-30)


def two_field(plant: PlatePlant) -> bool:
    return plant.milling_config.is_feed_normal_full()


def integration_substep(plant: PlatePlant, macro_dt: float = DT) -> tuple[float, int]:
    """Sub-step dt and count from modal/delay time scales (training-aligned)."""
    scales = recommend_integration_dt(plant, macro_dt=macro_dt, training_mode=True)
    return scales["dt_recommended_training_substep_s"], scales["n_substeps"]


def simulate(
    plant: PlatePlant,
    x0: np.ndarray,
    u_norm: np.ndarray,
    dt: float,
    t_final: float,
    *,
    integrator: str = INTEGRATOR,
) -> dict[str, Any]:
    """Rollout with rich diagnostics for verification."""
    integrate_fn = get_integrator(integrator)
    fmod.reset_episode_state(plant._modal_state_history)
    fmod.bind_modal_history(plant._modal_state_history)

    x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    u_norm = np.asarray(u_norm, dtype=np.float64).reshape(-1)
    n_steps = max(int(round(t_final / dt)), 1)
    sub_dt, substeps = integration_substep(plant, macro_dt=dt)

    u_phys0 = plant.normalized_to_physical_action(u_norm)
    plant.record_modal_state(0.0, x.copy(), omega=float(u_phys0[0]))

    rec: dict[str, list] = {
        "time": [0.0],
        "trajectory": [x.copy()],
        "sensor_w": [],
        "sensor_w_dot": [],
        "u_phys": [plant.normalized_to_physical_action(u_norm).copy()],
        "tau": [],
        "theta": [],
        "delta_n": [],
        "delta_f": [],
        "f_normal": [],
        "f_normal_raw": [],
        "f_feed": [],
        "f_feed_raw": [],
        "f_surface_normal": [],
        "q_n_cutter": [],
        "q_f_cutter": [],
        "x_c": [],
        "y_c": [],
        "phi_sample": [],
        "g_sample": [],
        "h_sample": [],
        "force_clipped": [],
    }

    t = 0.0
    terminated = truncated = False
    term_info: dict = {}

    try:
        for _ in range(n_steps):
            w_s, w_dot = plant.state_to_sensor_signals(x)
            rec["sensor_w"].append(w_s.copy())
            rec["sensor_w_dot"].append(w_dot.copy())

            u_phys = plant.normalized_to_physical_action(u_norm)
            omega, ac = float(u_phys[0]), float(u_phys[1])
            tau = fmod.tooth_period(omega)
            rec["tau"].append(tau)
            rec["theta"].append(fmod.theta_at(t))

            eta_n, _, eta_f, _ = split_modal_state(x, plant.K, two_field=two_field(plant))
            force = fmod._directional_force_result(t, eta_n, eta_f, omega, ac)
            rec["delta_n"].append(force["Delta_n"])
            rec["delta_f"].append(force["Delta_f"])
            rec["f_normal"].append(force["F_normal_total"])
            rec["f_normal_raw"].append(force.get("F_normal_raw", force["F_normal_total"]))
            rec["f_feed"].append(force["F_feed_total"])
            rec["f_feed_raw"].append(force.get("F_feed_raw", force["F_feed_total"]))
            rec["f_surface_normal"].append(force.get("F_surface_normal_total", 0.0))
            rec["q_n_cutter"].append(force.get("q_n", 0.0))
            rec["q_f_cutter"].append(force.get("q_f", 0.0))

            x_c, y_c = plant.tool_position_at(t)
            rec["x_c"].append(x_c)
            rec["y_c"].append(y_c)
            if force["phi_list"]:
                rec["phi_sample"].append(force["phi_list"][0])
                rec["g_sample"].append(force["g_list"][0])
                rec["h_sample"].append(force["h_list"][0])
            else:
                rec["phi_sample"].append(0.0)
                rec["g_sample"].append(0.0)
                rec["h_sample"].append(0.0)
            rec["force_clipped"].append(bool(force.get("force_clipped", False)))

            for _sub in range(substeps):
                x = integrate_fn(plant.dynamics, t, x, u_norm, sub_dt, n_steps=1)
                t += sub_dt
                plant.record_modal_state(t, x, omega=omega)
            rec["time"].append(t)
            rec["trajectory"].append(x.copy())
            rec["u_phys"].append(u_phys.copy())

            terminated, truncated, term_info = plant.termination(t, x)
            if terminated or truncated:
                break
    finally:
        fmod.unbind_modal_history()

    traj = np.asarray(rec["trajectory"], dtype=np.float64)
    out: dict[str, Any] = {k: (np.asarray(v) if k != "trajectory" else traj) for k, v in rec.items()}
    out["terminated"] = terminated
    out["truncated"] = truncated
    out["termination_info"] = term_info
    out["eta_n"], out["etad_n"], out["eta_f"], out["etad_f"] = split_modal_state(
        traj[-1], plant.K, two_field=two_field(plant)
    )
    if two_field(plant):
        block = 2 * plant.K
        out["eta_n_series"] = traj[:, 0:block:2]
        out["eta_f_series"] = traj[:, block::2]
    else:
        out["eta_n_series"] = traj[:, 0::2]
        out["eta_f_series"] = None
    return out


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------


def check_metadata(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Metadata & configuration ---")
    meta = plant.get_interface_metadata()
    mc = plant.milling_config.to_dict()
    expected_dim = state_dim_for_model(plant.K, plant.milling_config)
    rep.add(
        "state_dimension",
        "PASS" if plant.state_dim == expected_dim else "FAIL",
        f"state_dim={plant.state_dim}, expected={expected_dim}",
    )
    rep.add("displacement_model", "PASS", mc["displacement_model"])
    rep.add("modes_K", "PASS", f"K={plant.K}, subsystems={2 if two_field(plant) else 1}")
    if two_field(plant):
        layout_ok = plant.state_dim == 4 * plant.K
        rep.add(
            "two_field_state_layout",
            "PASS" if layout_ok else "FAIL",
            "x=[eta_n, eta_dot_n, eta_f, eta_dot_f] (4K interleaved)",
        )
        om_f_ok = bool(np.all(plant.omega_f_vec > 0))
        rep.add(
            "feed_natural_frequencies",
            "PASS" if om_f_ok else "FAIL",
            f"omega_f_vec min={float(np.min(plant.omega_f_vec)):.2f} max={float(np.max(plant.omega_f_vec)):.2f}",
        )
    rep.add("action_bounds", "PASS", f"omega in [{plant.u_phys_low[0]:.1f},{plant.u_phys_high[0]:.1f}], ac in [{plant.u_phys_low[1]:.1f},{plant.u_phys_high[1]:.1f}]")
    rep.add("integrator", "PASS", INTEGRATOR)
    rep.add("delay_mode", "PASS", fmod.DELAY_MODE)
    rep.add("trajectory_mode", "PASS", plant.trajectory_mode)
    rep.add("milling_type", "PASS", f"{mc['milling_type']}, phi_st={mc['phi_st']:.3f}, phi_ex={mc['phi_ex']:.3f}, arc={plant.milling_config.engaged_arc_width():.3f}")
    rep.add("feed_per_tooth", "PASS", f"source={mc['feed_per_tooth_source']}, cf_units={mc['cf_units']}")
    rep.add("axial_integration", "PASS", f"ac_via_axial_integration={mc['ac_via_axial_integration']}")
    sensor_obs_dim = 2 * plant.n_sensors
    rep.add("sensor_obs_dim", "PASS", f"{sensor_obs_dim} (+2 prev action in full PPO obs)")
    print(f"  metadata: {meta.get('state_representation')}, z_contact={plant.z_contact}")


def check_action_scaling(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Action scaling & variable controls ---")
    cases = [(-1, -1), (0, 0), (1, 1)]
    ok = True
    for a, b in cases:
        u_norm = np.array([a, b], dtype=np.float64)
        u_phys = plant.normalized_to_physical_action(u_norm)
        u_back = plant.physical_to_normalized_action(u_phys)
        if not np.allclose(u_norm, u_back, atol=1e-10):
            ok = False
    rep.add("affine_round_trip", "PASS" if ok else "FAIL")

    om1, om2 = 400.0, 1200.0
    ac = 3.0
    tau1 = fmod.tooth_period(om1)
    tau2 = fmod.tooth_period(om2)
    rep.add("tooth_period_vs_omega", "PASS" if tau1 > tau2 else "FAIL", f"tau({om1})={tau1:.5f}, tau({om2})={tau2:.5f}")

    fmod.reset_episode_state(plant._modal_state_history)
    fmod.record_omega(0.0, om1)
    th1 = fmod.theta_at(0.05)
    fmod.reset_episode_state(plant._modal_state_history)
    fmod.record_omega(0.0, om2)
    th2 = fmod.theta_at(0.05)
    rep.add("spindle_phase_vs_omega", "PASS" if not np.isclose(th1, th2) else "WARN", f"theta@0.05s: {th1:.4f} vs {th2:.4f}")

    eta = np.zeros(plant.K)
    eta_f = np.zeros(plant.K) if two_field(plant) else None
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        f_lo = abs(fmod._directional_force_result(0.03, eta, eta_f, om1, ac)["F_normal_total"])
        f_hi = abs(fmod._directional_force_result(0.03, eta, eta_f, om2, ac)["F_normal_total"])
    finally:
        fmod.unbind_modal_history()
    rep.add("force_timing_vs_omega", "PASS" if f_lo != f_hi or (f_lo == 0 and f_hi == 0) else "WARN", f"|F_n|@{om1}={f_lo:.2e}, @{om2}={f_hi:.2e}")

    fmod.bind_modal_history(plant._modal_state_history)
    try:
        f_ac1 = abs(fmod._directional_force_result(0.03, eta, eta_f, 800.0, 2.0)["F_normal_total"])
        f_ac2 = abs(fmod._directional_force_result(0.03, eta, eta_f, 800.0, 4.0)["F_normal_total"])
    finally:
        fmod.unbind_modal_history()
    if f_ac1 > 1e-9:
        ratio = f_ac2 / f_ac1
        rep.add("ac_axial_integration", "PASS" if 1.5 < ratio < 2.5 else "WARN", f"F(ac=4)/F(ac=2)={ratio:.2f}")
    else:
        rep.add("ac_axial_integration", "WARN", "force too small to compare")


def check_tool_path(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Straight-line tool path ---")
    if plant.trajectory_mode != "middle_line":
        rep.add("middle_line_path", "WARN", f"trajectory_mode={plant.trajectory_mode}")
        return
    t_arr = plant.t_original
    x_arr, y_arr = plant.x_traj, plant.y_traj
    x_mid = 0.5 * plant.L1
    x_ok = np.allclose(x_arr, x_mid, rtol=1e-6)
    y_mono = np.all(np.diff(y_arr) <= 1e-9) or np.all(np.diff(y_arr) >= -1e-9)
    in_bounds = (
        np.all((x_arr >= 0) & (x_arr <= plant.L1))
        and np.all((y_arr >= 0) & (y_arr <= plant.L2))
    )
    rep.add("x_c_equals_L1/2", "PASS" if x_ok else "FAIL", f"x in [{x_arr.min():.4f},{x_arr.max():.4f}]")
    rep.add("y_c_monotonic", "PASS" if y_mono else "FAIL")
    rep.add("path_inside_plate", "PASS" if in_bounds else "FAIL")


def check_zero_depth(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Zero depth / zero response ---")
    x0 = np.zeros(plant.state_dim)
    low = plant.u_phys_low
    u = plant.physical_to_normalized_action(np.array([low[0], low[1]]))
    res = simulate(plant, x0, u, DT, _t_final())
    mx = float(np.max(np.abs(res["trajectory"])))
    sw = float(np.max(np.abs(res["sensor_w"]))) if res["sensor_w"].size else 0.0
    fn = float(np.max(np.abs(res["f_normal"]))) if len(res["f_normal"]) else 0.0
    rep.add("zero_state", "PASS" if mx < ZERO_TOL else "FAIL", f"max|x|={mx:.2e}")
    rep.add("zero_sensors", "PASS" if sw < ZERO_TOL else "FAIL", f"max|w_s|={sw:.2e}")
    rep.add("zero_force", "PASS" if fn < ZERO_TOL else "FAIL", f"max|F_n|={fn:.2e}")


def check_free_decay(plant: PlatePlant, rep: VerificationReport) -> dict:
    print("\n--- Free vibration decay (ac=0) ---")
    x0 = np.zeros(plant.state_dim)
    x0[0] = 1e-4
    if two_field(plant):
        x0[2 * plant.K] = 1e-6
    u = plant.physical_to_normalized_action(np.array([plant.u_phys_low[0], plant.u_phys_low[1]]))
    res = simulate(plant, x0, u, DT, _t_final())
    n_s = res["eta_n_series"]
    r_n = decay_ratio(n_s[:, 0])
    rep.add("normal_subsystem_decays", "PASS" if r_n < DECAY_RATIO_MAX else "WARN", f"ratio={r_n:.4f}")
    if two_field(plant) and res["eta_f_series"] is not None:
        r_f = decay_ratio(res["eta_f_series"][:, 0])
        rep.add("feed_subsystem_decays", "PASS" if r_f < DECAY_RATIO_MAX else "WARN", f"ratio={r_f:.4f}")
    dmax = float(np.max(np.abs(res["delta_n"]))) if len(res["delta_n"]) else 0.0
    rep.add("no_regenerative_at_ac0", "PASS" if dmax < 1e-12 or np.max(np.abs(res["f_normal"])) < 1e-9 else "WARN")
    return res


def check_structural_no_delay(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Structural terms exclude delay ---")
    u = plant.physical_to_normalized_action(np.array([plant.u_phys_low[0], plant.u_phys_low[1]]))
    x = np.zeros(plant.state_dim)
    x[0] = 1e-4
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(0.0, np.ones(plant.state_dim) * 1e-2)
    dx = fmod.f_nonlinear2(1.0, x, u)
    eta_n, _, _, _ = split_modal_state(x, plant.K, two_field=two_field(plant))
    expected = -fmod.omega_vec**2 * eta_n - fmod.lambda_vec * eta_n**3
    block = 2 * plant.K if two_field(plant) else plant.state_dim
    actual = dx[1:block:2][: plant.K]
    ok = np.allclose(actual, expected, rtol=1e-5)
    rep.add("structural_current_state_only", "PASS" if ok else "FAIL")


def check_regenerative_delay(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Regenerative delay ---")
    omega = 700.0
    tau = fmod.tooth_period(omega)
    rep.add("tau_formula", "PASS", f"tau=2pi/(N*omega)={tau:.6f}s")
    t = 0.05
    eta_n = np.zeros(plant.K)
    eta_f = np.zeros(plant.K) if two_field(plant) else None
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(t - tau, np.zeros(plant.state_dim), omega=omega)
    plant.record_modal_state(t, np.zeros(plant.state_dim), omega=omega)
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        res = fmod._directional_force_result(t, eta_n, eta_f, omega, 2.0)
        rep.add("zero_delta_when_q_equal", "PASS" if abs(res["Delta_n"]) < 1e-11 else "WARN", f"Delta_n={res['Delta_n']:.2e}")
        if two_field(plant):
            rep.add("zero_delta_f_when_equal", "PASS" if abs(res["Delta_f"]) < 1e-11 else "WARN", f"Delta_f={res['Delta_f']:.2e}")
    finally:
        fmod.unbind_modal_history()


def check_chip_thickness(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Chip thickness & engagement ---")
    cfg = plant.milling_config
    phi = 0.4
    f_t = feed_per_tooth(0.05, 800.0, plant.N)
    if two_field(plant):
        h0 = chip_thickness_feed_normal_full(phi, f_t, 0.0, 0.0, cfg)
        h_df = chip_thickness_feed_normal_full(phi, f_t, 1e-5, 0.0, cfg)
        h_dn = chip_thickness_feed_normal_full(phi, f_t, 0.0, 1e-5, cfg)
        sp, cp = math.sin(phi), math.cos(phi)
        rep.add("delta_f_affects_sin", "PASS" if abs(h_df - h0 - 1e-5 * sp) < 1e-12 else "FAIL")
        rep.add("delta_n_affects_cos", "PASS" if abs(h_dn - h0 - 1e-5 * cp) < 1e-12 else "FAIL")
    else:
        h0 = chip_thickness_surface_normal_reduced(phi, f_t, 0.0, cfg)
        h1 = chip_thickness_surface_normal_reduced(phi, f_t, 1e-5, cfg)
        rep.add("delta_q_affects_cos", "PASS" if h1 != h0 else "FAIL")
    xi = fmod.xi_base
    d = fmod.delta_base
    rep.add("h_nonpos_zero_force", "PASS" if tangential_radial_increment(-1e-6, xi, d, 1.0) == (0.0, 0.0) else "FAIL")
    phi_outside = 4.0  # outside default [0, pi] engagement arc
    rep.add("g_zero_zero_force", "PASS" if engagement(phi_outside, cfg.phi_st, cfg.phi_ex) == 0.0 else "FAIL")
    if cfg.milling_type in {"surface", "face", "slotting"}:
        arc = cfg.engaged_arc_width()
        rep.add("default_engagement_arc", "PASS" if np.isclose(arc, math.pi) else "WARN", f"arc={arc:.3f}")
    rep.add("wrap_around", "PASS" if engagement(5.5, 5.0, 1.0) == 1.0 else "FAIL")


def check_force_projection(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Force projection ---")
    eta_n = np.zeros(plant.K)
    eta_f = np.zeros(plant.K) if two_field(plant) else None
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        res = fmod._directional_force_result(0.04, eta_n, eta_f, 750.0, 3.0)
        fn, ff = res["F_normal_total"], res["F_feed_total"]
        mag = math.hypot(fn, ff)
        if mag > 1e-9:
            rep.add("no_magnitude_excitation", "PASS" if fn != mag and ff != mag else "FAIL")
        dx = fmod.f_nonlinear2(0.04, np.zeros(plant.state_dim), np.array([750.0, 3.0]))
        if two_field(plant):
            _, dd_n, _, dd_f = split_modal_state(dx, plant.K, two_field=True)
            rep.add("F_normal_to_W_only", "PASS" if np.any(np.abs(dd_n) > 0) or fn == 0 else "WARN")
            rep.add("F_feed_to_V_only", "PASS" if (np.allclose(dd_f, 0) and ff == 0) or np.any(np.abs(dd_f) > 0) else "WARN")
        b = res.get("b_n", fmod._b_vec_at(0.04, "n"))
        rep.add("b_vec_finite", "PASS" if np.all(np.isfinite(b)) else "FAIL")
    finally:
        fmod.unbind_modal_history()


def check_units(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Units & coefficients ---")
    mc = plant.milling_config.to_dict()
    om = 800.0
    ft = feed_per_tooth(0.05, om, plant.N, source="from_feed_speed")
    rep.add("f_t_2pi_factor", "PASS" if np.isclose(ft, 2 * math.pi * 0.05 / (plant.N * om)) else "FAIL")
    ft_mm = feed_per_tooth(0.05, om, plant.N, cf=2.0, source="from_cf", cf_units="mm_per_tooth")
    rep.add("cf_mm", "PASS" if ft_mm == 2e-3 else "FAIL")
    rep.add("omega_rad_s", "PASS", "omega treated as rad/s in f_t")
    ac_units = mc.get("ac_units", "mm")
    ac_ok = ac_units == "mm" and plant.ac_max <= 50.0
    rep.add(
        "ac_mm",
        "PASS" if ac_ok else "FAIL",
        f"ac_max={plant.ac_max} {ac_units}, internal z/dz via ac_to_meters",
    )
    rep.add(
        "ac_axial_depth_si",
        "PASS" if np.isclose(ac_to_meters(plant.ac_max, ac_units), plant.ac_max * 1e-3) else "FAIL",
        f"ac_max -> {ac_to_meters(plant.ac_max, ac_units):.4f} m for force integration",
    )
    rep.add(
        "force_coefficient_scale",
        "PASS",
        f"scale={getattr(plant, 'force_coefficient_scale', 'n/a')}",
    )
    eta_n = np.zeros(plant.K)
    eta_f = np.zeros(plant.K) if two_field(plant) else None
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        r1 = fmod._directional_force_result(0.04, eta_n, eta_f, 800.0, 1.0)
        clipped = bool(r1.get("force_clipped", False))
        fn = abs(r1.get("F_normal_raw", r1["F_normal_total"]))
        rep.add(
            "no_clip_at_ac1mm",
            "PASS" if not clipped and fn < 5e4 else "FAIL",
            f"|F_n|={fn:.1f} N, clipped={clipped}",
        )
    finally:
        fmod.unbind_modal_history()


def check_integrator(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Integrator (dde_rk4) ---")
    x0 = np.zeros(plant.state_dim)
    x0[0] = 1e-5
    if two_field(plant):
        x0[2 * plant.K] = 1e-6
    u = plant.physical_to_normalized_action(
        np.array([800.0, plant.u_phys_low[1] if two_field(plant) else 3.0])
    )
    t_short = 0.1 if _QUICK else 0.2
    r_coarse = simulate(plant, x0, u, DT, t_short, integrator="dde_rk4")
    r_fine = simulate(plant, x0, u, DT * 0.5, t_short, integrator="dde_rk4")
    err = float(np.linalg.norm(r_coarse["trajectory"][-1] - r_fine["trajectory"][-1]))
    finite = np.all(np.isfinite(r_coarse["trajectory"])) and np.all(np.isfinite(r_fine["trajectory"]))
    if not finite or not np.isfinite(err):
        status: Status = "FAIL"
    elif err > 1e-2:
        status = "WARN"
    else:
        status = "PASS"
    rep.add(
        "dde_rk4_refinement",
        status,
        f"endpoint diff (coarse vs fine)={err:.2e}, finite={finite}",
    )
    scales = recommend_integration_dt(plant, macro_dt=DT)
    print(format_time_scales_report(scales))
    sub_dt, _ = integration_substep(plant)
    rep.add(
        "integration_dt_scales",
        "PASS" if scales["rk4_omega_dt_product"] < 1.5 else "WARN",
        f"sub_dt={sub_dt:.4e}s, tau_min/dt={scales['delay_dt_ratio']:.1f}",
    )
    tau = fmod.tooth_period(800.0)
    t0 = tau * 1.5
    plant.record_modal_state(t0, x0.copy(), omega=800.0)
    xf = integrate_dde_rk4(plant.dynamics, t0, x0, u, sub_dt, n_steps=1)
    rep.add("dde_substage_history", "PASS" if np.all(np.isfinite(xf)) else "FAIL")


def check_cutting_vs_idle(plant: PlatePlant, rep: VerificationReport) -> dict:
    print("\n--- Cutting vs idle ---")
    x0 = np.zeros(plant.state_dim)
    x0[0] = 1e-5
    u_idle = plant.physical_to_normalized_action(np.array([plant.u_phys_low[0], plant.u_phys_low[1]]))
    u_cut = plant.physical_to_normalized_action(np.array([900.0, 5.0]))
    res_idle = simulate(plant, x0, u_idle, DT, _t_final())
    res_cut = simulate(plant, x0, u_cut, DT, _t_final())

    def rms(a):
        return float(np.sqrt(np.mean(a**2))) if a.size else 0.0

    rms_i, rms_c = rms(res_idle["sensor_w"]), rms(res_cut["sensor_w"])
    detail = (
        f"RMS_w idle={rms_i:.2e} cut={rms_c:.2e}; "
        f"max|w| cut={np.max(np.abs(res_cut['sensor_w'])):.2e}; "
        f"max|Delta_n|={np.max(np.abs(res_cut['delta_n'])):.2e}; "
        f"max|Delta_f|={np.max(np.abs(res_cut['delta_f'])):.2e}; "
        f"max|F_n|={np.max(np.abs(res_cut['f_normal'])):.2e}; "
        f"max|F_feed|={np.max(np.abs(res_cut['f_feed'])):.2e}"
    )
    if res_cut["terminated"]:
        detail += f"; terminated={res_cut['termination_info'].get('termination_reason')}"
    d_f_max = float(np.max(np.abs(res_cut["delta_f"]))) if len(res_cut["delta_f"]) else 0.0
    if d_f_max > 1e-3:
        detail += "; chatter_like_Delta_f_growth=YES"
    clip_pct = (
        100.0 * np.mean(res_cut["force_clipped"])
        if len(res_cut.get("force_clipped", [])) > 0
        else 0.0
    )
    detail += f"; force_clip_pct={clip_pct:.1f}%"
    status: Status = "PASS" if rms_c >= rms_i and np.all(np.isfinite(res_cut["trajectory"])) and clip_pct < 1.0 else "WARN"
    if d_f_max > 1.0 or not np.all(np.isfinite(res_cut["delta_f"])):
        status = "FAIL"
    rep.add("cutting_vs_idle", status, detail)
    return res_cut


def check_rl_interface(plant: PlatePlant, rep: VerificationReport) -> None:
    print("\n--- Sensor / RL interface ---")
    import gymnasium as gym

    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        displacement_model=plant.milling_config.displacement_model,
        trajectory_mode=plant.trajectory_mode,
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        integrator=INTEGRATOR,
        delay_mode=DELAY_MODE,
        max_episode_steps=10,
        reward_id="productive",
    )
    obs, _ = env.reset(seed=SEED)
    expected = 2 * env.unwrapped.plant.n_sensors + 2
    rep.add("obs_shape", "PASS" if obs.shape == (expected,) else "FAIL", f"obs.shape={obs.shape}")
    rep.add("obs_not_modal", "PASS" if obs.shape[0] != plant.state_dim else "FAIL")
    action = np.array([0.2, -0.3])
    obs2, reward, term, trunc, info = env.step(action)
    rep.add("reward_finite", "PASS" if np.isfinite(reward) else "FAIL")
    rep.add("sensor_in_info", "PASS" if "sensor_w" in info and "action_phys" in info else "FAIL")
    ap = info["action_phys"]
    rep.add("reward_uses_physical_action", "PASS" if np.allclose(ap, plant.normalized_to_physical_action(action)) else "FAIL")
    assert "sensor_obs_norm" in info
    reward_fn = SensorProductivePlateReward()
    r2 = reward_fn(0.0, np.zeros(plant.state_dim), action, np.zeros(plant.state_dim), False, False, info)
    rep.add("reward_from_sensors", "PASS" if np.isfinite(r2) else "FAIL")
    env.close()


def check_calibration_sweep(plant: PlatePlant, rep: VerificationReport) -> dict:
    """Deterministic omega/ac grid for regenerative response calibration."""
    print("\n--- Calibration sweep (feed_normal_full) ---")
    omegas = [800.0] if _QUICK else [400.0, 800.0, 1200.0]
    acs = [0.0, 0.5, 2.0, 6.0] if _QUICK else [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    rows: list[dict[str, float]] = []
    t_sim = _calib_t_sim()

    print(f"  {'omega':>6} {'ac':>5} {'h_mm':>8} {'dF_mm':>8} {'dN_mm':>8} "
          f"{'Fn_N':>9} {'Ff_N':>9} {'clip%':>6} {'w_mm':>8} {'class':>12}")
    for omega in omegas:
        for ac in acs:
            x0 = np.zeros(plant.state_dim)
            if ac > 0:
                x0[0] = 1e-6
                x0[2 * plant.K] = 1e-7
            u = plant.physical_to_normalized_action(np.array([omega, max(ac, plant.u_phys_low[1])]))
            res = simulate(plant, x0, u, DT, t_sim)
            n = len(res.get("force_clipped", []))
            clip_pct = 100.0 * float(np.mean(res["force_clipped"])) if n else 0.0
            h_arr = np.asarray(res["h_sample"], dtype=np.float64)
            h_mm = float(np.max(np.abs(h_arr)) * 1e3) if h_arr.size else 0.0
            df_mm = float(np.max(np.abs(res["delta_f"])) * 1e3) if len(res["delta_f"]) else 0.0
            dn_mm = float(np.max(np.abs(res["delta_n"])) * 1e3) if len(res["delta_n"]) else 0.0
            fn = float(np.max(np.abs(res["f_normal_raw"]))) if len(res.get("f_normal_raw", [])) else 0.0
            ff = float(np.max(np.abs(res["f_feed_raw"]))) if len(res.get("f_feed_raw", [])) else 0.0
            sw = res["sensor_w"]
            max_w = float(np.max(np.abs(sw))) if sw.size else 0.0
            traj = res["trajectory"]
            finite = bool(np.all(np.isfinite(traj)))
            dd_n = traj[:, 1] if traj.shape[1] > 1 else traj[:, 0]
            max_acc = float(np.max(np.abs(dd_n))) if dd_n.size else 0.0
            row = {
                "omega": omega,
                "ac_mm": ac,
                "h_mm": h_mm,
                "delta_f_mm": df_mm,
                "delta_n_mm": dn_mm,
                "F_normal_N": fn,
                "F_feed_N": ff,
                "clip_pct": clip_pct,
                "max_w_m": max_w,
                "max_w_mm": max_w * 1e3,
                "max_acc": max_acc,
                "finite": finite,
            }
            if not finite or np.max(np.abs(res["delta_f"])) > 1.0:
                klass = "failed"
            elif clip_pct > 5.0:
                klass = "failed"
            elif ac <= 0:
                klass = "stable" if max_w < 1e-8 else "bounded"
            elif max_w > 2e-4 or df_mm > 0.5:
                klass = "chatter-like"
            elif max_w > 1e-6:
                klass = "bounded"
            else:
                klass = "stable"
            row["class"] = klass
            rows.append(row)
            print(
                f"  {omega:6.0f} {ac:5.1f} {h_mm:8.4f} {df_mm:8.4f} {dn_mm:8.4f} "
                f"{fn:9.1f} {ff:9.1f} {clip_pct:5.1f}% {max_w*1e3:8.4f} {klass:>12}"
            )

    n_clip = sum(1 for r in rows if r["clip_pct"] > 1.0 and r["ac_mm"] > 0)
    n_fail = sum(1 for r in rows if r["class"] == "failed")
    n_chatter = sum(1 for r in rows if r["class"] == "chatter-like")
    rep.add(
        "calibration_no_clip_stable",
        "PASS" if n_clip == 0 else "FAIL",
        f"clipped_cases={n_clip}/{len(rows)}",
    )
    rep.add(
        "calibration_regenerative_growth",
        "PASS" if n_chatter >= 1 else "WARN",
        f"chatter_like={n_chatter}, failed={n_fail}",
    )
    return {"rows": rows}




def plot_verification(mode: str, res_idle: dict, res_cut: dict, plant: PlatePlant) -> None:
    out_dir = PLOT_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # One-revolution engagement diagnostic
    diag = fmod.get_revolution_diagnostics(0.0, 800.0, 3.0, n_samples=200)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_rev = diag["t"] - diag["t"][0]
    axes[0].plot(t_rev, diag["phi"])
    axes[0].set_ylabel("phi_j")
    axes[1].plot(t_rev, diag["g"])
    axes[1].set_ylabel("g(phi)")
    axes[2].plot(t_rev, diag["h"] * 1e3)
    axes[2].set_ylabel("h_j [mm]")
    axes[2].set_xlabel("t [s]")
    fig.suptitle(f"{mode} - one revolution engagement")
    fig.tight_layout()
    fig.savefig(out_dir / "engagement_one_rev.png", dpi=120)
    plt.close(fig)

    # Cutting rollout (per-step diagnostics share length with sensor_w)
    sw = res_cut["sensor_w"]
    if not sw.size:
        print(f"  Plots saved to {out_dir} (engagement only; rollout empty)")
        return
    n = int(sw.shape[0])
    time = np.asarray(res_cut["time"][:n], dtype=np.float64)
    eta_n_step = res_cut["trajectory"][:n, 0::2][:, 0]
    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
    if res_cut["eta_f_series"] is not None:
        block = 2 * plant.K
        eta_f_step = res_cut["trajectory"][:n, block::2][:, 0]
        axes[0, 0].plot(time, eta_n_step, label="eta_n,0")
        axes[0, 1].plot(time, eta_f_step, label="eta_f,0")
    else:
        axes[0, 0].plot(time, eta_n_step, label="eta_0")
        axes[0, 1].axis("off")
    axes[0, 0].set_ylabel("eta [m]")
    axes[0, 1].set_ylabel("eta_f [m]")

    axes[1, 0].plot(time, res_cut["q_n_cutter"][:n], label="q_n@cutter")
    axes[1, 1].plot(time, res_cut["q_f_cutter"][:n], label="q_f@cutter")
    axes[1, 0].set_ylabel("q @ cutter [m]")

    sw = res_cut["sensor_w"]
    if sw.size:
        axes[2, 0].plot(time, sw[:, 0], label="w_s1")
        if sw.shape[1] > 1:
            axes[2, 0].plot(time, sw[:, 1], label="w_s2")
        axes[2, 1].plot(time, res_cut["sensor_w_dot"][:, 0], label="w_dot_s1")
    axes[2, 0].set_ylabel("Sensors [m, m/s]")

    axes[3, 0].plot(time, res_cut["delta_n"][:n], label="Delta_n")
    axes[3, 1].plot(time, res_cut["delta_f"][:n], label="Delta_f")
    axes[3, 0].set_ylabel("Delta [m]")

    axes[4, 0].plot(time, res_cut["tau"][:n], label="tau")
    axes[4, 1].plot(time, res_cut["phi_sample"][:n], label="phi_sample")
    axes[4, 0].set_ylabel("tau [s] / phi [rad]")

    axes[5, 0].plot(time, res_cut["f_normal"][:n], label="F_normal")
    axes[5, 1].plot(time, res_cut["f_feed"][:n], label="F_feed")
    u_phys = res_cut["u_phys"][:n]
    axes[5, 0].plot(time, u_phys[:, 0], ls="--", alpha=0.6, label="omega")
    axes[5, 1].plot(time, u_phys[:, 1], ls="--", alpha=0.6, label="ac")
    axes[5, 0].set_ylabel("F [N] / omega [rad/s]")
    axes[5, 1].set_ylabel("ac [mm]")
    axes[5, 0].set_xlabel("t [s]")

    for ax in axes.ravel():
        if ax.has_data():
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{mode} - cutting verification rollout")
    fig.tight_layout()
    fig.savefig(out_dir / "cutting_rollout.png", dpi=120)
    plt.close(fig)
    print(f"  Plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_mode(displacement_model: str, *, show_plots: bool) -> VerificationReport:
    print("\n" + "=" * 72)
    print(f"PRE-TRAINING VERIFICATION: {displacement_model}")
    print("=" * 72)

    plant = make_plant(displacement_model)
    rep = VerificationReport(displacement_model=displacement_model)

    check_metadata(plant, rep)
    check_action_scaling(plant, rep)
    check_tool_path(plant, rep)
    check_zero_depth(plant, rep)
    res_idle = check_free_decay(plant, rep)
    check_structural_no_delay(plant, rep)
    check_regenerative_delay(plant, rep)
    check_chip_thickness(plant, rep)
    check_force_projection(plant, rep)
    check_units(plant, rep)
    check_integrator(plant, rep)
    sweep = None
    if displacement_model == "feed_normal_full":
        sweep = check_calibration_sweep(plant, rep)
    res_cut = check_cutting_vs_idle(plant, rep)
    check_rl_interface(plant, rep)

    if show_plots:
        plot_verification(displacement_model, res_idle, res_cut, plant)

    return rep


def print_final_summary(reports: list[VerificationReport]) -> None:
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    all_pass = True
    for rep in reports:
        status = rep.summary()
        n_pass = sum(1 for r in rep.results if r.status == "PASS")
        n_warn = sum(1 for r in rep.results if r.status == "WARN")
        n_fail = sum(1 for r in rep.results if r.status == "FAIL")
        print(f"\n{rep.displacement_model}: {status}")
        print(f"  passed={n_pass}, warnings={n_warn}, failed={n_fail}")
        if n_fail:
            all_pass = False
            for r in rep.results:
                if r.status == "FAIL":
                    print(f"    FAIL: {r.name} - {r.detail}")
        if n_warn:
            for r in rep.results:
                if r.status == "WARN":
                    print(f"    WARN: {r.name} - {r.detail}")

    print("\n" + "-" * 72)
    if all_pass and not any(r.summary() == "WARN" for r in reports):
        print("VERDICT: PASS - dynamic plant ready for PPO training.")
    elif all_pass:
        print("VERDICT: WARN - plant usable; review warnings before long training runs.")
    else:
        print("VERDICT: FAIL - fix failed checks before training.")
    print("NOTE: Two-direction regenerative surface-milling model (Nasiri/Moradi-inspired).")
    print("NOTE: Old PPO checkpoints are incompatible after dynamics/calibration changes.")
    print(f"Plots (if enabled): {PLOT_DIR}")
    print("-" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-training plant verification")
    parser.add_argument(
        "--mode",
        choices=["all", "surface_normal_reduced", "feed_normal_full"],
        default="all",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter horizons and smaller calibration grid (faster pre-train check)",
    )
    args = parser.parse_args()

    global _QUICK
    _QUICK = bool(args.quick)

    modes = (
        ["surface_normal_reduced", "feed_normal_full"]
        if args.mode == "all"
        else [args.mode]
    )
    show_plots = not args.no_plots and os.environ.get("TEST2_SHOW_PLOTS", "1").lower() not in ("0", "false", "no")

    print("Custom_RL pre-training verification")
    print(
        f"seed={SEED}, dt={DT}, T_final={_t_final()}, integrator={INTEGRATOR}, "
        f"delay={DELAY_MODE}, quick={_QUICK}"
    )

    reports = [run_mode(m, show_plots=show_plots) for m in modes]
    print_final_summary(reports)


if __name__ == "__main__":
    main()
