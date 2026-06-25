"""Compute and plot a no-control stability lobe for CustomODEPlate-v0.

This script estimates the boundary between stable machining and
chatter/unstable machining by sweeping spindle speed and axial depth of cut
through time-domain simulations of the uncontrolled PlatePlant simulator.

IMPORTANT — this is a practical time-domain estimate of the no-control
stability boundary, not an analytical lobe diagram. Its accuracy depends on
the simulator correctly representing regenerative delay dynamics through the
difference between current and delayed displacement (STATE_DELAY in
f_nonlinear2). RL, PPO, SAC, reward learning, and trained controllers are
NOT involved in this calculation.

Run from the project root:

    python scripts/compute_stability_lobe.py
    python scripts/compute_stability_lobe.py --rpm-min 1000 --rpm-max 15000 --rpm-points 12

Example single-condition check (same physics, different script):

    python scripts/check_plate_random_policy.py --fixed-action --omega 500 --ac 5

3D stability surface (optional):

    python scripts/compute_stability_lobe.py --surface-3d --y0-min 0.05 --y0-max 0.45 --y0-points 5

Stochastic multi-y0 lobe (optional, 2D curves with MC mean +/- std bands):

    python scripts/compute_stability_lobe.py --stochastic-lobe --y0 0.1 0.2 0.3 \\
        --n-mc 5 --dynamics-uncertainty-std 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from custom_rl import register_envs
from custom_rl.plants.plate import estimate_pass_episode_steps

ENV_ID = "CustomODEPlate-v0"


def rpm_to_omega(rpm: float) -> float:
    """Convert spindle speed from rpm to rad/s."""
    return float(rpm) * 2.0 * np.pi / 60.0


def omega_to_rpm(omega: float) -> float:
    """Convert spindle speed from rad/s to rpm."""
    return float(omega) * 60.0 / (2.0 * np.pi)


def physical_to_normalized(
    omega: float,
    ac: float,
    omega_min: float,
    omega_max: float,
    ac_min: float,
    ac_max: float,
) -> np.ndarray:
    """Map physical [omega, ac] to normalized action in [-1, 1]^2."""
    low = np.array([omega_min, ac_min], dtype=np.float64)
    high = np.array([omega_max, ac_max], dtype=np.float64)
    u_phys = np.array([omega, ac], dtype=np.float64)
    frac = (u_phys - low) / (high - low)
    return np.clip(2.0 * frac - 1.0, -1.0, 1.0)


@dataclass
class TrialMetrics:
    """Diagnostic metrics from one fixed-action simulation."""

    rpm: float
    omega_rad_s: float
    ac_mm: float
    stable: bool
    steps: int
    sim_time_s: float
    max_abs_w_m: float
    rms_w_m: float
    rms_w_first_half_m: float
    rms_w_second_half_m: float
    rms_growth_ratio: float
    terminated: bool
    truncated: bool
    termination_reason: str | None


def _sensor_displacement_series(info_history: list[dict]) -> np.ndarray:
    """Return max absolute physical sensor displacement at each step."""
    values: list[float] = []
    for info in info_history:
        if "w_sensor" in info:
            w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        else:
            values.append(0.0)
            continue
        values.append(float(np.max(np.abs(w))))
    return np.asarray(values, dtype=np.float64)


def classify_stability(
    w_series: np.ndarray,
    *,
    w_limit: float,
    terminated: bool,
    termination_reason: str | None,
    rms_growth_threshold: float,
    w_limit_fraction: float,
) -> tuple[bool, float, float, float, float, float]:
    """
    Classify a simulation as stable or chatter/unstable.

    Unstable if:
      - environment terminated for excessive physical displacement/velocity,
      - max physical displacement exceeds the safety threshold,
      - or the vibration RMS envelope clearly grows over time.
    """
    if w_series.size == 0:
        return True, 0.0, 0.0, 0.0, 0.0, 1.0

    max_abs_w = float(np.max(w_series))
    rms_w = float(np.sqrt(np.mean(w_series ** 2)))

    mid = max(w_series.size // 2, 1)
    first = w_series[:mid]
    second = w_series[mid:]
    rms_first = float(np.sqrt(np.mean(first ** 2))) if first.size else 0.0
    rms_second = float(np.sqrt(np.mean(second ** 2))) if second.size else 0.0
    growth_ratio = rms_second / max(rms_first, 1e-12)

    unstable_reasons = (
        termination_reason
        in {
            "excessive_sensor_displacement",
            "excessive_sensor_velocity",
            "invalid_state",
        }
        or max_abs_w > w_limit * w_limit_fraction
        or (
            growth_ratio >= rms_growth_threshold
            and rms_second > 0.25 * w_limit
        )
    )

    stable = not unstable_reasons
    return stable, max_abs_w, rms_w, rms_first, rms_second, growth_ratio


def run_fixed_action_trial(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ac_mm: float,
    max_steps: int,
    seed: int,
    rms_growth_threshold: float,
    w_limit_fraction: float,
    reset_options: dict | None = None,
) -> TrialMetrics:
    """Simulate one fixed (omega, ac) pair with no controller."""
    plant = env.unwrapped.plant
    action = physical_to_normalized(
        omega_rad_s,
        ac_mm,
        plant.omega_min,
        plant.omega_max,
        plant.ac_min,
        plant.ac_max,
    )

    obs, _ = env.reset(seed=seed, options=reset_options)
    info_history: list[dict] = []
    terminated = truncated = False
    termination_reason: str | None = None
    steps = 0

    while steps < max_steps and not (terminated or truncated):
        obs, _reward, terminated, truncated, info = env.step(action)
        info_history.append(dict(info))
        steps += 1
        termination_reason = info.get("termination_reason")

    w_series = _sensor_displacement_series(info_history)
    stable, max_abs_w, rms_w, rms_first, rms_second, growth_ratio = classify_stability(
        w_series,
        w_limit=plant.w_limit,
        terminated=terminated,
        termination_reason=termination_reason,
        rms_growth_threshold=rms_growth_threshold,
        w_limit_fraction=w_limit_fraction,
    )

    sim_time_s = float(info_history[-1].get("t", steps * env.unwrapped._step_dt)) if info_history else 0.0

    return TrialMetrics(
        rpm=omega_to_rpm(omega_rad_s),
        omega_rad_s=omega_rad_s,
        ac_mm=ac_mm,
        stable=stable,
        steps=steps,
        sim_time_s=sim_time_s,
        max_abs_w_m=max_abs_w,
        rms_w_m=rms_w,
        rms_w_first_half_m=rms_first,
        rms_w_second_half_m=rms_second,
        rms_growth_ratio=growth_ratio,
        terminated=terminated,
        truncated=truncated,
        termination_reason=termination_reason,
    )


def find_max_stable_ac(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ac_min: float,
    ac_max: float,
    max_steps: int,
    seed: int,
    ac_tol_mm: float,
    max_binary_iters: int,
    rms_growth_threshold: float,
    w_limit_fraction: float,
    reset_options: dict | None = None,
) -> tuple[float, TrialMetrics, TrialMetrics | None]:
    """
    Binary search for the largest axial depth of cut that remains stable.

    Returns:
        (ac_stable, stable_trial_at_boundary, unstable_trial_above_boundary_or_None)
    """
    plant = env.unwrapped.plant
    ac_lo = float(ac_min)
    ac_hi = float(ac_max)

    trial_lo = run_fixed_action_trial(
        env,
        omega_rad_s=omega_rad_s,
        ac_mm=ac_lo,
        max_steps=max_steps,
        seed=seed,
        rms_growth_threshold=rms_growth_threshold,
        w_limit_fraction=w_limit_fraction,
        reset_options=reset_options,
    )

    if not trial_lo.stable:
        return ac_lo, trial_lo, trial_lo

    trial_hi = run_fixed_action_trial(
        env,
        omega_rad_s=omega_rad_s,
        ac_mm=ac_hi,
        max_steps=max_steps,
        seed=seed + 1,
        rms_growth_threshold=rms_growth_threshold,
        w_limit_fraction=w_limit_fraction,
        reset_options=reset_options,
    )

    if trial_hi.stable:
        return ac_hi, trial_hi, None

    unstable_probe: TrialMetrics | None = trial_hi
    best_stable = trial_lo

    for _ in range(max_binary_iters):
        if ac_hi - ac_lo <= ac_tol_mm:
            break

        ac_mid = 0.5 * (ac_lo + ac_hi)
        trial_mid = run_fixed_action_trial(
            env,
            omega_rad_s=omega_rad_s,
            ac_mm=ac_mid,
            max_steps=max_steps,
            seed=seed + int(ac_mid * 1000),
            rms_growth_threshold=rms_growth_threshold,
            w_limit_fraction=w_limit_fraction,
            reset_options=reset_options,
        )

        if trial_mid.stable:
            ac_lo = ac_mid
            best_stable = trial_mid
        else:
            ac_hi = ac_mid
            unstable_probe = trial_mid

    return ac_lo, best_stable, unstable_probe


def simulation_steps_for_rpm(
    rpm: float,
    *,
    dt: float,
    n_substeps: int,
    plant,
    sim_seconds: float | None,
    max_sim_steps: int,
) -> int:
    """Choose a practical step budget for one lobe evaluation."""
    if sim_seconds is not None:
        steps = int(np.ceil(sim_seconds / max(dt * n_substeps, 1e-12)))
        return min(steps, max_sim_steps)

    omega = rpm_to_omega(rpm)
    pass_steps = estimate_pass_episode_steps(
        plant.L1,
        plant.cf,
        plant.N,
        omega,
        dt * n_substeps,
        margin=0.35,
    )
    return min(pass_steps, max_sim_steps)


def compute_lobe(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Sweep spindle speeds and estimate the stable ac boundary at each."""
    register_envs()

    env_kwargs: dict = {
        "reward_id": "sparse",
        "dt": args.dt,
        "n_substeps": args.n_substeps,
    }
    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(ENV_ID, **env_kwargs)
    plant = env.unwrapped.plant

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    rows: list[dict] = []

    print(
        "No-control stability lobe sweep\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm "
        f"({args.rpm_points} points)\n"
        f"  ac search     : {plant.ac_min:.3f} - {plant.ac_max:.3f} mm\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for index, rpm in enumerate(rpm_values):
        omega = rpm_to_omega(float(rpm))
        omega = float(np.clip(omega, plant.omega_min, plant.omega_max))
        rpm_clipped = omega_to_rpm(omega)

        max_steps = simulation_steps_for_rpm(
            rpm_clipped,
            dt=args.dt,
            n_substeps=args.n_substeps,
            plant=plant,
            sim_seconds=args.sim_seconds,
            max_sim_steps=args.max_sim_steps,
        )

        print(
            f"[{index + 1}/{len(rpm_values)}] rpm={rpm_clipped:.0f} "
            f"(omega={omega:.1f} rad/s), max_steps={max_steps}"
        )

        ac_stable, stable_trial, unstable_trial = find_max_stable_ac(
            env,
            omega_rad_s=omega,
            ac_min=plant.ac_min,
            ac_max=plant.ac_max,
            max_steps=max_steps,
            seed=args.seed + index,
            ac_tol_mm=args.ac_tol,
            max_binary_iters=args.binary_iters,
            rms_growth_threshold=args.rms_growth_threshold,
            w_limit_fraction=args.w_limit_fraction,
        )

        row = {
            "rpm": rpm_clipped,
            "omega_rad_s": omega,
            "ac_stable_mm": ac_stable,
            "max_abs_w_m": stable_trial.max_abs_w_m,
            "rms_w_m": stable_trial.rms_w_m,
            "rms_growth_ratio": stable_trial.rms_growth_ratio,
            "sim_steps": stable_trial.steps,
            "sim_time_s": stable_trial.sim_time_s,
            "termination_reason": stable_trial.termination_reason,
            "stable_at_boundary": stable_trial.stable,
            "unstable_ac_mm": None if unstable_trial is None else unstable_trial.ac_mm,
            "unstable_max_abs_w_m": None if unstable_trial is None else unstable_trial.max_abs_w_m,
        }
        rows.append(row)

        print(
            f"    -> stable boundary ac ~ {ac_stable:.3f} mm "
            f"(max|w|={stable_trial.max_abs_w_m:.4e} m, "
            f"rms growth={stable_trial.rms_growth_ratio:.2f})"
        )

    env.close()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "time_domain_no_control_binary_search",
        "note": (
            "Practical time-domain estimate of the no-control stability boundary. "
            "Accuracy depends on regenerative delay dynamics (STATE_DELAY) in "
            "f_nonlinear2. RL/control is not used."
        ),
        "env_id": ENV_ID,
        "dt": args.dt,
        "n_substeps": args.n_substeps,
        "sim_seconds": args.sim_seconds,
        "max_sim_steps": args.max_sim_steps,
        "rpm_min": args.rpm_min,
        "rpm_max": args.rpm_max,
        "rpm_points": args.rpm_points,
        "ac_min_mm": plant.ac_min,
        "ac_max_mm": plant.ac_max,
        "ac_tol_mm": args.ac_tol,
        "binary_iters": args.binary_iters,
        "w_limit_m": plant.w_limit,
        "w_limit_fraction": args.w_limit_fraction,
        "rms_growth_threshold": args.rms_growth_threshold,
        "seed": args.seed,
        "omega_min_rad_s": plant.omega_min,
        "omega_max_rad_s": plant.omega_max,
    }

    return rows, metadata


def sweep_rpm_boundary_at_y0(
    env: gym.Env,
    plant,
    rpm_values: np.ndarray,
    args: argparse.Namespace,
    *,
    y0_m: float,
    seed_offset: int,
) -> list[dict]:
    """
    Estimate the 2D stability boundary over rpm at a fixed milling start y0.

    Sets plant.set_milling_start_y(y0_m) so f_nonlinear2.y_cutter and the modal
    force projection W_k(x_c, y0) both follow the selected start point.
    """
    plant.set_milling_start_y(y0_m)
    rows: list[dict] = []

    for index, rpm in enumerate(rpm_values):
        omega = rpm_to_omega(float(rpm))
        omega = float(np.clip(omega, plant.omega_min, plant.omega_max))
        rpm_clipped = omega_to_rpm(omega)

        max_steps = simulation_steps_for_rpm(
            rpm_clipped,
            dt=args.dt,
            n_substeps=args.n_substeps,
            plant=plant,
            sim_seconds=args.sim_seconds,
            max_sim_steps=args.max_sim_steps,
        )

        print(
            f"    rpm={rpm_clipped:.0f} (omega={omega:.1f} rad/s), "
            f"max_steps={max_steps}"
        )

        ac_stable, stable_trial, unstable_trial = find_max_stable_ac(
            env,
            omega_rad_s=omega,
            ac_min=plant.ac_min,
            ac_max=plant.ac_max,
            max_steps=max_steps,
            seed=args.seed + seed_offset + index,
            ac_tol_mm=args.ac_tol,
            max_binary_iters=args.binary_iters,
            rms_growth_threshold=args.rms_growth_threshold,
            w_limit_fraction=args.w_limit_fraction,
        )

        row = {
            "rpm": rpm_clipped,
            "omega_rad_s": omega,
            "y0_m": y0_m,
            "ac_stable_mm": ac_stable,
            "max_abs_w_m": stable_trial.max_abs_w_m,
            "rms_w_m": stable_trial.rms_w_m,
            "rms_growth_ratio": stable_trial.rms_growth_ratio,
            "sim_steps": stable_trial.steps,
            "sim_time_s": stable_trial.sim_time_s,
            "termination_reason": stable_trial.termination_reason,
            "stable_at_boundary": stable_trial.stable,
            "unstable_ac_mm": None if unstable_trial is None else unstable_trial.ac_mm,
            "unstable_max_abs_w_m": None if unstable_trial is None else unstable_trial.max_abs_w_m,
        }
        rows.append(row)

        print(
            f"      -> stable boundary ac ~ {ac_stable:.3f} mm "
            f"(max|w|={stable_trial.max_abs_w_m:.4e} m, "
            f"rms growth={stable_trial.rms_growth_ratio:.2f})"
        )

    return rows


def compute_lobe_surface_3d(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """
    Sweep y0 and rpm to estimate ac_stable = f(rpm, y0).

    Practical time-domain estimate of the no-control stability boundary surface.
    Physical accuracy depends on the regenerative force model (STATE_DELAY) and
    on y0 correctly updating both cutter path and force projection via
    PlatePlant.set_milling_start_y -> f_nonlinear2.y_cutter / _compute_b_vec_at.
    RL/control is not used.
    """
    register_envs()

    env_kwargs: dict = {
        "reward_id": "sparse",
        "dt": args.dt,
        "n_substeps": args.n_substeps,
    }
    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(ENV_ID, **env_kwargs)
    plant = env.unwrapped.plant

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    y0_values = np.linspace(args.y0_min, args.y0_max, args.y0_points)
    rows: list[dict] = []

    print(
        "No-control 3D stability surface sweep\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm "
        f"({args.rpm_points} points)\n"
        f"  y0 range      : {args.y0_min:.3f} - {args.y0_max:.3f} m "
        f"({args.y0_points} points)\n"
        f"  ac search     : {plant.ac_min:.3f} - {plant.ac_max:.3f} mm\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for y_index, y0_m in enumerate(y0_values):
        y0_m = float(y0_m)
        print(
            f"[y0 {y_index + 1}/{len(y0_values)}] milling start y0={y0_m:.3f} m "
            f"(L2={plant.L2:.3f} m)"
        )

        y_rows = sweep_rpm_boundary_at_y0(
            env,
            plant,
            rpm_values,
            args,
            y0_m=y0_m,
            seed_offset=y_index * 1000,
        )
        rows.extend(y_rows)

    env.close()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "time_domain_no_control_binary_search_3d_surface",
        "note": (
            "Practical time-domain estimate of the no-control stability boundary "
            "surface ac_stable = f(rpm, y0). Accuracy depends on regenerative "
            "delay dynamics (STATE_DELAY) and on y0 updating both the cutter "
            "path and modal force projection (f_nonlinear2.y_cutter, "
            "_compute_b_vec_at). RL/control is not used."
        ),
        "env_id": ENV_ID,
        "dt": args.dt,
        "n_substeps": args.n_substeps,
        "sim_seconds": args.sim_seconds,
        "max_sim_steps": args.max_sim_steps,
        "rpm_min": args.rpm_min,
        "rpm_max": args.rpm_max,
        "rpm_points": args.rpm_points,
        "y0_min_m": args.y0_min,
        "y0_max_m": args.y0_max,
        "y0_points": args.y0_points,
        "L2_m": plant.L2,
        "ac_min_mm": plant.ac_min,
        "ac_max_mm": plant.ac_max,
        "ac_tol_mm": args.ac_tol,
        "binary_iters": args.binary_iters,
        "w_limit_m": plant.w_limit,
        "w_limit_fraction": args.w_limit_fraction,
        "rms_growth_threshold": args.rms_growth_threshold,
        "seed": args.seed,
        "omega_min_rad_s": plant.omega_min,
        "omega_max_rad_s": plant.omega_max,
    }

    return rows, metadata


def resolve_y0_values(args: argparse.Namespace) -> list[float]:
    """Return explicit --y0 list or a linspace from y0-min/max/points."""
    if args.y0:
        return [float(v) for v in args.y0]
    return [float(v) for v in np.linspace(args.y0_min, args.y0_max, args.y0_points)]


def find_max_stable_ac_mc(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ac_min: float,
    ac_max: float,
    max_steps: int,
    seed: int,
    ac_tol_mm: float,
    max_binary_iters: int,
    rms_growth_threshold: float,
    w_limit_fraction: float,
    n_mc: int,
    reset_options: dict | None = None,
) -> tuple[float, float, list[float], TrialMetrics, TrialMetrics | None]:
    """
    Estimate max stable ac with Monte Carlo repetitions.

    Each MC run performs a full binary search with a distinct seed so stochastic
    dynamics uncertainty produces independent boundary estimates.

    Returns:
        (ac_mean_mm, ac_std_mm, ac_samples_mm, representative_stable_trial,
         representative_unstable_trial)
    """
    ac_samples: list[float] = []
    last_stable: TrialMetrics | None = None
    last_unstable: TrialMetrics | None = None

    for mc in range(n_mc):
        ac_stable, stable_trial, unstable_trial = find_max_stable_ac(
            env,
            omega_rad_s=omega_rad_s,
            ac_min=ac_min,
            ac_max=ac_max,
            max_steps=max_steps,
            seed=seed + mc * 10_000,
            ac_tol_mm=ac_tol_mm,
            max_binary_iters=max_binary_iters,
            rms_growth_threshold=rms_growth_threshold,
            w_limit_fraction=w_limit_fraction,
            reset_options=reset_options,
        )
        ac_samples.append(ac_stable)
        last_stable = stable_trial
        if unstable_trial is not None:
            last_unstable = unstable_trial

    ac_arr = np.asarray(ac_samples, dtype=np.float64)
    ac_mean = float(np.mean(ac_arr))
    ac_std = float(np.std(ac_arr)) if ac_arr.size > 1 else 0.0

    assert last_stable is not None
    return ac_mean, ac_std, ac_samples, last_stable, last_unstable


def sweep_rpm_boundary_stochastic_at_y0(
    env: gym.Env,
    plant,
    rpm_values: np.ndarray,
    args: argparse.Namespace,
    *,
    y0_m: float,
    seed_offset: int,
) -> list[dict]:
    """Estimate stochastic stability boundary over rpm at fixed y0."""
    plant.set_milling_start_y(y0_m)
    reset_options = {"y0": y0_m}
    rows: list[dict] = []

    for index, rpm in enumerate(rpm_values):
        omega = rpm_to_omega(float(rpm))
        omega = float(np.clip(omega, plant.omega_min, plant.omega_max))
        rpm_clipped = omega_to_rpm(omega)

        max_steps = simulation_steps_for_rpm(
            rpm_clipped,
            dt=args.dt,
            n_substeps=args.n_substeps,
            plant=plant,
            sim_seconds=args.sim_seconds,
            max_sim_steps=args.max_sim_steps,
        )

        print(
            f"    rpm={rpm_clipped:.0f} (omega={omega:.1f} rad/s), "
            f"max_steps={max_steps}, n_mc={args.n_mc}"
        )

        ac_mean, ac_std, ac_samples, stable_trial, unstable_trial = find_max_stable_ac_mc(
            env,
            omega_rad_s=omega,
            ac_min=plant.ac_min,
            ac_max=plant.ac_max,
            max_steps=max_steps,
            seed=args.seed + seed_offset + index,
            ac_tol_mm=args.ac_tol,
            max_binary_iters=args.binary_iters,
            rms_growth_threshold=args.rms_growth_threshold,
            w_limit_fraction=args.w_limit_fraction,
            n_mc=args.n_mc,
            reset_options=reset_options,
        )

        row = {
            "rpm": rpm_clipped,
            "omega_rad_s": omega,
            "y0_m": y0_m,
            "ac_stable_mean_mm": ac_mean,
            "ac_stable_std_mm": ac_std,
            "ac_stable_samples_mm": json.dumps(ac_samples),
            "n_mc": args.n_mc,
            "max_abs_w_m": stable_trial.max_abs_w_m,
            "rms_w_m": stable_trial.rms_w_m,
            "rms_growth_ratio": stable_trial.rms_growth_ratio,
            "sim_steps": stable_trial.steps,
            "sim_time_s": stable_trial.sim_time_s,
            "termination_reason": stable_trial.termination_reason,
            "stable_at_boundary": stable_trial.stable,
            "unstable_ac_mm": None if unstable_trial is None else unstable_trial.ac_mm,
            "unstable_max_abs_w_m": None if unstable_trial is None else unstable_trial.max_abs_w_m,
        }
        rows.append(row)

        print(
            f"      -> stable boundary ac ~ {ac_mean:.3f} +/- {ac_std:.3f} mm "
            f"(samples={ac_samples})"
        )

    return rows


def compute_stochastic_lobe_multi_y0(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """
    Stochastic stability lobe: multiple y0 curves on one 2D rpm-ac plot.

    For each y0 and spindle speed, runs n_mc independent binary searches with
    optional dynamics uncertainty. Boundary depth is summarized as mean +/- std.
    """
    register_envs()

    env_kwargs: dict = {
        "reward_id": "sparse",
        "dt": args.dt,
        "n_substeps": args.n_substeps,
        "dynamics_uncertainty_std": args.dynamics_uncertainty_std,
        "randomize_y0": False,
    }
    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(ENV_ID, **env_kwargs)
    plant = env.unwrapped.plant

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    y0_values = resolve_y0_values(args)
    rows: list[dict] = []

    print(
        "Stochastic no-control stability lobe (multi-y0 curves)\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm "
        f"({args.rpm_points} points)\n"
        f"  y0 values     : {y0_values}\n"
        f"  n_mc          : {args.n_mc}\n"
        f"  dynamics unc. : {plant.dynamics_uncertainty_std}\n"
        f"  ac search     : {plant.ac_min:.3f} - {plant.ac_max:.3f} mm\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for y_index, y0_m in enumerate(y0_values):
        print(
            f"[y0 {y_index + 1}/{len(y0_values)}] milling start y0={y0_m:.3f} m "
            f"(L2={plant.L2:.3f} m)"
        )
        y_rows = sweep_rpm_boundary_stochastic_at_y0(
            env,
            plant,
            rpm_values,
            args,
            y0_m=y0_m,
            seed_offset=y_index * 100_000,
        )
        rows.extend(y_rows)

    env.close()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "time_domain_no_control_binary_search_stochastic_mc",
        "note": (
            "Stochastic time-domain estimate of the no-control stability boundary. "
            "For each (rpm, y0), n_mc independent binary searches with optional "
            "dynamics uncertainty; boundary reported as mean +/- std across MC runs. "
            "Physical displacement criteria and growth-ratio chatter detection are "
            "unchanged from the deterministic lobe. RL/control is not used."
        ),
        "env_id": ENV_ID,
        "dt": args.dt,
        "n_substeps": args.n_substeps,
        "sim_seconds": args.sim_seconds,
        "max_sim_steps": args.max_sim_steps,
        "rpm_min": args.rpm_min,
        "rpm_max": args.rpm_max,
        "rpm_points": args.rpm_points,
        "y0_values_m": y0_values,
        "y0_min_m": args.y0_min,
        "y0_max_m": args.y0_max,
        "y0_points": args.y0_points,
        "n_mc": args.n_mc,
        "dynamics_uncertainty_std": args.dynamics_uncertainty_std,
        "L2_m": plant.L2,
        "ac_min_mm": plant.ac_min,
        "ac_max_mm": plant.ac_max,
        "ac_tol_mm": args.ac_tol,
        "binary_iters": args.binary_iters,
        "w_limit_m": plant.w_limit,
        "w_limit_fraction": args.w_limit_fraction,
        "rms_growth_threshold": args.rms_growth_threshold,
        "seed": args.seed,
        "omega_min_rad_s": plant.omega_min,
        "omega_max_rad_s": plant.omega_max,
    }

    return rows, metadata


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_lobe(rows: list[dict], metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rpm = np.array([row["rpm"] for row in rows], dtype=np.float64)
    ac = np.array([row["ac_stable_mm"] for row in rows], dtype=np.float64)
    ac_max = float(metadata["ac_max_mm"])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(rpm, ac, "o-", color="#1f77b4", linewidth=2.0, markersize=6, label="Stable boundary")

    if rpm.size >= 2:
        ax.fill_between(
            rpm,
            0.0,
            ac,
            color="#1f77b4",
            alpha=0.12,
            label="Stable region (below curve)",
        )
        ax.fill_between(
            rpm,
            ac,
            ac_max,
            color="#d62728",
            alpha=0.08,
            label="Chatter / unstable region (above curve)",
        )

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut (mm)")
    ax.set_title("No-control stability lobe (time-domain estimate)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlim(float(np.min(rpm)), float(np.max(rpm)))
    ax.set_ylim(0.0, ac_max * 1.05)
    ax.legend(loc="best")

    note = (
        "Time-domain estimate from uncontrolled CustomODEPlate-v0 simulations.\n"
        "Below curve: stable. Above curve: chatter/unstable."
    )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lobe_surface_3d(rows: list[dict], metadata: dict, path: Path) -> None:
    """Plot stable boundary ac as a surface over rpm and milling start y0."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rpm_axis = np.array(sorted({row["rpm"] for row in rows}), dtype=np.float64)
    y0_axis = np.array(sorted({row["y0_m"] for row in rows}), dtype=np.float64)

    ac_grid = np.full((y0_axis.size, rpm_axis.size), np.nan, dtype=np.float64)
    rpm_index = {rpm: idx for idx, rpm in enumerate(rpm_axis)}
    y0_index = {y0: idx for idx, y0 in enumerate(y0_axis)}

    for row in rows:
        ac_grid[y0_index[row["y0_m"]], rpm_index[row["rpm"]]] = row["ac_stable_mm"]

    rpm_mesh, y0_mesh = np.meshgrid(rpm_axis, y0_axis)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        rpm_mesh,
        y0_mesh,
        ac_grid,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.2,
        antialiased=True,
    )
    ax.plot_wireframe(
        rpm_mesh,
        y0_mesh,
        ac_grid,
        color="black",
        linewidth=0.4,
        alpha=0.35,
    )

    for y0 in y0_axis:
        ac_line = ac_grid[y0_index[y0], :]
        ax.plot(
            rpm_axis,
            np.full_like(rpm_axis, y0),
            ac_line,
            color="#d62728",
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Milling start point y0 (m)")
    ax.set_zlabel("Stable axial depth ac (mm)")
    ax.set_title("No-control 3D stability lobe surface (time-domain estimate)")
    fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1, label="ac stable (mm)")

    note = (
        "Surface: max stable ac = f(rpm, y0). Uncontrolled simulator only.\n"
        "Depends on regenerative model and y0 force-path coupling."
    )
    fig.text(
        0.02,
        0.02,
        note,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stochastic_lobe_multi_y0(rows: list[dict], metadata: dict, path: Path) -> None:
    """Plot MC mean stability boundaries with +/- std bands for several y0 values."""
    path.parent.mkdir(parents=True, exist_ok=True)

    y0_values = sorted({row["y0_m"] for row in rows})
    ac_max = float(metadata["ac_max_mm"])
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for idx, y0_m in enumerate(y0_values):
        y_rows = sorted(
            (row for row in rows if row["y0_m"] == y0_m),
            key=lambda row: row["rpm"],
        )
        rpm = np.array([row["rpm"] for row in y_rows], dtype=np.float64)
        ac_mean = np.array(
            [row["ac_stable_mean_mm"] for row in y_rows], dtype=np.float64
        )
        ac_std = np.array(
            [row["ac_stable_std_mm"] for row in y_rows], dtype=np.float64
        )

        color = cmap(idx % 10)
        label = f"y0 = {y0_m:.3f} m"
        ax.plot(
            rpm,
            ac_mean,
            "-",
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"{label} (mean)",
        )
        ax.fill_between(
            rpm,
            np.maximum(ac_mean - ac_std, 0.0),
            ac_mean + ac_std,
            color=color,
            alpha=0.22,
            label=f"{label} +/- std",
        )

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut (mm)")
    n_mc = metadata.get("n_mc", 1)
    unc = metadata.get("dynamics_uncertainty_std", 0.0)
    ax.set_title(
        "Stochastic no-control stability lobe "
        f"(MC mean +/- std, n_mc={n_mc}, uncertainty={unc})"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    if rows:
        all_rpm = [row["rpm"] for row in rows]
        ax.set_xlim(float(np.min(all_rpm)), float(np.max(all_rpm)))
    ax.set_ylim(0.0, ac_max * 1.05)

    handles, labels = ax.get_legend_handles_labels()
    # Keep one legend entry per y0 (mean line only) for clarity.
    mean_handles = [h for h, lab in zip(handles, labels) if "(mean)" in lab]
    mean_labels = [lab.replace(" (mean)", "") for lab in labels if "(mean)" in lab]
    ax.legend(mean_handles, mean_labels, loc="best", fontsize=9)

    note = (
        "Solid line: mean stable ac across MC runs. Shaded band: +/- 1 std.\n"
        "Below each curve: typically stable. Above: chatter/unstable (time-domain estimate)."
    )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a no-control stability lobe for CustomODEPlate-v0."
    )
    parser.add_argument("--rpm-min", type=float, default=500.0)
    parser.add_argument("--rpm-max", type=float, default=omega_to_rpm(4000.0))
    parser.add_argument("--rpm-points", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--n-substeps", type=int, default=1)
    parser.add_argument(
        "--sim-seconds",
        type=float,
        default=8.0,
        help="Simulated time per trial [s] (converted to steps)",
    )
    parser.add_argument(
        "--max-sim-steps",
        type=int,
        default=8000,
        help="Hard cap on steps per trial",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Env truncation limit (0 = auto from pass duration)",
    )
    parser.add_argument("--ac-tol", type=float, default=0.1, help="Binary-search tolerance [mm]")
    parser.add_argument("--binary-iters", type=int, default=12)
    parser.add_argument("--rms-growth-threshold", type=float, default=1.8)
    parser.add_argument("--w-limit-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/stability_lobe"),
    )
    parser.add_argument(
        "--surface-3d",
        action="store_true",
        help="Compute 3D stability surface ac=f(rpm, y0) instead of 2D lobe only",
    )
    parser.add_argument(
        "--y0-min",
        type=float,
        default=0.05,
        help="Minimum milling start y0 [m] for 3D surface mode",
    )
    parser.add_argument(
        "--y0-max",
        type=float,
        default=0.45,
        help="Maximum milling start y0 [m] for 3D surface mode",
    )
    parser.add_argument(
        "--y0-points",
        type=int,
        default=5,
        help="Number of y0 grid points for 3D surface or stochastic multi-y0 mode",
    )
    parser.add_argument(
        "--stochastic-lobe",
        action="store_true",
        help=(
            "Stochastic multi-y0 2D lobe: several y0 curves with MC mean +/- std "
            "bands on one rpm-ac figure (separate outputs from deterministic lobe)"
        ),
    )
    parser.add_argument(
        "--y0",
        type=float,
        nargs="*",
        default=None,
        help="Explicit milling start y0 values [m] for --stochastic-lobe",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=5,
        help="Monte Carlo repetitions per (rpm, y0) in --stochastic-lobe mode",
    )
    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=0.01,
        help="Gaussian disturbance on modal accelerations for stochastic lobe [0=off]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    if args.surface_3d and args.stochastic_lobe:
        raise SystemExit("Choose only one of --surface-3d or --stochastic-lobe.")

    if args.stochastic_lobe:
        rows, metadata = compute_stochastic_lobe_multi_y0(args)
        csv_path = out_dir / "stability_lobe_stochastic.csv"
        meta_path = out_dir / "stability_lobe_stochastic_metadata.json"
        plot_path = out_dir / "stability_lobe_stochastic.png"

        save_csv(rows, csv_path)
        save_metadata(metadata, meta_path)
        plot_stochastic_lobe_multi_y0(rows, metadata, plot_path)

        print("\nSaved stochastic lobe outputs:")
        print(f"  plot     : {plot_path.resolve()}")
        print(f"  csv      : {csv_path.resolve()}")
        print(f"  metadata : {meta_path.resolve()}")
        return

    if args.surface_3d:
        rows, metadata = compute_lobe_surface_3d(args)
        csv_path = out_dir / "stability_lobe_surface.csv"
        meta_path = out_dir / "stability_lobe_surface_metadata.json"
        plot_path = out_dir / "stability_lobe_surface.png"

        save_csv(rows, csv_path)
        save_metadata(metadata, meta_path)
        plot_lobe_surface_3d(rows, metadata, plot_path)

        print("\nSaved 3D surface outputs:")
        print(f"  plot     : {plot_path.resolve()}")
        print(f"  csv      : {csv_path.resolve()}")
        print(f"  metadata : {meta_path.resolve()}")
        return

    rows, metadata = compute_lobe(args)

    csv_path = out_dir / "stability_lobe.csv"
    meta_path = out_dir / "stability_lobe_metadata.json"
    plot_path = out_dir / "stability_lobe.png"

    save_csv(rows, csv_path)
    save_metadata(metadata, meta_path)
    plot_lobe(rows, metadata, plot_path)

    print("\nSaved outputs:")
    print(f"  plot     : {plot_path.resolve()}")
    print(f"  csv      : {csv_path.resolve()}")
    print(f"  metadata : {meta_path.resolve()}")


if __name__ == "__main__":
    main()
