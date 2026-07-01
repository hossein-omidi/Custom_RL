"""Compute and plot no-control stability-lobe estimates for the face-milling plate env.

This script is for *environment/model analysis only*. It does not use RL,
trained policies, rewards, PPO, SAC, or feedback control. It sweeps fixed
physical machining conditions and estimates the largest stable axial depth of
cut by time-domain simulation of ``CustomODEPlate-v0``.

Face-milling convention
-----------------------
The default face-milling plant action is

    u = [u_omega, u_ap]

where ``ap`` is the axial depth of cut [mm]. This is the face-milling equivalent
of the old peripheral-milling ``ac`` depth variable. The radial immersion
``ae`` is a fixed plant parameter by default. Use ``control_ae=True`` only if
an explicit third action ``ae`` is desired.

Outputs / modes
---------------
1) 2D lobe:
       ap_stable = f(spindle speed)

2) Moving-pass 3D surface:
       ap_stable = f(spindle speed, pass start x-position)

   This is a practical moving-pass/path-segment stability map for your
   environment model. For each selected x_start, the cutter begins at x_start
   and then moves in the real pass direction x=L1 free side -> x=0 clamped side.
   Therefore, this is not a local frozen-position SDM lobe; it is a time-domain
   validation surface for the remaining pass segment from x_start to the clamp.

3) Stochastic moving-pass multi-position lobe:
       several 2D curves, one per chosen pass start x-position, with MC mean +/- std bands

Important limitation
--------------------
This is a practical time-domain stability estimate. It is not an analytical
semi-discretization-method stability-lobe solver. It is useful for validating
and studying the current simulator dynamics. Its accuracy depends on the
face-milling force model, regenerative-delay history, numerical time step, and
classification thresholds.

Examples
--------
    python scripts/compute_stability_lobe.py

    python scripts/compute_stability_lobe.py --rpm-min 400 --rpm-max 2500 \
        --rpm-points 12 --ap-max 0.6

    python scripts/compute_stability_lobe.py --surface-3d \
        --x-min 0.1 --x-max 1.0 --x-points 6

    python scripts/compute_stability_lobe.py --stochastic-lobe \
        --x-position 0.2 0.5 0.8 --n-mc 3 --dynamics-uncertainty-std 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from custom_rl import register_envs
from custom_rl.eval.monte_carlo import MC_BAND_STD_MULT
from custom_rl.plants.plate import omega_to_rpm, rpm_to_omega

ENV_ID = "CustomODEPlate-v0"

# Defaults chosen for the unified reduced Ti-6Al-4V face-milling model.
# These are script search defaults, not hard-coded plant parameters.
DEFAULT_RPM_MIN = 400.0
DEFAULT_RPM_MAX = 2500.0
DEFAULT_AP_MIN_MM = 0.0
DEFAULT_AP_MAX_MM = 0.6


# ---------------------------------------------------------------------------
# Basic action / plant helpers
# ---------------------------------------------------------------------------


def _depth_bounds_from_plant(plant: Any) -> tuple[float, float]:
    """Return plant axial-depth action bounds using ap names, with ac fallback."""
    ap_min = getattr(plant, "ap_min", getattr(plant, "ac_min", 0.0))
    ap_max = getattr(plant, "ap_max", getattr(plant, "ac_max", 1.0))
    return float(ap_min), float(ap_max)


def _physical_action_bounds(plant: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return physical action bounds [omega, ap] or [omega, ap, ae]."""
    if hasattr(plant, "physical_action_bounds"):
        low, high = plant.physical_action_bounds()
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)

    ap_min, ap_max = _depth_bounds_from_plant(plant)
    if bool(getattr(plant, "control_ae", False)):
        low = np.array([plant.omega_min, ap_min, plant.ae_min], dtype=np.float64)
        high = np.array([plant.omega_max, ap_max, plant.ae_max], dtype=np.float64)
    else:
        low = np.array([plant.omega_min, ap_min], dtype=np.float64)
        high = np.array([plant.omega_max, ap_max], dtype=np.float64)
    return low, high


def physical_to_normalized_action(u_phys: np.ndarray, plant: Any) -> np.ndarray:
    """Map physical action [omega, ap] or [omega, ap, ae] to normalized action."""
    low, high = _physical_action_bounds(plant)
    u_phys = np.asarray(u_phys, dtype=np.float64).reshape(-1)

    if u_phys.size < low.size:
        padded = np.empty(low.size, dtype=np.float64)
        padded[:] = low
        padded[: u_phys.size] = u_phys
        u_phys = padded
    elif u_phys.size > low.size:
        u_phys = u_phys[: low.size]

    span = np.maximum(high - low, 1e-12)
    frac = (u_phys - low) / span
    return np.clip(2.0 * frac - 1.0, -1.0, 1.0)


def set_milling_line_y(plant: Any, y_m: float | None) -> float:
    """Set the fixed y-location of the x-direction milling line."""
    if y_m is None:
        return float(getattr(plant, "y_cutter", 0.0))

    y_m = float(np.clip(float(y_m), 0.0, float(plant.L2)))
    if hasattr(plant, "set_milling_line_y"):
        plant.set_milling_line_y(y_m)
    elif hasattr(plant, "set_milling_start_y"):
        plant.set_milling_start_y(y_m)
    else:
        plant.y_cutter = y_m
        if hasattr(plant, "_sync_face_milling_geometry"):
            plant._sync_face_milling_geometry()
    return y_m


def set_tool_start_x(plant: Any, x_position_m: float | None) -> float:
    """Set the pass start position x_c(0) while keeping x=L1 -> x=0 direction.

    The face-milling force module uses

        x_c(t) = L1 - (feed_distance(t) + x0_cutter)

    Therefore, choosing ``x0_cutter = L1 - x_position`` makes the reset/start
    point equal to ``x_position``. The pass still moves toward x=0.
    """
    if x_position_m is None:
        return float(getattr(plant, "L1", 0.0) - getattr(plant, "x0_cutter", 0.0))

    L1 = float(plant.L1)
    x_position_m = float(np.clip(float(x_position_m), 0.0, L1))
    plant.x0_cutter = float(L1 - x_position_m)
    if hasattr(plant, "_sync_face_milling_geometry"):
        plant._sync_face_milling_geometry()
    return x_position_m


def resolve_ap_search_bounds(args: argparse.Namespace, plant: Any) -> tuple[float, float]:
    """Return safe axial-depth search bounds inside the plant action bounds."""
    plant_ap_min, plant_ap_max = _depth_bounds_from_plant(plant)
    ap_min = float(args.ap_min)
    ap_max = float(args.ap_max)

    if ap_max <= ap_min:
        raise ValueError(f"ap_max must be greater than ap_min; got {ap_min}..{ap_max} mm.")

    if ap_min < plant_ap_min - 1e-12:
        print(
            f"Warning: requested ap_min={ap_min:g} mm is below plant bound "
            f"{plant_ap_min:g}; clipping."
        )
        ap_min = plant_ap_min

    if ap_max > plant_ap_max + 1e-12:
        print(
            f"Warning: requested ap_max={ap_max:g} mm is above plant bound "
            f"{plant_ap_max:g}; clipping."
        )
        ap_max = plant_ap_max

    if ap_max <= ap_min:
        raise ValueError(
            f"No valid ap search interval after clipping: {ap_min:g}..{ap_max:g} mm."
        )
    return ap_min, ap_max


def _reset_options_for_line_y(plant: Any, line_y_m: float | None) -> dict[str, float] | None:
    """Build reset options that prevent random y from overriding the selected line."""
    if line_y_m is None:
        return None
    return {"y0": float(np.clip(line_y_m, 0.0, float(plant.L2)))}


# ---------------------------------------------------------------------------
# Trial metric and classification
# ---------------------------------------------------------------------------


@dataclass
class TrialMetrics:
    """Diagnostic metrics from one fixed-action simulation."""

    rpm: float
    omega_rad_s: float
    ap_mm: float
    stable: bool
    steps: int
    sim_time_s: float
    x_position_m: float
    x_end_m: float
    remaining_path_m: float
    y_line_m: float
    max_abs_w_m: float
    rms_w_m: float
    rms_w_previous_window_m: float
    rms_w_last_window_m: float
    rms_growth_ratio: float
    terminated: bool
    truncated: bool
    pass_completed: bool
    termination_reason: str | None


@dataclass
class StabilityConfig:
    """Thresholds used to classify a fixed-depth trial."""

    rms_growth_threshold: float
    w_limit_fraction: float
    settling_fraction: float
    growth_window_fraction: float
    growth_min_w_fraction: float


def _sensor_displacement_series(info_history: list[dict]) -> np.ndarray:
    """Return max absolute physical sensor displacement at each step."""
    values: list[float] = []
    for info in info_history:
        if "w_sensor" in info:
            w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
            values.append(float(np.max(np.abs(w))))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float64)


def _window_rms(series: np.ndarray) -> float:
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    if series.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(series**2)))


def classify_stability(
    w_series: np.ndarray,
    *,
    w_limit: float,
    termination_reason: str | None,
    config: StabilityConfig,
) -> tuple[bool, float, float, float, float, float]:
    """Classify a simulation as stable or chatter/unstable.

    The classifier intentionally avoids comparing the very beginning of the
    simulation with the end, because a stable forced system can grow from zero
    initial condition into a bounded periodic steady response. Instead it
    compares two late windows after a settling interval.
    """
    w_series = np.asarray(w_series, dtype=np.float64).reshape(-1)
    if w_series.size == 0:
        return True, 0.0, 0.0, 0.0, 0.0, 1.0

    max_abs_w = float(np.max(w_series))
    rms_w = _window_rms(w_series)

    n = int(w_series.size)
    start = int(np.clip(config.settling_fraction, 0.0, 0.9) * n)
    tail = w_series[start:]
    if tail.size < 4:
        tail = w_series

    win = max(int(config.growth_window_fraction * tail.size), 2)
    win = min(win, max(tail.size // 2, 1))
    previous = tail[-2 * win : -win] if tail.size >= 2 * win else tail[: tail.size // 2]
    last = tail[-win:]

    rms_previous = _window_rms(previous)
    rms_last = _window_rms(last)
    growth_ratio = rms_last / max(rms_previous, 1e-12)

    severe_termination = termination_reason in {
        "excessive_sensor_displacement",
        "excessive_sensor_velocity",
        "invalid_state",
    }
    displacement_exceeded = max_abs_w > float(w_limit) * float(config.w_limit_fraction)
    growing_late_response = (
        growth_ratio >= float(config.rms_growth_threshold)
        and rms_last > float(config.growth_min_w_fraction) * float(w_limit)
        and rms_last > rms_previous
    )

    stable = not (severe_termination or displacement_exceeded or growing_late_response)
    return stable, max_abs_w, rms_w, rms_previous, rms_last, growth_ratio


# ---------------------------------------------------------------------------
# Fixed-action simulation and binary search
# ---------------------------------------------------------------------------


def run_fixed_action_trial(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ap_mm: float,
    max_steps: int,
    seed: int,
    stability_config: StabilityConfig,
    x_position_m: float | None = None,
    line_y_m: float | None = None,
) -> TrialMetrics:
    """Simulate one fixed [omega, ap] pair with no controller."""
    plant = env.unwrapped.plant

    x_selected = set_tool_start_x(plant, x_position_m)
    y_selected = set_milling_line_y(plant, line_y_m)
    reset_options = _reset_options_for_line_y(plant, y_selected)

    if bool(getattr(plant, "control_ae", False)):
        ae = float(getattr(plant, "ae_default", getattr(plant, "ae_min", 0.0)))
        u_phys = np.array([omega_rad_s, ap_mm, ae], dtype=np.float64)
    else:
        u_phys = np.array([omega_rad_s, ap_mm], dtype=np.float64)
    action = physical_to_normalized_action(u_phys, plant)

    env.reset(seed=seed, options=reset_options)

    info_history: list[dict] = []
    terminated = False
    truncated = False
    termination_reason: str | None = None
    steps = 0

    while steps < max_steps and not (terminated or truncated):
        _, _reward, terminated, truncated, info = env.step(action)
        info_history.append(dict(info))
        termination_reason = info.get("termination_reason")
        steps += 1

    w_series = _sensor_displacement_series(info_history)
    stable, max_abs_w, rms_w, rms_prev, rms_last, growth_ratio = classify_stability(
        w_series,
        w_limit=float(plant.w_limit),
        termination_reason=termination_reason,
        config=stability_config,
    )

    sim_time_s = (
        float(info_history[-1].get("t", steps * env.unwrapped._step_dt))
        if info_history
        else 0.0
    )

    return TrialMetrics(
        rpm=float(omega_to_rpm(omega_rad_s)),
        omega_rad_s=float(omega_rad_s),
        ap_mm=float(ap_mm),
        stable=bool(stable),
        steps=int(steps),
        sim_time_s=sim_time_s,
        x_position_m=float(x_selected),
        x_end_m=0.0,
        remaining_path_m=max(float(x_selected) - float(getattr(plant, "x_pass_end_tol", 0.0)), 0.0),
        y_line_m=float(y_selected),
        max_abs_w_m=max_abs_w,
        rms_w_m=rms_w,
        rms_w_previous_window_m=rms_prev,
        rms_w_last_window_m=rms_last,
        rms_growth_ratio=growth_ratio,
        terminated=bool(terminated),
        truncated=bool(truncated),
        pass_completed=(termination_reason == "pass_completed"),
        termination_reason=termination_reason,
    )


def find_max_stable_ap(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ap_min: float,
    ap_max: float,
    max_steps: int,
    seed: int,
    ap_tol_mm: float,
    max_binary_iters: int,
    stability_config: StabilityConfig,
    x_position_m: float | None = None,
    line_y_m: float | None = None,
) -> tuple[float, TrialMetrics, TrialMetrics | None]:
    """Binary search for the largest stable axial depth of cut ap."""
    ap_lo = float(ap_min)
    ap_hi = float(ap_max)

    trial_lo = run_fixed_action_trial(
        env,
        omega_rad_s=omega_rad_s,
        ap_mm=ap_lo,
        max_steps=max_steps,
        seed=seed,
        stability_config=stability_config,
        x_position_m=x_position_m,
        line_y_m=line_y_m,
    )

    if not trial_lo.stable:
        return ap_lo, trial_lo, trial_lo

    trial_hi = run_fixed_action_trial(
        env,
        omega_rad_s=omega_rad_s,
        ap_mm=ap_hi,
        max_steps=max_steps,
        seed=seed + 1,
        stability_config=stability_config,
        x_position_m=x_position_m,
        line_y_m=line_y_m,
    )

    if trial_hi.stable:
        return ap_hi, trial_hi, None

    unstable_probe: TrialMetrics | None = trial_hi
    best_stable = trial_lo

    for _ in range(int(max_binary_iters)):
        if ap_hi - ap_lo <= float(ap_tol_mm):
            break

        ap_mid = 0.5 * (ap_lo + ap_hi)
        trial_mid = run_fixed_action_trial(
            env,
            omega_rad_s=omega_rad_s,
            ap_mm=ap_mid,
            max_steps=max_steps,
            seed=seed + int(round(ap_mid * 10000.0)),
            stability_config=stability_config,
            x_position_m=x_position_m,
            line_y_m=line_y_m,
        )

        if trial_mid.stable:
            ap_lo = ap_mid
            best_stable = trial_mid
        else:
            ap_hi = ap_mid
            unstable_probe = trial_mid

    return ap_lo, best_stable, unstable_probe


def simulation_steps_for_rpm(
    rpm: float,
    *,
    dt: float,
    n_substeps: int,
    sim_seconds: float | None,
    max_sim_steps: int,
) -> int:
    """Choose a practical time-domain step budget for one lobe evaluation."""
    step_dt = max(float(dt) * int(n_substeps), 1e-12)
    if sim_seconds is None or sim_seconds <= 0.0:
        # Conservative fallback when no time is specified.
        return int(max_sim_steps)
    steps = int(np.ceil(float(sim_seconds) / step_dt))
    return max(1, min(steps, int(max_sim_steps)))


# ---------------------------------------------------------------------------
# Rows / metadata helpers
# ---------------------------------------------------------------------------


def _trial_row(
    *,
    rpm: float,
    omega: float,
    ap_stable: float,
    stable_trial: TrialMetrics,
    unstable_trial: TrialMetrics | None,
) -> dict[str, Any]:
    return {
        "rpm": float(rpm),
        "omega_rad_s": float(omega),
        "x_start_m": float(stable_trial.x_position_m),
        "x_position_m": float(stable_trial.x_position_m),  # backward-compatible alias for x_start_m
        "x_end_m": float(stable_trial.x_end_m),
        "remaining_path_m": float(stable_trial.remaining_path_m),
        "y_line_m": float(stable_trial.y_line_m),
        "ap_stable_mm": float(ap_stable),
        "max_abs_w_m": float(stable_trial.max_abs_w_m),
        "rms_w_m": float(stable_trial.rms_w_m),
        "rms_w_previous_window_m": float(stable_trial.rms_w_previous_window_m),
        "rms_w_last_window_m": float(stable_trial.rms_w_last_window_m),
        "rms_growth_ratio": float(stable_trial.rms_growth_ratio),
        "sim_steps": int(stable_trial.steps),
        "sim_time_s": float(stable_trial.sim_time_s),
        "termination_reason": stable_trial.termination_reason,
        "stable_at_boundary": bool(stable_trial.stable),
        "pass_completed_at_boundary": bool(stable_trial.pass_completed),
        "unstable_ap_mm": None if unstable_trial is None else float(unstable_trial.ap_mm),
        "unstable_max_abs_w_m": None if unstable_trial is None else float(unstable_trial.max_abs_w_m),
    }


def _metadata_common(args: argparse.Namespace, plant: Any, method: str, note: str) -> dict[str, Any]:
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "note": note,
        "env_id": ENV_ID,
        "dt": float(args.dt),
        "n_substeps": int(args.n_substeps),
        "sim_seconds": None if args.sim_seconds is None else float(args.sim_seconds),
        "max_sim_steps": int(args.max_sim_steps),
        "rpm_min": float(args.rpm_min),
        "rpm_max": float(args.rpm_max),
        "rpm_points": int(args.rpm_points),
        "ap_min_mm": float(ap_min),
        "ap_max_mm": float(ap_max),
        "ap_tol_mm": float(args.ap_tol),
        "binary_iters": int(args.binary_iters),
        "w_limit_m": float(plant.w_limit),
        "w_limit_fraction": float(args.w_limit_fraction),
        "rms_growth_threshold": float(args.rms_growth_threshold),
        "settling_fraction": float(args.settling_fraction),
        "growth_window_fraction": float(args.growth_window_fraction),
        "growth_min_w_fraction": float(args.growth_min_w_fraction),
        "seed": int(args.seed),
        "omega_min_rad_s": float(plant.omega_min),
        "omega_max_rad_s": float(plant.omega_max),
        "plant_ap_min_mm": _depth_bounds_from_plant(plant)[0],
        "plant_ap_max_mm": _depth_bounds_from_plant(plant)[1],
        "D_mm": float(getattr(plant, "D_mm", np.nan)),
        "ae_default_mm": float(getattr(plant, "ae_default", np.nan)),
        "feed_per_tooth_mm": float(getattr(plant, "feed_per_tooth_mm", np.nan)),
        "milling_mode": str(getattr(plant, "milling_mode", "")),
        "force_projection_mode": str(getattr(plant, "force_projection_mode", "")),
        "use_process_damping": bool(getattr(plant, "use_process_damping", False)),
    }


# ---------------------------------------------------------------------------
# Deterministic 2D lobe
# ---------------------------------------------------------------------------


def compute_lobe(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Sweep spindle speed and estimate ap_stable at one x-position/y-line."""
    register_envs()
    env = _make_env(args)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    stability_config = _stability_config_from_args(args)

    line_y = set_milling_line_y(plant, args.line_y)
    x_position = _resolve_single_x_position(args, plant)

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    rows: list[dict] = []

    print(
        "No-control face-milling stability lobe sweep\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm "
        f"({args.rpm_points} points)\n"
        f"  ap search     : {ap_min:.4g} - {ap_max:.4g} mm\n"
        f"  x start       : {x_position:.4g} m (free side L1={plant.L1:.4g}, clamped x=0)\n"
        f"  y line        : {line_y:.4g} m\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = simulation_steps_for_rpm(
            rpm_clipped,
            dt=args.dt,
            n_substeps=args.n_substeps,
            sim_seconds=args.sim_seconds,
            max_sim_steps=args.max_sim_steps,
        )

        print(
            f"[{index + 1}/{len(rpm_values)}] rpm={rpm_clipped:.0f} "
            f"(omega={omega:.2f} rad/s), max_steps={max_steps}"
        )

        ap_stable, stable_trial, unstable_trial = find_max_stable_ap(
            env,
            omega_rad_s=omega,
            ap_min=ap_min,
            ap_max=ap_max,
            max_steps=max_steps,
            seed=args.seed + index,
            ap_tol_mm=args.ap_tol,
            max_binary_iters=args.binary_iters,
            stability_config=stability_config,
            x_position_m=x_position,
            line_y_m=line_y,
        )
        rows.append(
            _trial_row(
                rpm=rpm_clipped,
                omega=omega,
                ap_stable=ap_stable,
                stable_trial=stable_trial,
                unstable_trial=unstable_trial,
            )
        )

        print(
            f"    -> stable boundary ap ~ {ap_stable:.4g} mm "
            f"(max|w|={stable_trial.max_abs_w_m:.3e} m, "
            f"late growth={stable_trial.rms_growth_ratio:.2f})"
        )

    metadata = _metadata_common(
        args,
        plant,
        method="time_domain_no_control_binary_search_face_milling_full_moving_pass_2d",
        note=(
            "Practical time-domain estimate of ap_stable=f(rpm) for the fixed "
            "face-milling plant. RL/control is not used. ap is axial depth of cut."
        ),
    )
    metadata["x_position_m"] = float(x_position)
    metadata["y_line_m"] = float(line_y)

    env.close()
    return rows, metadata


# ---------------------------------------------------------------------------
# Deterministic 3D surface over x-position
# ---------------------------------------------------------------------------


def _resolve_x_axis(args: argparse.Namespace, plant: Any) -> list[float]:
    if args.x_position:
        return [float(np.clip(v, 0.0, float(plant.L1))) for v in args.x_position]

    x_min = args.x_min
    x_max = args.x_max
    if x_min is None:
        # For moving-pass validation, starting extremely close to the clamp can
        # finish almost immediately and falsely look stable.  Use at least 10%
        # of the pass length by default; users can override with --x-min.
        x_min = max(2.0 * float(getattr(plant, "x_pass_end_tol", 0.0)), 0.10 * float(plant.L1))
    if x_max is None:
        x_max = float(plant.L1)

    x_min = float(np.clip(x_min, 0.0, float(plant.L1)))
    x_max = float(np.clip(x_max, 0.0, float(plant.L1)))
    if x_max < x_min:
        x_min, x_max = x_max, x_min

    x_values = [float(v) for v in np.linspace(x_min, x_max, int(args.x_points))]
    pass_end_tol = float(getattr(plant, "x_pass_end_tol", 0.0))
    if any(x <= pass_end_tol for x in x_values):
        print(
            "Warning: at least one x_start is at or below the pass-completion tolerance. "
            "Those trials may terminate immediately and overestimate stability."
        )
    return x_values


def _resolve_single_x_position(args: argparse.Namespace, plant: Any) -> float:
    if args.x_single is not None:
        return float(np.clip(args.x_single, 0.0, float(plant.L1)))
    # Default 2D lobe starts from the free side, matching the actual pass reset.
    return float(plant.L1)


def sweep_rpm_boundary_at_x(
    env: gym.Env,
    plant: Any,
    rpm_values: np.ndarray,
    args: argparse.Namespace,
    *,
    x_position_m: float,
    line_y_m: float,
    seed_offset: int,
) -> list[dict]:
    """Estimate ap_stable over rpm for a moving pass starting at x_position_m."""
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    stability_config = _stability_config_from_args(args)
    rows: list[dict] = []

    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = simulation_steps_for_rpm(
            rpm_clipped,
            dt=args.dt,
            n_substeps=args.n_substeps,
            sim_seconds=args.sim_seconds,
            max_sim_steps=args.max_sim_steps,
        )

        print(
            f"    rpm={rpm_clipped:.0f} (omega={omega:.2f} rad/s), max_steps={max_steps}"
        )

        ap_stable, stable_trial, unstable_trial = find_max_stable_ap(
            env,
            omega_rad_s=omega,
            ap_min=ap_min,
            ap_max=ap_max,
            max_steps=max_steps,
            seed=args.seed + seed_offset + index,
            ap_tol_mm=args.ap_tol,
            max_binary_iters=args.binary_iters,
            stability_config=stability_config,
            x_position_m=x_position_m,
            line_y_m=line_y_m,
        )
        rows.append(
            _trial_row(
                rpm=rpm_clipped,
                omega=omega,
                ap_stable=ap_stable,
                stable_trial=stable_trial,
                unstable_trial=unstable_trial,
            )
        )

        print(
            f"      -> stable boundary ap ~ {ap_stable:.4g} mm "
            f"(max|w|={stable_trial.max_abs_w_m:.3e} m, "
            f"late growth={stable_trial.rms_growth_ratio:.2f})"
        )

    return rows


def compute_lobe_surface_3d(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Sweep pass start x-position and rpm for moving-pass stability validation."""
    register_envs()
    env = _make_env(args)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    x_values = _resolve_x_axis(args, plant)
    line_y = set_milling_line_y(plant, args.line_y)
    rows: list[dict] = []

    print(
        "No-control 3D moving-pass face-milling stability surface sweep\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm "
        f"({args.rpm_points} points)\n"
        f"  x start values: {x_values}\n"
        f"  y line        : {line_y:.4g} m\n"
        f"  ap search     : {ap_min:.4g} - {ap_max:.4g} mm\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for x_index, x_position in enumerate(x_values):
        print(
            f"[x {x_index + 1}/{len(x_values)}] pass start x={x_position:.4g} m "
            f"(L1={plant.L1:.4g} m; free side x=L1, clamped side x=0)"
        )
        rows.extend(
            sweep_rpm_boundary_at_x(
                env,
                plant,
                rpm_values,
                args,
                x_position_m=x_position,
                line_y_m=line_y,
                seed_offset=x_index * 1000,
            )
        )

    metadata = _metadata_common(
        args,
        plant,
        method="time_domain_no_control_binary_search_face_milling_3d_moving_pass_surface_start_x",
        note=(
            "Practical time-domain moving-pass estimate of ap_stable=f(rpm,x_start). "
            "For each x_start, the cutter begins at x_start and continues toward x=0 "
            "along the free-to-clamped pass. This is intended as a pass/path-segment "
            "validation map for the environment model, not a frozen local-position SDM lobe. "
            "RL/control is not used."
        ),
    )
    metadata["x_start_values_m"] = x_values
    metadata["x_positions_m"] = x_values  # backward-compatible alias
    metadata["y_line_m"] = float(line_y)
    metadata["L1_m"] = float(plant.L1)

    env.close()
    return rows, metadata


# ---------------------------------------------------------------------------
# Stochastic multi-x lobe
# ---------------------------------------------------------------------------


def find_max_stable_ap_mc(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ap_min: float,
    ap_max: float,
    max_steps: int,
    seed: int,
    ap_tol_mm: float,
    max_binary_iters: int,
    stability_config: StabilityConfig,
    n_mc: int,
    x_position_m: float,
    line_y_m: float,
) -> tuple[float, float, list[float], TrialMetrics, TrialMetrics | None]:
    """Estimate max stable ap with repeated binary searches."""
    ap_samples: list[float] = []
    last_stable: TrialMetrics | None = None
    last_unstable: TrialMetrics | None = None

    for mc in range(int(n_mc)):
        ap_stable, stable_trial, unstable_trial = find_max_stable_ap(
            env,
            omega_rad_s=omega_rad_s,
            ap_min=ap_min,
            ap_max=ap_max,
            max_steps=max_steps,
            seed=seed + mc * 10_000,
            ap_tol_mm=ap_tol_mm,
            max_binary_iters=max_binary_iters,
            stability_config=stability_config,
            x_position_m=x_position_m,
            line_y_m=line_y_m,
        )
        ap_samples.append(float(ap_stable))
        last_stable = stable_trial
        if unstable_trial is not None:
            last_unstable = unstable_trial

    ap_arr = np.asarray(ap_samples, dtype=np.float64)
    assert last_stable is not None
    return (
        float(np.mean(ap_arr)),
        float(np.std(ap_arr)) if ap_arr.size > 1 else 0.0,
        ap_samples,
        last_stable,
        last_unstable,
    )


def sweep_rpm_boundary_stochastic_at_x(
    env: gym.Env,
    plant: Any,
    rpm_values: np.ndarray,
    args: argparse.Namespace,
    *,
    x_position_m: float,
    line_y_m: float,
    seed_offset: int,
) -> list[dict]:
    """Estimate stochastic ap boundary over rpm for a moving pass starting at x_position_m."""
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    stability_config = _stability_config_from_args(args)
    rows: list[dict] = []

    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = simulation_steps_for_rpm(
            rpm_clipped,
            dt=args.dt,
            n_substeps=args.n_substeps,
            sim_seconds=args.sim_seconds,
            max_sim_steps=args.max_sim_steps,
        )

        print(
            f"    rpm={rpm_clipped:.0f} (omega={omega:.2f} rad/s), "
            f"max_steps={max_steps}, n_mc={args.n_mc}"
        )

        ap_mean, ap_std, ap_samples, stable_trial, unstable_trial = find_max_stable_ap_mc(
            env,
            omega_rad_s=omega,
            ap_min=ap_min,
            ap_max=ap_max,
            max_steps=max_steps,
            seed=args.seed + seed_offset + index,
            ap_tol_mm=args.ap_tol,
            max_binary_iters=args.binary_iters,
            stability_config=stability_config,
            n_mc=args.n_mc,
            x_position_m=x_position_m,
            line_y_m=line_y_m,
        )

        row = _trial_row(
            rpm=rpm_clipped,
            omega=omega,
            ap_stable=ap_mean,
            stable_trial=stable_trial,
            unstable_trial=unstable_trial,
        )
        row.update(
            {
                "ap_stable_mean_mm": ap_mean,
                "ap_stable_std_mm": ap_std,
                "ap_stable_samples_mm": json.dumps(ap_samples),
                "n_mc": int(args.n_mc),
            }
        )
        rows.append(row)

        print(
            f"      -> stable boundary ap ~ {ap_mean:.4g} +/- {ap_std:.4g} mm "
            f"(samples={ap_samples})"
        )

    return rows


def compute_stochastic_lobe_multi_x(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Stochastic moving-pass lobe: multiple x-start curves on one rpm-ap plot."""
    register_envs()
    env = _make_env(args, stochastic=True)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    x_values = _resolve_x_axis(args, plant)
    line_y = set_milling_line_y(plant, args.line_y)
    rows: list[dict] = []

    print(
        "Stochastic no-control moving-pass stability lobe (multi-x-start curves)\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm "
        f"({args.rpm_points} points)\n"
        f"  x start values: {x_values}\n"
        f"  y line        : {line_y:.4g} m\n"
        f"  n_mc          : {args.n_mc}\n"
        f"  dynamics unc. : {plant.dynamics_uncertainty_std}\n"
        f"  ap search     : {ap_min:.4g} - {ap_max:.4g} mm\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for x_index, x_position in enumerate(x_values):
        print(
            f"[x {x_index + 1}/{len(x_values)}] pass start x={x_position:.4g} m "
            f"(L1={plant.L1:.4g} m)"
        )
        rows.extend(
            sweep_rpm_boundary_stochastic_at_x(
                env,
                plant,
                rpm_values,
                args,
                x_position_m=x_position,
                line_y_m=line_y,
                seed_offset=x_index * 100_000,
            )
        )

    metadata = _metadata_common(
        args,
        plant,
        method="time_domain_no_control_binary_search_face_milling_stochastic_mc_multi_x_start_moving_pass",
        note=(
            "Stochastic time-domain moving-pass estimate of the no-control stability boundary. "
            "For each (rpm,x_start), n_mc independent binary searches are run while the cutter "
            "moves from x_start toward x=0. Boundary is reported as mean +/- std across MC runs. "
            "RL/control is not used."
        ),
    )
    metadata["x_start_values_m"] = x_values
    metadata["x_positions_m"] = x_values  # backward-compatible alias
    metadata["y_line_m"] = float(line_y)
    metadata["n_mc"] = int(args.n_mc)
    metadata["dynamics_uncertainty_std"] = float(args.dynamics_uncertainty_std)
    metadata["L1_m"] = float(plant.L1)

    env.close()
    return rows, metadata


# ---------------------------------------------------------------------------
# Saving / plotting
# ---------------------------------------------------------------------------


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
    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda row: row["rpm"])
    rpm = np.array([row["rpm"] for row in rows_sorted], dtype=np.float64)
    ap = np.array([row["ap_stable_mm"] for row in rows_sorted], dtype=np.float64)
    ap_max = float(metadata["ap_max_mm"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(rpm, ap, "o-", linewidth=2.0, markersize=6, label="Stable boundary")

    if rpm.size >= 2:
        ax.fill_between(rpm, 0.0, ap, alpha=0.12, label="Stable region")
        ax.fill_between(rpm, ap, ap_max, alpha=0.08, label="Chatter / unstable region")

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut ap (mm)")
    ax.set_title("No-control face-milling stability lobe (time-domain estimate)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlim(float(np.min(rpm)), float(np.max(rpm)))
    ax.set_ylim(0.0, ap_max * 1.05)
    ax.legend(loc="best")

    note = (
        "Fixed-action time-domain estimate from CustomODEPlate-v0.\n"
        "Below curve: stable by displacement/growth criteria. Above: chatter/unstable."
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
    """Plot stable boundary ap as a surface over rpm and pass start x-position."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    rpm_axis = np.array(sorted({row["rpm"] for row in rows}), dtype=np.float64)
    x_axis = np.array(sorted({row["x_position_m"] for row in rows}), dtype=np.float64)

    ap_grid = np.full((x_axis.size, rpm_axis.size), np.nan, dtype=np.float64)
    rpm_index = {rpm: idx for idx, rpm in enumerate(rpm_axis)}
    x_index = {x: idx for idx, x in enumerate(x_axis)}
    for row in rows:
        ap_grid[x_index[row["x_position_m"]], rpm_index[row["rpm"]]] = row["ap_stable_mm"]

    rpm_mesh, x_mesh = np.meshgrid(rpm_axis, x_axis)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        rpm_mesh,
        x_mesh,
        ap_grid,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.2,
        antialiased=True,
    )
    ax.plot_wireframe(rpm_mesh, x_mesh, ap_grid, color="black", linewidth=0.4, alpha=0.35)

    for x_position in x_axis:
        ap_line = ap_grid[x_index[x_position], :]
        ax.plot(
            rpm_axis,
            np.full_like(rpm_axis, x_position),
            ap_line,
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Pass start x-position (m)")
    ax.set_zlabel("Stable axial depth ap (mm)")
    ax.set_title("No-control 3D moving-pass face-milling stability surface")
    fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1, label="ap stable (mm)")

    note = (
        "Surface: max stable ap = f(rpm, x_start) for a remaining moving pass.\n"
        "Each trial moves from x_start toward x=0; x=L1 is the free side."
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


def plot_stochastic_lobe_multi_x(rows: list[dict], metadata: dict, path: Path) -> None:
    """Plot MC mean moving-pass stability boundaries with +/- std bands for x-start values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    x_values = sorted({row["x_position_m"] for row in rows})
    ap_max = float(metadata["ap_max_mm"])
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for idx, x_position in enumerate(x_values):
        x_rows = sorted(
            (row for row in rows if row["x_position_m"] == x_position),
            key=lambda row: row["rpm"],
        )
        rpm = np.array([row["rpm"] for row in x_rows], dtype=np.float64)
        ap_mean = np.array([row["ap_stable_mean_mm"] for row in x_rows], dtype=np.float64)
        ap_std = np.array([row["ap_stable_std_mm"] for row in x_rows], dtype=np.float64)
        band = MC_BAND_STD_MULT * ap_std

        color = cmap(idx % 10)
        label = f"x_start = {x_position:.3f} m"
        ax.plot(rpm, ap_mean, "-", color=color, linewidth=2.0, marker="o", markersize=4, label=label)
        ax.fill_between(rpm, np.maximum(ap_mean - band, 0.0), ap_mean + band, color=color, alpha=0.22)

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut ap (mm)")
    ax.set_title(
        "Stochastic no-control moving-pass face-milling stability lobe "
        f"(mean +/- {MC_BAND_STD_MULT:.0f}σ)"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    all_rpm = [row["rpm"] for row in rows]
    ax.set_xlim(float(np.min(all_rpm)), float(np.max(all_rpm)))
    ax.set_ylim(0.0, ap_max * 1.05)
    ax.legend(loc="best", fontsize=9)

    note = (
        "Solid line: mean stable ap across MC runs. Shaded band: +/- std multiplier.\n"
        "Each curve is one pass start x-position; the cutter then moves toward the clamp."
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


# ---------------------------------------------------------------------------
# Environment / args / main
# ---------------------------------------------------------------------------


def _make_env(args: argparse.Namespace, *, stochastic: bool = False) -> gym.Env:
    env_kwargs: dict[str, Any] = {
        "reward_id": "sparse",
        "dt": args.dt,
        "n_substeps": args.n_substeps,
        "randomize_y0": False,
        "dynamics_uncertainty_std": float(args.dynamics_uncertainty_std if stochastic else 0.0),
    }
    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    # Optional model overrides for lobe-specific experiments. None means use the
    # plant/repo default, which is usually preferred once the model is finalized.
    for key in (
        "E",
        "nu",
        "rho",
        "rho_type",
        "Kt",
        "Kr",
        "Ka",
        "Kte",
        "Kre",
        "Kae",
        "use_process_damping",
        "Ksp",
        "mu",
        "VB",
        "lambda_L_deg",
        "ap_min",
        "ap_max",
    ):
        value = getattr(args, key, None)
        if value is not None:
            env_kwargs[key] = value

    return gym.make(ENV_ID, **env_kwargs)


def _stability_config_from_args(args: argparse.Namespace) -> StabilityConfig:
    return StabilityConfig(
        rms_growth_threshold=float(args.rms_growth_threshold),
        w_limit_fraction=float(args.w_limit_fraction),
        settling_fraction=float(args.settling_fraction),
        growth_window_fraction=float(args.growth_window_fraction),
        growth_min_w_fraction=float(args.growth_min_w_fraction),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute no-control stability-lobe estimates for CustomODEPlate-v0."
    )
    parser.add_argument("--rpm-min", type=float, default=DEFAULT_RPM_MIN)
    parser.add_argument("--rpm-max", type=float, default=DEFAULT_RPM_MAX)
    parser.add_argument("--rpm-points", type=int, default=12)
    parser.add_argument("--ap-min", type=float, default=DEFAULT_AP_MIN_MM)
    parser.add_argument("--ap-max", type=float, default=DEFAULT_AP_MAX_MM)
    parser.add_argument("--ap-tol", type=float, default=0.02, help="Binary-search tolerance [mm].")
    parser.add_argument("--binary-iters", type=int, default=12)

    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--n-substeps", type=int, default=1)
    parser.add_argument("--sim-seconds", type=float, default=2.0, help="Simulated time per trial [s].")
    parser.add_argument("--max-sim-steps", type=int, default=20000)
    parser.add_argument("--max-episode-steps", type=int, default=0)

    parser.add_argument("--rms-growth-threshold", type=float, default=1.8)
    parser.add_argument("--w-limit-fraction", type=float, default=0.95)
    parser.add_argument("--settling-fraction", type=float, default=0.25)
    parser.add_argument("--growth-window-fraction", type=float, default=0.25)
    parser.add_argument("--growth-min-w-fraction", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/stability_lobe"))

    parser.add_argument("--surface-3d", action="store_true", help="Compute moving-pass ap=f(rpm,x_start) surface.")
    parser.add_argument("--stochastic-lobe", action="store_true", help="Compute multi-x-start moving-pass MC curves.")

    parser.add_argument("--x-single", type=float, default=None, help="Single pass start x-position [m] for 2D lobe.")
    parser.add_argument("--x-min", type=float, default=None, help="Minimum pass start x-position [m] for 3D/stochastic mode.")
    parser.add_argument("--x-max", type=float, default=None, help="Maximum pass start x-position [m] for 3D/stochastic mode.")
    parser.add_argument("--x-points", type=int, default=5)
    parser.add_argument("--x-position", type=float, nargs="*", default=None, help="Explicit pass start x-position values [m].")
    parser.add_argument("--line-y", type=float, default=None, help="Fixed milling line y [m]. Default: plant.y_cutter.")

    parser.add_argument("--n-mc", type=int, default=5)
    parser.add_argument("--dynamics-uncertainty-std", type=float, default=0.0)

    # Optional model overrides. Use these only for controlled parameter studies;
    # otherwise keep repo/plant defaults.
    # Material/cutting defaults below are the unified reduced Ti-6Al-4V setup.
    # Geometry is intentionally not overridden here.
    parser.add_argument("--E", type=float, default=113.8e9)
    parser.add_argument("--nu", type=float, default=0.34)
    parser.add_argument("--rho", type=float, default=4430.0)
    parser.add_argument("--rho-type", dest="rho_type", default="volumetric")
    parser.add_argument("--Kt", type=float, default=0.0)
    parser.add_argument("--Kr", type=float, default=0.0)
    parser.add_argument("--Ka", type=float, default=4790.9)
    parser.add_argument("--Kte", type=float, default=0.0)
    parser.add_argument("--Kre", type=float, default=0.0)
    parser.add_argument("--Kae", type=float, default=360.6)
    pd_group = parser.add_mutually_exclusive_group()
    pd_group.add_argument("--use-process-damping", dest="use_process_damping", action="store_true")
    pd_group.add_argument("--no-process-damping", dest="use_process_damping", action="store_false")
    parser.set_defaults(use_process_damping=True)
    parser.add_argument("--Ksp", type=float, default=30000.0)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--VB", type=float, default=0.08)
    parser.add_argument("--lambda-L-deg", dest="lambda_L_deg", type=float, default=45.0)

    args = parser.parse_args()
    if args.surface_3d and args.stochastic_lobe:
        raise SystemExit("Choose only one of --surface-3d or --stochastic-lobe.")
    if args.rpm_points <= 0:
        raise SystemExit("--rpm-points must be positive.")
    if args.x_points <= 0:
        raise SystemExit("--x-points must be positive.")
    if args.n_mc <= 0:
        raise SystemExit("--n-mc must be positive.")
    if args.ap_max > 1.0:
        print(
            "Warning: ap_max above 1 mm is outside the current Ti-6Al-4V reference-calibrated "
            "range. Use it only if you have re-identified cutting coefficients."
        )
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.stochastic_lobe:
        rows, metadata = compute_stochastic_lobe_multi_x(args)
        csv_path = out_dir / "stability_lobe_stochastic.csv"
        meta_path = out_dir / "stability_lobe_stochastic_metadata.json"
        plot_path = out_dir / "stability_lobe_stochastic.png"
        save_csv(rows, csv_path)
        save_metadata(metadata, meta_path)
        plot_stochastic_lobe_multi_x(rows, metadata, plot_path)
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
