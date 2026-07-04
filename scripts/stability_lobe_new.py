"""Compute no-control stability-lobe estimates for the face-milling plate env.

This script is for environment/model analysis only. It does not use RL,
trained policies, rewards, PPO, SAC, or feedback control. It sweeps fixed
physical machining conditions and estimates the largest stable axial depth of
cut by time-domain simulation of ``CustomODEPlate-v0``.

Stability convention
--------------------
The classification follows the environment's own pass logic:

    stable   -> the pass reaches the plant's pass-completion condition
    unstable -> the plant terminates by physical displacement instability or
                invalid state before pass completion

No reward is used for classification.

Outputs / modes
---------------
1) 2D lobe:
       ap_stable = f(spindle speed) at one milling line y=a

2) 3D pass-line surface:
       ap_stable = f(spindle speed, milling line y)

   For every selected line y=a, the cutter starts from the same x-start
   position and moves in the real pass direction x=L1 -> x=0. This is the
   required path-line-dependent stability map, not a moving x-start map.

3) Stochastic pass-line lobe:
       several 2D curves, one per selected milling line y=a, with MC mean/std
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from custom_rl import register_envs
from custom_rl.eval.monte_carlo import MC_BAND_STD_MULT
from custom_rl.plants.plate import omega_to_rpm, rpm_to_omega

ENV_ID = "CustomODEPlate-v0"

# Search defaults. Model/material/cutting coefficients are NOT overridden by
# default; the script uses current project/plant defaults unless the user
# explicitly supplies CLI overrides.
DEFAULT_RPM_MIN = 400.0
DEFAULT_RPM_MAX = 4000.0
DEFAULT_AP_MIN_MM = 0.0
DEFAULT_AP_MAX_MM = None
DEFAULT_DT = 1.0e-3
DEFAULT_MAX_SIM_STEPS = 500_000
DEFAULT_Y_POINTS = 5
PASS_TIME_MARGIN = 1.05

PASS_COMPLETED_REASONS = {"pass_completed_90percent", "pass_completed"}
UNSTABLE_REASONS = {
    "excessive_sensor_displacement",
    "excessive_sensor_velocity",  # backward/future compatibility
    "invalid_state",
}


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


def set_tool_start_x(plant: Any, x_start_m: float | None) -> float:
    """Set the pass start x_c(0), keeping the x=L1 -> x=0 pass direction."""
    if x_start_m is None:
        return float(getattr(plant, "L1", 0.0) - getattr(plant, "x0_cutter", 0.0))

    L1 = float(plant.L1)
    x_start_m = float(np.clip(float(x_start_m), 0.0, L1))
    plant.x0_cutter = float(L1 - x_start_m)
    if hasattr(plant, "_sync_face_milling_geometry"):
        plant._sync_face_milling_geometry()
    return x_start_m


def resolve_ap_search_bounds(args: argparse.Namespace, plant: Any) -> tuple[float, float]:
    """Return safe axial-depth search bounds inside the plant action bounds."""
    plant_ap_min, plant_ap_max = _depth_bounds_from_plant(plant)
    ap_min = plant_ap_min if args.ap_min is None else float(args.ap_min)
    ap_max = plant_ap_max if args.ap_max is None else float(args.ap_max)

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
        raise ValueError(f"No valid ap search interval after clipping: {ap_min:g}..{ap_max:g} mm.")
    return ap_min, ap_max


def _reset_options_for_line_y(plant: Any, line_y_m: float) -> dict[str, float]:
    """Build reset options that force the selected milling line y=a."""
    return {"y0": float(np.clip(line_y_m, 0.0, float(plant.L2)))}


def _pass_end_x(plant: Any) -> float:
    return float(getattr(plant, "x_pass_end_m", getattr(plant, "x_pass_end_tol", 0.1 * plant.L1)))


def _feed_rate_m_s(plant: Any, omega_rad_s: float) -> float:
    ft_mm = float(getattr(plant, "feed_per_tooth_mm", getattr(plant, "cf", 0.0)))
    n_teeth = int(getattr(plant, "N", 1))
    omega_rad_s = max(float(omega_rad_s), 1e-12)
    return (ft_mm / 1000.0) * n_teeth * omega_rad_s / (2.0 * np.pi)


def simulation_steps_for_trial(
    plant: Any,
    *,
    omega_rad_s: float,
    x_start_m: float,
    dt: float,
    n_substeps: int,
    max_sim_steps: int,
) -> tuple[int, int]:
    """Return (usable_steps, required_steps) for the remaining pass."""
    step_dt = max(float(dt) * int(n_substeps), 1e-12)
    remaining = max(float(x_start_m) - _pass_end_x(plant), 0.0)
    feed_rate = _feed_rate_m_s(plant, omega_rad_s)

    if remaining <= 0.0:
        required = 1
    else:
        required = int(np.ceil(PASS_TIME_MARGIN * remaining / max(feed_rate, 1e-12) / step_dt)) + 5

    usable = max(1, min(int(required), int(max_sim_steps)))
    return usable, int(required)


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
    x_start_m: float
    x_end_m: float
    remaining_path_m: float
    y_line_m: float
    max_abs_w_m: float
    rms_w_m: float
    terminated: bool
    truncated: bool
    pass_completed: bool
    termination_reason: str | None


def _update_w_metrics(info: dict[str, Any], max_abs_w: float, sum_sq: float, count: int) -> tuple[float, float, int]:
    if "w_sensor" not in info:
        return max_abs_w, sum_sq, count
    w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.all(np.isfinite(w)):
        return max_abs_w, sum_sq, count
    max_abs_w = max(float(max_abs_w), float(np.max(np.abs(w))))
    sum_sq += float(np.mean(w**2))
    count += 1
    return max_abs_w, sum_sq, count


def run_fixed_action_trial(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ap_mm: float,
    max_steps: int,
    seed: int,
    x_start_m: float | None = None,
    line_y_m: float | None = None,
) -> TrialMetrics:
    """Simulate one fixed [omega, ap] pair with no controller on one line y=a."""
    plant = env.unwrapped.plant

    x_selected = set_tool_start_x(plant, x_start_m)
    y_selected = set_milling_line_y(plant, line_y_m)
    reset_options = _reset_options_for_line_y(plant, y_selected)

    if bool(getattr(plant, "control_ae", False)):
        ae = float(getattr(plant, "ae_default", getattr(plant, "ae_min", 0.0)))
        u_phys = np.array([omega_rad_s, ap_mm, ae], dtype=np.float64)
    else:
        u_phys = np.array([omega_rad_s, ap_mm], dtype=np.float64)
    action = physical_to_normalized_action(u_phys, plant)

    # This reset is the critical line-selection step.  Passing y0 forces the
    # plant reset to keep this trial on exactly the selected line y=a.
    env.reset(seed=seed, options=reset_options)

    terminated = False
    truncated = False
    termination_reason: str | None = None
    pass_completed = False
    steps = 0
    final_cutter_x = float(x_selected)
    max_abs_w = 0.0
    sum_sq_w = 0.0
    count_w = 0
    safety_limit = max(1, int(max_steps))

    while steps < safety_limit and not (terminated or truncated):
        _, _reward, terminated, truncated, info = env.step(action)
        termination_reason = info.get("termination_reason", termination_reason)
        if info.get("pass_completed") or termination_reason in PASS_COMPLETED_REASONS:
            pass_completed = True
        final_cutter_x = float(info.get("cutter_x", final_cutter_x))
        max_abs_w, sum_sq_w, count_w = _update_w_metrics(info, max_abs_w, sum_sq_w, count_w)
        steps += 1

    if pass_completed:
        stable = True
        if termination_reason is None:
            termination_reason = "pass_completed_90percent"
    elif termination_reason in UNSTABLE_REASONS:
        stable = False
    else:
        stable = False
        if termination_reason is None:
            termination_reason = "safety_max_steps_reached"

    x_end_threshold = _pass_end_x(plant)
    remaining_path = max(float(final_cutter_x) - x_end_threshold, 0.0)
    rms_w = float(np.sqrt(sum_sq_w / count_w)) if count_w > 0 else 0.0

    return TrialMetrics(
        rpm=float(omega_to_rpm(omega_rad_s)),
        omega_rad_s=float(omega_rad_s),
        ap_mm=float(ap_mm),
        stable=bool(stable),
        steps=int(steps),
        sim_time_s=float(steps * env.unwrapped._step_dt),
        x_start_m=float(x_selected),
        x_end_m=float(final_cutter_x),
        remaining_path_m=float(remaining_path),
        y_line_m=float(y_selected),
        max_abs_w_m=float(max_abs_w),
        rms_w_m=float(rms_w),
        terminated=bool(terminated),
        truncated=bool(truncated),
        pass_completed=bool(pass_completed),
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
    x_start_m: float | None = None,
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
        x_start_m=x_start_m,
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
        x_start_m=x_start_m,
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
            seed=seed + int(round(ap_mid * 10_000.0)),
            x_start_m=x_start_m,
            line_y_m=line_y_m,
        )

        if trial_mid.stable:
            ap_lo = ap_mid
            best_stable = trial_mid
        else:
            ap_hi = ap_mid
            unstable_probe = trial_mid

    return ap_lo, best_stable, unstable_probe


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
        "x_start_m": float(stable_trial.x_start_m),
        "x_position_m": float(stable_trial.x_start_m),  # backward-compatible alias
        "x_end_m": float(stable_trial.x_end_m),
        "remaining_path_m": float(stable_trial.remaining_path_m),
        "y_line_m": float(stable_trial.y_line_m),
        "y_position_m": float(stable_trial.y_line_m),  # plot/CSV alias
        "line_y_m": float(stable_trial.y_line_m),
        "ap_stable_mm": float(ap_stable),
        "max_abs_w_m": float(stable_trial.max_abs_w_m),
        "rms_w_m": float(stable_trial.rms_w_m),
        "sim_steps": int(stable_trial.steps),
        "sim_time_s": float(stable_trial.sim_time_s),
        "termination_reason": stable_trial.termination_reason,
        "stable_at_boundary": bool(stable_trial.stable),
        "pass_completed_at_boundary": bool(stable_trial.pass_completed),
        "unstable_ap_mm": None if unstable_trial is None else float(unstable_trial.ap_mm),
        "unstable_max_abs_w_m": None if unstable_trial is None else float(unstable_trial.max_abs_w_m),
        "unstable_reason": None if unstable_trial is None else unstable_trial.termination_reason,
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
        "max_sim_steps": int(args.max_sim_steps),
        "pass_time_margin": float(PASS_TIME_MARGIN),
        "rpm_min": float(args.rpm_min),
        "rpm_max": float(args.rpm_max),
        "rpm_points": int(args.rpm_points),
        "ap_min_mm": float(ap_min),
        "ap_max_mm": float(ap_max),
        "ap_tol_mm": float(args.ap_tol),
        "binary_iters": int(args.binary_iters),
        "w_limit_m": float(plant.w_limit),
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
        "Ksp": None if getattr(plant, "Ksp", None) is None else float(getattr(plant, "Ksp")),
        "VB": float(getattr(plant, "VB", np.nan)),
    }


# ---------------------------------------------------------------------------
# Axis resolution
# ---------------------------------------------------------------------------


def _resolve_single_x_start(args: argparse.Namespace, plant: Any) -> float:
    if args.x_single is not None:
        return float(np.clip(args.x_single, 0.0, float(plant.L1)))
    return float(plant.L1)


def _resolve_y_axis(args: argparse.Namespace, plant: Any) -> list[float]:
    """Resolve milling-line values y=a for 3D/stochastic stability maps."""
    if args.y_position:
        return [float(np.clip(v, 0.0, float(plant.L2))) for v in args.y_position]

    y_min = args.y_min
    y_max = args.y_max
    if y_min is None:
        y_min = 0.05 * float(plant.L2)
    if y_max is None:
        y_max = 0.95 * float(plant.L2)

    y_min = float(np.clip(y_min, 0.0, float(plant.L2)))
    y_max = float(np.clip(y_max, 0.0, float(plant.L2)))
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    if y_min <= 0.0 or y_max >= float(plant.L2):
        print(
            "Warning: y-line values include plate edges. For line y=a, usually choose "
            "0 < a < L2 to avoid edge-contact interpretations."
        )

    return [float(v) for v in np.linspace(y_min, y_max, int(args.y_points))]


def _steps_for_point(args: argparse.Namespace, plant: Any, omega: float, x_start: float) -> int:
    max_steps, required_steps = simulation_steps_for_trial(
        plant,
        omega_rad_s=omega,
        x_start_m=x_start,
        dt=args.dt,
        n_substeps=args.n_substeps,
        max_sim_steps=args.max_sim_steps,
    )
    if required_steps > max_steps:
        print(
            f"Warning: required pass steps ({required_steps}) exceed max_sim_steps "
            f"({max_steps}); this point may be classified as safety-limited/unstable."
        )
    return max_steps


# ---------------------------------------------------------------------------
# Deterministic 2D lobe
# ---------------------------------------------------------------------------


def compute_lobe(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Sweep spindle speed and estimate ap_stable at one fixed y-line."""
    register_envs()
    env = _make_env(args)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    line_y = set_milling_line_y(plant, args.line_y)
    x_start = _resolve_single_x_start(args, plant)

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    rows: list[dict] = []

    print(
        "No-control face-milling stability lobe sweep\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm ({args.rpm_points} points)\n"
        f"  ap search     : {ap_min:.4g} - {ap_max:.4g} mm\n"
        f"  x start       : {x_start:.4g} m\n"
        f"  y line        : {line_y:.4g} m\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = _steps_for_point(args, plant, omega, x_start)

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
            x_start_m=x_start,
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
            f"(max|w|={stable_trial.max_abs_w_m:.3e} m, reason={stable_trial.termination_reason})"
        )

    metadata = _metadata_common(
        args,
        plant,
        method="time_domain_no_control_binary_search_face_milling_2d_fixed_y_line",
        note=(
            "Practical time-domain estimate of ap_stable=f(rpm) on one milling line y=a. "
            "Stability is pass completion versus displacement/invalid-state termination."
        ),
    )
    metadata["x_start_m"] = float(x_start)
    metadata["y_line_m"] = float(line_y)

    env.close()
    return rows, metadata


# ---------------------------------------------------------------------------
# Deterministic 3D surface over milling line y
# ---------------------------------------------------------------------------


def sweep_rpm_boundary_at_y(
    env: gym.Env,
    plant: Any,
    rpm_values: np.ndarray,
    args: argparse.Namespace,
    *,
    line_y_m: float,
    x_start_m: float,
    seed_offset: int,
) -> list[dict]:
    """Estimate ap_stable over rpm for one fixed milling line y=a."""
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rows: list[dict] = []

    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = _steps_for_point(args, plant, omega, x_start_m)

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
            x_start_m=x_start_m,
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
            f"(max|w|={stable_trial.max_abs_w_m:.3e} m, reason={stable_trial.termination_reason})"
        )

    return rows


def compute_lobe_surface_3d(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Sweep milling line y and rpm for ap_stable=f(rpm,y)."""
    register_envs()
    env = _make_env(args)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    y_values = _resolve_y_axis(args, plant)
    x_start = _resolve_single_x_start(args, plant)
    rows: list[dict] = []

    print(
        "No-control 3D milling-line face-milling stability surface sweep\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm ({args.rpm_points} points)\n"
        f"  y line values : {y_values}\n"
        f"  fixed x start : {x_start:.4g} m\n"
        f"  ap search     : {ap_min:.4g} - {ap_max:.4g} mm\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for y_index, line_y in enumerate(y_values):
        print(f"[y {y_index + 1}/{len(y_values)}] milling line y={line_y:.4g} m")
        rows.extend(
            sweep_rpm_boundary_at_y(
                env,
                plant,
                rpm_values,
                args,
                line_y_m=line_y,
                x_start_m=x_start,
                seed_offset=y_index * 1000,
            )
        )

    metadata = _metadata_common(
        args,
        plant,
        method="time_domain_no_control_binary_search_face_milling_3d_surface_y_line",
        note=(
            "Practical estimate of ap_stable=f(rpm,y_line). For each milling line y=a, "
            "the pass starts from the same x_start and moves toward x=0. Stability is "
            "based on pass-completion versus displacement/invalid-state termination."
        ),
    )
    metadata["y_line_values_m"] = y_values
    metadata["y_positions_m"] = y_values
    metadata["x_start_m"] = float(x_start)
    metadata["L1_m"] = float(plant.L1)
    metadata["L2_m"] = float(plant.L2)

    env.close()
    return rows, metadata


# ---------------------------------------------------------------------------
# Stochastic multi-y-line lobe
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
    n_mc: int,
    x_start_m: float,
    line_y_m: float,
) -> tuple[float, float, list[float], TrialMetrics, TrialMetrics | None]:
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
            x_start_m=x_start_m,
            line_y_m=line_y_m,
        )
        ap_samples.append(float(ap_stable))
        last_stable = stable_trial
        if unstable_trial is not None:
            last_unstable = unstable_trial

    ap_arr = np.asarray(ap_samples, dtype=np.float64)
    if last_stable is None:
        raise RuntimeError("No MC trials were executed.")
    return (
        float(np.mean(ap_arr)),
        float(np.std(ap_arr)) if ap_arr.size > 1 else 0.0,
        ap_samples,
        last_stable,
        last_unstable,
    )


def sweep_rpm_boundary_stochastic_at_y(
    env: gym.Env,
    plant: Any,
    rpm_values: np.ndarray,
    args: argparse.Namespace,
    *,
    line_y_m: float,
    x_start_m: float,
    seed_offset: int,
) -> list[dict]:
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rows: list[dict] = []

    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = _steps_for_point(args, plant, omega, x_start_m)

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
            n_mc=args.n_mc,
            x_start_m=x_start_m,
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
        print(f"      -> stable boundary ap ~ {ap_mean:.4g} +/- {ap_std:.4g} mm")

    return rows


def compute_stochastic_lobe_multi_y(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Stochastic lobe: several rpm-ap curves, one per line y=a."""
    register_envs()
    env = _make_env(args, stochastic=True)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    y_values = _resolve_y_axis(args, plant)
    x_start = _resolve_single_x_start(args, plant)
    rows: list[dict] = []

    print(
        "Stochastic no-control milling-line stability lobe\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm ({args.rpm_points} points)\n"
        f"  y line values : {y_values}\n"
        f"  fixed x start : {x_start:.4g} m\n"
        f"  n_mc          : {args.n_mc}\n"
        f"  dynamics unc. : {plant.dynamics_uncertainty_std}\n"
        f"  ap search     : {ap_min:.4g} - {ap_max:.4g} mm\n"
    )

    for y_index, line_y in enumerate(y_values):
        print(f"[y {y_index + 1}/{len(y_values)}] milling line y={line_y:.4g} m")
        rows.extend(
            sweep_rpm_boundary_stochastic_at_y(
                env,
                plant,
                rpm_values,
                args,
                line_y_m=line_y,
                x_start_m=x_start,
                seed_offset=y_index * 100_000,
            )
        )

    metadata = _metadata_common(
        args,
        plant,
        method="time_domain_no_control_binary_search_face_milling_stochastic_mc_multi_y_line",
        note=(
            "Stochastic estimate of the no-control stability boundary. Each curve is one "
            "milling line y=a; each (rpm,y_line) uses n_mc independent binary searches."
        ),
    )
    metadata["y_line_values_m"] = y_values
    metadata["y_positions_m"] = y_values
    metadata["x_start_m"] = float(x_start)
    metadata["n_mc"] = int(args.n_mc)
    metadata["dynamics_uncertainty_std"] = float(args.dynamics_uncertainty_std)
    metadata["L1_m"] = float(plant.L1)
    metadata["L2_m"] = float(plant.L2)

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
    ax.set_title("No-control face-milling stability lobe on fixed milling line")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlim(float(np.min(rpm)), float(np.max(rpm)))
    ax.set_ylim(0.0, ap_max * 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lobe_surface_3d(rows: list[dict], metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    rpm_axis = np.array(sorted({row["rpm"] for row in rows}), dtype=np.float64)
    y_axis = np.array(sorted({row["y_line_m"] for row in rows}), dtype=np.float64)
    ap_grid = np.full((y_axis.size, rpm_axis.size), np.nan, dtype=np.float64)
    rpm_index = {rpm: idx for idx, rpm in enumerate(rpm_axis)}
    y_index = {y: idx for idx, y in enumerate(y_axis)}
    for row in rows:
        ap_grid[y_index[row["y_line_m"]], rpm_index[row["rpm"]]] = row["ap_stable_mm"]

    rpm_mesh, y_mesh = np.meshgrid(rpm_axis, y_axis)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        rpm_mesh,
        y_mesh,
        ap_grid,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.2,
        antialiased=True,
    )
    ax.plot_wireframe(rpm_mesh, y_mesh, ap_grid, color="black", linewidth=0.4, alpha=0.35)
    for line_y in y_axis:
        ap_line = ap_grid[y_index[line_y], :]
        ax.plot(rpm_axis, np.full_like(rpm_axis, line_y), ap_line, linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Milling line y (m)")
    ax.set_zlabel("Stable axial depth ap (mm)")
    ax.set_title("No-control 3D face-milling stability surface over milling line y")
    fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1, label="ap stable (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stochastic_lobe_multi_y(rows: list[dict], metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    y_values = sorted({row["y_line_m"] for row in rows})
    ap_max = float(metadata["ap_max_mm"])
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for idx, line_y in enumerate(y_values):
        y_rows = sorted((row for row in rows if row["y_line_m"] == line_y), key=lambda row: row["rpm"])
        rpm = np.array([row["rpm"] for row in y_rows], dtype=np.float64)
        ap_mean = np.array([row["ap_stable_mean_mm"] for row in y_rows], dtype=np.float64)
        ap_std = np.array([row["ap_stable_std_mm"] for row in y_rows], dtype=np.float64)
        band = MC_BAND_STD_MULT * ap_std
        color = cmap(idx % 10)
        ax.plot(rpm, ap_mean, "-", color=color, linewidth=2.0, marker="o", markersize=4, label=f"y = {line_y:.3f} m")
        ax.fill_between(rpm, np.maximum(ap_mean - band, 0.0), ap_mean + band, color=color, alpha=0.22)

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut ap (mm)")
    ax.set_title(f"Stochastic no-control face-milling stability lobes by y-line (mean +/- {MC_BAND_STD_MULT:.0f}σ)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    all_rpm = [row["rpm"] for row in rows]
    ax.set_xlim(float(np.min(all_rpm)), float(np.max(all_rpm)))
    ax.set_ylim(0.0, ap_max * 1.05)
    ax.legend(loc="best", fontsize=9)
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
        # Keep ODEControlEnv's internal time-limit from truncating before the
        # stability script's own safety limit/pass-completion logic.
        "max_episode_steps": int(args.max_episode_steps if args.max_episode_steps > 0 else args.max_sim_steps),
    }

    # Optional model overrides. Defaults are None, so current project/plant
    # parameters are used unless the user explicitly supplies a value.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute no-control stability-lobe estimates for CustomODEPlate-v0.")
    parser.add_argument("--rpm-min", type=float, default=DEFAULT_RPM_MIN)
    parser.add_argument("--rpm-max", type=float, default=DEFAULT_RPM_MAX)
    parser.add_argument("--rpm-points", type=int, default=12)
    parser.add_argument("--ap-min", type=float, default=DEFAULT_AP_MIN_MM)
    parser.add_argument("--ap-max", type=float, default=DEFAULT_AP_MAX_MM)
    parser.add_argument("--ap-tol", type=float, default=0.05, help="Binary-search tolerance [mm].")
    parser.add_argument("--binary-iters", type=int, default=10)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--n-substeps", type=int, default=1)
    parser.add_argument("--max-sim-steps", type=int, default=DEFAULT_MAX_SIM_STEPS, help="Safety fallback max steps per trial.")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/stability_lobe"))

    parser.add_argument("--surface-3d", action="store_true", help="Compute ap=f(rpm,y_line) surface.")
    parser.add_argument("--stochastic-lobe", action="store_true", help="Compute MC rpm-ap curves for multiple y-lines.")
    parser.add_argument("--x-single", type=float, default=None, help="Fixed pass start x-position [m]. Default: L1/free side.")
    parser.add_argument("--line-y", type=float, default=None, help="Fixed milling line y [m] for 2D lobe. Default: plant.y_cutter.")

    parser.add_argument("--y-min", type=float, default=None, help="Minimum milling line y [m] for 3D/stochastic mode.")
    parser.add_argument("--y-max", type=float, default=None, help="Maximum milling line y [m] for 3D/stochastic mode.")
    parser.add_argument("--y-points", type=int, default=None)
    parser.add_argument("--y-position", type=float, nargs="*", default=None, help="Explicit milling line y values [m].")

    # Deprecated aliases kept so old commands do not fail. They now map to y-axis
    # controls because the third stability-lobe parameter is the milling line y.
    parser.add_argument("--x-min", dest="deprecated_x_min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--x-max", dest="deprecated_x_max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--x-points", dest="deprecated_x_points", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--x-position", dest="deprecated_x_position", type=float, nargs="*", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--n-mc", type=int, default=5)
    parser.add_argument("--dynamics-uncertainty-std", type=float, default=0.0)

    # Optional model overrides. Leave as None to use current project defaults.
    parser.add_argument("--E", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--rho-type", dest="rho_type", default=None)
    parser.add_argument("--Kt", type=float, default=None)
    parser.add_argument("--Kr", type=float, default=None)
    parser.add_argument("--Ka", type=float, default=None)
    parser.add_argument("--Kte", type=float, default=None)
    parser.add_argument("--Kre", type=float, default=None)
    parser.add_argument("--Kae", type=float, default=None)
    pd_group = parser.add_mutually_exclusive_group()
    pd_group.add_argument("--use-process-damping", dest="use_process_damping", action="store_true")
    pd_group.add_argument("--no-process-damping", dest="use_process_damping", action="store_false")
    parser.set_defaults(use_process_damping=None)
    parser.add_argument("--Ksp", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--VB", type=float, default=None)
    parser.add_argument("--lambda-L-deg", dest="lambda_L_deg", type=float, default=None)

    args = parser.parse_args()

    # Apply deprecated x-axis aliases as y-axis inputs.
    if args.y_min is None and args.deprecated_x_min is not None:
        args.y_min = args.deprecated_x_min
        print("Warning: --x-min is deprecated for 3D/stochastic mode; interpreting it as --y-min.")
    if args.y_max is None and args.deprecated_x_max is not None:
        args.y_max = args.deprecated_x_max
        print("Warning: --x-max is deprecated for 3D/stochastic mode; interpreting it as --y-max.")
    if args.y_points is None and args.deprecated_x_points is not None:
        args.y_points = args.deprecated_x_points
        print("Warning: --x-points is deprecated for 3D/stochastic mode; interpreting it as --y-points.")
    if args.y_position is None and args.deprecated_x_position is not None:
        args.y_position = args.deprecated_x_position
        print("Warning: --x-position is deprecated for 3D/stochastic mode; interpreting it as --y-position.")
    if args.y_points is None:
        args.y_points = DEFAULT_Y_POINTS

    if args.surface_3d and args.stochastic_lobe:
        raise SystemExit("Choose only one of --surface-3d or --stochastic-lobe.")
    if args.rpm_points <= 0:
        raise SystemExit("--rpm-points must be positive.")
    if args.y_points <= 0:
        raise SystemExit("--y-points must be positive.")
    if args.n_mc <= 0:
        raise SystemExit("--n-mc must be positive.")
    if args.dt <= 0.0:
        raise SystemExit("--dt must be positive.")
    if args.n_substeps <= 0:
        raise SystemExit("--n-substeps must be positive.")
    if args.max_sim_steps <= 0:
        raise SystemExit("--max-sim-steps must be positive.")
    if args.use_process_damping is True and args.Ksp is None:
        raise SystemExit("--use-process-damping requires --Ksp for the current plant wiring.")
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.stochastic_lobe:
        rows, metadata = compute_stochastic_lobe_multi_y(args)
        csv_path = out_dir / "stability_lobe_stochastic_yline.csv"
        meta_path = out_dir / "stability_lobe_stochastic_yline_metadata.json"
        plot_path = out_dir / "stability_lobe_stochastic_yline.png"
    elif args.surface_3d:
        rows, metadata = compute_lobe_surface_3d(args)
        csv_path = out_dir / "stability_lobe_surface_yline.csv"
        meta_path = out_dir / "stability_lobe_surface_yline_metadata.json"
        plot_path = out_dir / "stability_lobe_surface_yline.png"
    else:
        rows, metadata = compute_lobe(args)
        csv_path = out_dir / "stability_lobe.csv"
        meta_path = out_dir / "stability_lobe_metadata.json"
        plot_path = out_dir / "stability_lobe.png"

    save_csv(rows, csv_path)
    save_metadata(metadata, meta_path)
    if args.stochastic_lobe:
        plot_stochastic_lobe_multi_y(rows, metadata, plot_path)
    elif args.surface_3d:
        plot_lobe_surface_3d(rows, metadata, plot_path)
    else:
        plot_lobe(rows, metadata, plot_path)

    print(f"\nSaved outputs to {out_dir.resolve()}")
    print(f"  plot     : {plot_path.resolve()}")
    print(f"  csv      : {csv_path.resolve()}")
    print(f"  metadata : {meta_path.resolve()}")


if __name__ == "__main__":
    main()
