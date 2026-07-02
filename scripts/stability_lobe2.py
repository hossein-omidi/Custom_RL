"""Compute and plot no-control stability-lobe estimates for the face-milling plate env.
This script is for environment/model analysis only. It does not use RL,
trained policies, rewards, PPO, SAC, or feedback control. It sweeps fixed
physical machining conditions and estimates the largest stable axial depth of
cut by time-domain simulation of `CustomODEPlate-v0`.

CRITICAL UPDATE:
Stability is now determined STRICTLY by the environment's termination() function.
- If termination_reason == "pass_completed_90percent": STABLE
- If termination_reason == "excessive_sensor_displacement": UNSTABLE
This removes the dependency on arbitrary time windows (RMS growth) and ensures 
the binary search stays exactly on the requested y-line by overriding randomization.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from custom_rl import register_envs
from custom_rl.eval.monte_carlo import MC_BAND_STD_MULT
from custom_rl.plants.plate import omega_to_rpm, rpm_to_omega

ENV_ID = "CustomODEPlate-v0"

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
    """Set the pass start position x_c(0) while keeping x=L1 -> x=0 direction."""
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
        ap_min = plant_ap_min
    if ap_max > plant_ap_max + 1e-12:
        ap_max = plant_ap_max

    if ap_max <= ap_min:
        raise ValueError(f"No valid ap search interval after clipping: {ap_min}..{ap_max} mm.")
    return ap_min, ap_max


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
    terminated: bool
    truncated: bool
    pass_completed: bool
    termination_reason: str | None


def run_fixed_action_trial(
    env: gym.Env,
    *,
    omega_rad_s: float,
    ap_mm: float,
    max_steps: int,
    seed: int,
    x_position_m: float | None = None,
    line_y_m: float | None = None,
) -> TrialMetrics:
    """Simulate one fixed [omega, ap] pair with no controller.
    
    Stability is determined STRICTLY by the environment's termination() function:
    - If termination_reason == "pass_completed_90percent": STABLE
    - If termination_reason == "excessive_sensor_displacement": UNSTABLE
    """
    plant = env.unwrapped.plant
    x_selected = set_tool_start_x(plant, x_position_m)
    y_selected = set_milling_line_y(plant, line_y_m)
    
    # CRITICAL FIX: Force the environment to use this exact y-line.
    # By passing "y0" in options, the plate's reset() function will execute:
    #   if "y0" in options: self.set_milling_line_y(float(options["y0"]))
    # This completely bypasses the `randomize_y0` logic.
    reset_options = {"y0": float(np.clip(y_selected, 0.0, float(plant.L2)))}

    if bool(getattr(plant, "control_ae", False)):
        ae = float(getattr(plant, "ae_default", getattr(plant, "ae_min", 0.0)))
        u_phys = np.array([omega_rad_s, ap_mm, ae], dtype=np.float64)
    else:
        u_phys = np.array([omega_rad_s, ap_mm], dtype=np.float64)
    action = physical_to_normalized_action(u_phys, plant)

    env.reset(seed=seed, options=reset_options)

    terminated = False
    truncated = False
    termination_reason: str | None = None
    steps = 0
    final_cutter_x = float(x_selected)
    
    # Safety fallback to prevent infinite loops, but we expect natural termination.
    safety_limit = max_steps if max_steps > 0 else 1_000_000

    while steps < safety_limit and not (terminated or truncated):
        _, _reward, terminated, truncated, info = env.step(action)
        termination_reason = info.get("termination_reason")
        final_cutter_x = float(info.get("cutter_x", final_cutter_x))
        steps += 1

    # --- THE CORE LOGIC: Time-Independent Stability Classification ---
    if termination_reason == "pass_completed_90percent":
        stable = True
    elif termination_reason in {"excessive_sensor_displacement", "invalid_state"}:
        stable = False
    else:
        # Fallback if it hit the safety limit without terminating
        stable = False
        if termination_reason is None:
            termination_reason = "safety_max_steps_reached"

    # Calculate remaining path for diagnostics
    x_end_threshold = 0.1 * plant.L1
    remaining_path = max(final_cutter_x - x_end_threshold, 0.0)

    return TrialMetrics(
        rpm=float(omega_to_rpm(omega_rad_s)),
        omega_rad_s=float(omega_rad_s),
        ap_mm=float(ap_mm),
        stable=bool(stable),
        steps=int(steps),
        sim_time_s=float(steps * env.unwrapped._step_dt),
        x_position_m=float(x_selected),
        x_end_m=float(final_cutter_x),
        remaining_path_m=float(remaining_path),
        y_line_m=float(y_selected),
        max_abs_w_m=float(plant.w_limit) if not stable else 0.0,
        terminated=bool(terminated),
        truncated=bool(truncated),
        pass_completed=(termination_reason == "pass_completed_90percent"),
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
    x_position_m: float | None = None,
    line_y_m: float | None = None,
) -> tuple[float, TrialMetrics, TrialMetrics | None]:
    """Binary search for the largest stable axial depth of cut ap."""
    ap_lo = float(ap_min)
    ap_hi = float(ap_max)
    trial_lo = run_fixed_action_trial(
        env, omega_rad_s=omega_rad_s, ap_mm=ap_lo, max_steps=max_steps,
        seed=seed, x_position_m=x_position_m, line_y_m=line_y_m,
    )

    if not trial_lo.stable:
        return ap_lo, trial_lo, trial_lo

    trial_hi = run_fixed_action_trial(
        env, omega_rad_s=omega_rad_s, ap_mm=ap_hi, max_steps=max_steps,
        seed=seed + 1, x_position_m=x_position_m, line_y_m=line_y_m,
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
            env, omega_rad_s=omega_rad_s, ap_mm=ap_mid, max_steps=max_steps,
            seed=seed + int(round(ap_mid * 10000.0)),
            x_position_m=x_position_m, line_y_m=line_y_m,
        )

        if trial_mid.stable:
            ap_lo = ap_mid
            best_stable = trial_mid
        else:
            ap_hi = ap_mid
            unstable_probe = trial_mid

    return ap_lo, best_stable, unstable_probe


def simulation_steps_for_rpm(
    max_sim_steps: int,
) -> int:
    """Choose a practical time-domain step budget for one lobe evaluation."""
    # We no longer limit by sim_seconds. We just use max_sim_steps as a safety fallback.
    return int(max_sim_steps)


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
        "x_position_m": float(stable_trial.x_position_m),
        "x_end_m": float(stable_trial.x_end_m),
        "remaining_path_m": float(stable_trial.remaining_path_m),
        "y_line_m": float(stable_trial.y_line_m),
        "ap_stable_mm": float(ap_stable),
        "max_abs_w_m": float(stable_trial.max_abs_w_m),
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
        "max_sim_steps": int(args.max_sim_steps),
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
    }


# ---------------------------------------------------------------------------
# Deterministic 2D lobe
# ---------------------------------------------------------------------------
def _resolve_single_x_position(args: argparse.Namespace, plant: Any) -> float:
    if args.x_single is not None:
        return float(np.clip(args.x_single, 0.0, float(plant.L1)))
    return float(plant.L1)

def compute_lobe(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Sweep spindle speed and estimate ap_stable at one x-position/y-line."""
    register_envs()
    env = _make_env(args)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    line_y = set_milling_line_y(plant, args.line_y)
    x_position = _resolve_single_x_position(args, plant)

    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    rows: list[dict] = []

    print(
        "No-control face-milling stability lobe sweep\n"
        f"  rpm range     : {args.rpm_min:.0f} - {args.rpm_max:.0f} rpm ({args.rpm_points} points)\n"
        f"  ap search     : {ap_min:.4g} - {ap_max:.4g} mm\n"
        f"  x start       : {x_position:.4g} m (free side L1={plant.L1:.4g}, clamped x=0)\n"
        f"  y line        : {line_y:.4g} m\n"
        f"  dt            : {args.dt} s, substeps={args.n_substeps}\n"
        f"  w_limit       : {plant.w_limit} m\n"
    )

    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = simulation_steps_for_rpm(max_sim_steps=args.max_sim_steps)

        print(f"[{index + 1}/{len(rpm_values)}] rpm={rpm_clipped:.0f} (omega={omega:.2f} rad/s)")

        ap_stable, stable_trial, unstable_trial = find_max_stable_ap(
            env, omega_rad_s=omega, ap_min=ap_min, ap_max=ap_max, max_steps=max_steps,
            seed=args.seed + index, ap_tol_mm=args.ap_tol, max_binary_iters=args.binary_iters,
            x_position_m=x_position, line_y_m=line_y,
        )
        rows.append(_trial_row(rpm=rpm_clipped, omega=omega, ap_stable=ap_stable,
                               stable_trial=stable_trial, unstable_trial=unstable_trial))
        print(f"    -> stable boundary ap ~ {ap_stable:.4g} mm")

    metadata = _metadata_common(args, plant, method="time_domain_no_control_binary_search_face_milling_full_moving_pass_2d",
                                note="Practical time-domain estimate. Stability defined by environment termination.")
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
    x_min = args.x_min if args.x_min is not None else 0.10 * float(plant.L1)
    x_max = args.x_max if args.x_max is not None else float(plant.L1)
    x_min = float(np.clip(x_min, 0.0, float(plant.L1)))
    x_max = float(np.clip(x_max, 0.0, float(plant.L1)))
    if x_max < x_min: x_min, x_max = x_max, x_min
    return [float(v) for v in np.linspace(x_min, x_max, int(args.x_points))]

def sweep_rpm_boundary_at_x(
    env: gym.Env, plant: Any, rpm_values: np.ndarray, args: argparse.Namespace,
    *, x_position_m: float, line_y_m: float, seed_offset: int,
) -> list[dict]:
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rows: list[dict] = []
    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = simulation_steps_for_rpm(max_sim_steps=args.max_sim_steps)

        ap_stable, stable_trial, unstable_trial = find_max_stable_ap(
            env, omega_rad_s=omega, ap_min=ap_min, ap_max=ap_max, max_steps=max_steps,
            seed=args.seed + seed_offset + index, ap_tol_mm=args.ap_tol, max_binary_iters=args.binary_iters,
            x_position_m=x_position_m, line_y_m=line_y_m,
        )
        rows.append(_trial_row(rpm=rpm_clipped, omega=omega, ap_stable=ap_stable,
                               stable_trial=stable_trial, unstable_trial=unstable_trial))
    return rows

def compute_lobe_surface_3d(args: argparse.Namespace) -> tuple[list[dict], dict]:
    register_envs()
    env = _make_env(args)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    x_values = _resolve_x_axis(args, plant)
    line_y = set_milling_line_y(plant, args.line_y)
    rows: list[dict] = []

    for x_index, x_position in enumerate(x_values):
        rows.extend(sweep_rpm_boundary_at_x(env, plant, rpm_values, args,
                                            x_position_m=x_position, line_y_m=line_y, seed_offset=x_index * 1000))

    metadata = _metadata_common(args, plant, method="time_domain_no_control_binary_search_face_milling_3d_moving_pass_surface_start_x",
                                note="Stability defined by environment termination.")
    metadata["x_start_values_m"] = x_values
    metadata["y_line_m"] = float(line_y)
    metadata["L1_m"] = float(plant.L1)
    env.close()
    return rows, metadata


# ---------------------------------------------------------------------------
# Stochastic multi-x lobe
# ---------------------------------------------------------------------------
def find_max_stable_ap_mc(
    env: gym.Env, *, omega_rad_s: float, ap_min: float, ap_max: float,
    max_steps: int, seed: int, ap_tol_mm: float, max_binary_iters: int,
    n_mc: int, x_position_m: float, line_y_m: float,
) -> tuple[float, float, list[float], TrialMetrics, TrialMetrics | None]:
    ap_samples: list[float] = []
    last_stable: TrialMetrics | None = None
    last_unstable: TrialMetrics | None = None
    for mc in range(int(n_mc)):
        ap_stable, stable_trial, unstable_trial = find_max_stable_ap(
            env, omega_rad_s=omega_rad_s, ap_min=ap_min, ap_max=ap_max,
            max_steps=max_steps, seed=seed + mc * 10_000, ap_tol_mm=ap_tol_mm,
            max_binary_iters=max_binary_iters, x_position_m=x_position_m, line_y_m=line_y_m,
        )
        ap_samples.append(float(ap_stable))
        last_stable = stable_trial
        if unstable_trial is not None: last_unstable = unstable_trial

    ap_arr = np.asarray(ap_samples, dtype=np.float64)
    return (float(np.mean(ap_arr)), float(np.std(ap_arr)) if ap_arr.size > 1 else 0.0,
            ap_samples, last_stable, last_unstable)

def sweep_rpm_boundary_stochastic_at_x(
    env: gym.Env, plant: Any, rpm_values: np.ndarray, args: argparse.Namespace,
    *, x_position_m: float, line_y_m: float, seed_offset: int,
) -> list[dict]:
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rows: list[dict] = []
    for index, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_clipped = float(omega_to_rpm(omega))
        max_steps = simulation_steps_for_rpm(max_sim_steps=args.max_sim_steps)

        ap_mean, ap_std, ap_samples, stable_trial, unstable_trial = find_max_stable_ap_mc(
            env, omega_rad_s=omega, ap_min=ap_min, ap_max=ap_max, max_steps=max_steps,
            seed=args.seed + seed_offset + index, ap_tol_mm=args.ap_tol, max_binary_iters=args.binary_iters,
            n_mc=args.n_mc, x_position_m=x_position_m, line_y_m=line_y_m,
        )
        row = _trial_row(rpm=rpm_clipped, omega=omega, ap_stable=ap_mean,
                         stable_trial=stable_trial, unstable_trial=unstable_trial)
        row.update({"ap_stable_mean_mm": ap_mean, "ap_stable_std_mm": ap_std,
                    "ap_stable_samples_mm": json.dumps(ap_samples), "n_mc": int(args.n_mc)})
        rows.append(row)
    return rows

def compute_stochastic_lobe_multi_x(args: argparse.Namespace) -> tuple[list[dict], dict]:
    register_envs()
    env = _make_env(args, stochastic=True)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_search_bounds(args, plant)
    rpm_values = np.linspace(args.rpm_min, args.rpm_max, args.rpm_points)
    x_values = _resolve_x_axis(args, plant)
    line_y = set_milling_line_y(plant, args.line_y)
    rows: list[dict] = []

    for x_index, x_position in enumerate(x_values):
        rows.extend(sweep_rpm_boundary_stochastic_at_x(env, plant, rpm_values, args,
                                                       x_position_m=x_position, line_y_m=line_y, seed_offset=x_index * 100_000))

    metadata = _metadata_common(args, plant, method="time_domain_no_control_binary_search_face_milling_stochastic_mc_multi_x_start_moving_pass",
                                note="Stability defined by environment termination.")
    metadata["x_start_values_m"] = x_values
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
    if not rows: return
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
    ax.set_title("No-control face-milling stability lobe (Time-Independent)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlim(float(np.min(rpm)), float(np.max(rpm)))
    ax.set_ylim(0.0, ap_max * 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_lobe_surface_3d(rows: list[dict], metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
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
    surface = ax.plot_surface(rpm_mesh, x_mesh, ap_grid, cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.2, antialiased=True)
    ax.plot_wireframe(rpm_mesh, x_mesh, ap_grid, color="black", linewidth=0.4, alpha=0.35)
    for x_position in x_axis:
        ap_line = ap_grid[x_index[x_position], :]
        ax.plot(rpm_axis, np.full_like(rpm_axis, x_position), ap_line, linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Pass start x-position (m)")
    ax.set_zlabel("Stable axial depth ap (mm)")
    ax.set_title("No-control 3D moving-pass face-milling stability surface")
    fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1, label="ap stable (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_stochastic_lobe_multi_x(rows: list[dict], metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    x_values = sorted({row["x_position_m"] for row in rows})
    ap_max = float(metadata["ap_max_mm"])
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for idx, x_position in enumerate(x_values):
        x_rows = sorted((row for row in rows if row["x_position_m"] == x_position), key=lambda row: row["rpm"])
        rpm = np.array([row["rpm"] for row in x_rows], dtype=np.float64)
        ap_mean = np.array([row["ap_stable_mean_mm"] for row in x_rows], dtype=np.float64)
        ap_std = np.array([row["ap_stable_std_mm"] for row in x_rows], dtype=np.float64)
        band = MC_BAND_STD_MULT * ap_std
        color = cmap(idx % 10)
        ax.plot(rpm, ap_mean, "-", color=color, linewidth=2.0, marker="o", markersize=4, label=f"x_start = {x_position:.3f} m")
        ax.fill_between(rpm, np.maximum(ap_mean - band, 0.0), ap_mean + band, color=color, alpha=0.22)

    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut ap (mm)")
    ax.set_title(f"Stochastic no-control moving-pass face-milling stability lobe (mean +/- {MC_BAND_STD_MULT:.0f}σ)")
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
        "randomize_y0": False, # We control y explicitly via reset options
        "dynamics_uncertainty_std": float(args.dynamics_uncertainty_std if stochastic else 0.0),
    }
    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    for key in ("E", "nu", "rho", "rho_type", "Kt", "Kr", "Ka", "Kte", "Kre", "Kae",
                "use_process_damping", "Ksp", "mu", "VB", "lambda_L_deg", "ap_min", "ap_max"):
        value = getattr(args, key, None)
        if value is not None:
            env_kwargs[key] = value
    return gym.make(ENV_ID, **env_kwargs)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute no-control stability-lobe estimates.")
    parser.add_argument("--rpm-min", type=float, default=DEFAULT_RPM_MIN)
    parser.add_argument("--rpm-max", type=float, default=DEFAULT_RPM_MAX)
    parser.add_argument("--rpm-points", type=int, default=12)
    parser.add_argument("--ap-min", type=float, default=DEFAULT_AP_MIN_MM)
    parser.add_argument("--ap-max", type=float, default=DEFAULT_AP_MAX_MM)
    parser.add_argument("--ap-tol", type=float, default=0.02)
    parser.add_argument("--binary-iters", type=int, default=12)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--n-substeps", type=int, default=1)
    parser.add_argument("--max-sim-steps", type=int, default=100000, help="Safety fallback max steps.")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/stability_lobe"))

    parser.add_argument("--surface-3d", action="store_true")
    parser.add_argument("--stochastic-lobe", action="store_true")
    parser.add_argument("--x-single", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--x-points", type=int, default=5)
    parser.add_argument("--x-position", type=float, nargs="*", default=None)
    parser.add_argument("--line-y", type=float, default=None)
    parser.add_argument("--n-mc", type=int, default=5)
    parser.add_argument("--dynamics-uncertainty-std", type=float, default=0.0)

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
    return args

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.stochastic_lobe:
        rows, metadata = compute_stochastic_lobe_multi_x(args)
        csv_path, meta_path, plot_path = out_dir/"stability_lobe_stochastic.csv", out_dir/"stability_lobe_stochastic_metadata.json", out_dir/"stability_lobe_stochastic.png"
    elif args.surface_3d:
        rows, metadata = compute_lobe_surface_3d(args)
        csv_path, meta_path, plot_path = out_dir/"stability_lobe_surface.csv", out_dir/"stability_lobe_surface_metadata.json", out_dir/"stability_lobe_surface.png"
    else:
        rows, metadata = compute_lobe(args)
        csv_path, meta_path, plot_path = out_dir/"stability_lobe.csv", out_dir/"stability_lobe_metadata.json", out_dir/"stability_lobe.png"

    save_csv(rows, csv_path)
    save_metadata(metadata, meta_path)
    if args.stochastic_lobe: plot_stochastic_lobe_multi_x(rows, metadata, plot_path)
    elif args.surface_3d: plot_lobe_surface_3d(rows, metadata, plot_path)
    else: plot_lobe(rows, metadata, plot_path)
    print(f"\nSaved outputs to {out_dir.resolve()}")

if __name__ == "__main__":
    main()