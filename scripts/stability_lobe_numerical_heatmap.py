"""Numerical stability-lobe heat map for the CustomODEPlate face-milling case study.

This script estimates the practical no-control stability map of the face-milling
cantilever-plate environment by direct time-domain simulation of the same
black-box Gym environment used for RL/MPC.  It is intended for the paper's case
study and complements analytical/linear SDM or ZOA calculations.

Why this script is preferred for the present project
---------------------------------------------------
Classical SLDs are usually obtained from a linearized regenerative model.  That is
useful for the theoretical chatter-onset boundary, but the current project uses a
nonlinear face-milling simulator with tooth engagement, regenerative history,
modal workpiece dynamics, displacement-based termination, and pass-completion
logic.  Therefore the most consistent case-study stability map is obtained by
simulating each fixed cutting condition ``(rpm, ap)`` in the actual environment.

Classification convention
-------------------------
For each grid point the cutter is run with a fixed physical action.

    stable   : the pass reaches the environment pass-completion condition
               before physical instability.
    unstable : excessive displacement/invalid state occurs before pass completion.
    unknown  : the maximum simulation step cap is reached before either event.

No reward, PPO policy, or feedback control is used.  The map is a baseline
process-stability estimate for fixed cutting parameters.

Outputs
-------
1. CSV table of all simulated grid points.
2. Heat map of normalized vibration severity ``max|w|/w_limit`` with the
   stable/unstable contour overlaid.
3. Optional boundary CSV extracted from the grid.

Example
-------
    python scripts/stability_lobe_numerical_heatmap.py \
        --rpm-min 4000 --rpm-max 40000 --rpm-points 60 \
        --ap-min 0 --ap-max 20 --ap-points 50 \
        --dt 1e-4 --n-substeps 10 --max-sim-steps 80000 \
        --line-y 0.20 --out-dir plots/stability_lobe_case_study
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from custom_rl import register_envs
from custom_rl.plants.plate import omega_to_rpm, rpm_to_omega

ENV_ID = "CustomODEPlate-v0"

PASS_COMPLETED_REASONS = {"pass_completed_90percent", "pass_completed"}
UNSTABLE_REASONS = {
    "excessive_sensor_displacement",
    "excessive_sensor_velocity",  # backward/future compatibility
    "invalid_state",
}


@dataclass
class TrialMetrics:
    rpm: float
    omega_rad_s: float
    ap_mm: float
    stable: bool
    status_code: int   # 1 stable, 0 unstable, -1 unknown/time cap
    steps: int
    sim_time_s: float
    x_start_m: float
    x_end_m: float
    y_line_m: float
    feed_progress: float
    max_abs_w_m: float
    rms_w_m: float
    w_limit_m: float
    severity_max_w_over_limit: float
    terminated: bool
    truncated: bool
    pass_completed: bool
    termination_reason: str | None


def _depth_bounds_from_plant(plant: Any) -> tuple[float, float]:
    ap_min = getattr(plant, "ap_min", getattr(plant, "ac_min", 0.0))
    ap_max = getattr(plant, "ap_max", getattr(plant, "ac_max", 1.0))
    return float(ap_min), float(ap_max)


def _physical_action_bounds(plant: Any) -> tuple[np.ndarray, np.ndarray]:
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
    low, high = _physical_action_bounds(plant)
    u_phys = np.asarray(u_phys, dtype=np.float64).reshape(-1)
    if u_phys.size < low.size:
        padded = low.copy()
        padded[: u_phys.size] = u_phys
        u_phys = padded
    elif u_phys.size > low.size:
        u_phys = u_phys[: low.size]
    span = np.maximum(high - low, 1e-12)
    return np.clip(2.0 * (u_phys - low) / span - 1.0, -1.0, 1.0)


def set_milling_line_y(plant: Any, y_m: float | None) -> float:
    if y_m is None:
        return float(getattr(plant, "y_cutter", 0.5 * plant.L2))
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
    """Set the starting cutter x-location while preserving the x=L1 -> x=0 pass."""
    if x_start_m is None:
        return float(getattr(plant, "L1", 0.0) - getattr(plant, "x0_cutter", 0.0))
    L1 = float(plant.L1)
    x_start_m = float(np.clip(float(x_start_m), 0.0, L1))
    # The force model commonly uses x_c = L1 - (feed_distance + x0_cutter).
    plant.x0_cutter = float(L1 - x_start_m)
    if hasattr(plant, "_sync_face_milling_geometry"):
        plant._sync_face_milling_geometry()
    return x_start_m


def resolve_ap_bounds(args: argparse.Namespace, plant: Any) -> tuple[float, float]:
    plant_ap_min, plant_ap_max = _depth_bounds_from_plant(plant)
    ap_min = plant_ap_min if args.ap_min is None else float(args.ap_min)
    ap_max = plant_ap_max if args.ap_max is None else float(args.ap_max)
    ap_min = max(ap_min, plant_ap_min)
    ap_max = min(ap_max, plant_ap_max)
    if not ap_max > ap_min:
        raise ValueError(f"No valid ap interval after clipping to plant bounds: {ap_min:g}..{ap_max:g} mm")
    return float(ap_min), float(ap_max)


def _feed_rate_m_s(plant: Any, omega_rad_s: float) -> float:
    ft_mm = float(getattr(plant, "feed_per_tooth_mm", getattr(plant, "cf", 0.0)))
    n_teeth = int(getattr(plant, "N", 1))
    return (ft_mm / 1000.0) * n_teeth * max(float(omega_rad_s), 1e-12) / (2.0 * np.pi)


def _pass_end_x(plant: Any) -> float:
    return float(getattr(plant, "x_pass_end_m", getattr(plant, "x_pass_end_tol", 0.1 * plant.L1)))


def steps_for_trial(plant: Any, *, omega_rad_s: float, x_start_m: float, dt: float,
                    n_substeps: int, max_sim_steps: int) -> int:
    """Compute enough steps for pass completion at the selected rpm, then cap."""
    step_dt = max(float(dt) * int(n_substeps), 1e-12)
    x_end = _pass_end_x(plant)
    distance = max(float(x_start_m) - x_end, 0.0)
    feed = max(_feed_rate_m_s(plant, omega_rad_s), 1e-12)
    required = int(np.ceil(1.10 * distance / (feed * step_dt))) + 10
    return int(min(max(required, 1), max_sim_steps))


def make_env(args: argparse.Namespace) -> gym.Env:
    env_kwargs: dict[str, Any] = {
        "reward_id": "sparse",
        "dt": float(args.dt),
        "n_substeps": int(args.n_substeps),
        "randomize_y0": False,
        "dynamics_uncertainty_std": 0.0,
        "max_episode_steps": int(args.max_sim_steps),
    }
    # Optional physical/model overrides if the project env exposes them.
    for key in (
        "E", "nu", "rho", "rho_type", "Kt", "Kr", "Ka", "Kte", "Kre", "Kae",
        "use_process_damping", "Ksp", "mu", "VB", "lambda_L_deg",
        "ap_min", "ap_max", "omega_min", "omega_max",
    ):
        value = getattr(args, key, None)
        if value is not None:
            env_kwargs[key] = value
    return gym.make(ENV_ID, **env_kwargs)


def update_w_metrics(info: dict[str, Any], max_abs_w: float, sum_sq: float, count: int) -> tuple[float, float, int]:
    if "w_sensor" not in info:
        return max_abs_w, sum_sq, count
    w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.all(np.isfinite(w)):
        return max_abs_w, sum_sq, count
    max_abs_w = max(float(max_abs_w), float(np.max(np.abs(w))))
    sum_sq += float(np.mean(w**2))
    count += 1
    return max_abs_w, sum_sq, count


def run_fixed_action_trial(env: gym.Env, *, omega_rad_s: float, ap_mm: float, max_steps: int,
                           seed: int, x_start_m: float | None, line_y_m: float | None) -> TrialMetrics:
    plant = env.unwrapped.plant
    x_selected = set_tool_start_x(plant, x_start_m)
    y_selected = set_milling_line_y(plant, line_y_m)

    if bool(getattr(plant, "control_ae", False)):
        ae = float(getattr(plant, "ae_default", getattr(plant, "ae_min", 0.0)))
        u_phys = np.array([omega_rad_s, ap_mm, ae], dtype=np.float64)
    else:
        u_phys = np.array([omega_rad_s, ap_mm], dtype=np.float64)
    action = physical_to_normalized_action(u_phys, plant)

    reset_options = {"y0": y_selected}
    env.reset(seed=int(seed), options=reset_options)

    terminated = truncated = False
    pass_completed = False
    termination_reason: str | None = None
    steps = 0
    max_abs_w = 0.0
    sum_sq_w = 0.0
    count_w = 0
    final_x = float(x_selected)
    feed_progress = 0.0

    while steps < int(max_steps) and not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        termination_reason = info.get("termination_reason", termination_reason)
        if info.get("pass_completed") or termination_reason in PASS_COMPLETED_REASONS:
            pass_completed = True
        final_x = float(info.get("cutter_x", final_x))
        feed_progress = float(info.get("feed_progress", feed_progress))
        max_abs_w, sum_sq_w, count_w = update_w_metrics(info, max_abs_w, sum_sq_w, count_w)
        steps += 1

    if pass_completed:
        stable = True
        status_code = 1
        if termination_reason is None:
            termination_reason = "pass_completed"
    elif termination_reason in UNSTABLE_REASONS:
        stable = False
        status_code = 0
    else:
        stable = False
        status_code = -1
        if termination_reason is None:
            termination_reason = "safety_max_steps_reached"

    step_dt = float(getattr(env.unwrapped, "_step_dt", float(getattr(plant, "dt", 1e-4)) * int(getattr(plant, "n_substeps", 1))))
    rms_w = float(np.sqrt(sum_sq_w / count_w)) if count_w else 0.0
    w_limit = float(getattr(plant, "w_limit", 1.0e-3))
    severity = max_abs_w / max(w_limit, 1e-30)
    return TrialMetrics(
        rpm=float(omega_to_rpm(omega_rad_s)),
        omega_rad_s=float(omega_rad_s),
        ap_mm=float(ap_mm),
        stable=bool(stable),
        status_code=int(status_code),
        steps=int(steps),
        sim_time_s=float(steps * step_dt),
        x_start_m=float(x_selected),
        x_end_m=float(final_x),
        y_line_m=float(y_selected),
        feed_progress=float(feed_progress),
        max_abs_w_m=float(max_abs_w),
        rms_w_m=float(rms_w),
        w_limit_m=float(w_limit),
        severity_max_w_over_limit=float(severity),
        terminated=bool(terminated),
        truncated=bool(truncated),
        pass_completed=bool(pass_completed),
        termination_reason=termination_reason,
    )


def compute_grid(args: argparse.Namespace) -> tuple[list[TrialMetrics], dict[str, Any]]:
    register_envs()
    env = make_env(args)
    plant = env.unwrapped.plant
    ap_min, ap_max = resolve_ap_bounds(args, plant)
    rpm_values = np.linspace(float(args.rpm_min), float(args.rpm_max), int(args.rpm_points))
    ap_values = np.linspace(ap_min, ap_max, int(args.ap_points))
    x_start = set_tool_start_x(plant, args.x_start)
    y_line = set_milling_line_y(plant, args.line_y)

    rows: list[TrialMetrics] = []
    print("Numerical face-milling stability heat-map sweep")
    print(f"  rpm: {rpm_values[0]:.0f}..{rpm_values[-1]:.0f} ({rpm_values.size})")
    print(f"  ap : {ap_values[0]:.3g}..{ap_values[-1]:.3g} mm ({ap_values.size})")
    print(f"  x_start={x_start:.4g} m, y_line={y_line:.4g} m")
    print(f"  dt={args.dt:g}, n_substeps={args.n_substeps}, max_sim_steps={args.max_sim_steps}")

    for i_rpm, rpm in enumerate(rpm_values):
        omega = float(np.clip(rpm_to_omega(float(rpm)), plant.omega_min, plant.omega_max))
        rpm_actual = float(omega_to_rpm(omega))
        max_steps = steps_for_trial(plant, omega_rad_s=omega, x_start_m=x_start,
                                    dt=args.dt, n_substeps=args.n_substeps,
                                    max_sim_steps=args.max_sim_steps)
        for i_ap, ap in enumerate(ap_values):
            row = run_fixed_action_trial(
                env, omega_rad_s=omega, ap_mm=float(ap), max_steps=max_steps,
                seed=int(args.seed + i_rpm * 1000 + i_ap), x_start_m=x_start, line_y_m=y_line,
            )
            rows.append(row)
        stable_count = sum(r.stable for r in rows[-ap_values.size:])
        print(f"  [{i_rpm+1:>3}/{rpm_values.size}] rpm={rpm_actual:>7.0f}: stable {stable_count}/{ap_values.size}")

    env.close()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "black_box_time_domain_fixed_action_grid",
        "classification": "stable=pass_completed; unstable=displacement/invalid before pass; unknown=time cap",
        "rpm_values": rpm_values.tolist(),
        "ap_values_mm": ap_values.tolist(),
        "x_start_m": float(x_start),
        "y_line_m": float(y_line),
        "dt": float(args.dt),
        "n_substeps": int(args.n_substeps),
        "max_sim_steps": int(args.max_sim_steps),
        "w_limit_m": float(getattr(plant, "w_limit", np.nan)),
        "L1_m": float(getattr(plant, "L1", np.nan)),
        "L2_m": float(getattr(plant, "L2", np.nan)),
        "N_teeth": int(getattr(plant, "N", -1)),
        "feed_per_tooth_mm": float(getattr(plant, "feed_per_tooth_mm", np.nan)),
    }
    return rows, meta


def rows_to_grids(rows: list[TrialMetrics]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rpm_axis = np.array(sorted({r.rpm for r in rows}), dtype=np.float64)
    ap_axis = np.array(sorted({r.ap_mm for r in rows}), dtype=np.float64)
    stable = np.full((ap_axis.size, rpm_axis.size), np.nan)
    severity = np.full_like(stable, np.nan, dtype=np.float64)
    progress = np.full_like(stable, np.nan, dtype=np.float64)
    r_index = {v: i for i, v in enumerate(rpm_axis)}
    a_index = {v: i for i, v in enumerate(ap_axis)}
    for r in rows:
        ai = a_index[r.ap_mm]
        ri = r_index[r.rpm]
        stable[ai, ri] = r.status_code
        severity[ai, ri] = r.severity_max_w_over_limit
        progress[ai, ri] = r.feed_progress
    return rpm_axis, ap_axis, stable, severity, progress


def boundary_from_grid(rpm_axis: np.ndarray, ap_axis: np.ndarray, stable_grid: np.ndarray) -> list[dict[str, float]]:
    boundary = []
    for ri, rpm in enumerate(rpm_axis):
        column = stable_grid[:, ri]
        stable_aps = ap_axis[column == 1]
        if stable_aps.size:
            ap_lim = float(np.max(stable_aps))
        else:
            ap_lim = 0.0
        boundary.append({"rpm": float(rpm), "ap_stable_grid_mm": ap_lim})
    return boundary


def save_rows(rows: list[TrialMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_boundary(boundary: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rpm", "ap_stable_grid_mm"])
        writer.writeheader()
        writer.writerows(boundary)


def plot_heatmap(rows: list[TrialMetrics], meta: dict[str, Any], out_dir: Path) -> None:
    rpm_axis, ap_axis, stable_grid, severity_grid, progress_grid = rows_to_grids(rows)
    R, A = np.meshgrid(rpm_axis, ap_axis)

    # log severity makes both stable and unstable regions visible.
    sev_plot = np.log10(np.maximum(severity_grid, 1e-4))
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    im = ax.pcolormesh(R, A, sev_plot, shading="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\max|w|/w_{lim})$")

    # Stability contour: between stable code=1 and unstable/unknown <=0.
    binary = np.where(stable_grid == 1, 1.0, 0.0)
    if np.nanmin(binary) < 0.5 < np.nanmax(binary):
        cs = ax.contour(R, A, binary, levels=[0.5], colors="white", linewidths=2.5)
        cs.collections[0].set_label("Stable/unstable boundary")

    # Unit vibration severity line max|w|=w_limit.
    if np.nanmin(severity_grid) <= 1.0 <= np.nanmax(severity_grid):
        ax.contour(R, A, severity_grid, levels=[1.0], colors="red", linestyles="--", linewidths=1.6)

    boundary = boundary_from_grid(rpm_axis, ap_axis, stable_grid)
    ap_lim = np.array([b["ap_stable_grid_mm"] for b in boundary], dtype=np.float64)
    ax.plot(rpm_axis, ap_lim, color="black", linewidth=2.0, marker="o", markersize=3,
            label="Grid-extracted SLD line")

    ax.set_xlabel("Spindle speed [rpm]")
    ax.set_ylabel("Axial depth of cut $a_p$ [mm]")
    ax.set_title("Numerical face-milling stability heat map for the flexible cantilever plate\n"
                 "fixed-parameter pass simulation; heat = vibration severity, line = stability boundary")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stability_lobe_numerical_heatmap.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    im = ax.pcolormesh(R, A, progress_grid, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Final feed progress")
    ax.plot(rpm_axis, ap_lim, color="white", linewidth=2.0, marker="o", markersize=3,
            label="Grid-extracted SLD line")
    ax.set_xlabel("Spindle speed [rpm]")
    ax.set_ylabel("Axial depth of cut $a_p$ [mm]")
    ax.set_title("Pass-progress heat map for fixed cutting conditions")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "stability_lobe_progress_heatmap.png", dpi=170)
    plt.close(fig)

    save_boundary(boundary, out_dir / "stability_lobe_boundary_from_grid.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("plots/stability_lobe_case_study"))
    p.add_argument("--rpm-min", type=float, default=4000.0)
    p.add_argument("--rpm-max", type=float, default=5000.0)
    p.add_argument("--rpm-points", type=int, default=10)
    p.add_argument("--ap-min", type=float, default=0.0)
    p.add_argument("--ap-max", type=float, default=None)
    p.add_argument("--ap-points", type=int, default=50)
    p.add_argument("--dt", type=float, default=1.0e-4)
    p.add_argument("--n-substeps", type=int, default=10)
    p.add_argument("--max-sim-steps", type=int, default=500)
    p.add_argument("--x-start", type=float, default=None, help="Starting cutter x-position [m]. Default: plant current/free-side start.")
    p.add_argument("--line-y", type=float, default=None, help="Fixed milling-line y-position [m]. Default: plant default.")
    p.add_argument("--seed", type=int, default=0)
    # Optional overrides.
    p.add_argument("--E", type=float, default=None)
    p.add_argument("--nu", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--rho-type", type=str, default=None)
    p.add_argument("--Kt", type=float, default=None)
    p.add_argument("--Kr", type=float, default=None)
    p.add_argument("--Ka", type=float, default=None)
    p.add_argument("--Kte", type=float, default=None)
    p.add_argument("--Kre", type=float, default=None)
    p.add_argument("--Kae", type=float, default=None)
    p.add_argument("--use-process-damping", action="store_true")
    p.add_argument("--Ksp", type=float, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--VB", type=float, default=None)
    p.add_argument("--lambda-L-deg", type=float, default=None)
    p.add_argument("--omega-min", type=float, default=None, help="Optional omega lower bound [rad/s] for env creation.")
    p.add_argument("--omega-max", type=float, default=None, help="Optional omega upper bound [rad/s] for env creation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows, meta = compute_grid(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out / "stability_lobe_grid_results.csv")
    (out / "stability_lobe_grid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_heatmap(rows, meta, out)
    print(f"\nsaved numerical stability-lobe heat maps and CSV files to: {out}")


if __name__ == "__main__":
    main()
