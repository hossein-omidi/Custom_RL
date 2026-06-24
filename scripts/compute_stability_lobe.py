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
        return True, 0.0, 0.0, 0.0, 1.0

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

    obs, _ = env.reset(seed=seed)
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
        f"  ac search     : {plant.ac_min:.3f} – {plant.ac_max:.3f} mm\n"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, metadata = compute_lobe(args)

    out_dir = args.out_dir
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
