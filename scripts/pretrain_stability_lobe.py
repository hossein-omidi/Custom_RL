#!/usr/bin/env python3
"""
Pre-training stability lobe — uncontrolled (omega, ac) grid, no RL policy.

Conventional stability diagram: at each spindle speed omega, estimate the critical
depth of cut ac where the process crosses from stable to chatter-like, using Monte
Carlo rollouts over plant stochasticity (geometry, sensors, pass-line start L).

Research-aligned criterion (time-domain, regenerative milling):
  - **Stable**: finite state, bounded post-transient vibration, no sustained growth.
  - **Chatter-like**: sustained post-transient amplitude growth and/or large vibration
    after transients (regenerative feedback), or physical termination from excessive
    displacement without numerical failure.
  - **Failed**: non-finite state, force-clip dominated, or |Delta_f| blow-up.

Boundary extraction: P(chatter-like) from MC rollouts at each grid point; critical
ac(omega) is where P crosses ``--threshold`` (default 0.5), with ac-scan monotonicity
enforced. Report mean critical ac and MC std across rollouts.

Start position L: random pass-line starts are averaged via MC. A 3D lobe (omega, ac, L)
is only needed if L strongly shifts the boundary (see ``start_position_sensitivity``).

Usage (from project root)::

    python scripts/pretrain_stability_lobe.py
    python scripts/pretrain_stability_lobe.py --quick
    python scripts/pretrain_stability_lobe.py --n-rollouts 10 --t-final 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_rl.analysis.milling_rollout import rollout_uncontrolled
from custom_rl.analysis.response_classification import (
    classify_milling_response,
    is_instability_label,
    metrics_from_rollout_dict,
)
from custom_rl.analysis.stability_lobe import (
    critical_ac_curve_with_mc_std,
    monotonicity_violations,
    start_position_sensitivity,
)
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.time_scales import recommend_integration_dt, tooth_period

PLOT_DIR = _ROOT / "plots" / "pretrain_stability_lobe"
MACRO_DT = 0.002


def make_plant(
    *,
    geometry_uncertainty: bool = True,
    sensor_uncertainty: bool = False,
    process_noise: bool = False,
    path_y_start: float | None = None,
    pass_sampling: str = "random",
) -> PlatePlant:
    return PlatePlant(
        displacement_model="feed_normal_full",
        trajectory_mode="middle_line",
        enable_geometry_uncertainty=geometry_uncertainty,
        enable_sensor_uncertainty=sensor_uncertainty,
        enable_process_noise=process_noise,
        pass_sampling=pass_sampling,
        path_y_start=path_y_start,
    )


def resolve_t_final(plant: PlatePlant, omega_grid: np.ndarray, t_final: float | None) -> float:
    if t_final is not None and t_final > 0.0:
        return float(t_final)
    om_min = float(np.min(omega_grid))
    om_max = float(np.max(omega_grid))
    tau_max = tooth_period(om_max, plant.N)
    t_rev_min = 2.0 * math.pi / om_min
    return max(3.0, 30.0 * tau_max, 8.0 * t_rev_min)


def sweep_mc_grid(
    plant: PlatePlant,
    omega_grid: np.ndarray,
    ac_grid: np.ndarray,
    *,
    t_final: float,
    macro_dt: float,
    seeds: list[int],
    n_rollouts: int,
    perturb: float,
    transient_fraction: float,
    growth_factor: float,
    threshold: float,
) -> dict:
    """Monte Carlo grid sweep; returns arrays and per-rollout records."""
    disp_limit = float(plant.displacement_failure_limit)
    n_om, n_ac = len(omega_grid), len(ac_grid)
    instability = np.zeros((n_om, n_ac, n_rollouts), dtype=bool)
    p_chatter = np.zeros((n_om, n_ac), dtype=np.float64)
    p_failed = np.zeros((n_om, n_ac), dtype=np.float64)
    rms_mean = np.zeros((n_om, n_ac), dtype=np.float64)
    rms_std = np.zeros((n_om, n_ac), dtype=np.float64)
    all_records: list[dict] = []

    scales = recommend_integration_dt(plant, macro_dt=macro_dt, training_mode=True)
    print(
        f"  sub_dt={scales['dt_recommended_training_substep_s']:.2e}s  "
        f"n_substeps={scales['n_substeps']}  rollouts/cell={n_rollouts}"
    )

    for i, omega in enumerate(omega_grid):
        for j, ac in enumerate(ac_grid):
            classes: list[str] = []
            rms_vals: list[float] = []
            for r in range(n_rollouts):
                seed = int(seeds[r % len(seeds)] + 1000 * i + 10 * j + r)
                rng = np.random.default_rng(seed)
                raw = rollout_uncontrolled(
                    plant, float(omega), float(ac),
                    t_final=t_final, macro_dt=macro_dt, rng=rng, perturb=perturb,
                )
                raw["transient_fraction"] = transient_fraction
                metrics = metrics_from_rollout_dict(raw)
                klass = classify_milling_response(
                    metrics,
                    transient_fraction=transient_fraction,
                    displacement_limit_m=disp_limit,
                    growth_factor=growth_factor,
                )
                classes.append(klass)
                rms_vals.append(metrics.rms_w_post_m)
                unstable = is_instability_label(klass)
                instability[i, j, r] = unstable
                all_records.append({
                    "omega_rad_s": float(omega),
                    "ac_mm": float(ac),
                    "seed": seed,
                    "class": klass,
                    "max_w_mm": metrics.max_w_mm,
                    "rms_w_post_um": metrics.rms_w_post_m * 1e3,
                    "clip_pct": metrics.clip_pct,
                    "delta_f_mm": metrics.delta_f_mm,
                    "path_y_start_m": raw.get("path_y_start_m"),
                    "finite": metrics.finite,
                })
            p_chatter[i, j] = float(np.mean([c == "chatter-like" for c in classes]))
            p_failed[i, j] = float(np.mean([c == "failed" for c in classes]))
            rms_mean[i, j] = float(np.mean(rms_vals))
            rms_std[i, j] = float(np.std(rms_vals))

        row_preview = " ".join(
            f"{ac_grid[j]:.1f}:{p_chatter[i, j]:.2f}" for j in range(n_ac)
        )
        print(f"  omega={omega:6.0f}  P(chatter) [{row_preview}]")

    critical_mean, critical_unc, critical_std, p_mono = critical_ac_curve_with_mc_std(
        omega_grid, ac_grid, instability, threshold=threshold,
    )
    start_sens = start_position_sensitivity(all_records, omega_grid, ac_grid)
    mono_v = monotonicity_violations(p_mono)

    return {
        "omega_grid": omega_grid,
        "ac_grid": ac_grid,
        "p_chatter": p_chatter,
        "p_failed": p_failed,
        "p_mono": p_mono,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "instability_labels": instability,
        "critical_ac": critical_mean,
        "critical_ac_uncertainty": critical_unc,
        "critical_ac_std": critical_std,
        "threshold": threshold,
        "t_final_s": t_final,
        "macro_dt_s": macro_dt,
        "records": all_records,
        "start_position": start_sens,
        "monotonicity_violations": mono_v,
    }


def plot_lobe(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    og = data["omega_grid"]
    crit = data["critical_ac"]
    unc = data["critical_ac_uncertainty"]
    std = data["critical_ac_std"]
    ok = np.isfinite(crit)

    fig, ax = plt.subplots(figsize=(8, 5))
    if np.any(ok):
        ax.plot(og[ok], crit[ok], "b-o", lw=2, ms=5, label="E[critical ac]")
        ax.fill_between(
            og[ok],
            crit[ok] - unc[ok],
            crit[ok] + unc[ok],
            alpha=0.2,
            color="blue",
            label="grid interpolation band",
        )
        if np.any(np.isfinite(std[ok])):
            ax.fill_between(
                og[ok],
                crit[ok] - std[ok],
                crit[ok] + std[ok],
                alpha=0.15,
                color="orange",
                label="MC std across rollouts",
            )
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("critical ac [mm]")
    ax.set_title(
        f"Stability lobe (uncontrolled, P(chatter)={data['threshold']:.2f})"
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "stability_lobe_boundary.png", dpi=150)
    plt.close(fig)

    ag = data["ac_grid"]
    pc = data["p_mono"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, omega in enumerate(og):
        ax2.plot(ag, pc[i, :], "-o", ms=3, lw=1.2, label=f"{omega:.0f} rad/s")
    ax2.axhline(data["threshold"], color="k", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("ac [mm]")
    ax2.set_ylabel("P(chatter-like), monotone")
    ax2.set_title("Instability probability vs depth of cut")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "p_chatter_vs_ac.png", dpi=150)
    plt.close(fig2)


def save_outputs(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    og, ag = data["omega_grid"], data["ac_grid"]

    np.savez_compressed(
        out_dir / "stability_lobe_data.npz",
        omega_grid=og,
        ac_grid=ag,
        p_chatter=data["p_chatter"],
        p_failed=data["p_failed"],
        p_mono=data["p_mono"],
        rms_mean=data["rms_mean"],
        rms_std=data["rms_std"],
        critical_ac=data["critical_ac"],
        critical_ac_uncertainty=data["critical_ac_uncertainty"],
        critical_ac_std=data["critical_ac_std"],
        threshold=np.array(data["threshold"]),
        t_final_s=np.array(data["t_final_s"]),
    )

    with open(out_dir / "grid_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "omega_rad_s", "ac_mm", "p_chatter", "p_failed",
            "rms_mean_m", "rms_std_m",
        ])
        for i, om in enumerate(og):
            for j, ac in enumerate(ag):
                w.writerow([
                    om, ac,
                    data["p_chatter"][i, j],
                    data["p_failed"][i, j],
                    data["rms_mean"][i, j],
                    data["rms_std"][i, j],
                ])

    with open(out_dir / "critical_ac_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "omega_rad_s", "critical_ac_mm", "interpolation_uncertainty_mm", "mc_std_mm",
        ])
        for om, ac_c, u, s in zip(
            og, data["critical_ac"], data["critical_ac_uncertainty"], data["critical_ac_std"],
        ):
            w.writerow([
                om,
                ac_c if np.isfinite(ac_c) else "",
                u if np.isfinite(u) else "",
                s if np.isfinite(s) else "",
            ])

    with open(out_dir / "rollout_records.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data["records"][0].keys()))
        w.writeheader()
        w.writerows(data["records"])

    report = {
        "map_type": "uncontrolled_stability_lobe_pretrain",
        "requires_trained_policy": False,
        "stability_criterion": (
            "Monte Carlo time-domain classification per (omega, ac): stable / bounded / "
            "chatter-like / failed using post-transient vibration growth, amplitude "
            "limits, force-clip %, |Delta_f| blow-up, and finite-state checks "
            "(Altintas / regenerative chatter time-domain philosophy)."
        ),
        "boundary_definition": (
            "critical ac(omega): interpolated ac where monotone P(chatter-like) "
            "crosses threshold (default 0.5). mc_std: std of per-rollout crossing "
            "estimates at each omega."
        ),
        "stochastic_sources": {
            "geometry_uncertainty": "episode L1, L2, h, E, rho resampling",
            "sensor_uncertainty": "sensor position jitter each reset",
            "process_noise": "diagonal modal noise after integration",
            "pass_sampling": "random pass-line / start position L each reset",
            "initial_perturbation": "small eta perturbation at ac > 0",
        },
        "start_position_L": data["start_position"],
        "recommend_3d_lobe": data["start_position"].get("recommend_3d_lobe", False),
        "monotonicity_violations_ac_scan": data["monotonicity_violations"],
        "t_final_s": data["t_final_s"],
        "threshold": data["threshold"],
    }
    with open(out_dir / "stability_lobe_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-training stability lobe (MC uncontrolled omega/ac grid)",
    )
    parser.add_argument("--quick", action="store_true", help="small grid for smoke test")
    parser.add_argument("--t-final", type=float, default=None, help="simulation horizon [s]")
    parser.add_argument("--macro-dt", type=float, default=MACRO_DT)
    parser.add_argument("--n-rollouts", type=int, default=8, help="MC rollouts per grid cell")
    parser.add_argument("--threshold", type=float, default=0.5, help="P(chatter) crossing")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--perturb", type=float, default=1e-6)
    parser.add_argument("--transient-fraction", type=float, default=0.35)
    parser.add_argument("--growth-factor", type=float, default=3.0)
    parser.add_argument("--omega", type=float, nargs="+", default=None)
    parser.add_argument("--ac", type=float, nargs="+", default=None)
    parser.add_argument("--no-geometry-uncertainty", action="store_true")
    parser.add_argument("--sensor-uncertainty", action="store_true")
    parser.add_argument("--process-noise", action="store_true")
    parser.add_argument("--fixed-path-y", type=float, default=None, help="fix start L [m]")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.quick:
        omega_grid = np.array([400.0, 600.0, 800.0, 1000.0, 1200.0], dtype=np.float64)
        ac_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], dtype=np.float64)
        n_rollouts = min(args.n_rollouts, 4)
    else:
        omega_grid = np.asarray(
            args.omega if args.omega else np.linspace(400.0, 1600.0, 13),
            dtype=np.float64,
        )
        ac_grid = np.asarray(
            args.ac if args.ac else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            dtype=np.float64,
        )
        n_rollouts = args.n_rollouts

    seeds = list(args.seeds) if args.seeds else [args.seed + k for k in range(n_rollouts)]

    plant = make_plant(
        geometry_uncertainty=not args.no_geometry_uncertainty,
        sensor_uncertainty=args.sensor_uncertainty,
        process_noise=args.process_noise,
        path_y_start=args.fixed_path_y,
    )
    plant.reset(np.random.default_rng(0))
    t_final = resolve_t_final(plant, omega_grid, args.t_final)

    warnings.simplefilter("ignore", RuntimeWarning)

    print(
        f"Pre-training stability lobe: {len(omega_grid)} omega x {len(ac_grid)} ac, "
        f"t_final={t_final:.2f}s (no RL policy)"
    )
    data = sweep_mc_grid(
        plant,
        omega_grid,
        ac_grid,
        t_final=t_final,
        macro_dt=args.macro_dt,
        seeds=seeds,
        n_rollouts=n_rollouts,
        perturb=args.perturb,
        transient_fraction=args.transient_fraction,
        growth_factor=args.growth_factor,
        threshold=args.threshold,
    )

    crit = data["critical_ac"]
    ok = np.isfinite(crit)
    print(f"\nSummary:")
    print(f"  critical ac range: {np.nanmin(crit):.2f}–{np.nanmax(crit):.2f} mm" if ok.any() else "  no finite boundary")
    if ok.any():
        print(f"  MC std range: {np.nanmin(data['critical_ac_std'][ok]):.3f}–"
              f"{np.nanmax(data['critical_ac_std'][ok]):.3f} mm")
    print(f"  ac-scan monotonicity violations: {data['monotonicity_violations']}")
    sp = data["start_position"]
    print(f"  3D lobe (omega, ac, L) recommended: {sp.get('recommend_3d_lobe', False)}")
    if ok.any() and len(np.unique(np.round(crit[ok], 3))) == 1:
        print("  WARN: flat boundary — use denser ac grid, longer t_final, more rollouts")

    save_outputs(data, PLOT_DIR)
    if not args.no_plot:
        plot_lobe(data, PLOT_DIR)
        print(f"Plots: {PLOT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
