#!/usr/bin/env python
"""Compare PPO evaluation learning curves across Fast/Balanced/Precise sweeps.

Each seed contributes its mean over evaluation episodes at each checkpoint.
The plotted line is the mean across independent training seeds; the band is
plus/minus one sample standard deviation across those seed-level means.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_seed_curve(run_dir: Path, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = run_dir / "logs" / f"seed_{seed}" / "evaluations.npz"
    if not path.is_file():
        print(f"WARNING: missing evaluation history: {path}")
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if not {"timesteps", "results"}.issubset(data.files):
                raise ValueError("expected arrays 'timesteps' and 'results'")
            steps = np.asarray(data["timesteps"], dtype=float).reshape(-1)
            results = np.asarray(data["results"], dtype=float)
    except Exception as exc:
        print(f"WARNING: cannot read {path}: {exc}")
        return None
    if results.ndim == 1:
        results = results.reshape(-1, 1)
    if results.ndim != 2 or results.shape[0] != steps.size:
        print(f"WARNING: invalid shapes in {path}: steps={steps.shape}, results={results.shape}")
        return None
    results[~np.isfinite(results)] = np.nan
    finite_counts = np.isfinite(results).sum(axis=1)
    seed_mean = np.divide(
        np.nansum(results, axis=1),
        finite_counts,
        out=np.full(steps.shape, np.nan, dtype=float),
        where=finite_counts > 0,
    )
    valid = np.isfinite(steps) & np.isfinite(seed_mean)
    steps, seed_mean = steps[valid], seed_mean[valid]
    if not steps.size:
        return None
    order = np.argsort(steps, kind="stable")
    steps, seed_mean = steps[order], seed_mean[order]
    # Keep the last value if a resumed history contains duplicate step values.
    last: dict[float, int] = {}
    for i, step in enumerate(steps):
        last[float(step)] = i
    keep = np.asarray(sorted(last.values()), dtype=int)
    return steps[keep], seed_mean[keep]


def build_common_grid(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, bool]:
    """Build a shared grid; interpolate only within the common observed range."""
    if not curves:
        raise ValueError("No valid learning curves were found.")
    lower = max(float(x[0]) for x, _ in curves)
    upper = min(float(x[-1]) for x, _ in curves)
    if upper < lower:
        raise ValueError("The configurations have no overlapping evaluation-timestep range.")
    common = curves[0][0][(curves[0][0] >= lower) & (curves[0][0] <= upper)]
    for x, _ in curves[1:]:
        common = np.intersect1d(common, x[(x >= lower) & (x <= upper)])
    if common.size:
        return common.astype(float), False
    grid = np.unique(np.concatenate([x[(x >= lower) & (x <= upper)] for x, _ in curves]))
    if not grid.size:
        raise ValueError("Could not construct a common evaluation-timestep grid.")
    return grid.astype(float), True


def aggregate_seed_curves(
    curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.vstack([np.interp(grid, x, y) for x, y in curves])
    mean = np.mean(matrix, axis=0)
    if matrix.shape[0] >= 2:
        sd = np.std(matrix, axis=0, ddof=1)
    else:
        sd = np.full(grid.shape, np.nan, dtype=float)
    return mean, sd


def _write_raw_csv(path: Path, raw_rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "training_seed", "training_timesteps", "evaluation_episode", "episode_return"])
        writer.writerows(raw_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the cross-seed learning-curve comparison for a PPO sweep.")
    parser.add_argument("--manifest", required=True, help="Path to sweep_manifest.json created by sweep_ppo.py")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <sweep-root>/plots)")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Sweep manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in manifest.get("seeds", [])]
    entries = manifest.get("configurations", [])
    if not seeds or not entries:
        raise SystemExit("Manifest does not contain training seeds and configurations.")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else manifest_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    histories: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    labels: dict[str, str] = {}
    raw_rows: list[list[Any]] = []
    all_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for entry in entries:
        name = str(entry.get("name"))
        label = str(entry.get("label") or name.title())
        run_value = entry.get("run_dir")
        if not run_value:
            print(f"WARNING: {name} has no completed run directory in the manifest; skipped.")
            continue
        run_dir = Path(run_value).expanduser().resolve()
        seed_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for seed in seeds:
            curve = load_seed_curve(run_dir, seed)
            if curve is None:
                continue
            seed_curves[seed] = curve
            # Preserve the raw per-episode evaluation returns for auditability.
            npz_path = run_dir / "logs" / f"seed_{seed}" / "evaluations.npz"
            with np.load(npz_path, allow_pickle=False) as data:
                steps = np.asarray(data["timesteps"], dtype=float).reshape(-1)
                results = np.asarray(data["results"], dtype=float)
            if results.ndim == 1:
                results = results.reshape(-1, 1)
            for step, row in zip(steps, results):
                for ep_index, value in enumerate(row, start=1):
                    if np.isfinite(step) and np.isfinite(value):
                        raw_rows.append([name, seed, int(step), ep_index, float(value)])
        if seed_curves:
            histories[name] = seed_curves
            labels[name] = label
            all_curves.extend(seed_curves.values())
            if len(seed_curves) < 2:
                print(f"WARNING: {name} has only {len(seed_curves)} seed; no sample-SD band can be estimated.")

    if not histories:
        raise SystemExit("No evaluation histories were found. Ensure the sweep training runs completed.")

    grid, interpolated = build_common_grid(all_curves)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    summary_rows: list[list[Any]] = []
    stats_rows: list[list[Any]] = []

    for index, entry in enumerate(entries):
        name = str(entry.get("name"))
        if name not in histories:
            continue
        seed_curves = histories[name]
        ordered_seeds = sorted(seed_curves)
        aligned = [seed_curves[s] for s in ordered_seeds]
        mean, sd = aggregate_seed_curves(aligned, grid)
        color = color_cycle[index % len(color_cycle)]
        ax.plot(grid, mean, color=color, linewidth=2.0,
                label=f"{labels[name]} (n={len(ordered_seeds)} seeds)")
        if len(ordered_seeds) >= 2:
            ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        for j, step in enumerate(grid):
            per_seed_values = [
                float(np.interp(step, seed_curves[s][0], seed_curves[s][1])) if s in seed_curves else ""
                for s in sorted(seeds)
            ]
            stats_rows.append([
                name, int(step), len(ordered_seeds), float(mean[j]),
                float(sd[j]) if np.isfinite(sd[j]) else "",
                *per_seed_values,
            ])
        best_i = int(np.nanargmax(mean))
        summary_rows.append([
            name, labels[name], len(ordered_seeds), int(grid[-1]), float(mean[-1]),
            float(sd[-1]) if np.isfinite(sd[-1]) else "",
            int(grid[best_i]), float(mean[best_i]),
        ])

    ax.set_xlabel("PPO training transitions")
    ax.set_ylabel("Mean evaluation episode return")
    ax.set_title("PPO configuration comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out_dir / f"ppo_sweep_learning_curve{suffix}", dpi=300 if suffix == ".png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    raw_path = out_dir / "ppo_sweep_raw_evaluation_returns.csv"
    _write_raw_csv(raw_path, raw_rows)
    stats_path = out_dir / "ppo_sweep_learning_curve_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "configuration", "training_timesteps", "n_training_seeds", "mean_eval_return",
            "sample_sd_across_seeds", *[f"seed_{s}_mean_return" for s in sorted(seeds)],
        ])
        writer.writerows(stats_rows)
    summary_path = out_dir / "ppo_sweep_final_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "configuration", "label", "n_training_seeds", "final_timestep", "final_mean_return",
            "final_sample_sd", "best_timestep", "best_mean_return",
        ])
        writer.writerows(summary_rows)

    print(f"Saved: {out_dir / 'ppo_sweep_learning_curve.png'}")
    print(f"Saved: {out_dir / 'ppo_sweep_learning_curve.pdf'}")
    print(f"Saved: {raw_path}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {summary_path}")
    if interpolated:
        print("Note: checkpoint grids differed; interpolation was restricted to the shared observed timestep range.")
    print("Band: mean ± 1 sample SD across independent training-seed means (not across MC episodes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
