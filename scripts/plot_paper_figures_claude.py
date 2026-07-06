"""Publication-quality figures from training logs and eval_policy.py trajectories.

Reads only generated artifacts (SB3 monitor CSV, evaluations.npz, trajectory JSON).
Does not modify training or evaluation scripts.

Figures (paper-oriented, no duplicates of casual plot_results.py views):
  fig01  Training return vs environment steps (+ eval checkpoints)
  fig02  Closed-loop control: spindle speed, depth of cut, peak vibration
  fig03  Actuator trajectories across evaluation runs (normalized pass time)
  fig04  Operating map: joint density of (rpm, a_c) with marginals,
         optionally overlaid with the theoretical no-control stability
         boundary from scripts/compute_stability_lobe.py (--lobe-csv)
  fig05  Vibration field: sensor location vs normalized pass progress
  fig06  Evaluation summary: return distribution and pass completion by seed
  fig07  Per-sensor vibration ensemble: mean +/- std |w| per sensor,
         labeled by physical (x, y) location, vs normalized pass progress
  fig08  Cutting-process ensemble: mean +/- std engaged chip thickness
         (mean/max) vs normalized pass progress
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_PLOT_DIR, DEFAULT_TRAJ_DIR
from custom_rl.eval.monte_carlo import MC_BAND_STD_MULT
from custom_rl.eval.pipeline import (
    discover_log_seeds,
    discover_trajectory_seeds,
    primary_episode_record,
)
from custom_rl.plants.plate import RPM_MAX, RPM_MIN, omega_to_rpm

PAPER_OUT_DIR = Path(DEFAULT_PLOT_DIR) / "paper"
# Auto-discovery locations for a previously computed stability-lobe CSV
# (scripts/compute_stability_lobe.py output). Used only as a convenience
# default for --lobe-csv; never required for the other figures.
DEFAULT_LOBE_CSV_CANDIDATES = (
    Path(DEFAULT_PLOT_DIR) / "stability_lobe" / "stability_lobe.csv",
    Path(DEFAULT_PLOT_DIR) / "stability_lobe_test" / "stability_lobe.csv",
)

# Journal-style palette (colorblind-friendly).
C_PRIMARY = "#1F4E79"
C_ACCENT = "#C44E52"
C_FILL = "#4C72B0"
C_GRID = (0, 0, 0, 0.22)
# Per-sensor palette for ensemble figures (extend if more sensors are used).
SENSOR_COLORS = ["#1F4E79", "#C44E52", "#3B7D3E", "#8E5B9C", "#B8860B"]

ACTION_CMAP = LinearSegmentedColormap.from_list(
    "action_seq",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)
VIB_CMAP = LinearSegmentedColormap.from_list(
    "vibration",
    ["#FFFFE5", "#FD8D3C", "#E31A1C", "#800026"],
)


@dataclass
class EpisodeData:
    seed: int
    episode: int
    y0: float | None
    return_: float
    termination_reason: str | None
    pass_completed: bool
    times: np.ndarray
    rpm: np.ndarray
    ac: np.ndarray
    w_sensor: np.ndarray | None  # (T, n_sensors)
    chip_mean_mm: np.ndarray | None  # (T,) mean engaged chip thickness proxy
    chip_max_mm: np.ndarray | None  # (T,) max engaged chip thickness proxy
    metadata: dict




def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"Saved {path}")


def _resolve_seeds(
    seeds: list[int] | None,
    log_dir: Path,
    traj_dir: Path,
) -> list[int]:
    if seeds:
        return list(seeds)
    combined = sorted(set(discover_log_seeds(log_dir)) | set(discover_trajectory_seeds(traj_dir)))
    return combined or [0]


def _load_monitor_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=2)
    except Exception:
        return np.array([]), np.array([]), np.array([])

    if data.size == 0:
        return np.array([]), np.array([]), np.array([])

    if data.ndim == 1:
        data = data.reshape(1, -1)

    returns = data[:, 0]
    lengths = data[:, 1] if data.shape[1] > 1 else np.full_like(returns, 1.0)
    wall_times = data[:, 2] if data.shape[1] > 2 else np.arange(len(returns), dtype=np.float64)
    return returns, lengths, wall_times


def load_seed_episodes(log_dir: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seed_dir = log_dir / f"seed_{seed}"
    if not seed_dir.is_dir():
        return np.array([]), np.array([])

    monitor_files = list(seed_dir.glob("*.monitor.csv"))
    if not monitor_files and (seed_dir / "monitor.csv").is_file():
        monitor_files = [seed_dir / "monitor.csv"]

    episodes: list[tuple[float, float, float]] = []
    for monitor_file in sorted(monitor_files):
        returns, lengths, wall_times = _load_monitor_csv(monitor_file)
        for ret, length, wall_t in zip(returns, lengths, wall_times):
            episodes.append((float(wall_t), float(ret), float(length)))

    if not episodes:
        return np.array([]), np.array([])

    episodes.sort(key=lambda row: row[0])
    returns = np.array([row[1] for row in episodes], dtype=np.float64)
    lengths = np.array([row[2] for row in episodes], dtype=np.float64)
    return returns, lengths


def _load_evaluations(log_dir: Path, seeds: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        path = log_dir / f"seed_{seed}" / "evaluations.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        timesteps = np.asarray(data["timesteps"], dtype=np.float64).reshape(-1)
        results = np.asarray(data["results"], dtype=np.float64)
        if results.ndim > 1:
            results = np.nanmean(results, axis=1)
        else:
            results = results.reshape(-1)
        n = min(timesteps.size, results.size)
        if n:
            out[seed] = (timesteps[:n], results[:n])
    return out


def load_episodes(traj_dir: Path, seeds: list[int]) -> list[EpisodeData]:
    """Load eval_policy.py trajectory JSON into compact episode records."""
    episodes: list[EpisodeData] = []

    for seed in seeds:
        path = traj_dir / f"trajectories_seed{seed}.json"
        if not path.is_file():
            continue

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        for ep in payload:
            record = primary_episode_record(ep)
            metadata = dict(ep.get("metadata", {}))

            physical_actions = np.asarray(
                record.get("physical_actions", ep.get("physical_actions", [])),
                dtype=np.float64,
            )
            if physical_actions.size == 0:
                continue
            if physical_actions.ndim == 1:
                physical_actions = physical_actions.reshape(-1, 1)

            times = np.asarray(record.get("times", ep.get("times", [])), dtype=np.float64)
            n = min(len(times), physical_actions.shape[0])
            if n <= 0:
                continue

            times = times[:n]
            physical_actions = physical_actions[:n]

            rpm = omega_to_rpm(physical_actions[:, 0]) if physical_actions.shape[1] >= 1 else None
            ac = physical_actions[:, 1] if physical_actions.shape[1] >= 2 else None
            if rpm is None or ac is None:
                continue

            w_hist = record.get("w_sensor") or ep.get("w_sensor")
            w_sensor = None
            if w_hist:
                w_arr = np.asarray(w_hist, dtype=np.float64)
                if w_arr.ndim == 1:
                    w_arr = w_arr.reshape(-1, 1)
                w_sensor = w_arr[:n]

            process = record.get("process") or ep.get("process") or {}
            chip_mean_mm = None
            chip_max_mm = None
            if isinstance(process, dict):
                cm = process.get("mean_chip_mm")
                cx = process.get("max_chip_mm")
                if cm is not None:
                    cm_arr = np.asarray(cm, dtype=np.float64).reshape(-1)
                    chip_mean_mm = cm_arr[:n] if cm_arr.size >= n else None
                if cx is not None:
                    cx_arr = np.asarray(cx, dtype=np.float64).reshape(-1)
                    chip_max_mm = cx_arr[:n] if cx_arr.size >= n else None

            reason = record.get("termination_reason") or ep.get("termination_reason")
            pass_completed = bool(
                reason == "pass_completed"
                or record.get("pass_completed")
                or ep.get("pass_completed")
            )

            episodes.append(
                EpisodeData(
                    seed=int(ep.get("seed", seed)),
                    episode=int(ep.get("episode", 0)),
                    y0=record.get("y0", ep.get("y0")),
                    return_=float(record.get("return", ep.get("return", np.nan))),
                    termination_reason=reason,
                    pass_completed=pass_completed,
                    times=times,
                    rpm=rpm,
                    ac=ac,
                    w_sensor=w_sensor,
                    chip_mean_mm=chip_mean_mm,
                    chip_max_mm=chip_max_mm,
                    metadata=metadata,
                )
            )

    return episodes


def _nan_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.any(~np.isnan(values), axis=0)
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)
    if np.any(valid):
        mean[valid] = np.nanmean(values[:, valid], axis=0)
        std[valid] = np.nanstd(values[:, valid], axis=0)
    return mean, std, valid


def _time_grid(T: np.ndarray, max_t: int) -> np.ndarray:
    valid = np.any(~np.isnan(T), axis=0)
    grid = np.arange(max_t, dtype=np.float64)
    if np.any(valid):
        grid[valid] = np.nanmean(T[:, valid], axis=0)
    return grid


def resample_unit_interval(
    times: np.ndarray,
    values: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Linear resample values onto uniform pass progress in [0, 1]."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or times.size == 0:
        return np.full(n_bins, np.nan, dtype=np.float64)
    if values.size == 1:
        return np.full(n_bins, values[0], dtype=np.float64)

    n = min(times.size, values.size)
    times = times[:n]
    values = values[:n]

    t0, t1 = float(times[0]), float(times[-1])
    if t1 <= t0:
        return np.full(n_bins, values[-1], dtype=np.float64)

    tau = (times - t0) / (t1 - t0)
    grid = np.linspace(0.0, 1.0, n_bins)
    return np.interp(grid, tau, values)


def _run_label(ep: EpisodeData) -> str:
    y0_mm = ep.y0 * 1e3 if ep.y0 is not None else float("nan")
    if np.isfinite(y0_mm):
        return f"s{ep.seed} ep{ep.episode}\n$y_0$={y0_mm:.0f} mm"
    return f"s{ep.seed} ep{ep.episode}"


def load_stability_lobe_csv(path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (rpm, ap_stable_mm) boundary points from a compute_stability_lobe.py CSV.

    Returns None if the path is missing or the file does not have the expected
    columns; the caller treats this as "no lobe overlay available" rather than
    an error, since the overlay is optional context, not a required input.
    """
    if path is None or not Path(path).is_file():
        return None

    rpm_vals: list[float] = []
    ap_vals: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "rpm" not in reader.fieldnames or "ap_stable_mm" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                rpm_vals.append(float(row["rpm"]))
                ap_vals.append(float(row["ap_stable_mm"]))
            except (TypeError, ValueError):
                continue

    if not rpm_vals:
        return None

    rpm_arr = np.asarray(rpm_vals, dtype=np.float64)
    ap_arr = np.asarray(ap_vals, dtype=np.float64)
    order = np.argsort(rpm_arr)
    return rpm_arr[order], ap_arr[order]


def _resolve_lobe_csv(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for candidate in DEFAULT_LOBE_CSV_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _load_lobe_metadata(csv_path: Path | None) -> dict | None:
    """Load the metadata JSON that compute_stability_lobe.py saves alongside its CSV."""
    if csv_path is None:
        return None
    meta_path = csv_path.with_name(csv_path.stem + "_metadata.json")
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def check_lobe_consistency(lobe_meta: dict | None, episode_meta: dict) -> list[str]:
    """Return human-readable mismatches between the lobe CSV's plant config and the
    eval trajectories' plant config.

    A stability-lobe sweep run under different physical parameters (displacement
    limit, spindle-speed range, axial-depth range, cutter geometry, ...) than the
    evaluated policy is not a valid boundary for that policy's operating map, even
    though nothing about loading/plotting it will raise an error. This check turns
    a silent, misleading overlay into an explicit warning.
    """
    if lobe_meta is None:
        return []

    mismatches: list[str] = []

    def _cmp(label: str, lobe_val, ep_val, rel_tol: float = 0.02) -> None:
        if lobe_val is None or ep_val is None:
            return
        try:
            lobe_f, ep_f = float(lobe_val), float(ep_val)
        except (TypeError, ValueError):
            return
        if ep_f == 0.0:
            return
        if abs(lobe_f - ep_f) / max(abs(ep_f), 1e-12) > rel_tol:
            mismatches.append(f"{label}: lobe={lobe_f:g} vs eval-env={ep_f:g}")

    _cmp("w_limit_m", lobe_meta.get("w_limit_m"), episode_meta.get("w_limit_m") or episode_meta.get("w_limit"))
    _cmp("rpm_min", lobe_meta.get("rpm_min"), episode_meta.get("rpm_min"))
    _cmp("rpm_max", lobe_meta.get("rpm_max"), episode_meta.get("rpm_max"))
    lobe_ap_max = lobe_meta.get("plant_ap_max_mm", lobe_meta.get("ap_max_mm"))
    ep_ap_max = episode_meta.get("ap_max_mm")
    if ep_ap_max is None:
        phys_high = episode_meta.get("physical_action_high")
        if isinstance(phys_high, (list, tuple)) and len(phys_high) > 1:
            ep_ap_max = phys_high[1]
    _cmp("ap_max_mm", lobe_ap_max, ep_ap_max)
    _cmp("D_mm", lobe_meta.get("D_mm"), episode_meta.get("D_mm"))

    return mismatches


# ---------------------------------------------------------------------------
# Figure 1 — learning curve
# ---------------------------------------------------------------------------


def figure_learning_curve(
    log_dir: Path,
    seeds: list[int],
    out_dir: Path,
    *,
    smooth: int,
    formats: tuple[str, ...],
) -> None:
    data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        returns, lengths = load_seed_episodes(log_dir, seed)
        if returns.size:
            data[seed] = (returns, lengths)

    if not data:
        print(f"Skip fig01: no monitor data under {log_dir}")
        return

    max_steps = min(np.sum(lengths) for _, lengths in data.values())
    if max_steps <= 0:
        return

    step_grid = np.linspace(0, max_steps, num=250)
    curves = []
    for seed in sorted(data.keys()):
        returns, lengths = data[seed]
        steps = np.concatenate([[0.0], np.cumsum(lengths)])
        returns_ext = np.concatenate([[returns[0]], returns])
        curves.append(np.interp(step_grid, steps, returns_ext))

    matrix = np.asarray(curves, dtype=np.float64)
    mean_r = np.mean(matrix, axis=0)
    std_r = np.std(matrix, axis=0)

    if smooth > 1:
        k = min(smooth, max(len(mean_r) // 4, 1))
        kernel = np.ones(k) / k
        mean_r = np.convolve(mean_r, kernel, mode="same")
        std_r = np.convolve(std_r, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.fill_between(
        step_grid,
        mean_r - MC_BAND_STD_MULT * std_r,
        mean_r + MC_BAND_STD_MULT * std_r,
        color=C_FILL,
        alpha=0.22,
        linewidth=0,
        label=f"Train ±{MC_BAND_STD_MULT:.0f}σ",
    )
    ax.plot(step_grid, mean_r, color=C_PRIMARY, lw=1.4, label="Train mean")

    eval_data = _load_evaluations(log_dir, seeds)
    if eval_data:
        eval_steps: list[float] = []
        eval_means: list[float] = []
        for seed in sorted(eval_data.keys()):
            ts, res = eval_data[seed]
            eval_steps.extend(ts.tolist())
            eval_means.extend(res.tolist())
        ax.scatter(
            eval_steps,
            eval_means,
            s=16,
            c=C_ACCENT,
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
            label="Eval checkpoint",
        )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("PPO training progress")
    ax.grid(True, alpha=C_GRID[3], color=C_GRID[:3])
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig01_learning_curve", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — closed-loop time series
# ---------------------------------------------------------------------------


def figure_closed_loop_response(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
) -> None:
    if not episodes:
        print("Skip fig02: no trajectory episodes")
        return

    metadata = episodes[0].metadata
    w_limit = metadata.get("w_limit")

    max_t = max(len(ep.times) for ep in episodes)
    T = np.full((len(episodes), max_t), np.nan, dtype=np.float64)
    rpm_series = np.full((len(episodes), max_t), np.nan, dtype=np.float64)
    ac_series = np.full((len(episodes), max_t), np.nan, dtype=np.float64)
    peak_w_um = np.full((len(episodes), max_t), np.nan, dtype=np.float64)

    for i, ep in enumerate(episodes):
        n = len(ep.times)
        T[i, :n] = ep.times
        rpm_series[i, :n] = ep.rpm
        ac_series[i, :n] = ep.ac
        if ep.w_sensor is not None and ep.w_sensor.size:
            peak_w_um[i, :n] = np.max(np.abs(ep.w_sensor), axis=1) * 1e6

    t_grid = _time_grid(T, max_t)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.4), sharex=True)
    panels = [
        (rpm_series, "Spindle speed (rpm)", None),
        (ac_series, r"Axial depth of cut $a_c$ (mm)", None),
        (peak_w_um, r"Peak $|w|$ across sensors ($\mu$m)", w_limit),
    ]

    for ax, (series, ylabel, limit_m), label in zip(axes, panels, ("(a)", "(b)", "(c)")):
        mean_y, std_y, valid = _nan_mean_std(series)
        band = MC_BAND_STD_MULT * std_y
        ax.fill_between(
            t_grid[valid],
            mean_y[valid] - band[valid],
            mean_y[valid] + band[valid],
            color=C_FILL,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(t_grid[valid], mean_y[valid], color=C_PRIMARY, lw=1.3)

        if limit_m is not None and np.isfinite(limit_m):
            lim_um = float(limit_m) * 1e6
            ax.axhline(+lim_um, color=C_ACCENT, ls="--", lw=0.9)
            ax.axhline(-lim_um, color=C_ACCENT, ls="--", lw=0.9)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=C_GRID[3], color=C_GRID[:3])
        _panel_label(ax, label)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Closed-loop policy response ({RPM_MIN:.0f}–{RPM_MAX:.0f} rpm)",
        y=1.01,
        fontsize=9,
    )
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig02_closed_loop_response", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — actuator heatmaps across runs
# ---------------------------------------------------------------------------


def figure_action_run_heatmap(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    n_bins: int,
    formats: tuple[str, ...],
) -> None:
    if not episodes:
        print("Skip fig03: no trajectory episodes")
        return

    # Sort by seed then episode for readable row order.
    episodes = sorted(episodes, key=lambda e: (e.seed, e.episode))
    rpm_mat = np.vstack([resample_unit_interval(ep.times, ep.rpm, n_bins) for ep in episodes])
    ac_mat = np.vstack([resample_unit_interval(ep.times, ep.ac, n_bins) for ep in episodes])

    progress = np.linspace(0.0, 1.0, n_bins)
    labels = [_run_label(ep) for ep in episodes]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, max(2.8, 0.22 * len(episodes) + 1.2)), sharey=True)

    for ax, mat, title, cbar_label in zip(
        axes,
        (rpm_mat, ac_mat),
        ("Spindle speed (rpm)", r"Depth of cut $a_c$ (mm)"),
        ("rpm", "mm"),
    ):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            extent=[progress[0], progress[1], len(episodes) - 0.5, -0.5],
            cmap=ACTION_CMAP,
            interpolation="nearest",
        )
        ax.set_xlabel("Normalized pass progress")
        ax.set_title(title)
        ax.set_yticks(np.arange(len(episodes)))
        ax.set_yticklabels(labels, fontsize=6.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=8)

    axes[0].set_ylabel("Evaluation run")
    fig.suptitle("Actuator commands across evaluation rollouts", y=1.02, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig03_action_run_heatmap", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — operating map (rpm vs a_c)
# ---------------------------------------------------------------------------


def figure_operating_map(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
    lobe: tuple[np.ndarray, np.ndarray] | None = None,
    lobe_consistent: bool = True,
) -> None:
    if not episodes:
        print("Skip fig04: no trajectory episodes")
        return

    metadata = episodes[0].metadata
    rpm_all = np.concatenate([ep.rpm for ep in episodes])
    ac_all = np.concatenate([ep.ac for ep in episodes])

    rpm_lo = float(metadata.get("rpm_min", RPM_MIN))
    rpm_hi = float(metadata.get("rpm_max", RPM_MAX))
    ac_lo, ac_hi = 0.0, float(metadata.get("physical_action_high", [0, 20.0])[1])
    # Widen the depth-of-cut axis if the (optional) stability-lobe boundary
    # extends past the policy's own action range, so the overlay is not
    # clipped out of view.
    if lobe is not None and lobe[1].size:
        ac_hi = max(ac_hi, float(np.nanmax(lobe[1])) * 1.05)

    fig = plt.figure(figsize=(3.6, 3.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    h, xedges, yedges = np.histogram2d(
        rpm_all,
        ac_all,
        bins=[48, 36],
        range=[[rpm_lo, rpm_hi], [ac_lo, ac_hi]],
        density=True,
    )
    h = h.T
    pcm = ax_main.pcolormesh(
        xedges,
        yedges,
        h,
        cmap=ACTION_CMAP,
        shading="auto",
        rasterized=True,
    )
    ax_main.set_xlabel("Spindle speed (rpm)")
    ax_main.set_ylabel(r"Depth of cut $a_c$ (mm)")
    ax_main.set_xlim(rpm_lo, rpm_hi)
    ax_main.set_ylim(ac_lo, ac_hi)

    if lobe is not None and lobe[0].size >= 2:
        lobe_rpm, lobe_ap = lobe
        # Only draw/shade within the current rpm axis range.
        in_range = (lobe_rpm >= rpm_lo) & (lobe_rpm <= rpm_hi)
        if np.count_nonzero(in_range) >= 2:
            lr, la = lobe_rpm[in_range], lobe_ap[in_range]
            lobe_label = "Predicted stability boundary (no control)"
            lobe_color = C_ACCENT
            if not lobe_consistent:
                lobe_label += " [param mismatch!]"
            ax_main.plot(
                lr, la, color=lobe_color, lw=1.4, ls="--", zorder=6,
                label=lobe_label,
            )
            ax_main.fill_between(
                lr, la, ac_hi, color=lobe_color, alpha=0.10, zorder=1,
                label="Predicted unstable region",
            )
            ax_main.legend(loc="upper right", fontsize=6, frameon=True, framealpha=0.9)
            if not lobe_consistent:
                ax_main.text(
                    0.02, 0.02,
                    "Caution: lobe computed with\ndifferent plant parameters",
                    transform=ax_main.transAxes,
                    fontsize=6, color=C_ACCENT, va="bottom", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=C_ACCENT, linewidth=0.6),
                )

    ax_top.hist(rpm_all, bins=48, range=(rpm_lo, rpm_hi), color=C_PRIMARY, alpha=0.85, density=True)
    ax_right.hist(
        ac_all,
        bins=36,
        range=(ac_lo, ac_hi),
        orientation="horizontal",
        color=C_PRIMARY,
        alpha=0.85,
        density=True,
    )

    for ax in (ax_top, ax_right):
        ax.axis("off")

    cbar = fig.colorbar(pcm, ax=ax_main, fraction=0.046, pad=0.08)
    cbar.set_label("Empirical density", fontsize=8)

    title = "Policy operating region (all eval steps)"
    if lobe is not None:
        title += " vs. predicted stability boundary"
    fig.suptitle(title, y=0.98, fontsize=9)
    _save_figure(fig, out_dir, "fig04_operating_map", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — vibration field (sensor x pass progress)
# ---------------------------------------------------------------------------


def figure_vibration_field(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    n_bins: int,
    formats: tuple[str, ...],
) -> None:
    episodes_with_w = [ep for ep in episodes if ep.w_sensor is not None and ep.w_sensor.size]
    if not episodes_with_w:
        print("Skip fig05: no w_sensor histories in trajectories")
        return

    metadata = episodes_with_w[0].metadata
    sensor_points = np.asarray(metadata.get("sensor_points", []), dtype=np.float64)
    n_sensors = int(metadata.get("n_sensors", episodes_with_w[0].w_sensor.shape[1]))

    if sensor_points.size >= 2 * n_sensors:
        x_vals = sensor_points[:n_sensors, 0]
        y_vals = sensor_points[:n_sensors, 1]
        # Default sensor layout varies in y at (near-)fixed x (both default
        # sensors sit at x=0.83, y=0.20/0.83). A heatmap axis built from the
        # coordinate that does not vary across sensors collapses to a single
        # row (matplotlib's "identical low and high ylims" warning) and is
        # uninformative regardless of how much data is plotted. Pick whichever
        # coordinate actually varies; fall back to a sensor index if neither
        # does (e.g. a single sensor).
        if np.ptp(y_vals) > np.ptp(x_vals):
            x_pos = y_vals
            axis_label = "Sensor $y$ position (m)"
        elif np.ptp(x_vals) > 0:
            x_pos = x_vals
            axis_label = "Sensor $x$ position (m)"
        else:
            x_pos = np.arange(n_sensors, dtype=np.float64)
            axis_label = "Sensor index"
    else:
        x_pos = np.linspace(0.1, 0.9, n_sensors)
        axis_label = "Sensor $x$ position (m)"

    field = np.full((n_sensors, n_bins), np.nan, dtype=np.float64)
    counts = np.zeros((n_sensors, n_bins), dtype=np.float64)

    for ep in episodes_with_w:
        w_um = np.abs(ep.w_sensor) * 1e6
        for s in range(min(n_sensors, w_um.shape[1])):
            resampled = resample_unit_interval(ep.times, w_um[:, s], n_bins)
            valid = np.isfinite(resampled)
            field[s, valid] = np.nan_to_num(field[s, valid], nan=0.0) + resampled[valid]
            counts[s, valid] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(counts > 0, field / counts, np.nan)

    progress = np.linspace(0.0, 1.0, n_bins)
    y_lo, y_hi = float(np.min(x_pos)), float(np.max(x_pos))
    if y_hi <= y_lo:
        # Single sensor or genuinely coincident sensor coordinates: pad a
        # small margin so imshow does not warn/collapse on identical limits.
        pad = 0.5 if x_pos.size <= 1 else max(abs(y_lo) * 0.05, 1e-3)
        y_lo, y_hi = y_lo - pad, y_hi + pad

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    im = ax.imshow(
        field,
        aspect="auto",
        origin="lower",
        extent=[progress[0], progress[1], y_lo, y_hi],
        cmap=VIB_CMAP,
        interpolation="bilinear",
    )

    w_limit = metadata.get("w_limit")
    if w_limit is not None and np.isfinite(w_limit):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.axhline(float(w_limit) * 1e6, color="white", lw=1.2, ls="--")
        cbar.set_label(r"$|w|$ ($\mu$m)", fontsize=8)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|w|$ ($\mu$m)", fontsize=8)

    ax.set_xlabel("Normalized pass progress")
    ax.set_ylabel(axis_label)
    ax.set_title("Mean vibration field across evaluation runs")
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig05_vibration_field", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — per-sensor vibration ensemble (mean +/- std across episodes)
# ---------------------------------------------------------------------------


def figure_sensor_ensemble(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    n_bins: int,
    formats: tuple[str, ...],
) -> None:
    """Per-sensor mean +/- std |w| over normalized pass progress.

    fig05 shows the same underlying data as a 2-row heatmap, which is
    hard to read quantitatively with only a handful of discrete sensors.
    This figure gives each sensor its own labeled line + uncertainty band,
    which is the more informative view for a small, fixed sensor count.
    """
    episodes_with_w = [ep for ep in episodes if ep.w_sensor is not None and ep.w_sensor.size]
    if not episodes_with_w:
        print("Skip fig07: no w_sensor histories in trajectories")
        return

    metadata = episodes_with_w[0].metadata
    n_sensors = int(metadata.get("n_sensors", episodes_with_w[0].w_sensor.shape[1]))
    sensor_points = np.asarray(metadata.get("sensor_points", []), dtype=np.float64)
    w_limit = metadata.get("w_limit")

    progress = np.linspace(0.0, 1.0, n_bins)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    for s in range(n_sensors):
        rows = []
        for ep in episodes_with_w:
            if ep.w_sensor.shape[1] <= s:
                continue
            w_um = np.abs(ep.w_sensor[:, s]) * 1e6
            rows.append(resample_unit_interval(ep.times, w_um, n_bins))
        if not rows:
            continue
        mat = np.vstack(rows)
        mean_y, std_y, valid = _nan_mean_std(mat)
        color = SENSOR_COLORS[s % len(SENSOR_COLORS)]

        if sensor_points.size >= 2 * n_sensors:
            xs, ys = sensor_points[s, 0], sensor_points[s, 1]
            label = f"Sensor {s + 1} (x={xs:.2f}, y={ys:.2f} m)"
        else:
            label = f"Sensor {s + 1}"

        band = MC_BAND_STD_MULT * std_y
        ax.fill_between(
            progress[valid], (mean_y - band)[valid], (mean_y + band)[valid],
            color=color, alpha=0.20, linewidth=0,
        )
        ax.plot(progress[valid], mean_y[valid], color=color, lw=1.4, label=label)

    if w_limit is not None and np.isfinite(w_limit):
        ax.axhline(float(w_limit) * 1e6, color=C_ACCENT, ls="--", lw=0.9, label="Displacement limit")

    ax.set_xlabel("Normalized pass progress")
    ax.set_ylabel(r"$|w|$ ($\mu$m)")
    ax.set_title(f"Per-sensor vibration ensemble (mean $\\pm{MC_BAND_STD_MULT:.0f}\\sigma$, n={len(episodes_with_w)})")
    ax.grid(True, alpha=C_GRID[3], color=C_GRID[:3])
    ax.legend(loc="upper left", fontsize=6.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig07_sensor_ensemble", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — cutting-process ensemble (engaged chip thickness)
# ---------------------------------------------------------------------------


def figure_process_ensemble(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    n_bins: int,
    formats: tuple[str, ...],
) -> None:
    """Mean +/- std engaged chip thickness over normalized pass progress.

    Chip thickness is the direct physical driver of both the cutting force
    (hence vibration) and the material-removal rate the reward's
    productivity term is meant to proxy, so it ties the vibration and
    productivity stories together in one place.
    """
    episodes_with_chip = [ep for ep in episodes if ep.chip_mean_mm is not None and ep.chip_mean_mm.size]
    if not episodes_with_chip:
        print("Skip fig08: no chip-thickness process data in trajectories")
        return

    progress = np.linspace(0.0, 1.0, n_bins)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))

    series_specs = [
        ("chip_mean_mm", "Mean engaged chip thickness", C_PRIMARY),
        ("chip_max_mm", "Max engaged chip thickness", C_ACCENT),
    ]
    for attr, label, color in series_specs:
        rows = [
            resample_unit_interval(ep.times, getattr(ep, attr), n_bins)
            for ep in episodes_with_chip
            if getattr(ep, attr) is not None
        ]
        if not rows:
            continue
        mat = np.vstack(rows)
        mean_y, std_y, valid = _nan_mean_std(mat)
        band = MC_BAND_STD_MULT * std_y
        ax.fill_between(
            progress[valid], (mean_y - band)[valid], (mean_y + band)[valid],
            color=color, alpha=0.18, linewidth=0,
        )
        ax.plot(progress[valid], mean_y[valid], color=color, lw=1.4, label=label)

    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("Normalized pass progress")
    ax.set_ylabel("Chip thickness (mm)")
    ax.set_title(f"Cutting-process ensemble (mean $\\pm{MC_BAND_STD_MULT:.0f}\\sigma$, n={len(episodes_with_chip)})")
    ax.grid(True, alpha=C_GRID[3], color=C_GRID[:3])
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig08_process_ensemble", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — evaluation summary
# ---------------------------------------------------------------------------


def figure_eval_summary(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
) -> None:
    if not episodes:
        print("Skip fig06: no trajectory episodes")
        return

    seeds = sorted({ep.seed for ep in episodes})
    seed_returns: dict[int, list[float]] = {s: [] for s in seeds}
    seed_pass: dict[int, list[bool]] = {s: [] for s in seeds}

    for ep in episodes:
        seed_returns[ep.seed].append(ep.return_)
        seed_pass[ep.seed].append(ep.pass_completed)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel (a): return distribution per seed.
    ax = axes[0]
    positions = np.arange(len(seeds))
    box_data = [seed_returns[s] for s in seeds]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color=C_ACCENT, linewidth=1.2),
        boxprops=dict(facecolor=C_FILL, alpha=0.35, edgecolor=C_PRIMARY),
        whiskerprops=dict(color=C_PRIMARY, linewidth=0.9),
        capprops=dict(color=C_PRIMARY, linewidth=0.9),
    )
    _ = bp  # patch_artist boxplot reference

    for i, s in enumerate(seeds):
        jitter = 0.06 * (np.random.default_rng(s).random(len(seed_returns[s])) - 0.5)
        ax.scatter(
            np.full(len(seed_returns[s]), positions[i]) + jitter,
            seed_returns[s],
            s=12,
            c=C_PRIMARY,
            alpha=0.65,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Episode return")
    ax.set_title("Return distribution")
    ax.grid(True, axis="y", alpha=C_GRID[3], color=C_GRID[:3])
    _panel_label(ax, "(a)")

    # Panel (b): pass completion rate.
    ax = axes[1]
    pass_rates = [100.0 * np.mean(seed_pass[s]) for s in seeds]
    bars = ax.bar(positions, pass_rates, width=0.55, color=C_PRIMARY, alpha=0.85, edgecolor="white")
    ax.set_ylim(0, 105)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Pass completion (%)")
    ax.set_title("Task success rate")
    ax.grid(True, axis="y", alpha=C_GRID[3], color=C_GRID[:3])
    _panel_label(ax, "(b)")

    for bar, rate in zip(bars, pass_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Closed-loop evaluation summary", y=1.02, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig06_eval_summary", formats)
    plt.close(fig)


def write_metrics_summary(episodes: list[EpisodeData], out_dir: Path) -> None:
    if not episodes:
        print("Skip metrics summary: no trajectory episodes")
        return

    metrics: dict = {"seeds": {}, "aggregate": {}}
    all_returns: list[float] = []
    pass_complete = 0

    by_seed: dict[int, list[EpisodeData]] = {}
    for ep in episodes:
        by_seed.setdefault(ep.seed, []).append(ep)

    for seed, eps in sorted(by_seed.items()):
        returns = [e.return_ for e in eps]
        n_pass = sum(1 for e in eps if e.pass_completed)
        all_returns.extend(returns)
        pass_complete += n_pass
        metrics["seeds"][str(seed)] = {
            "n_episodes": len(eps),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "pass_completion_rate": n_pass / len(eps),
        }

    metrics["aggregate"] = {
        "n_episodes": len(episodes),
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "pass_completion_rate": pass_complete / len(episodes),
    }

    path = out_dir / "paper_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from training logs and eval trajectories"
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument(
        "--out-dir",
        default=str(PAPER_OUT_DIR),
        help=f"Output directory (default: {PAPER_OUT_DIR})",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds to include (default: auto-discover)",
    )
    parser.add_argument("--smooth", type=int, default=15)
    parser.add_argument(
        "--time-bins",
        type=int,
        default=120,
        help="Bins for normalized pass-progress heatmaps",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated figure formats (e.g. pdf,png)",
    )
    parser.add_argument(
        "--figures",
        default="all",
        help="Comma-separated figure ids: 1,2,3,4,5,6,7,8 or 'all'",
    )
    parser.add_argument(
        "--lobe-csv",
        default=None,
        help=(
            "Path to a stability_lobe*.csv from scripts/compute_stability_lobe.py, "
            "overlaid as a predicted stability boundary on fig04. If omitted, "
            f"auto-discovers one of: {[str(p) for p in DEFAULT_LOBE_CSV_CANDIDATES]}"
        ),
    )

    args = parser.parse_args()
    apply_paper_style()

    log_dir = Path(args.log_dir).resolve()
    traj_dir = Path(args.traj_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    formats = tuple(f.strip().lstrip(".") for f in args.formats.split(",") if f.strip())

    if args.figures.strip().lower() == "all":
        figure_ids = {1, 2, 3, 4, 5, 6, 7, 8}
    else:
        figure_ids = {int(x.strip()) for x in args.figures.split(",") if x.strip()}

    seeds = _resolve_seeds(args.seeds, log_dir, traj_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Paper figures -> {out_dir}")
    print(f"Seeds          : {seeds}")
    print(f"Trajectory dir : {traj_dir}")

    episodes = load_episodes(traj_dir, seeds)
    if episodes:
        print(f"Loaded {len(episodes)} evaluation episode(s) from trajectory JSON")
    else:
        print("Warning: no trajectory JSON found — fig02-fig08 will be skipped")

    lobe_path = _resolve_lobe_csv(args.lobe_csv)
    lobe = load_stability_lobe_csv(lobe_path)
    lobe_consistent = True
    if lobe_path is not None:
        print(f"Stability lobe : {lobe_path} ({'loaded' if lobe is not None else 'failed to parse'})")
        if lobe is not None and episodes:
            lobe_meta = _load_lobe_metadata(lobe_path)
            mismatches = check_lobe_consistency(lobe_meta, episodes[0].metadata)
            if mismatches:
                lobe_consistent = False
                print(
                    "WARNING: stability-lobe CSV was computed with different plant "
                    "parameters than the evaluated environment. The overlay in fig04 "
                    "is NOT a valid stability boundary for this policy until you "
                    "regenerate the lobe with matching settings:"
                )
                for m in mismatches:
                    print(f"    - {m}")
    else:
        print("Stability lobe : none found (fig04 will show the operating map without a boundary overlay)")

    if 1 in figure_ids:
        figure_learning_curve(log_dir, seeds, out_dir, smooth=args.smooth, formats=formats)
    if 2 in figure_ids:
        figure_closed_loop_response(episodes, out_dir, formats=formats)
    if 3 in figure_ids:
        figure_action_run_heatmap(episodes, out_dir, n_bins=args.time_bins, formats=formats)
    if 4 in figure_ids:
        figure_operating_map(episodes, out_dir, formats=formats, lobe=lobe, lobe_consistent=lobe_consistent)
    if 5 in figure_ids:
        figure_vibration_field(episodes, out_dir, n_bins=args.time_bins, formats=formats)
    if 6 in figure_ids:
        figure_eval_summary(episodes, out_dir, formats=formats)
    if 7 in figure_ids:
        figure_sensor_ensemble(episodes, out_dir, n_bins=args.time_bins, formats=formats)
    if 8 in figure_ids:
        figure_process_ensemble(episodes, out_dir, n_bins=args.time_bins, formats=formats)

    write_metrics_summary(episodes, out_dir)


if __name__ == "__main__":
    main()