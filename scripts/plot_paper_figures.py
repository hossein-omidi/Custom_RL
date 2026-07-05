"""Publication-quality figures from training logs and eval_policy.py trajectories.

Reads only generated artifacts (SB3 monitor CSV, evaluations.npz, trajectory JSON).
Does not modify training or evaluation scripts. Run eval_policy.py first so
trajectory JSON exists before generating fig02-fig09.

Figures (paper-oriented, no duplicates of casual plot_results.py views):
  fig01  Training return vs environment steps (+ eval checkpoints)
  fig02  Closed-loop control: spindle speed, depth of cut, peak vibration
  fig03  Actuator trajectories across evaluation runs (normalized pass time)
  fig04  Operating map: joint density of (rpm, a_p) with marginals
  fig05  Vibration field: sensor location vs normalized pass progress
  fig06  Evaluation summary: return distribution and pass completion by seed
  fig07  Reward decomposition vs normalized pass progress (mean +/- std)
  fig08  Cutting force vs normalized pass progress (mean +/- std)
  fig09  Vibration robustness vs milling pass line y=a across episodes
"""

from __future__ import annotations

import argparse
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

# Plant termination reason strings that indicate the milling pass completed
# without triggering an instability/safety-margin termination. eval_policy.py
# already resolves "pass_completed" against this same set and saves the
# resulting bool, so this is only needed as a fallback for older trajectory
# JSON that predates that field.
PASS_COMPLETED_REASONS = {"pass_completed_90percent", "pass_completed"}

# Journal-style palette (colorblind-friendly).
C_PRIMARY = "#1F4E79"
C_ACCENT = "#C44E52"
C_FILL = "#4C72B0"
C_GRID = (0, 0, 0, 0.22)

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
    y_line_m: float | None
    return_: float
    termination_reason: str | None
    pass_completed: bool
    times: np.ndarray
    rpm: np.ndarray
    ap: np.ndarray
    w_sensor: np.ndarray | None  # (T, n_sensors)
    reward_terms: list[dict] | None
    force_N: np.ndarray | None  # (T, 4): Fx, Fy, Fz, |F|
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
            ap = physical_actions[:, 1] if physical_actions.shape[1] >= 2 else None
            if rpm is None or ap is None:
                continue

            w_hist = record.get("w_sensor") or ep.get("w_sensor")
            w_sensor = None
            if w_hist:
                w_arr = np.asarray(w_hist, dtype=np.float64)
                if w_arr.ndim == 1:
                    w_arr = w_arr.reshape(-1, 1)
                w_sensor = w_arr[:n]

            reward_terms_hist = record.get("reward_terms") or ep.get("reward_terms")
            reward_terms = reward_terms_hist[:n] if reward_terms_hist else None

            process = record.get("process") or ep.get("process") or {}
            force_N = None
            if all(k in process for k in ("Fx_N", "Fy_N", "Fz_N")):
                fx = np.asarray(process["Fx_N"], dtype=np.float64)
                fy = np.asarray(process["Fy_N"], dtype=np.float64)
                fz = np.asarray(process["Fz_N"], dtype=np.float64)
                fmag = np.asarray(
                    process.get("F_mag_N", np.sqrt(fx**2 + fy**2 + fz**2)),
                    dtype=np.float64,
                )
                m = min(n, fx.size, fy.size, fz.size, fmag.size)
                if m > 0:
                    force_N = np.stack([fx[:m], fy[:m], fz[:m], fmag[:m]], axis=1)

            reason = record.get("termination_reason") or ep.get("termination_reason")
            pass_completed = bool(
                record.get("pass_completed")
                or ep.get("pass_completed")
                or reason in PASS_COMPLETED_REASONS
            )

            episodes.append(
                EpisodeData(
                    seed=int(ep.get("seed", seed)),
                    episode=int(ep.get("episode", 0)),
                    y_line_m=record.get("y_line_m", ep.get("y_line_m")),
                    return_=float(record.get("return", ep.get("return", np.nan))),
                    termination_reason=reason,
                    pass_completed=pass_completed,
                    times=times,
                    rpm=rpm,
                    ap=ap,
                    w_sensor=w_sensor,
                    reward_terms=reward_terms,
                    force_N=force_N,
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
    y_mm = ep.y_line_m * 1e3 if ep.y_line_m is not None else float("nan")
    if np.isfinite(y_mm):
        return f"s{ep.seed} ep{ep.episode}\n$y$={y_mm:.0f} mm"
    return f"s{ep.seed} ep{ep.episode}"


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
    ap_series = np.full((len(episodes), max_t), np.nan, dtype=np.float64)
    peak_w_um = np.full((len(episodes), max_t), np.nan, dtype=np.float64)

    for i, ep in enumerate(episodes):
        n = len(ep.times)
        T[i, :n] = ep.times
        rpm_series[i, :n] = ep.rpm
        ap_series[i, :n] = ep.ap
        if ep.w_sensor is not None and ep.w_sensor.size:
            peak_w_um[i, :n] = np.max(np.abs(ep.w_sensor), axis=1) * 1e6

    t_grid = _time_grid(T, max_t)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.4), sharex=True)
    panels = [
        (rpm_series, "Spindle speed (rpm)", None),
        (ap_series, r"Axial depth of cut $a_p$ (mm)", None),
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
    ap_mat = np.vstack([resample_unit_interval(ep.times, ep.ap, n_bins) for ep in episodes])

    progress = np.linspace(0.0, 1.0, n_bins)
    labels = [_run_label(ep) for ep in episodes]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, max(2.8, 0.22 * len(episodes) + 1.2)), sharey=True)

    for ax, mat, title, cbar_label in zip(
        axes,
        (rpm_mat, ap_mat),
        ("Spindle speed (rpm)", r"Axial depth of cut $a_p$ (mm)"),
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
) -> None:
    if not episodes:
        print("Skip fig04: no trajectory episodes")
        return

    metadata = episodes[0].metadata
    rpm_all = np.concatenate([ep.rpm for ep in episodes])
    ap_all = np.concatenate([ep.ap for ep in episodes])

    rpm_lo = float(metadata.get("rpm_min", RPM_MIN))
    rpm_hi = float(metadata.get("rpm_max", RPM_MAX))
    ap_lo = float(metadata.get("ap_min_mm", 0.0))
    ap_hi = float(metadata.get("ap_max_mm", 20.0))

    fig = plt.figure(figsize=(3.6, 3.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    h, xedges, yedges = np.histogram2d(
        rpm_all,
        ap_all,
        bins=[48, 36],
        range=[[rpm_lo, rpm_hi], [ap_lo, ap_hi]],
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
    ax_main.set_ylabel(r"Axial depth of cut $a_p$ (mm)")
    ax_main.set_xlim(rpm_lo, rpm_hi)
    ax_main.set_ylim(ap_lo, ap_hi)

    ax_top.hist(rpm_all, bins=48, range=(rpm_lo, rpm_hi), color=C_PRIMARY, alpha=0.85, density=True)
    ax_right.hist(
        ap_all,
        bins=36,
        range=(ap_lo, ap_hi),
        orientation="horizontal",
        color=C_PRIMARY,
        alpha=0.85,
        density=True,
    )

    for ax in (ax_top, ax_right):
        ax.axis("off")

    cbar = fig.colorbar(pcm, ax=ax_main, fraction=0.046, pad=0.08)
    cbar.set_label("Empirical density", fontsize=8)

    fig.suptitle("Policy operating region (all eval steps)", y=0.98, fontsize=9)
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
        x_pos = sensor_points[:n_sensors, 0]
    else:
        x_pos = np.linspace(0.1, 0.9, n_sensors)

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
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    im = ax.imshow(
        field,
        aspect="auto",
        origin="lower",
        extent=[progress[0], progress[1], float(np.min(x_pos)), float(np.max(x_pos))],
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
    ax.set_ylabel("Sensor $x$ position (m)")
    ax.set_title("Mean vibration field across evaluation runs")
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig05_vibration_field", formats)
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


# ---------------------------------------------------------------------------
# Figure 7 — reward decomposition vs pass progress
# ---------------------------------------------------------------------------

REWARD_TERM_SPECS = (
    ("productivity", "Productivity", C_PRIMARY),
    ("vibration_w_cost", "Displacement cost", C_ACCENT),
    ("vibration_wdot_cost", "Velocity cost", "#55A868"),
    ("omega_cost", "Spindle-speed cost", "#8172B2"),
)


def figure_reward_decomposition(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    n_bins: int,
    formats: tuple[str, ...],
) -> None:
    episodes_with_terms = [ep for ep in episodes if ep.reward_terms]
    if not episodes_with_terms:
        print("Skip fig07: no reward_terms in trajectories (re-run eval_policy.py)")
        return

    progress = np.linspace(0.0, 1.0, n_bins)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    for key, label, color in REWARD_TERM_SPECS:
        series = []
        for ep in episodes_with_terms:
            values = np.array(
                [float(step.get(key, np.nan)) for step in ep.reward_terms],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            series.append(resample_unit_interval(ep.times, values, n_bins))
        if not series:
            continue
        mat = np.vstack(series)
        mean_y, std_y, valid = _nan_mean_std(mat)
        band = MC_BAND_STD_MULT * std_y
        ax.plot(progress[valid], mean_y[valid], color=color, lw=1.3, label=label)
        ax.fill_between(
            progress[valid],
            mean_y[valid] - band[valid],
            mean_y[valid] + band[valid],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("Normalized pass progress")
    ax.set_ylabel("Reward term value")
    ax.set_title("Dense reward decomposition")
    ax.grid(True, alpha=C_GRID[3], color=C_GRID[:3])
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig07_reward_decomposition", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — cutting force vs pass progress
# ---------------------------------------------------------------------------


def figure_cutting_force(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    n_bins: int,
    formats: tuple[str, ...],
) -> None:
    episodes_with_force = [ep for ep in episodes if ep.force_N is not None and ep.force_N.size]
    if not episodes_with_force:
        print("Skip fig08: no force data in trajectories (re-run eval_policy.py)")
        return

    progress = np.linspace(0.0, 1.0, n_bins)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True)

    specs = [
        (2, r"Axial cutting force $F_z$ (N)", axes[0]),
        (3, r"Resultant force $|F|$ (N)", axes[1]),
    ]
    for col, ylabel, ax in specs:
        series = [
            resample_unit_interval(ep.times, ep.force_N[:, col], n_bins)
            for ep in episodes_with_force
        ]
        mat = np.vstack(series)
        mean_y, std_y, valid = _nan_mean_std(mat)
        band = MC_BAND_STD_MULT * std_y
        ax.fill_between(
            progress[valid],
            mean_y[valid] - band[valid],
            mean_y[valid] + band[valid],
            color=C_FILL,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(progress[valid], mean_y[valid], color=C_PRIMARY, lw=1.3)
        ax.set_xlabel("Normalized pass progress")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=C_GRID[3], color=C_GRID[:3])

    fig.suptitle("Face-milling cutting force across evaluation rollouts", y=1.02, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig08_cutting_force", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9 — vibration robustness across milling pass lines y=a
# ---------------------------------------------------------------------------


def figure_pass_line_robustness(
    episodes: list[EpisodeData],
    out_dir: Path,
    *,
    formats: tuple[str, ...],
) -> None:
    episodes_with_y = [
        ep for ep in episodes if ep.y_line_m is not None and ep.w_sensor is not None and ep.w_sensor.size
    ]
    if not episodes_with_y:
        print("Skip fig09: no y_line_m/w_sensor data in trajectories")
        return

    y_mm = np.array([float(ep.y_line_m) * 1e3 for ep in episodes_with_y])
    peak_w_um = np.array(
        [float(np.max(np.abs(ep.w_sensor))) * 1e6 for ep in episodes_with_y]
    )
    completed = np.array([bool(ep.pass_completed) for ep in episodes_with_y])

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.scatter(
        y_mm[completed],
        peak_w_um[completed],
        s=18,
        c=C_PRIMARY,
        alpha=0.75,
        label="Pass completed",
        edgecolors="white",
        linewidths=0.3,
    )
    ax.scatter(
        y_mm[~completed],
        peak_w_um[~completed],
        s=18,
        c=C_ACCENT,
        alpha=0.75,
        label="Terminated early",
        marker="x",
    )

    if y_mm.size >= 3:
        order = np.argsort(y_mm)
        n_bins = max(min(10, y_mm.size // 2), 2)
        edges = np.linspace(y_mm.min(), y_mm.max(), n_bins + 1)
        centers, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (y_mm >= lo) & (y_mm <= hi)
            if np.any(mask):
                centers.append(0.5 * (lo + hi))
                means.append(np.mean(peak_w_um[mask]))
        if centers:
            ax.plot(centers, means, color="black", lw=1.2, ls="--", alpha=0.6, label="Binned mean")

    metadata = episodes_with_y[0].metadata
    w_limit = metadata.get("w_limit")
    if w_limit is not None and np.isfinite(w_limit):
        ax.axhline(float(w_limit) * 1e6, color=C_ACCENT, ls=":", lw=1.0, alpha=0.8)

    ax.set_xlabel("Milling pass line $y=a$ (mm)")
    ax.set_ylabel(r"Peak $|w|$ across sensors ($\mu$m)")
    ax.set_title("Vibration robustness across pass lines")
    ax.grid(True, alpha=C_GRID[3], color=C_GRID[:3])
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig09_pass_line_robustness", formats)
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
        help="Comma-separated figure ids: 1..9 or 'all'",
    )

    args = parser.parse_args()
    apply_paper_style()

    log_dir = Path(args.log_dir).resolve()
    traj_dir = Path(args.traj_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    formats = tuple(f.strip().lstrip(".") for f in args.formats.split(",") if f.strip())

    if args.figures.strip().lower() == "all":
        figure_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9}
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
        print("Warning: no trajectory JSON found — fig02-fig09 will be skipped. Run eval_policy.py first.")

    if 1 in figure_ids:
        figure_learning_curve(log_dir, seeds, out_dir, smooth=args.smooth, formats=formats)
    if 2 in figure_ids:
        figure_closed_loop_response(episodes, out_dir, formats=formats)
    if 3 in figure_ids:
        figure_action_run_heatmap(episodes, out_dir, n_bins=args.time_bins, formats=formats)
    if 4 in figure_ids:
        figure_operating_map(episodes, out_dir, formats=formats)
    if 5 in figure_ids:
        figure_vibration_field(episodes, out_dir, n_bins=args.time_bins, formats=formats)
    if 6 in figure_ids:
        figure_eval_summary(episodes, out_dir, formats=formats)
    if 7 in figure_ids:
        figure_reward_decomposition(episodes, out_dir, n_bins=args.time_bins, formats=formats)
    if 8 in figure_ids:
        figure_cutting_force(episodes, out_dir, n_bins=args.time_bins, formats=formats)
    if 9 in figure_ids:
        figure_pass_line_robustness(episodes, out_dir, formats=formats)

    write_metrics_summary(episodes, out_dir)


if __name__ == "__main__":
    main()
