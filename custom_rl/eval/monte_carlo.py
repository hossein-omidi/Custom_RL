"""Monte Carlo rollout aggregation and plotting for physical sensor signals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Shaded MC bands use mean +/- k*std (~95% for k=3, Gaussian).
MC_BAND_STD_MULT = 3.0


def physical_w_from_info_or_obs(
    info: dict,
    obs: np.ndarray,
    *,
    n_sensors: int,
    w_obs_scale: float,
) -> np.ndarray:
    """Return physical sensor displacement vector [m]."""
    if "w_sensor" in info:
        return np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)

    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    scale = w_obs_scale if abs(w_obs_scale) > 1e-12 else 1.0
    return obs_arr[:n_sensors] * scale


def physical_wdot_from_info_or_obs(
    info: dict,
    obs: np.ndarray,
    *,
    n_sensors: int,
    wdot_obs_scale: float,
) -> np.ndarray:
    """Return physical sensor velocity vector [m/s]."""
    if "wdot_sensor" in info:
        return np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)

    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    scale = wdot_obs_scale if abs(wdot_obs_scale) > 1e-12 else 1.0
    return obs_arr[n_sensors : 2 * n_sensors] * scale


def _as_2d_sensor_series(arr: np.ndarray) -> np.ndarray:
    """Ensure sensor history is (T, n_sensors)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D sensor series, got shape {arr.shape}")


def aggregate_mc_sensor_runs(
    runs: list[np.ndarray],
    times_runs: list[np.ndarray] | None = None,
    *,
    default_dt: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Aggregate Monte Carlo sensor histories with NaN padding.

    Unlike truncating to the shortest rollout, incomplete runs remain valid
    for mean/std up to their length; longer runs are not collapsed to one step.

    Returns:
        times (max_len,), mean (max_len, n_sensors), std (max_len, n_sensors),
        n_runs_used
    """
    if not runs:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.reshape(0, 0), empty.reshape(0, 0), 0

    runs_2d = [_as_2d_sensor_series(r) for r in runs if r.size > 0]
    if not runs_2d:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.reshape(0, 0), empty.reshape(0, 0), 0

    n_sensors = max(r.shape[1] for r in runs_2d)
    runs_2d = [
        r if r.shape[1] == n_sensors else np.pad(
            r,
            ((0, 0), (0, n_sensors - r.shape[1])),
            mode="constant",
            constant_values=np.nan,
        )
        for r in runs_2d
    ]

    max_len = max(r.shape[0] for r in runs_2d)
    stacked = np.full((len(runs_2d), max_len, n_sensors), np.nan, dtype=np.float64)
    for i, run in enumerate(runs_2d):
        stacked[i, : run.shape[0], :] = run

    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)

    times = _build_common_times(times_runs, max_len, default_dt=default_dt)
    return times, mean, std, len(runs_2d)


def _build_common_times(
    times_runs: list[np.ndarray] | None,
    max_len: int,
    *,
    default_dt: float,
) -> np.ndarray:
    """Build a shared time axis for MC plots."""
    if times_runs:
        valid = [np.asarray(t, dtype=np.float64).reshape(-1) for t in times_runs if t.size > 0]
        if valid:
            ref = valid[0]
            if ref.size >= max_len:
                return ref[:max_len]
            if ref.size > 1:
                dt = float(np.median(np.diff(ref)))
            else:
                dt = default_dt
            t0 = float(ref[0]) if ref.size else 0.0
            return t0 + np.arange(max_len, dtype=np.float64) * dt

    return np.arange(max_len, dtype=np.float64) * default_dt


def plot_mc_sensor_bands(
    times: np.ndarray,
    mean_series: np.ndarray,
    std_series: np.ndarray,
    *,
    labels: list[str],
    ylabel: str,
    title: str,
    out_path: Path,
    w_limit: float | None = None,
) -> None:
    """Plot mean solid line with +/- k*std shaded band for each sensor channel."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    mean_series = np.asarray(mean_series, dtype=np.float64)
    std_series = np.asarray(std_series, dtype=np.float64)

    if mean_series.ndim == 1:
        mean_series = mean_series.reshape(-1, 1)
        std_series = std_series.reshape(-1, 1)

    n_len = min(len(times), mean_series.shape[0])
    if n_len == 0:
        return

    times = times[:n_len]
    mean_series = mean_series[:n_len]
    std_series = std_series[:n_len]
    n_channels = mean_series.shape[1]

    n_cols = 2
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.0 * n_rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for idx in range(n_channels):
        ax = axes[idx]
        mean = mean_series[:, idx]
        std = std_series[:, idx]
        label = labels[idx] if idx < len(labels) else f"sensor {idx + 1}"

        valid = np.isfinite(mean)
        if np.count_nonzero(valid) < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient MC samples",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        else:
            band = MC_BAND_STD_MULT * std
            ax.plot(times, mean, lw=1.5, color="#1f77b4", label=f"{label} mean")
            ax.fill_between(
                times,
                mean - band,
                mean + band,
                color="#1f77b4",
                alpha=0.25,
                label=f"{label} +/- {MC_BAND_STD_MULT:.0f}σ (~95%)",
            )

        if w_limit is not None and np.isfinite(w_limit) and ylabel.startswith("Displacement"):
            ax.axhline(+w_limit, linestyle="--", color="tab:red", alpha=0.6, lw=1)
            ax.axhline(-w_limit, linestyle="--", color="tab:red", alpha=0.6, lw=1)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    for ax in axes[n_channels:]:
        ax.axis("off")

    axes[min(n_channels - 1, len(axes) - 1)].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_monte_carlo_rollouts(
    rollouts: list[dict],
    *,
    n_sensors: int,
    w_limit: float | None,
    out_dir: Path,
    title_suffix: str,
    default_dt: float = 0.002,
) -> None:
    """
    Plot MC mean +/- std for displacement and velocity from rollout dicts.

    Each rollout dict must contain keys: times, w_sensor, wdot_sensor.
    """
    w_runs = [r["w_sensor"] for r in rollouts if r.get("w_sensor") is not None]
    wdot_runs = [r["wdot_sensor"] for r in rollouts if r.get("wdot_sensor") is not None]
    times_runs = [r["times"] for r in rollouts if r.get("times") is not None]

    if not w_runs or not wdot_runs:
        return

    w_times, w_mean, w_std, n_used = aggregate_mc_sensor_runs(
        w_runs, times_runs, default_dt=default_dt
    )
    v_times, v_mean, v_std, _ = aggregate_mc_sensor_runs(
        wdot_runs, times_runs, default_dt=default_dt
    )

    w_labels = [f"w_sensor_{i + 1} (m)" for i in range(n_sensors)]
    v_labels = [f"wdot_sensor_{i + 1} (m/s)" for i in range(n_sensors)]

    plot_mc_sensor_bands(
        w_times,
        w_mean,
        w_std,
        labels=w_labels,
        ylabel="Displacement (m)",
        title=f"Monte Carlo physical displacement ({title_suffix}, n={n_used})",
        out_path=out_dir / "pretrain_physical_sensor_response_mc.png",
        w_limit=w_limit,
    )
    plot_mc_sensor_bands(
        v_times,
        v_mean,
        v_std,
        labels=v_labels,
        ylabel="Velocity (m/s)",
        title=f"Monte Carlo physical velocity ({title_suffix}, n={n_used})",
        out_path=out_dir / "pretrain_physical_velocity_mc.png",
    )
