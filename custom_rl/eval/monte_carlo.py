"""Monte Carlo rollout aggregation and plotting for the face-milling plate plant.

This module is intentionally evaluation-only.  It does not change training,
plant dynamics, or the RK4 integrator.  It only converts observations back to
physical sensor displacement/velocity and aggregates rollout histories.

Face-milling convention
-----------------------
The default plant control is

    u = [u_omega, u_ap]

where ``ap`` is the axial depth of cut [mm].  The radial immersion/depth
``ae`` is a fixed plant parameter by default.  In the current face-milling
plant this is ``plant.ae_default``; with the usual D=50 mm and ae_default=25 mm
this corresponds to half immersion, ae/D = 0.5.  Only when the plant is created
with ``control_ae=True`` should ``ae`` be treated as a third action component.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


# Shaded MC bands use mean +/- k*std.  The previous code used k=3 and this is
# kept for backward-compatible plots.
MC_BAND_STD_MULT = 3.0


def physical_w_from_info_or_obs(
    info: dict,
    obs: np.ndarray,
    *,
    n_sensors: int,
    w_obs_scale: float,
) -> np.ndarray:
    """Return physical sensor displacement vector [m].

    Preferred source is ``info["w_sensor"]`` produced by ``PlatePlant`` through
    its modal-to-physical map

        w_sensor = Phi @ eta.

    If the key is absent, fall back to the scaled observation convention used by
    the plant:

        obs[:n_sensors] = w_sensor / w_obs_scale.
    """
    n_sensors = int(n_sensors)
    if n_sensors <= 0:
        return np.zeros(0, dtype=np.float64)

    if "w_sensor" in info:
        w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
        return w[:n_sensors]

    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    scale = float(w_obs_scale) if abs(float(w_obs_scale)) > 1e-12 else 1.0
    return obs_arr[:n_sensors] * scale


def physical_wdot_from_info_or_obs(
    info: dict,
    obs: np.ndarray,
    *,
    n_sensors: int,
    wdot_obs_scale: float,
) -> np.ndarray:
    """Return physical sensor velocity vector [m/s].

    Preferred source is ``info["wdot_sensor"]`` produced by ``PlatePlant``:

        wdot_sensor = Phi @ eta_dot.

    If the key is absent, fall back to the scaled observation convention:

        obs[n_sensors:2*n_sensors] = wdot_sensor / wdot_obs_scale.
    """
    n_sensors = int(n_sensors)
    if n_sensors <= 0:
        return np.zeros(0, dtype=np.float64)

    if "wdot_sensor" in info:
        wd = np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1)
        return wd[:n_sensors]

    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    scale = float(wdot_obs_scale) if abs(float(wdot_obs_scale)) > 1e-12 else 1.0
    return obs_arr[n_sensors : 2 * n_sensors] * scale


def face_milling_process_from_info(info: dict[str, Any], plant: Any | None = None) -> dict[str, float]:
    """Extract physical face-milling process values from step ``info``.

    Returns a dictionary containing available physical values.  ``ae_mm`` is
    resolved as follows:

    1. ``info["ae_mm"]`` if the environment supplied it;
    2. ``plant.ae_default`` if a plant object is supplied;
    3. ``D_mm/2`` if only cutter diameter is known, corresponding to 50% radial
       immersion;
    4. NaN if none of the above is available.

    This function does not make ``ae`` a control variable.  By default, only
    ``omega`` and ``ap`` are controlled by the agent; ``ae`` remains a fixed
    plant/process parameter unless ``control_ae=True`` in ``PlatePlant``.
    """
    out: dict[str, float] = {}

    def _float_or_nan(value: Any) -> float:
        try:
            value_f = float(value)
        except Exception:
            return float("nan")
        return value_f if np.isfinite(value_f) else float("nan")

    omega_rad_s = info.get("omega_rad_s", getattr(plant, "_last_omega", np.nan))
    out["omega_rad_s"] = _float_or_nan(omega_rad_s)

    omega_rpm = info.get("omega_rpm", np.nan)
    if not np.isfinite(_float_or_nan(omega_rpm)) and np.isfinite(out["omega_rad_s"]):
        omega_rpm = out["omega_rad_s"] * 60.0 / (2.0 * np.pi)
    out["omega_rpm"] = _float_or_nan(omega_rpm)

    out["ap_mm"] = _float_or_nan(info.get("ap_mm", getattr(plant, "_last_ap", np.nan)))

    ae_value = info.get("ae_mm", getattr(plant, "ae_default", np.nan))
    if not np.isfinite(_float_or_nan(ae_value)):
        d_mm = info.get("D_mm", getattr(plant, "D_mm", np.nan))
        d_mm_f = _float_or_nan(d_mm)
        ae_value = 0.5 * d_mm_f if np.isfinite(d_mm_f) else np.nan
    out["ae_mm"] = _float_or_nan(ae_value)

    if "mean_chip_mm" in info:
        out["mean_chip_mm"] = _float_or_nan(info["mean_chip_mm"])
    if "max_chip_mm" in info:
        out["max_chip_mm"] = _float_or_nan(info["max_chip_mm"])

    return out


def physical_action_from_normalized(action: np.ndarray, plant: Any) -> np.ndarray:
    """Convert normalized action to physical face-milling action using the plant.

    The returned vector is [omega_rad_s, ap_mm] for the default plant and
    [omega_rad_s, ap_mm, ae_mm] when ``control_ae=True``.
    """
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)

    if hasattr(plant, "physical_action_bounds"):
        low, high = plant.physical_action_bounds()
        low = np.asarray(low, dtype=np.float64).reshape(-1)
        high = np.asarray(high, dtype=np.float64).reshape(-1)
    else:
        if bool(getattr(plant, "control_ae", False)):
            low = np.array([plant.omega_min, plant.ap_min, plant.ae_min], dtype=np.float64)
            high = np.array([plant.omega_max, plant.ap_max, plant.ae_max], dtype=np.float64)
        else:
            low = np.array([plant.omega_min, plant.ap_min], dtype=np.float64)
            high = np.array([plant.omega_max, plant.ap_max], dtype=np.float64)

    dim = low.size
    if action_arr.size < dim:
        padded = np.zeros(dim, dtype=np.float64)
        padded[: action_arr.size] = action_arr
        action_arr = padded
    action_arr = np.clip(action_arr[:dim], -1.0, 1.0)
    return low + 0.5 * (action_arr + 1.0) * (high - low)


def _as_2d_sensor_series(arr: np.ndarray) -> np.ndarray:
    """Ensure a sensor history has shape (T, n_channels)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D series, got shape {arr.shape}.")


def _non_empty_arrays(values: Iterable[Any]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.size > 0:
            arrays.append(arr)
    return arrays


def aggregate_mc_sensor_runs(
    runs: list[np.ndarray],
    times_runs: list[np.ndarray] | None = None,
    *,
    default_dt: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate Monte Carlo sensor histories with NaN padding.

    Incomplete episodes remain valid up to their own length.  Longer episodes
    are not truncated to the shortest rollout.

    Returns
    -------
    times : ndarray, shape (T,)
    mean : ndarray, shape (T, n_channels)
    std : ndarray, shape (T, n_channels)
    n_runs_used : int
    """
    valid_runs = _non_empty_arrays(runs)
    if not valid_runs:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.reshape(0, 0), empty.reshape(0, 0), 0

    runs_2d = [_as_2d_sensor_series(r) for r in valid_runs]
    n_channels = max(r.shape[1] for r in runs_2d)
    max_len = max(r.shape[0] for r in runs_2d)

    stacked = np.full((len(runs_2d), max_len, n_channels), np.nan, dtype=np.float64)
    for i, run in enumerate(runs_2d):
        stacked[i, : run.shape[0], : run.shape[1]] = run

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
    """Build a shared time axis for Monte Carlo plots."""
    if max_len <= 0:
        return np.zeros(0, dtype=np.float64)

    if times_runs:
        valid = _non_empty_arrays(times_runs)
        if valid:
            ref = np.asarray(valid[0], dtype=np.float64).reshape(-1)
            if ref.size >= max_len:
                return ref[:max_len]
            if ref.size > 1:
                dt = float(np.nanmedian(np.diff(ref)))
                if not np.isfinite(dt) or dt <= 0.0:
                    dt = float(default_dt)
            else:
                dt = float(default_dt)
            t0 = float(ref[0]) if ref.size else 0.0
            return t0 + np.arange(max_len, dtype=np.float64) * dt

    return np.arange(max_len, dtype=np.float64) * float(default_dt)


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
    """Plot mean response with +/- MC_BAND_STD_MULT standard-deviation bands."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    mean_series = np.asarray(mean_series, dtype=np.float64)
    std_series = np.asarray(std_series, dtype=np.float64)

    if mean_series.ndim == 1:
        mean_series = mean_series.reshape(-1, 1)
    if std_series.ndim == 1:
        std_series = std_series.reshape(-1, 1)

    n_len = min(len(times), mean_series.shape[0], std_series.shape[0])
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
        std = std_series[:, idx] if idx < std_series.shape[1] else np.zeros_like(mean)
        label = labels[idx] if idx < len(labels) else f"channel {idx + 1}"

        valid = np.isfinite(times) & np.isfinite(mean)
        if np.count_nonzero(valid) < 2:
            ax.text(0.5, 0.5, "Insufficient MC samples", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            t_valid = times[valid]
            mean_valid = mean[valid]
            std_valid = np.nan_to_num(std[valid], nan=0.0, posinf=0.0, neginf=0.0)
            band = MC_BAND_STD_MULT * std_valid
            ax.plot(t_valid, mean_valid, lw=1.5, label=f"{label} mean")
            ax.fill_between(
                t_valid,
                mean_valid - band,
                mean_valid + band,
                alpha=0.25,
                label=f"{label} +/- {MC_BAND_STD_MULT:.0f}σ",
            )

        if w_limit is not None and np.isfinite(w_limit) and ylabel.lower().startswith("displacement"):
            ax.axhline(+float(w_limit), linestyle="--", alpha=0.6, lw=1)
            ax.axhline(-float(w_limit), linestyle="--", alpha=0.6, lw=1)
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


def _collect_rollout_key(rollouts: list[dict], key: str) -> list[np.ndarray]:
    return _non_empty_arrays([r.get(key) for r in rollouts])


def plot_monte_carlo_rollouts(
    rollouts: list[dict],
    *,
    n_sensors: int,
    w_limit: float | None,
    out_dir: Path,
    title_suffix: str,
    default_dt: float = 0.002,
    plot_process_signals: bool = True,
) -> None:
    """Plot Monte Carlo physical sensor and optional process signals.

    Required rollout keys for sensor plots:
        ``times``, ``w_sensor``, ``wdot_sensor``

    Optional process keys, if present:
        ``omega_rpm``, ``ap_mm``, ``ae_mm``, ``mean_chip_mm``, ``max_chip_mm``

    The function is compatible with the updated face-milling plant, where the
    default controlled inputs are spindle speed and axial depth of cut ``ap``.
    ``ae`` is plotted only if the rollout supplies it; otherwise it is treated as
    a fixed plant parameter and is not inferred here.
    """
    out_dir = Path(out_dir)

    w_runs = _collect_rollout_key(rollouts, "w_sensor")
    wdot_runs = _collect_rollout_key(rollouts, "wdot_sensor")
    times_runs = _collect_rollout_key(rollouts, "times")

    if w_runs:
        w_times, w_mean, w_std, n_used = aggregate_mc_sensor_runs(
            w_runs, times_runs, default_dt=default_dt
        )
        w_labels = [f"w_sensor_{i + 1} (m)" for i in range(int(n_sensors))]
        plot_mc_sensor_bands(
            w_times,
            w_mean,
            w_std,
            labels=w_labels,
            ylabel="Displacement (m)",
            title=f"Face-milling MC sensor displacement ({title_suffix}, n={n_used})",
            out_path=out_dir / "pretrain_physical_sensor_response_mc.png",
            w_limit=w_limit,
        )

    if wdot_runs:
        v_times, v_mean, v_std, n_used_v = aggregate_mc_sensor_runs(
            wdot_runs, times_runs, default_dt=default_dt
        )
        v_labels = [f"wdot_sensor_{i + 1} (m/s)" for i in range(int(n_sensors))]
        plot_mc_sensor_bands(
            v_times,
            v_mean,
            v_std,
            labels=v_labels,
            ylabel="Velocity (m/s)",
            title=f"Face-milling MC sensor velocity ({title_suffix}, n={n_used_v})",
            out_path=out_dir / "pretrain_physical_velocity_mc.png",
        )

    if plot_process_signals:
        _plot_optional_process_rollouts(
            rollouts,
            times_runs=times_runs,
            out_dir=out_dir,
            title_suffix=title_suffix,
            default_dt=default_dt,
        )


def _plot_optional_process_rollouts(
    rollouts: list[dict],
    *,
    times_runs: list[np.ndarray],
    out_dir: Path,
    title_suffix: str,
    default_dt: float,
) -> None:
    """Plot optional face-milling process signals if they exist in rollouts."""
    process_specs = [
        ("omega_rpm", "Spindle speed (rpm)", "mc_spindle_speed_rpm.png"),
        ("ap_mm", "Axial depth of cut ap (mm)", "mc_axial_depth_ap_mm.png"),
        ("ae_mm", "Radial immersion ae (mm)", "mc_radial_immersion_ae_mm.png"),
        ("mean_chip_mm", "Mean effective chip thickness (mm)", "mc_mean_chip_mm.png"),
        ("max_chip_mm", "Max effective chip thickness (mm)", "mc_max_chip_mm.png"),
    ]

    for key, ylabel, filename in process_specs:
        runs = _collect_rollout_key(rollouts, key)
        if not runs:
            continue
        times, mean, std, n_used = aggregate_mc_sensor_runs(
            runs, times_runs, default_dt=default_dt
        )
        plot_mc_sensor_bands(
            times,
            mean,
            std,
            labels=[key],
            ylabel=ylabel,
            title=f"Face-milling MC {ylabel} ({title_suffix}, n={n_used})",
            out_path=out_dir / filename,
        )


__all__ = [
    "MC_BAND_STD_MULT",
    "physical_w_from_info_or_obs",
    "physical_wdot_from_info_or_obs",
    "face_milling_process_from_info",
    "physical_action_from_normalized",
    "aggregate_mc_sensor_runs",
    "plot_mc_sensor_bands",
    "plot_monte_carlo_rollouts",
]
