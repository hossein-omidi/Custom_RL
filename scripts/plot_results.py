"""Plot training results: mean±std learning curves and physical sensor trajectory summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_PLOT_DIR, DEFAULT_TRAJ_DIR
from custom_rl.eval.monte_carlo import MC_BAND_STD_MULT
from custom_rl.eval.pipeline import (
    discover_log_seeds,
    discover_trajectory_seeds,
    physical_action_names,
    primary_episode_record,
)
from custom_rl.plants.plate import RPM_MAX, RPM_MIN, omega_to_rpm


def load_monitor_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load episode returns, lengths, and wall-clock timestamps from SB3 Monitor CSV."""
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
    """
    Load all finished episodes for one seed (all parallel env monitor files).

    Episodes from multiple *.monitor.csv files are merged in wall-clock order so
    cumulative environment steps reflect parallel training correctly.
    """
    seed_dir = log_dir / f"seed_{seed}"
    if not seed_dir.is_dir():
        return np.array([]), np.array([])

    monitor_files = list(seed_dir.glob("*.monitor.csv"))
    if not monitor_files:
        single = seed_dir / "monitor.csv"
        if single.is_file():
            monitor_files = [single]

    episodes: list[tuple[float, float, float]] = []
    for monitor_file in sorted(monitor_files):
        returns, lengths, wall_times = load_monitor_csv(monitor_file)
        for r, length, wall_t in zip(returns, lengths, wall_times):
            episodes.append((float(wall_t), float(r), float(length)))

    if not episodes:
        return np.array([]), np.array([])

    episodes.sort(key=lambda row: row[0])
    returns = np.array([row[1] for row in episodes], dtype=np.float64)
    lengths = np.array([row[2] for row in episodes], dtype=np.float64)
    return returns, lengths


def plot_learning_curve(
    log_dir: Path,
    out_dir: Path,
    seeds: list[int],
    smooth: int = 10,
) -> None:
    """Episode return vs cumulative environment steps with mean±std across seeds."""
    data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        returns, lengths = load_seed_episodes(log_dir, seed)
        if returns.size > 0:
            data[seed] = (returns, lengths)

    if not data:
        print(f"No monitor data in {log_dir} for seeds {seeds}")
        return

    missing = [s for s in seeds if s not in data]
    if missing:
        print(f"Note: no training logs for seed(s) {missing} under {log_dir}")

    valid = [(returns, lengths) for returns, lengths in data.values() if len(returns) > 0]

    if not valid:
        print(f"No valid episode data in {log_dir}")
        return

    max_steps = min(np.sum(lengths) for _, lengths in valid)

    if max_steps <= 0:
        return

    step_grid = np.linspace(0, max_steps, num=200)
    returns_interp = []

    for seed in sorted(data.keys()):
        returns, lengths = data[seed]

        if len(returns) == 0:
            continue

        steps = np.concatenate([[0.0], np.cumsum(lengths)])
        returns_ext = np.concatenate([[returns[0]], returns])
        returns_interp.append(np.interp(step_grid, steps, returns_ext))

    if not returns_interp:
        return

    returns_matrix = np.asarray(returns_interp, dtype=np.float64)

    mean_return = np.mean(returns_matrix, axis=0)
    std_return = np.std(returns_matrix, axis=0)

    if smooth > 1:
        kernel_size = min(smooth, len(mean_return) // 2 or 1)
        kernel = np.ones(kernel_size) / kernel_size
        mean_return = np.convolve(mean_return, kernel, mode="same")
        std_return = np.convolve(std_return, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(
        step_grid,
        mean_return - MC_BAND_STD_MULT * std_return,
        mean_return + MC_BAND_STD_MULT * std_return,
        alpha=0.3,
        label=f"±{MC_BAND_STD_MULT:.0f}σ (~99.7%)",
    )
    ax.plot(step_grid, mean_return, lw=2)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title(f"Training: Episode return (mean ± {MC_BAND_STD_MULT:.0f}σ)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = out_dir / "learning_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def load_trajectories(traj_dir: Path, seeds: list[int]) -> dict[int, list]:
    """Load saved trajectory JSON files."""
    out = {}

    for seed in seeds:
        path = traj_dir / f"trajectories_seed{seed}.json"

        if path.exists():
            with open(path, encoding="utf-8") as f:
                out[seed] = json.load(f)

    return out


def _physical_signal_labels(n_sensors: int) -> list[str]:
    """Create labels for physical sensor displacement and velocity."""
    labels = []

    for i in range(n_sensors):
        labels.append(f"w_sensor_{i + 1} (m)")

    for i in range(n_sensors):
        labels.append(f"wdot_sensor_{i + 1} (m/s)")

    return labels


def _nan_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean/std over trajectories while avoiding all-NaN warnings."""
    valid = np.any(~np.isnan(values), axis=0)

    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)

    if np.any(valid):
        mean[valid] = np.nanmean(values[:, valid], axis=0)
        std[valid] = np.nanstd(values[:, valid], axis=0)

    return mean, std, valid


def _physical_signals_from_episode(ep: dict) -> np.ndarray:
    """
    Return unscaled physical signals from one saved episode.

    Preferred new format:
        ep["physical_signals"] = [w_sensor, wdot_sensor]

    Fallback:
        ep["observations"] begins with scaled [w_sensor, wdot_sensor], so unscale
        using metadata["w_obs_scale"] and metadata["wdot_obs_scale"].  Any
        additional observation terms, such as normalized cutter path coordinates,
        are ignored for physical sensor plots.

    Old trajectory files with only ep["states"] are intentionally skipped because
    they may contain modal coordinates and should not be relabeled as physical
    sensor response.
    """
    record = primary_episode_record(ep)
    metadata = ep.get("metadata", record.get("metadata", {}))

    if "physical_signals" in record and len(record["physical_signals"]) > 0:
        signals = np.asarray(record["physical_signals"], dtype=np.float64)
    elif "physical_signals" in ep and len(ep["physical_signals"]) > 0:
        signals = np.asarray(ep["physical_signals"], dtype=np.float64)

    elif "observations" in record and len(record["observations"]) > 0:
        obs = np.asarray(record["observations"], dtype=np.float64)

        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        n_sensors = int(metadata.get("n_sensors", obs.shape[1] // 2))
        w_scale = float(metadata.get("w_obs_scale", 1.0))
        wdot_scale = float(metadata.get("wdot_obs_scale", 1.0))

        if not np.isfinite(w_scale) or abs(w_scale) < 1e-12:
            w_scale = 1.0

        if not np.isfinite(wdot_scale) or abs(wdot_scale) < 1e-12:
            wdot_scale = 1.0

        if n_sensors > 0 and obs.shape[1] >= 2 * n_sensors:
            signals = obs[:, : 2 * n_sensors].copy()
            signals[:, :n_sensors] *= w_scale
            signals[:, n_sensors : 2 * n_sensors] *= wdot_scale
        else:
            return np.asarray([], dtype=np.float64)

    else:
        return np.asarray([], dtype=np.float64)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    return signals


def _non_empty_sequence(value) -> bool:
    """Return True for non-empty saved list/array-like values."""
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def _saved_actions_from_episode(ep: dict, metadata: dict) -> tuple[np.ndarray, bool]:
    """Return actions for plotting and whether they are already physical.

    New eval files save both normalized ``actions`` and ``physical_actions``.
    Prefer the physical values.  For older files, fall back to normalized
    actions instead of incorrectly assuming that ``actions`` are physical just
    because the metadata describes the plant's physical action convention.
    """
    record = primary_episode_record(ep)

    physical = record.get("physical_actions", ep.get("physical_actions", []))
    if _non_empty_sequence(physical):
        return np.asarray(physical, dtype=np.float64), True

    actions = record.get("actions", ep.get("actions", []))
    if _non_empty_sequence(actions):
        return np.asarray(actions, dtype=np.float64), False

    return np.asarray([], dtype=np.float64), False


def _collect_trajectory_arrays(
    traj_dir: Path,
    seeds: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """Collect physical sensor signals, actions, and times into padded arrays."""
    data = load_trajectories(traj_dir, seeds)

    if not data:
        print(f"No trajectory data in {traj_dir}")
        return None, None, None, {}

    all_trajs = []
    metadata: dict = {}
    selected_actions_are_physical: bool | None = None

    for seed in sorted(data.keys()):
        for ep in data[seed]:
            if not metadata and ep.get("metadata"):
                metadata = dict(ep["metadata"])

            record = primary_episode_record(ep)
            states = _physical_signals_from_episode(ep)
            actions, action_is_physical = _saved_actions_from_episode(ep, metadata)

            # Do not mix physical and normalized actions in one summary plot.
            if selected_actions_are_physical is None:
                selected_actions_are_physical = bool(action_is_physical)
            elif bool(action_is_physical) != bool(selected_actions_are_physical):
                continue

            times = np.asarray(
                record.get("times", ep.get("times", np.arange(len(states)))),
                dtype=np.float64,
            )

            if states.size == 0 or actions.size == 0:
                continue

            if states.ndim == 1:
                states = states.reshape(-1, 1)

            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)

            n_steps = min(states.shape[0], actions.shape[0], times.shape[0])
            if n_steps <= 0:
                continue

            states = states[:n_steps]
            actions = actions[:n_steps]
            times = times[:n_steps]

            all_trajs.append((states, actions, times))

    if not all_trajs:
        print(f"No valid trajectories in {traj_dir}")
        return None, None, None, metadata

    state_dim = all_trajs[0][0].shape[1]
    action_dim = all_trajs[0][1].shape[1]
    max_t = max(states.shape[0] for states, actions, times in all_trajs)

    S = np.full((len(all_trajs), max_t, state_dim), np.nan, dtype=np.float64)
    A = np.full((len(all_trajs), max_t, action_dim), np.nan, dtype=np.float64)
    T = np.full((len(all_trajs), max_t), np.nan, dtype=np.float64)

    valid_count = 0

    for states, actions, times in all_trajs:
        if states.shape[1] != state_dim or actions.shape[1] != action_dim:
            continue

        idx = valid_count
        valid_count += 1

        n = min(states.shape[0], actions.shape[0], times.shape[0])
        S[idx, :n, :] = states[:n]
        A[idx, :n, :] = actions[:n]
        T[idx, :n] = times[:n]

    if valid_count == 0:
        print("No trajectories with consistent dimensions.")
        return None, None, None, metadata

    metadata["_plot_actions_are_physical"] = bool(selected_actions_are_physical)
    return S[:valid_count], A[:valid_count], T[:valid_count], metadata


def _time_grid(T: np.ndarray | None, max_t: int) -> tuple[np.ndarray, str]:
    """Return representative time grid."""
    if T is None:
        return np.arange(max_t), "Step"

    valid = np.any(~np.isnan(T), axis=0)

    if not np.any(valid):
        return np.arange(max_t), "Step"

    grid = np.arange(max_t, dtype=np.float64)
    grid[valid] = np.nanmean(T[:, valid], axis=0)

    return grid, "Time (s)"


def plot_state_trajectories(
    S: np.ndarray,
    T: np.ndarray | None,
    metadata: dict,
    out_dir: Path,
) -> None:
    """Plot physical sensor signals with mean±std and displacement safety bounds."""
    state_dim = S.shape[2]
    n_sensors = int(metadata.get("n_sensors", state_dim // 2))

    if n_sensors <= 0:
        n_sensors = state_dim // 2 if state_dim >= 2 else state_dim

    labels = _physical_signal_labels(n_sensors)

    if len(labels) < state_dim:
        labels.extend(f"signal{idx + 1}" for idx in range(len(labels), state_dim))

    t_grid, x_label = _time_grid(T, S.shape[1])

    w_limit = metadata.get("w_limit", None)

    n_cols = 2
    n_rows = int(np.ceil(state_dim / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 3.0 * n_rows),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for dim in range(state_dim):
        ax = axes[dim]

        mean_s, std_s, valid = _nan_mean_std(S[:, :, dim])

        ax.fill_between(
            t_grid[valid],
            mean_s[valid] - MC_BAND_STD_MULT * std_s[valid],
            mean_s[valid] + MC_BAND_STD_MULT * std_s[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_s[valid], lw=1.5)

        # Show physical displacement limit only for displacement channels.
        if w_limit is not None and np.isfinite(w_limit) and dim < n_sensors:
            ax.axhline(+float(w_limit), linestyle="--", linewidth=1)
            ax.axhline(-float(w_limit), linestyle="--", linewidth=1)

        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.3)

    for ax in axes[state_dim:]:
        ax.axis("off")

    axes[min(state_dim - 1, len(axes) - 1)].set_xlabel(x_label)

    fig.suptitle(
        f"Evaluation trajectories: physical sensor response "
        f"(mean ± {MC_BAND_STD_MULT:.0f}σ)"
    )
    fig.tight_layout()

    out_path = out_dir / "trajectory_physical_sensor_response.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def _physical_action_plot_label(name: str) -> str:
    """Human-readable axis label for physical action channels."""
    if name == "omega_rad_s":
        return "Spindle speed (rpm)"
    if name == "ap_mm":
        return "Axial depth of cut ap (mm)"
    if name == "ae_mm":
        return "Radial immersion ae (mm)"
    return name


def _normalized_action_plot_label(name: str, dim: int) -> str:
    """Human-readable axis label for normalized action channels."""
    if name == "omega_rad_s":
        return "u_omega (normalized)"
    if name == "ap_mm":
        return "u_ap (normalized)"
    if name == "ae_mm":
        return "u_ae (normalized)"
    return f"u_{dim} (normalized)"


def _maybe_convert_action_channel_for_plot(
    values: np.ndarray,
    *,
    action_name: str,
    actions_are_physical: bool,
) -> np.ndarray:
    """Convert omega from rad/s to rpm for plotting; leave other channels unchanged."""
    if actions_are_physical and action_name == "omega_rad_s":
        return omega_to_rpm(values)
    return values


def plot_action_trajectories(
    A: np.ndarray,
    T: np.ndarray | None,
    metadata: dict,
    out_dir: Path,
) -> None:
    """Plot policy actions with correct face-milling labels and bounds."""
    action_dim = A.shape[2]
    t_grid, x_label = _time_grid(T, A.shape[1])
    actions_are_physical_for_plot = bool(metadata.get("_plot_actions_are_physical", False))
    names = physical_action_names(metadata, n_dims=action_dim)

    if actions_are_physical_for_plot:
        action_labels = [_physical_action_plot_label(name) for name in names]
    else:
        action_labels = [_normalized_action_plot_label(name, idx) for idx, name in enumerate(names)]

    physical_low = metadata.get("physical_action_low", None)
    physical_high = metadata.get("physical_action_high", None)

    if physical_low is not None:
        physical_low = np.asarray(physical_low, dtype=np.float64)

    if physical_high is not None:
        physical_high = np.asarray(physical_high, dtype=np.float64)

    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(8, 3.0 * action_dim),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for dim in range(action_dim):
        ax = axes[dim]
        action_name = names[dim] if dim < len(names) else f"action_{dim}"

        series = _maybe_convert_action_channel_for_plot(
            A[:, :, dim].copy(),
            action_name=action_name,
            actions_are_physical=actions_are_physical_for_plot,
        )

        mean_a, std_a, valid = _nan_mean_std(series)
        band = MC_BAND_STD_MULT * std_a

        ax.fill_between(
            t_grid[valid],
            mean_a[valid] - band[valid],
            mean_a[valid] + band[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_a[valid], lw=1.5)

        if actions_are_physical_for_plot:
            if (
                physical_low is not None
                and physical_high is not None
                and dim < len(physical_low)
                and dim < len(physical_high)
            ):
                low = float(physical_low[dim])
                high = float(physical_high[dim])
                if action_name == "omega_rad_s":
                    low = float(omega_to_rpm(low))
                    high = float(omega_to_rpm(high))
                ax.axhline(low, linestyle="--", linewidth=1)
                ax.axhline(high, linestyle="--", linewidth=1)
        else:
            ax.axhline(-1.0, linestyle="--", linewidth=1)
            ax.axhline(+1.0, linestyle="--", linewidth=1)

        ax.set_ylabel(action_labels[dim])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)

    title_mode = "physical control actions" if actions_are_physical_for_plot else "normalized control actions"
    fig.suptitle(f"Evaluation trajectories: {title_mode} (mean ± {MC_BAND_STD_MULT:.0f}σ)")
    fig.tight_layout()

    out_path = out_dir / "trajectory_actions_physical.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


PROCESS_PLOT_SPECS = [
    ("cutter_x", "Cutter x-position (m)"),
    ("cutter_y", "Milling line y-position (m)"),
    ("feed_progress", "Feed progress"),
    ("omega_rpm", "Spindle speed (rpm)"),
    ("ap_mm", "Axial depth of cut ap (mm)"),
    ("ae_mm", "Radial immersion ae (mm)"),
    ("mean_chip_mm", "Mean chip thickness (mm)"),
    ("max_chip_mm", "Max chip thickness (mm)"),
]


def _process_dict_from_episode(ep: dict) -> dict:
    record = primary_episode_record(ep)
    process = record.get("process", ep.get("process", {}))
    return process if isinstance(process, dict) else {}


def _collect_process_arrays(
    traj_dir: Path,
    seeds: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None, list[str], dict]:
    """Collect process histories saved by eval_policy.py into padded arrays."""
    data = load_trajectories(traj_dir, seeds)
    if not data:
        return None, None, [], {}

    metadata: dict = {}
    records: list[tuple[dict, np.ndarray]] = []
    available_keys: set[str] = set()

    for seed in sorted(data.keys()):
        for ep in data[seed]:
            if not metadata and ep.get("metadata"):
                metadata = dict(ep["metadata"])
            record = primary_episode_record(ep)
            process = _process_dict_from_episode(ep)
            if not process:
                continue
            times = np.asarray(record.get("times", ep.get("times", [])), dtype=np.float64)
            if times.size == 0:
                continue
            non_empty = {
                key for key, values in process.items()
                if _non_empty_sequence(values) and np.asarray(values, dtype=np.float64).size > 0
            }
            if non_empty:
                available_keys.update(non_empty)
                records.append((process, times))

    keys = [key for key, _label in PROCESS_PLOT_SPECS if key in available_keys]
    if not records or not keys:
        return None, None, [], metadata

    max_t = max(min(len(times), max(len(process.get(key, [])) for key in keys)) for process, times in records)
    P = np.full((len(records), max_t, len(keys)), np.nan, dtype=np.float64)
    T = np.full((len(records), max_t), np.nan, dtype=np.float64)

    valid_count = 0
    for process, times in records:
        n = min(len(times), max_t)
        if n <= 0:
            continue
        idx = valid_count
        valid_count += 1
        T[idx, :n] = times[:n]
        for key_index, key in enumerate(keys):
            values = np.asarray(process.get(key, []), dtype=np.float64).reshape(-1)
            m = min(n, values.size)
            if m > 0:
                P[idx, :m, key_index] = values[:m]

    if valid_count == 0:
        return None, None, [], metadata

    return P[:valid_count], T[:valid_count], keys, metadata


def plot_process_trajectories(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Plot saved face-milling process signals, if present in eval outputs."""
    P, T, keys, _metadata = _collect_process_arrays(traj_dir, seeds)
    if P is None or T is None or not keys:
        print("No process histories found for plotting.")
        return

    label_lookup = dict(PROCESS_PLOT_SPECS)
    t_grid, x_label = _time_grid(T, P.shape[1])
    n_channels = P.shape[2]
    n_cols = 2
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 3.0 * n_rows),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for dim, key in enumerate(keys):
        ax = axes[dim]
        mean_p, std_p, valid = _nan_mean_std(P[:, :, dim])
        band = MC_BAND_STD_MULT * std_p

        ax.fill_between(
            t_grid[valid],
            mean_p[valid] - band[valid],
            mean_p[valid] + band[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_p[valid], lw=1.5)
        ax.set_ylabel(label_lookup.get(key, key))
        ax.grid(True, alpha=0.3)

    for ax in axes[n_channels:]:
        ax.axis("off")

    axes[min(n_channels - 1, len(axes) - 1)].set_xlabel(x_label)
    fig.suptitle(f"Evaluation trajectories: face-milling process signals (mean ± {MC_BAND_STD_MULT:.0f}σ)")
    fig.tight_layout()

    out_path = out_dir / "trajectory_process_signals.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_trajectory_summary(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Mean±std over time for physical sensor signals and physical actions."""
    available = [s for s in seeds if (traj_dir / f"trajectories_seed{s}.json").is_file()]
    missing = [s for s in seeds if s not in available]

    if missing:
        print(f"Note: no trajectory file for seed(s) {missing} in {traj_dir}")

    if not available:
        print(f"No trajectory JSON files in {traj_dir}")
        return

    S, A, T, metadata = _collect_trajectory_arrays(traj_dir, available)

    if S is None or A is None:
        return

    plot_state_trajectories(S, T, metadata, out_dir)
    plot_action_trajectories(A, T, metadata, out_dir)
    plot_process_trajectories(traj_dir, out_dir, available)


def _resolve_plot_seeds(
    seeds: list[int] | None,
    log_dir: Path,
    traj_dir: Path,
) -> tuple[list[int], list[int]]:
    """Return (log_seeds, traj_seeds), auto-discovering when seeds is None."""
    if seeds:
        return list(seeds), list(seeds)

    log_seeds = discover_log_seeds(log_dir)
    traj_seeds = discover_trajectory_seeds(traj_dir)

    if log_seeds or traj_seeds:
        combined = sorted(set(log_seeds) | set(traj_seeds))
        return combined, combined

    return [0, 1, 2], [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and trajectory results")

    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"SB3 monitor logs (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--traj-dir",
        default=DEFAULT_TRAJ_DIR,
        help=f"eval_policy.py output (default: {DEFAULT_TRAJ_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_PLOT_DIR,
        help=f"Plot output directory (default: {DEFAULT_PLOT_DIR})",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds to plot (default: auto-discover from log-dir and traj-dir)",
    )
    parser.add_argument("--smooth", type=int, default=15)

    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    traj_dir = Path(args.traj_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    log_seeds, traj_seeds = _resolve_plot_seeds(args.seeds, log_dir, traj_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Log directory  : {log_dir}")
    print(f"Trajectory dir : {traj_dir}")
    print(f"Plot output    : {out_dir}")
    print(f"Learning curve seeds: {log_seeds}")
    print(f"Trajectory seeds    : {traj_seeds}")

    plot_learning_curve(log_dir, out_dir, log_seeds, smooth=args.smooth)
    plot_trajectory_summary(traj_dir, out_dir, traj_seeds)


if __name__ == "__main__":
    main()