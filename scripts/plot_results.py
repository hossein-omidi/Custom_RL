"""Plot training results: mean±std learning curves and plate trajectory summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_PLOT_DIR, DEFAULT_TRAJ_DIR


def load_monitor_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load episode lengths and returns from SB3 Monitor CSV."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=2)
    except Exception:
        return np.array([]), np.array([])

    if data.size == 0:
        return np.array([]), np.array([])

    if data.ndim == 1:
        data = data.reshape(1, -1)

    returns = data[:, 0]
    lengths = data[:, 1] if data.shape[1] > 1 else np.full_like(returns, 1.0)

    return returns, lengths


def load_all_seed_logs(
    log_dir: Path,
    seeds: list[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load and aggregate monitor logs for each seed."""
    out = {}

    for seed in seeds:
        seed_dir = log_dir / f"seed_{seed}"

        if not seed_dir.exists():
            continue

        monitor_files = list(seed_dir.glob("*.monitor.csv"))

        if not monitor_files:
            single = seed_dir / "monitor.csv"
            if single.exists():
                monitor_files = [single]

        if not monitor_files:
            continue

        all_returns = []
        all_lengths = []

        for monitor_file in sorted(monitor_files):
            returns, lengths = load_monitor_csv(monitor_file)
            if len(returns) > 0:
                all_returns.append(returns)
                all_lengths.append(lengths)

        if all_returns:
            out[seed] = (
                np.concatenate(all_returns),
                np.concatenate(all_lengths),
            )

    return out


def plot_learning_curve(
    log_dir: Path,
    out_dir: Path,
    seeds: list[int],
    smooth: int = 10,
) -> None:
    """Episode return vs environment steps with mean±std across seeds."""
    data = load_all_seed_logs(log_dir, seeds)

    if not data:
        print(f"No monitor data in {log_dir}")
        return

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
        mean_return - std_return,
        mean_return + std_return,
        alpha=0.3,
    )
    ax.plot(step_grid, mean_return, lw=2)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("Training: Episode return (mean ± std)")
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


def _state_labels(state_dim: int) -> list[str]:
    """Create labels for plate modal state."""
    labels = []

    for dim in range(state_dim):
        mode_index = dim // 2 + 1

        if dim % 2 == 0:
            labels.append(f"eta{mode_index}")
        else:
            labels.append(f"eta{mode_index}_dot")

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


def _collect_trajectory_arrays(
    traj_dir: Path,
    seeds: list[int],
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict,
]:
    """Collect trajectory states, physical actions, and times into padded arrays."""
    data = load_trajectories(traj_dir, seeds)

    if not data:
        print(f"No trajectory data in {traj_dir}")
        return None, None, None, None, None, None, None, {}

    all_trajs = []
    metadata = {}

    for seed in sorted(data.keys()):
        for ep in data[seed]:
            states = np.asarray(ep["states"], dtype=np.float64)

            # Prefer physical actions. Fall back to normalized actions for old files.
            actions = np.asarray(
                ep.get("physical_actions", ep.get("actions", [])),
                dtype=np.float64,
            )

            times = np.asarray(
                ep.get("times", np.arange(len(states))),
                dtype=np.float64,
            )

            if states.size == 0 or actions.size == 0:
                continue

            # Reward components (for plotting metrics).
            rc_list = ep.get("reward_components", None)
            prod_series = np.full((states.shape[0],), np.nan, dtype=np.float64)
            vib_series = np.full((states.shape[0],), np.nan, dtype=np.float64)
            if isinstance(rc_list, list) and len(rc_list) == states.shape[0]:
                for i, d in enumerate(rc_list):
                    if isinstance(d, dict):
                        prod_series[i] = d.get("productivity_score", np.nan)
                        vib_series[i] = d.get("vibration_cost", np.nan)

            if states.ndim == 1:
                states = states.reshape(-1, 1)

            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)

            if times.size != states.shape[0]:
                times = np.arange(states.shape[0], dtype=np.float64)

            if not metadata and "metadata" in ep:
                metadata = ep["metadata"]

            # Optional physical sensor signals (preferred for plotting).
            sensor_w = ep.get("sensor_w", None)
            sensor_w_dot = ep.get("sensor_w_dot", None)
            SW = None
            SV = None
            if sensor_w is not None and sensor_w_dot is not None:
                try:
                    SW = np.asarray(sensor_w, dtype=np.float64)
                    SV = np.asarray(sensor_w_dot, dtype=np.float64)
                except Exception:
                    SW = None
                    SV = None

            all_trajs.append((states, actions, times, prod_series, vib_series, SW, SV))

    if not all_trajs:
        print(f"No valid trajectories in {traj_dir}")
        return None, None, None, None, None, None, None, metadata

    state_dim = all_trajs[0][0].shape[1]
    action_dim = all_trajs[0][1].shape[1]
    max_t = max(states.shape[0] for states, _, _, _, _ in all_trajs)

    S = np.full((len(all_trajs), max_t, state_dim), np.nan, dtype=np.float64)
    A = np.full((len(all_trajs), max_t, action_dim), np.nan, dtype=np.float64)
    T = np.full((len(all_trajs), max_t), np.nan, dtype=np.float64)
    P = np.full((len(all_trajs), max_t), np.nan, dtype=np.float64)  # productivity_score
    V = np.full((len(all_trajs), max_t), np.nan, dtype=np.float64)  # vibration_cost

    # Physical sensor arrays: (n_trajs, max_t, n_sensors)
    # If sensor_w is missing, these remain None.
    SW_out = None
    SV_out = None
    for item in all_trajs:
        SW = item[5]
        SV = item[6]
        if SW is not None and SV is not None and SW.ndim == 2 and SW.shape[1] > 0:
            n_sensors = SW.shape[1]
            SW_out = np.full((len(all_trajs), max_t, n_sensors), np.nan, dtype=np.float64)
            SV_out = np.full((len(all_trajs), max_t, n_sensors), np.nan, dtype=np.float64)
            break

    valid_count = 0

    for states, actions, times, prod_series, vib_series, SW, SV in all_trajs:
        if states.shape[1] != state_dim or actions.shape[1] != action_dim:
            continue

        idx = valid_count
        valid_count += 1

        state_steps = states.shape[0]
        action_steps = actions.shape[0]
        time_steps = times.shape[0]

        S[idx, :state_steps, :] = states
        A[idx, :action_steps, :] = actions
        T[idx, :time_steps] = times
        P[idx, :state_steps] = prod_series
        V[idx, :state_steps] = vib_series
        if SW_out is not None and SW is not None and SV is not None:
            if SW.ndim == 2:
                SW_out[idx, :SW.shape[0], :] = SW
            if SV.ndim == 2:
                SV_out[idx, :SV.shape[0], :] = SV

    if valid_count == 0:
        print("No trajectories with consistent dimensions.")
        return None, None, None, None, None, None, None, metadata

    return (
        S[:valid_count],
        A[:valid_count],
        T[:valid_count],
        P[:valid_count],
        V[:valid_count],
        SW_out[:valid_count] if SW_out is not None else None,
        SV_out[:valid_count] if SV_out is not None else None,
        metadata,
    )


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
    sensor_w: np.ndarray | None = None,
    sensor_w_dot: np.ndarray | None = None,
) -> None:
    """Plot physical sensor displacement/velocity (preferred) or reconstructed fallback."""
    if sensor_w is not None and sensor_w_dot is not None:
        n_sensors = sensor_w.shape[2]
        disp_failure = metadata.get("displacement_failure_limit", None)

        t_grid, x_label = _time_grid(T, S.shape[1])
        n_plots = 2 * n_sensors
        n_cols = 2
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(10, 3.0 * n_rows),
            sharex=True,
        )
        axes = np.asarray(axes).reshape(-1)

        for s_idx in range(n_sensors):
            # Displacement subplot
            ax = axes[s_idx]
            mean_s, std_s, valid = _nan_mean_std(sensor_w[:, :, s_idx])
            ax.fill_between(
                t_grid[valid],
                mean_s[valid] - std_s[valid],
                mean_s[valid] + std_s[valid],
                alpha=0.3,
            )
            ax.plot(t_grid[valid], mean_s[valid], lw=1.5)
            if disp_failure is not None and np.isfinite(disp_failure):
                ax.axhline(float(+disp_failure), linestyle="--", linewidth=1)
                ax.axhline(float(-disp_failure), linestyle="--", linewidth=1)
            ax.set_ylabel(f"w_s{s_idx + 1} (physical)")
            ax.grid(True, alpha=0.3)

            # Velocity subplot
            axv = axes[n_sensors + s_idx]
            mean_sv, std_sv, valid_v = _nan_mean_std(sensor_w_dot[:, :, s_idx])
            axv.fill_between(
                t_grid[valid_v],
                mean_sv[valid_v] - std_sv[valid_v],
                mean_sv[valid_v] + std_sv[valid_v],
                alpha=0.3,
            )
            axv.plot(t_grid[valid_v], mean_sv[valid_v], lw=1.5)
            axv.set_ylabel(f"w_dot_s{s_idx + 1} (physical)")
            axv.grid(True, alpha=0.3)

        for ax in axes[n_plots:]:
            ax.axis("off")
        axes[min(n_plots - 1, len(axes) - 1)].set_xlabel(x_label)

        fig.suptitle("Evaluation trajectories: physical sensor displacement & velocity (mean ± std)")
        fig.tight_layout()

        out_path = out_dir / "trajectory_states.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")
        return

    # Fallback: reconstructed from normalized PPO observation (may include obs clipping).
    state_dim = S.shape[2]
    n_sensors = len(metadata.get("sensor_coords", []))
    if n_sensors <= 0:
        # Fallback: plate env default is 2 sensors.
        n_sensors = 2

    # Observation layout:
    #   [w_s1_norm..w_sN_norm, w_dot_s1_norm..w_dot_sN_norm, prev_omega_norm, prev_ac_norm]
    sensor_norm_dim = 2 * n_sensors
    if sensor_norm_dim > state_dim:
        sensor_norm_dim = state_dim

    disp_norm_scale = float(metadata.get("disp_norm_scale", 1.0))
    vel_norm_scale = float(metadata.get("vel_norm_scale", 1.0))
    disp_failure = metadata.get("displacement_failure_limit", None)

    t_grid, x_label = _time_grid(T, S.shape[1])

    n_plots = min(sensor_norm_dim, 2 * n_sensors)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 3.0 * n_rows),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for dim in range(n_plots):
        ax = axes[dim]

        mean_s, std_s, valid = _nan_mean_std(S[:, :, dim])

        # Convert from normalized to physical units for sensor signals.
        if dim < n_sensors:
            mean_phys = mean_s * disp_norm_scale
            std_phys = std_s * disp_norm_scale
        else:
            mean_phys = mean_s * vel_norm_scale
            std_phys = std_s * vel_norm_scale

        ax.fill_between(
            t_grid[valid],
            mean_phys[valid] - std_phys[valid],
            mean_phys[valid] + std_phys[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_phys[valid], lw=1.5)

        # Draw termination bounds only for displacement signals.
        if (
            dim < n_sensors
            and disp_failure is not None
            and np.isfinite(disp_failure)
        ):
            ax.axhline(float(+disp_failure), linestyle="--", linewidth=1)
            ax.axhline(float(-disp_failure), linestyle="--", linewidth=1)

        if dim < n_sensors:
            ax.set_ylabel(f"w_s{dim + 1} (physical)")
        else:
            ax.set_ylabel(f"w_dot_s{dim - n_sensors + 1} (physical)")

        ax.grid(True, alpha=0.3)

    for ax in axes[n_plots:]:
        ax.axis("off")

    axes[min(n_plots - 1, len(axes) - 1)].set_xlabel(x_label)

    fig.suptitle("Evaluation trajectories: reconstructed sensor signals")
    fig.tight_layout()

    out_path = out_dir / "trajectory_states.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_action_trajectories(
    A: np.ndarray,
    T: np.ndarray | None,
    metadata: dict,
    out_dir: Path,
) -> None:
    """Plot physical policy actions [omega, ac] with their bounds."""
    action_dim = A.shape[2]
    t_grid, x_label = _time_grid(T, A.shape[1])

    if action_dim == 2:
        action_labels = ["omega [rad/s]", "ac [mm]"]
    else:
        action_labels = [f"action{idx}" for idx in range(action_dim)]

    physical_low = metadata.get("physical_action_low", None)
    physical_high = metadata.get("physical_action_high", None)

    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(8, 3.0 * action_dim),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for dim in range(action_dim):
        ax = axes[dim]

        mean_a, std_a, valid = _nan_mean_std(A[:, :, dim])

        ax.fill_between(
            t_grid[valid],
            mean_a[valid] - std_a[valid],
            mean_a[valid] + std_a[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_a[valid], lw=1.5)

        if physical_low is not None and physical_high is not None:
            ax.axhline(float(physical_low[dim]), linestyle="--", linewidth=1)
            ax.axhline(float(physical_high[dim]), linestyle="--", linewidth=1)

        ax.set_ylabel(action_labels[dim])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)

    fig.suptitle("Evaluation trajectories: physical control actions")
    fig.tight_layout()

    out_path = out_dir / "trajectory_actions_physical.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_reward_metrics(
    P: np.ndarray,
    V: np.ndarray,
    T: np.ndarray | None,
    metadata: dict,
    out_dir: Path,
) -> None:
    """Plot mean±std productivity score and vibration cost over time."""
    max_t = P.shape[1]
    t_grid, x_label = _time_grid(T, max_t)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    # Productivity
    mean_p = np.nanmean(P, axis=0)
    std_p = np.nanstd(P, axis=0)
    valid_p = ~np.isnan(mean_p)
    if np.any(valid_p):
        axes[0].fill_between(t_grid[valid_p], mean_p[valid_p] - std_p[valid_p], mean_p[valid_p] + std_p[valid_p], alpha=0.3)
        axes[0].plot(t_grid[valid_p], mean_p[valid_p], lw=1.5)
    axes[0].set_ylabel("productivity_score")
    axes[0].grid(True, alpha=0.3)

    # Vibration cost
    mean_v = np.nanmean(V, axis=0)
    std_v = np.nanstd(V, axis=0)
    valid_v = ~np.isnan(mean_v)
    if np.any(valid_v):
        axes[1].fill_between(t_grid[valid_v], mean_v[valid_v] - std_v[valid_v], mean_v[valid_v] + std_v[valid_v], alpha=0.3)
        axes[1].plot(t_grid[valid_v], mean_v[valid_v], lw=1.5)
    axes[1].set_ylabel("vibration_cost")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Reward metrics over evaluation trajectories")
    fig.tight_layout()

    out_path = out_dir / "reward_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")

    print(f"Saved {out_path}")


def load_mc_trajectories(mc_dir: Path, seeds: list[int]) -> list[dict]:
    """Load Monte Carlo trajectory JSON files."""
    episodes: list[dict] = []
    for seed in seeds:
        path = mc_dir / f"mc_trajectories_seed{seed}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                episodes.extend(json.load(f))
    return episodes


def load_mc_summaries(mc_dir: Path, seeds: list[int]) -> list[dict]:
    summaries = []
    for seed in seeds:
        path = mc_dir / f"mc_summary_seed{seed}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                summaries.append(json.load(f))
    return summaries


def _pad_mc_sensor_series(
    episodes: list[dict],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length sensor series to (n_ep, max_t, n_sensors)."""
    valid = [ep for ep in episodes if key in ep and ep[key]]
    if not valid:
        return np.empty((0, 0, 0)), np.empty((0, 0))

    n_sensors = len(valid[0][key][0])
    max_t = max(len(ep[key]) for ep in valid)
    n_ep = len(valid)

    data = np.full((n_ep, max_t, n_sensors), np.nan, dtype=np.float64)
    times = np.full((n_ep, max_t), np.nan, dtype=np.float64)

    for i, ep in enumerate(valid):
        arr = np.asarray(ep[key], dtype=np.float64)
        data[i, : arr.shape[0], :] = arr
        t = np.asarray(ep.get("times", np.arange(arr.shape[0] + 1)), dtype=np.float64)
        if t.size >= arr.shape[0] + 1:
            times[i, : arr.shape[0]] = t[1 : arr.shape[0] + 1]
        elif t.size == arr.shape[0]:
            times[i, : arr.shape[0]] = t
        else:
            times[i, : arr.shape[0]] = np.arange(arr.shape[0], dtype=np.float64)

    return data, times


def plot_mc_return_distribution(mc_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Histogram and boxplot of episode returns under Monte Carlo uncertainty."""
    episodes = load_mc_trajectories(mc_dir, seeds)
    if not episodes:
        print(f"No MC trajectories in {mc_dir}")
        return

    returns = np.asarray([ep["return"] for ep in episodes], dtype=np.float64)
    reasons = [ep.get("termination_reason", "unknown") for ep in episodes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(returns, bins=min(25, max(5, len(returns) // 4)), density=True, alpha=0.75, edgecolor="k")
    axes[0].axvline(float(np.mean(returns)), color="C1", lw=2, label=f"mean={np.mean(returns):.3f}")
    axes[0].axvline(float(np.median(returns)), color="C2", ls="--", lw=1.5, label=f"median={np.median(returns):.3f}")
    axes[0].set_xlabel("Episode return")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Monte Carlo return distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    unique_reasons = sorted(set(reasons))
    counts = [reasons.count(r) for r in unique_reasons]
    axes[1].bar(unique_reasons, counts, color="steelblue", edgecolor="k")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Termination reasons")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = out_dir / "mc_return_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pass_line_heatmap(mc_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """
    Heatmap of mean return vs pass-line index (spatial robustness across plate width).

    Rows: pass_line_index; color: mean episode return; annotations: sample count.
    """
    episodes = load_mc_trajectories(mc_dir, seeds)
    if not episodes:
        return

    pass_returns: dict[int, list[float]] = {}
    for ep in episodes:
        ctx = ep.get("episode_context", {})
        if "pass_line_index" not in ctx:
            continue
        idx = int(ctx["pass_line_index"])
        pass_returns.setdefault(idx, []).append(float(ep["return"]))

    if not pass_returns:
        return

    indices = sorted(pass_returns.keys())
    means = np.asarray([np.mean(pass_returns[i]) for i in indices])
    stds = np.asarray([np.std(pass_returns[i]) for i in indices])
    counts = np.asarray([len(pass_returns[i]) for i in indices])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    im = axes[0].imshow(
        means.reshape(1, -1),
        aspect="auto",
        cmap="RdYlGn",
        extent=[indices[0] - 0.5, indices[-1] + 0.5, 0, 1],
    )
    axes[0].set_yticks([])
    axes[0].set_title("Mean return by pass-line index (Monte Carlo)")
    fig.colorbar(im, ax=axes[0], label="mean return")

    axes[1].errorbar(indices, means, yerr=stds, fmt="o-", capsize=3, lw=1.2)
    for i, c in zip(indices, counts):
        axes[1].annotate(str(c), (i, means[indices.index(i)]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    axes[1].set_xlabel("Pass-line index")
    axes[1].set_ylabel("Return (mean ± std)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "mc_pass_line_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_geometry_scatter(mc_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Scatter of geometry perturbations vs episode return (epistemic layout)."""
    episodes = load_mc_trajectories(mc_dir, seeds)
    if not episodes:
        return

    deltas = {
        "delta_L1": [],
        "delta_L2": [],
        "delta_h": [],
        "return": [],
    }
    for ep in episodes:
        ctx = ep.get("episode_context", {})
        if not ctx.get("geometry_uncertainty_enabled", True):
            continue
        deltas["return"].append(float(ep["return"]))
        for key in ("delta_L1", "delta_L2", "delta_h"):
            deltas[key].append(float(ctx.get(key, 0.0)))

    if len(deltas["return"]) < 3:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, label in zip(
        axes,
        ("delta_L1", "delta_L2", "delta_h"),
        (r"$\delta L_1$", r"$\delta L_2$", r"$\delta h$"),
    ):
        x = np.asarray(deltas[key])
        y = np.asarray(deltas["return"])
        ax.scatter(x, y, alpha=0.65, s=28, edgecolors="k", linewidths=0.3)
        if np.std(x) > 1e-12:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(np.min(x), np.max(x), 50)
            ax.plot(xs, np.polyval(coef, xs), "r--", lw=1)
        ax.set_xlabel(f"Relative perturbation {label}")
        ax.set_ylabel("Episode return")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Return vs {label}")

    fig.suptitle("Geometry uncertainty vs policy performance")
    fig.tight_layout()
    out_path = out_dir / "mc_geometry_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sensor_psd(mc_dir: Path, out_dir: Path, seeds: list[int], dt: float = 0.001) -> None:
    """
    Welch PSD of sensor displacement — chatter frequency content (mean ± std across MC).

    PSD estimate: S_xx(f) from scipy.signal.welch per episode, then averaged.
    """
    try:
        from scipy.signal import welch
    except ImportError:
        print("scipy required for PSD plots; skipping mc_sensor_psd.png")
        return

    episodes = load_mc_trajectories(mc_dir, seeds)
    sensor_w, _ = _pad_mc_sensor_series(episodes, "sensor_w")
    if sensor_w.size == 0:
        return

    n_ep, max_t, n_sensors = sensor_w.shape
    fs = 1.0 / dt if dt > 0 else 1000.0

    psd_list = []
    freqs = None
    for i in range(n_ep):
        for s in range(n_sensors):
            sig = sensor_w[i, :, s]
            valid = ~np.isnan(sig)
            if np.sum(valid) < 64:
                continue
            f, pxx = welch(sig[valid], fs=fs, nperseg=min(256, np.sum(valid)))
            if freqs is None:
                freqs = f
            elif len(f) != len(freqs):
                continue
            psd_list.append(pxx)

    if not psd_list or freqs is None:
        return

    psd_mat = np.asarray(psd_list, dtype=np.float64)
    mean_psd = np.mean(psd_mat, axis=0)
    std_psd = np.std(psd_mat, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(freqs, mean_psd, lw=1.8, label="mean PSD")
    ax.fill_between(
        freqs,
        np.maximum(mean_psd - std_psd, 1e-20),
        mean_psd + std_psd,
        alpha=0.35,
        label="±1 std",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD $|w_s(f)|^2$")
    ax.set_title("Sensor displacement power spectral density (Monte Carlo)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "mc_sensor_psd.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_phase_portraits(mc_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Phase portraits w_s vs w_dot_s with MC cloud and mean trajectory."""
    episodes = load_mc_trajectories(mc_dir, seeds)
    sensor_w, times = _pad_mc_sensor_series(episodes, "sensor_w")
    sensor_wd, _ = _pad_mc_sensor_series(episodes, "sensor_w_dot")
    if sensor_w.size == 0 or sensor_wd.size == 0:
        return

    n_sensors = sensor_w.shape[2]
    fig, axes = plt.subplots(1, n_sensors, figsize=(5 * n_sensors, 4.5))
    if n_sensors == 1:
        axes = [axes]

    for s, ax in enumerate(axes):
        w = sensor_w[:, :, s].reshape(-1)
        wd = sensor_wd[:, :, s].reshape(-1)
        valid = ~(np.isnan(w) | np.isnan(wd))
        ax.scatter(w[valid], wd[valid], s=4, alpha=0.15, c="C0", edgecolors="none")

        mean_w, std_w, vmask = _nan_mean_std(sensor_w[:, :, s])
        mean_wd, std_wd, _ = _nan_mean_std(sensor_wd[:, :, s])
        t_grid, _ = _time_grid(times, sensor_w.shape[1])
        ax.plot(mean_w[vmask], mean_wd[vmask], "r-", lw=2, label="mean trajectory")
        ax.set_xlabel(r"$w_{s}$ (m)")
        ax.set_ylabel(r"$\dot{w}_{s}$ (m/s)")
        ax.set_title(f"Sensor {s + 1} phase portrait")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Displacement–velocity phase space (Monte Carlo ensemble)")
    fig.tight_layout()
    out_path = out_dir / "mc_phase_portraits.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mc_sensor_envelopes(mc_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Time-domain sensor envelopes with mean ± std across Monte Carlo rollouts."""
    episodes = load_mc_trajectories(mc_dir, seeds)
    sensor_w, times = _pad_mc_sensor_series(episodes, "sensor_w")
    sensor_wd, _ = _pad_mc_sensor_series(episodes, "sensor_w_dot")
    if sensor_w.size == 0:
        return

    n_sensors = sensor_w.shape[2]
    t_grid, x_label = _time_grid(times, sensor_w.shape[1])
    fig, axes = plt.subplots(2 * n_sensors, 1, figsize=(9, 2.8 * n_sensors), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for s in range(n_sensors):
        ax_w = axes[s]
        mean_w, std_w, valid = _nan_mean_std(sensor_w[:, :, s])
        ax_w.fill_between(t_grid[valid], mean_w[valid] - std_w[valid], mean_w[valid] + std_w[valid], alpha=0.35)
        ax_w.plot(t_grid[valid], mean_w[valid], lw=1.6)
        ax_w.set_ylabel(f"$w_{{s{s+1}}}$ (m)")
        ax_w.grid(True, alpha=0.3)

        ax_v = axes[n_sensors + s]
        mean_v, std_v, valid_v = _nan_mean_std(sensor_wd[:, :, s])
        ax_v.fill_between(t_grid[valid_v], mean_v[valid_v] - std_v[valid_v], mean_v[valid_v] + std_v[valid_v], alpha=0.35)
        ax_v.plot(t_grid[valid_v], mean_v[valid_v], lw=1.6)
        ax_v.set_ylabel(f"$\\dot{{w}}_{{s{s+1}}}$ (m/s)")
        ax_v.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    fig.suptitle("Monte Carlo sensor response envelopes (mean ± std)")
    fig.tight_layout()
    out_path = out_dir / "mc_sensor_envelopes.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mc_summary_table(mc_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Text summary figure with key MC statistics for paper-ready reporting."""
    summaries = load_mc_summaries(mc_dir, seeds)
    if not summaries:
        return

    lines = ["Monte Carlo evaluation summary", ""]
    for sm in summaries:
        lines.append(
            f"n={sm['n_episodes']}  return={sm['return_mean']:.4f}±{sm['return_std']:.4f}  "
            f"success={sm['success_rate']:.1%}  fail={sm['failure_rate']:.1%}"
        )
        lines.append(f"  RMS w_s: {sm['rms_sensor_w_mean']:.2e}±{sm['rms_sensor_w_std']:.2e}")
        lines.append(f"  reasons: {sm.get('termination_reason_counts', {})}")
        lines.append("")

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(lines) + 0.5))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "mc_summary_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mc_research_suite(mc_dir: Path, out_dir: Path, seeds: list[int], step_dt: float = 0.001) -> None:
    """Generate all Monte Carlo research plots."""
    plot_mc_return_distribution(mc_dir, out_dir, seeds)
    plot_pass_line_heatmap(mc_dir, out_dir, seeds)
    plot_geometry_scatter(mc_dir, out_dir, seeds)
    plot_mc_sensor_envelopes(mc_dir, out_dir, seeds)
    plot_phase_portraits(mc_dir, out_dir, seeds)
    plot_sensor_psd(mc_dir, out_dir, seeds, dt=step_dt)
    plot_mc_summary_table(mc_dir, out_dir, seeds)


def plot_trajectory_summary(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Mean±std over time for plate states and physical actions."""
    S, A, T, P, V, SW, SV, metadata = _collect_trajectory_arrays(traj_dir, seeds)

    if S is None or A is None:
        return

    plot_state_trajectories(S, T, metadata, out_dir, sensor_w=SW, sensor_w_dot=SV)
    plot_action_trajectories(A, T, metadata, out_dir)
    if P is not None and V is not None:
        plot_reward_metrics(P, V, T, metadata, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and trajectory results")

    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--mc-dir", default="eval_mc", help="Monte Carlo evaluation output directory")
    parser.add_argument("--mc-only", action="store_true", help="Only plot Monte Carlo research figures")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    traj_dir = Path(args.traj_dir)
    mc_dir = Path(args.mc_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.mc_only:
        plot_learning_curve(log_dir, out_dir, args.seeds, smooth=args.smooth)
        plot_trajectory_summary(traj_dir, out_dir, args.seeds)

    if mc_dir.exists():
        summaries = load_mc_summaries(mc_dir, args.seeds)
        step_dt = float(summaries[0]["step_dt"]) if summaries else 0.001
        plot_mc_research_suite(mc_dir, out_dir, args.seeds, step_dt=step_dt)


if __name__ == "__main__":
    main()