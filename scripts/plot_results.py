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
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """Collect trajectory states, physical actions, and times into padded arrays."""
    data = load_trajectories(traj_dir, seeds)

    if not data:
        print(f"No trajectory data in {traj_dir}")
        return None, None, None, {}

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

            if states.ndim == 1:
                states = states.reshape(-1, 1)

            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)

            if times.size != states.shape[0]:
                times = np.arange(states.shape[0], dtype=np.float64)

            if not metadata and "metadata" in ep:
                metadata = ep["metadata"]

            all_trajs.append((states, actions, times))

    if not all_trajs:
        print(f"No valid trajectories in {traj_dir}")
        return None, None, None, metadata

    state_dim = all_trajs[0][0].shape[1]
    action_dim = all_trajs[0][1].shape[1]
    max_t = max(states.shape[0] for states, _, _ in all_trajs)

    S = np.full((len(all_trajs), max_t, state_dim), np.nan, dtype=np.float64)
    A = np.full((len(all_trajs), max_t, action_dim), np.nan, dtype=np.float64)
    T = np.full((len(all_trajs), max_t), np.nan, dtype=np.float64)

    valid_count = 0

    for states, actions, times in all_trajs:
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

    if valid_count == 0:
        print("No trajectories with consistent dimensions.")
        return None, None, None, metadata

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
    """Plot all plate modal states with mean±std and displacement safety bounds."""
    state_dim = S.shape[2]
    labels = _state_labels(state_dim)

    t_grid, x_label = _time_grid(T, S.shape[1])

    eta_limit = metadata.get("eta_limit", None)

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
            mean_s[valid] - std_s[valid],
            mean_s[valid] + std_s[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_s[valid], lw=1.5)

        # Show displacement termination bounds only for eta states.
        if eta_limit is not None and np.isfinite(eta_limit) and dim % 2 == 0:
            ax.axhline(+eta_limit, linestyle="--", linewidth=1)
            ax.axhline(-eta_limit, linestyle="--", linewidth=1)

        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.3)

    for ax in axes[state_dim:]:
        ax.axis("off")

    axes[min(state_dim - 1, len(axes) - 1)].set_xlabel(x_label)

    fig.suptitle("Evaluation trajectories: plate modal states")
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
        action_labels = ["omega", "ac"]
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


def plot_trajectory_summary(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Mean±std over time for plate states and physical actions."""
    S, A, T, metadata = _collect_trajectory_arrays(traj_dir, seeds)

    if S is None or A is None:
        return

    plot_state_trajectories(S, T, metadata, out_dir)
    plot_action_trajectories(A, T, metadata, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and trajectory results")

    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--smooth", type=int, default=10)

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(log_dir, out_dir, args.seeds, smooth=args.smooth)
    plot_trajectory_summary(traj_dir, out_dir, args.seeds)


if __name__ == "__main__":
    main()