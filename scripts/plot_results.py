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
        # SB3 Monitor: first line #comment, second "r,l,t", then data
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
    """
    Create labels for plate modal state.

    State:
        [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]
    """
    labels = []

    for dim in range(state_dim):
        mode_index = dim // 2 + 1

        if dim % 2 == 0:
            labels.append(f"eta{mode_index}")
        else:
            labels.append(f"eta{mode_index}_dot")

    return labels


def _collect_trajectory_arrays(
    traj_dir: Path,
    seeds: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Collect trajectory states and actions into padded arrays."""
    data = load_trajectories(traj_dir, seeds)

    if not data:
        print(f"No trajectory data in {traj_dir}")
        return None, None

    all_trajs = []

    for seed in sorted(data.keys()):
        for ep in data[seed]:
            states = np.asarray(ep["states"], dtype=np.float64)
            actions = np.asarray(ep["actions"], dtype=np.float64)

            if states.size == 0 or actions.size == 0:
                continue

            if states.ndim == 1:
                states = states.reshape(-1, 1)

            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)

            all_trajs.append((states, actions))

    if not all_trajs:
        print(f"No valid trajectories in {traj_dir}")
        return None, None

    state_dim = all_trajs[0][0].shape[1]
    action_dim = all_trajs[0][1].shape[1]
    max_t = max(states.shape[0] for states, _ in all_trajs)

    S = np.full((len(all_trajs), max_t, state_dim), np.nan, dtype=np.float64)
    A = np.full((len(all_trajs), max_t, action_dim), np.nan, dtype=np.float64)

    valid_count = 0

    for states, actions in all_trajs:
        if states.shape[1] != state_dim or actions.shape[1] != action_dim:
            continue

        idx = valid_count
        valid_count += 1

        state_steps = states.shape[0]
        action_steps = actions.shape[0]

        S[idx, :state_steps, :] = states
        A[idx, :action_steps, :] = actions

    if valid_count == 0:
        print("No trajectories with consistent dimensions.")
        return None, None

    return S[:valid_count], A[:valid_count]


def plot_state_trajectories(S: np.ndarray, out_dir: Path) -> None:
    """Plot all plate modal states."""
    state_dim = S.shape[2]
    labels = _state_labels(state_dim)

    max_t = S.shape[1]
    t_grid = np.arange(max_t)

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

        mean_s = np.nanmean(S[:, :, dim], axis=0)
        std_s = np.nanstd(S[:, :, dim], axis=0)
        valid = ~np.isnan(mean_s)

        ax.fill_between(
            t_grid[valid],
            mean_s[valid] - std_s[valid],
            mean_s[valid] + std_s[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_s[valid], lw=1.5)

        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.3)

    for ax in axes[state_dim:]:
        ax.axis("off")

    axes[min(state_dim - 1, len(axes) - 1)].set_xlabel("Step")

    fig.suptitle("Evaluation trajectories: plate modal states")
    fig.tight_layout()

    out_path = out_dir / "trajectory_states.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_action_trajectories(A: np.ndarray, out_dir: Path) -> None:
    """Plot policy actions."""
    action_dim = A.shape[2]
    max_t = A.shape[1]
    t_grid = np.arange(max_t)

    action_labels = ["u_omega", "u_ac"]

    if action_dim != 2:
        action_labels = [f"action{idx}" for idx in range(action_dim)]

    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(8, 3.0 * action_dim),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for dim in range(action_dim):
        ax = axes[dim]

        mean_a = np.nanmean(A[:, :, dim], axis=0)
        std_a = np.nanstd(A[:, :, dim], axis=0)
        valid = ~np.isnan(mean_a)

        ax.fill_between(
            t_grid[valid],
            mean_a[valid] - std_a[valid],
            mean_a[valid] + std_a[valid],
            alpha=0.3,
        )
        ax.plot(t_grid[valid], mean_a[valid], lw=1.5)

        ax.set_ylabel(action_labels[dim])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")

    fig.suptitle("Evaluation trajectories: actions")
    fig.tight_layout()

    out_path = out_dir / "trajectory_actions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_trajectory_summary(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Mean±std over time for plate states and actions."""
    S, A = _collect_trajectory_arrays(traj_dir, seeds)

    if S is None or A is None:
        return

    plot_state_trajectories(S, out_dir)
    plot_action_trajectories(A, out_dir)


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