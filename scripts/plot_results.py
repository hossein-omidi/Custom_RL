"""Plot training results: mean±std learning curves and trajectory summaries."""

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
    # First cols: r, l (sometimes with t)
    returns = data[:, 0]
    lengths = data[:, 1] if data.shape[1] > 1 else np.full_like(returns, 1.0)
    return returns, lengths


def load_all_seed_logs(log_dir: Path, seeds: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load and aggregate monitor logs for each seed (supports multiple parallel env logs)."""
    out = {}
    for s in seeds:
        seed_dir = log_dir / f"seed_{s}"
        if not seed_dir.exists():
            continue

        # Collect all monitor files (single or parallel envs)
        monitor_files = list(seed_dir.glob("*.monitor.csv"))
        if not monitor_files:
            # Fallback: old-style single monitor.csv
            single = seed_dir / "monitor.csv"
            if single.exists():
                monitor_files = [single]

        if not monitor_files:
            continue

        # Load and concatenate all monitor logs (from parallel envs)
        all_returns, all_lengths = [], []
        for mf in sorted(monitor_files):
            r, l = load_monitor_csv(mf)
            if len(r) > 0:
                all_returns.append(r)
                all_lengths.append(l)

        if all_returns:
            # Concatenate episodes from all parallel envs
            returns = np.concatenate(all_returns)
            lengths = np.concatenate(all_lengths)
            out[s] = (returns, lengths)
    return out


def plot_learning_curve(log_dir: Path, out_dir: Path, seeds: list[int], smooth: int = 10) -> None:
    """Episode return vs environment steps with mean±std across seeds."""
    data = load_all_seed_logs(log_dir, seeds)
    if not data:
        print(f"No monitor data in {log_dir}")
        return

    valid = [(r, l) for r, l in data.values() if len(r) > 0]
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
        ret_ext = np.concatenate([[returns[0]], returns])
        r_interp = np.interp(step_grid, steps, ret_ext)
        returns_interp.append(r_interp)

    if not returns_interp:
        return

    returns_matrix = np.array(returns_interp)
    mean_ret = np.mean(returns_matrix, axis=0)
    std_ret = np.std(returns_matrix, axis=0)
    if smooth > 1:
        k = min(smooth, len(mean_ret) // 2 or 1)
        mean_ret = np.convolve(mean_ret, np.ones(k) / k, mode="same")
        std_ret = np.convolve(std_ret, np.ones(k) / k, mode="same")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(step_grid, mean_ret - std_ret, mean_ret + std_ret, alpha=0.3)
    ax.plot(step_grid, mean_ret, lw=2)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("Training: Episode return (mean ± std)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'learning_curve.png'}")


def load_trajectories(traj_dir: Path, seeds: list[int]) -> dict[int, list]:
    out = {}
    for s in seeds:
        p = traj_dir / f"trajectories_seed{s}.json"
        if p.exists():
            with open(p) as f:
                out[s] = json.load(f)
    return out


def plot_trajectory_summary(traj_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Mean±std over time for key states across seeds/episodes."""
    data = load_trajectories(traj_dir, seeds)
    if not data:
        print(f"No trajectory data in {traj_dir}")
        return

    # Stack trajectories: (seed, ep, t, state_dim)
    all_trajs = []
    for seed in sorted(data.keys()):
        for ep in data[seed]:
            states = np.array(ep["states"])
            actions = np.array(ep["actions"])
            if len(states) == 0:
                continue
            # state: [x, x_dot, theta, theta_dot]
            all_trajs.append((states, actions))

    if not all_trajs:
        return

    # Align by max T (pad shorter)
    max_t = max(s.shape[0] for s, a in all_trajs)
    n_seeds_eps = len(all_trajs)
    S = np.full((n_seeds_eps, max_t, 4), np.nan)
    A = np.full((n_seeds_eps, max_t, 1), np.nan)
    for i, (states, actions) in enumerate(all_trajs):
        T = states.shape[0]
        S[i, :T] = states
        A[i, :T] = actions

    t_grid = np.arange(max_t)
    labels = ["x (cart pos)", "x_dot", "theta (angle)", "theta_dot"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    for dim, (ax, lab) in enumerate(zip(axes[:4], labels)):
        mean_s = np.nanmean(S[:, :, dim], axis=0)
        std_s = np.nanstd(S[:, :, dim], axis=0)
        valid = ~np.isnan(mean_s)
        ax.fill_between(t_grid[valid], mean_s[valid] - std_s[valid], mean_s[valid] + std_s[valid], alpha=0.3)
        ax.plot(t_grid[valid], mean_s[valid], lw=1.5)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle("Evaluation trajectories: state (mean ± std)")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_states.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'trajectory_states.png'}")


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
