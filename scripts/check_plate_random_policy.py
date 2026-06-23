"""Check CustomODEPlate-v0 with random or fixed actions and pre-training plots.

Run from the project root:

    python scripts/check_plate_random_policy.py
    python scripts/check_plate_random_policy.py --fixed-action --omega 500 --ac 5

This script does not train any agent. It verifies that:
- environment registration works
- reset() / step() work
- observation/action shapes are compatible
- rewards and states stay finite
- modal vibration responds to fixed spindle speed and depth of cut
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from custom_rl import DEFAULT_PLOT_DIR, register_envs

ENV_ID = "CustomODEPlate-v0"


def physical_to_normalized(
    omega: float,
    ac: float,
    omega_min: float,
    omega_max: float,
    ac_min: float,
    ac_max: float,
) -> np.ndarray:
    """Map physical [omega, ac] to normalized action in [-1, 1]^2."""
    low = np.array([omega_min, ac_min], dtype=np.float64)
    high = np.array([omega_max, ac_max], dtype=np.float64)
    u_phys = np.array([omega, ac], dtype=np.float64)
    frac = (u_phys - low) / (high - low)
    return np.clip(2.0 * frac - 1.0, -1.0, 1.0)


def normalized_to_physical(
    action: np.ndarray,
    omega_min: float,
    omega_max: float,
    ac_min: float,
    ac_max: float,
) -> np.ndarray:
    """Map normalized action in [-1, 1]^2 to physical [omega, ac]."""
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
    low = np.array([omega_min, ac_min], dtype=np.float64)
    high = np.array([omega_max, ac_max], dtype=np.float64)
    return low + 0.5 * (action + 1.0) * (high - low)


def state_labels(state_dim: int) -> list[str]:
    labels: list[str] = []
    for dim in range(state_dim):
        mode_index = dim // 2 + 1
        if dim % 2 == 0:
            labels.append(f"eta{mode_index}")
        else:
            labels.append(f"eta{mode_index}_dot")
    return labels


def plot_rollout(
    times: np.ndarray,
    states: np.ndarray,
    actions_norm: np.ndarray,
    actions_phys: np.ndarray,
    rewards: np.ndarray,
    terminated_flags: np.ndarray,
    truncated_flags: np.ndarray,
    eta_limit: float | None,
    out_dir: Path,
    title_suffix: str,
) -> None:
    """Save pre-training diagnostic plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dim = states.shape[1]
    labels = state_labels(state_dim)

    # --- Modal states ---
    n_cols = 2
    n_rows = int(np.ceil(state_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.0 * n_rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for dim in range(state_dim):
        ax = axes[dim]
        ax.plot(times, states[:, dim], lw=1.2)
        if eta_limit is not None and np.isfinite(eta_limit) and dim % 2 == 0:
            ax.axhline(+eta_limit, linestyle="--", linewidth=1, color="tab:red", alpha=0.7)
            ax.axhline(-eta_limit, linestyle="--", linewidth=1, color="tab:red", alpha=0.7)
        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.3)

    for ax in axes[state_dim:]:
        ax.axis("off")

    axes[min(state_dim - 1, len(axes) - 1)].set_xlabel("Time (s)")
    fig.suptitle(f"Modal state trajectories ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_modal_states.png", dpi=150)
    plt.close(fig)

    # --- Physical actions ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for dim, name in enumerate(["omega (rad/s)", "ac (depth of cut)"]):
        axes[dim].plot(times, actions_phys[:, dim], lw=1.5)
        axes[dim].set_ylabel(name)
        axes[dim].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Physical control inputs ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_actions_physical.png", dpi=150)
    plt.close(fig)

    # --- Normalized actions ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for dim, name in enumerate(["u_omega", "u_ac"]):
        axes[dim].plot(times, actions_norm[:, dim], lw=1.5)
        axes[dim].axhline(-1.0, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
        axes[dim].axhline(+1.0, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
        axes[dim].set_ylabel(name)
        axes[dim].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Normalized policy actions ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_actions_normalized.png", dpi=150)
    plt.close(fig)

    # --- Reward ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, rewards, lw=1.2, color="tab:green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Step reward ({title_suffix})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_rewards.png", dpi=150)
    plt.close(fig)

    # --- Cumulative reward ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, np.cumsum(rewards), lw=1.5, color="tab:purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative reward")
    ax.set_title(f"Cumulative reward ({title_suffix})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_cumulative_reward.png", dpi=150)
    plt.close(fig)

    # --- Termination markers ---
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.step(times, terminated_flags.astype(float), where="post", label="terminated")
    ax.step(times, truncated_flags.astype(float), where="post", label="truncated")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flag")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Episode termination flags")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_termination_flags.png", dpi=150)
    plt.close(fig)

    print(f"Saved pre-training plots to {out_dir.resolve()}")


def run_rollout(
  env: gym.Env,
  max_steps: int,
  seed: int,
  fixed_action: np.ndarray | None,
) -> dict[str, np.ndarray]:
    obs, info = env.reset(seed=seed)
    plant = env.unwrapped.plant

    times: list[float] = [0.0]
    states: list[np.ndarray] = [obs.copy()]
    actions_norm: list[np.ndarray] = []
    actions_phys: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []

    for step in range(max_steps):
        if fixed_action is not None:
            action = fixed_action.copy()
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        actions_norm.append(np.asarray(action, dtype=np.float64))
        actions_phys.append(
            normalized_to_physical(
                action,
                plant.omega_min,
                plant.omega_max,
                plant.ac_min,
                plant.ac_max,
            )
        )
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))

        times.append(float(info.get("t", step + 1)))
        states.append(obs.copy())

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}.")
            print("  terminated:", terminated)
            print("  truncated:", truncated)
            if "termination_reason" in info:
                print("  reason:", info["termination_reason"])
            break

    return {
        "times": np.asarray(times[1:], dtype=np.float64),
        "states": np.asarray(states[1:], dtype=np.float64),
        "actions_norm": np.asarray(actions_norm, dtype=np.float64),
        "actions_phys": np.asarray(actions_phys, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "terminated_flags": np.asarray(terminated_flags, dtype=bool),
        "truncated_flags": np.asarray(truncated_flags, dtype=bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plate env smoke test with optional plots.")
    parser.add_argument("--reward", default="dense", choices=["dense", "sparse", "quadratic"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--out-dir", default=str(Path(DEFAULT_PLOT_DIR) / "pretrain"))
    parser.add_argument("--fixed-action", action="store_true", help="Use constant omega/ac.")
    parser.add_argument("--omega", type=float, default=500.0, help="Physical spindle speed.")
    parser.add_argument("--ac", type=float, default=5.0, help="Physical depth of cut.")
    args = parser.parse_args()

    register_envs()

    env = gym.make(
        ENV_ID,
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=1,
        max_episode_steps=args.max_episode_steps,
    )
    plant = env.unwrapped.plant

    fixed_action = None
    if args.fixed_action:
        fixed_action = physical_to_normalized(
            args.omega,
            args.ac,
            plant.omega_min,
            plant.omega_max,
            plant.ac_min,
            plant.ac_max,
        )

    print("Environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    if fixed_action is not None:
        print(
            f"Fixed physical action: omega={args.omega}, ac={args.ac} "
            f"(normalized={fixed_action})"
        )

    rollout = run_rollout(env, args.steps, args.seed, fixed_action)

    assert rollout["states"].shape[1] == env.observation_space.shape[0]
    assert np.all(np.isfinite(rollout["states"]))
    assert np.all(np.isfinite(rollout["rewards"]))

    env.close()

    policy_label = (
        f"fixed omega={args.omega}, ac={args.ac}"
        if fixed_action is not None
        else "random policy"
    )

    plot_rollout(
        times=rollout["times"],
        states=rollout["states"],
        actions_norm=rollout["actions_norm"],
        actions_phys=rollout["actions_phys"],
        rewards=rollout["rewards"],
        terminated_flags=rollout["terminated_flags"],
        truncated_flags=rollout["truncated_flags"],
        eta_limit=getattr(plant, "eta_limit", None),
        out_dir=Path(args.out_dir),
        title_suffix=policy_label,
    )

    print("Rollout test finished successfully.")
    print("Total reward:", float(np.sum(rollout["rewards"])))
    print("Final modal displacements:", rollout["states"][-1, 0::2])
    print("Final modal velocities:", rollout["states"][-1, 1::2])


if __name__ == "__main__":
    main()
