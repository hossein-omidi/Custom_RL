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
from custom_rl.eval.monte_carlo import (
    physical_w_from_info_or_obs,
    physical_wdot_from_info_or_obs,
    plot_monte_carlo_rollouts,
)

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


def obs_labels(n_sensors: int) -> list[str]:
    labels: list[str] = []
    for i in range(n_sensors):
        labels.append(f"w_sensor_{i + 1} (m)")
    for i in range(n_sensors):
        labels.append(f"wdot_sensor_{i + 1} (m/s)")
    return labels


def modal_labels(state_dim: int) -> list[str]:
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
    observations: np.ndarray,
    x_modal: np.ndarray | None,
    actions_norm: np.ndarray,
    actions_phys: np.ndarray,
    rewards: np.ndarray,
    terminated_flags: np.ndarray,
    truncated_flags: np.ndarray,
    w_limit: float | None,
    n_sensors: int,
    w_obs_scale: float,
    wdot_obs_scale: float,
    out_dir: Path,
    title_suffix: str,
) -> None:
    """Save pre-training diagnostic plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dim = observations.shape[1]
    labels = obs_labels(n_sensors)

    # --- Physical sensor observations (unscaled) ---
    obs_physical = observations.copy()
    if n_sensors > 0 and obs_dim >= 2 * n_sensors:
        obs_physical[:, :n_sensors] *= w_obs_scale
        obs_physical[:, n_sensors : 2 * n_sensors] *= wdot_obs_scale

    n_cols = 2
    n_rows = int(np.ceil(obs_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.0 * n_rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for dim in range(obs_dim):
        ax = axes[dim]
        ax.plot(times, obs_physical[:, dim], lw=1.2)
        if w_limit is not None and np.isfinite(w_limit) and dim < n_sensors:
            ax.axhline(+w_limit, linestyle="--", linewidth=1, color="tab:red", alpha=0.7)
            ax.axhline(-w_limit, linestyle="--", linewidth=1, color="tab:red", alpha=0.7)
        ax.set_ylabel(labels[dim] if dim < len(labels) else f"obs{dim}")
        ax.grid(True, alpha=0.3)

    for ax in axes[obs_dim:]:
        ax.axis("off")

    axes[min(obs_dim - 1, len(axes) - 1)].set_xlabel("Time (s)")
    fig.suptitle(f"Physical sensor response ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_physical_sensor_response.png", dpi=150)
    plt.close(fig)

    if x_modal is not None and x_modal.size > 0:
        n_align = min(len(times), x_modal.shape[0])
        times_m = times[:n_align]
        x_modal = x_modal[:n_align]
        modal_dim = x_modal.shape[1]
        modal_lbls = modal_labels(modal_dim)
        n_rows_m = int(np.ceil(modal_dim / n_cols))
        fig, axes = plt.subplots(
            n_rows_m, n_cols, figsize=(10, 3.0 * n_rows_m), sharex=True
        )
        axes = np.asarray(axes).reshape(-1)
        for dim in range(modal_dim):
            ax = axes[dim]
            ax.plot(times_m, x_modal[:, dim], lw=1.0, alpha=0.8)
            ax.set_ylabel(modal_lbls[dim])
            ax.grid(True, alpha=0.3)
        for ax in axes[modal_dim:]:
            ax.axis("off")
        axes[min(modal_dim - 1, len(axes) - 1)].set_xlabel("Time (s)")
        fig.suptitle(f"Modal state (debug) ({title_suffix})")
        fig.tight_layout()
        fig.savefig(out_dir / "pretrain_modal_states_debug.png", dpi=150)
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
  reset_options: dict | None = None,
) -> dict[str, np.ndarray]:
    if reset_options is not None:
        obs, info = env.reset(seed=seed, options=reset_options)
    else:
        obs, info = env.reset(seed=seed)
    plant = env.unwrapped.plant
    if "y0" in info:
        print(f"  reset y0={info['y0']:.3f} m")

    times: list[float] = [0.0]
    observations: list[np.ndarray] = [obs.copy()]
    x_modal_hist: list[np.ndarray] = []
    actions_norm: list[np.ndarray] = []
    actions_phys: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []

    w_sensor_hist: list[np.ndarray] = []
    wdot_sensor_hist: list[np.ndarray] = []
    reward_terms_hist: list[dict] = []

    for step in range(max_steps):
        if fixed_action is not None:
            action = fixed_action.copy()
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if "w_sensor" in info:
            w_sensor_hist.append(np.asarray(info["w_sensor"], dtype=np.float64))
        else:
            w_sensor_hist.append(
                physical_w_from_info_or_obs(
                    info,
                    obs,
                    n_sensors=plant.n_sensors,
                    w_obs_scale=plant.w_obs_scale,
                )
            )
        if "wdot_sensor" in info:
            wdot_sensor_hist.append(np.asarray(info["wdot_sensor"], dtype=np.float64))
        else:
            wdot_sensor_hist.append(
                physical_wdot_from_info_or_obs(
                    info,
                    obs,
                    n_sensors=plant.n_sensors,
                    wdot_obs_scale=plant.wdot_obs_scale,
                )
            )
        if "reward_terms" in info:
            reward_terms_hist.append(dict(info["reward_terms"]))

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
        observations.append(obs.copy())
        if "x_modal" in info:
            x_modal_hist.append(np.asarray(info["x_modal"], dtype=np.float64))

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}.")
            print("  terminated:", terminated)
            print("  truncated:", truncated)
            if "termination_reason" in info:
                print("  reason:", info["termination_reason"])
            if "feed_progress" in info:
                print("  feed_progress:", f"{info['feed_progress']:.3f}")
            if info.get("pass_completed"):
                print("  pass_completed: True")
            break

    x_modal_arr = None
    if x_modal_hist:
        x_modal_arr = np.asarray(x_modal_hist, dtype=np.float64)

    return {
        "times": np.asarray(times[1:], dtype=np.float64),
        "observations": np.asarray(observations[1:], dtype=np.float64),
        "x_modal": x_modal_arr,
        "w_sensor": np.asarray(w_sensor_hist, dtype=np.float64) if w_sensor_hist else None,
        "wdot_sensor": np.asarray(wdot_sensor_hist, dtype=np.float64) if wdot_sensor_hist else None,
        "reward_terms": reward_terms_hist,
        "actions_norm": np.asarray(actions_norm, dtype=np.float64),
        "actions_phys": np.asarray(actions_phys, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "terminated_flags": np.asarray(terminated_flags, dtype=bool),
        "truncated_flags": np.asarray(truncated_flags, dtype=bool),
    }


def plot_monte_carlo_physical(
    rollouts: list[dict],
    *,
    n_sensors: int,
    w_limit: float | None,
    out_dir: Path,
    title_suffix: str,
    default_dt: float = 0.002,
) -> None:
    """Plot mean +/- std bands for physical sensor displacement and velocity."""
    plot_monte_carlo_rollouts(
        rollouts,
        n_sensors=n_sensors,
        w_limit=w_limit,
        out_dir=out_dir,
        title_suffix=title_suffix,
        default_dt=default_dt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plate env smoke test with optional plots.")
    parser.add_argument("--reward", default="dense", choices=["dense", "sparse", "quadratic"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Env step limit (0 = auto from pass duration)",
    )
    parser.add_argument("--out-dir", default=str(Path(DEFAULT_PLOT_DIR) / "pretrain"))
    parser.add_argument("--fixed-action", action="store_true", help="Use constant omega/ac.")
    parser.add_argument("--omega", type=float, default=500.0, help="Physical spindle speed [rad/s].")
    parser.add_argument("--ac", type=float, default=5.0, help="Physical depth of cut [mm].")
    parser.add_argument(
        "--n-mc",
        type=int,
        default=1,
        help="Monte Carlo rollouts (mean/std bands when > 1)",
    )
    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=0.0,
        help="Gaussian disturbance on modal accelerations in dynamics [0=off]",
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample milling start y0 on each reset (uniform over y0 range)",
    )
    parser.add_argument("--y0", type=float, default=None, help="Fixed milling start y0 [m]")
    args = parser.parse_args()

    register_envs()

    env_kwargs: dict = {
        "reward_id": args.reward,
        "dt": args.dt,
        "n_substeps": 1,
        "dynamics_uncertainty_std": args.dynamics_uncertainty_std,
        "randomize_y0": args.randomize_y0,
    }
    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(ENV_ID, **env_kwargs)
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
    print(
        f"Omega bounds [rad/s]: {plant.omega_min} - {plant.omega_max} "
        f"({plant.omega_min * 60 / (2 * np.pi):.0f} - "
        f"{plant.omega_max * 60 / (2 * np.pi):.0f} rpm)"
    )
    print(f"Dynamics uncertainty std: {plant.dynamics_uncertainty_std}")
    if fixed_action is not None:
        print(
            f"Fixed physical action: omega={args.omega}, ac={args.ac} "
            f"(normalized={fixed_action})"
        )

    rollouts: list[dict] = []
    reset_options = {"y0": args.y0} if args.y0 is not None else None
    for mc in range(args.n_mc):
        rollout = run_rollout(
            env,
            args.steps,
            args.seed + mc,
            fixed_action,
            reset_options=reset_options,
        )
        rollouts.append(rollout)

    rollout = rollouts[0]

    assert rollout["observations"].shape[1] == env.observation_space.shape[0]
    assert np.all(np.isfinite(rollout["observations"]))
    assert np.all(np.isfinite(rollout["rewards"]))

    env.close()

    policy_label = (
        f"fixed omega={args.omega}, ac={args.ac}"
        if fixed_action is not None
        else "random policy"
    )

    plot_rollout(
        times=rollout["times"],
        observations=rollout["observations"],
        x_modal=rollout["x_modal"],
        actions_norm=rollout["actions_norm"],
        actions_phys=rollout["actions_phys"],
        rewards=rollout["rewards"],
        terminated_flags=rollout["terminated_flags"],
        truncated_flags=rollout["truncated_flags"],
        w_limit=getattr(plant, "w_limit", None),
        n_sensors=getattr(plant, "n_sensors", 2),
        w_obs_scale=getattr(plant, "w_obs_scale", 0.01),
        wdot_obs_scale=getattr(plant, "wdot_obs_scale", 1.0),
        out_dir=Path(args.out_dir),
        title_suffix=policy_label,
    )

    if args.n_mc > 1:
        plot_monte_carlo_physical(
            rollouts,
            n_sensors=getattr(plant, "n_sensors", 2),
            w_limit=getattr(plant, "w_limit", None),
            out_dir=Path(args.out_dir),
            title_suffix=f"{policy_label}, n_mc={args.n_mc}",
            default_dt=args.dt,
        )

    if rollout["reward_terms"]:
        terms = rollout["reward_terms"][-1]
        print("Last-step reward terms:", terms)

    print("Rollout test finished successfully.")
    print("Total reward:", float(np.sum(rollout["rewards"])))
    if rollout["x_modal"] is not None and rollout["x_modal"].size > 0:
        print("Final modal displacements:", rollout["x_modal"][-1, 0::2])
        print("Final modal velocities:", rollout["x_modal"][-1, 1::2])
    w_scale = getattr(plant, "w_obs_scale", 0.01)
    wdot_scale = getattr(plant, "wdot_obs_scale", 1.0)
    obs_last = rollout["observations"][-1]
    n_sensors = getattr(plant, "n_sensors", 2)
    print("Final sensor displacement (m):", obs_last[:n_sensors] * w_scale)
    print("Final sensor velocity (m/s):", obs_last[n_sensors : 2 * n_sensors] * wdot_scale)


if __name__ == "__main__":
    main()
