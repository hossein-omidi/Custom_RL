"""Simple smoke test for the face-milling CustomODEPlate-v0 environment.

Run from the project root:

    python scripts/check_plate_random_policy.py
    python scripts/check_plate_random_policy.py --fixed-action --rpm 1000 --ap 0.5

This script does not train an agent and does not run Monte Carlo evaluation.
It checks that:
- environment registration works
- reset() / step() work
- observation/action shapes are compatible
- rewards, states, and physical sensor signals stay finite
- modal states are converted to physical sensor displacement/velocity
- the second control input is axial depth of cut ap [mm]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from custom_rl import DEFAULT_PLOT_DIR, register_envs
from custom_rl.plants.plate import RPM_MAX, RPM_MIN, omega_to_rpm, rpm_to_omega

ENV_ID = "CustomODEPlate-v0"


def _depth_bounds_from_plant(plant: Any) -> tuple[float, float]:
    """Return axial-depth bounds using new ap names, with old ac fallback."""
    ap_min = getattr(plant, "ap_min", getattr(plant, "ac_min", 0.0))
    ap_max = getattr(plant, "ap_max", getattr(plant, "ac_max", 1.0))
    return float(ap_min), float(ap_max)


def _physical_action_bounds(plant: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return physical action bounds [omega, ap] or [omega, ap, ae]."""
    if hasattr(plant, "physical_action_bounds"):
        low, high = plant.physical_action_bounds()
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)

    ap_min, ap_max = _depth_bounds_from_plant(plant)
    low = np.array([plant.omega_min, ap_min], dtype=np.float64)
    high = np.array([plant.omega_max, ap_max], dtype=np.float64)
    return low, high


def physical_to_normalized(
    u_phys: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map physical action to normalized action in [-1, 1]."""
    u_phys = np.asarray(u_phys, dtype=np.float64).reshape(-1)
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)

    if u_phys.size != low.size or low.size != high.size:
        raise ValueError(
            f"Action/bounds shape mismatch: u={u_phys.shape}, "
            f"low={low.shape}, high={high.shape}."
        )

    span = np.maximum(high - low, 1e-12)
    frac = (u_phys - low) / span
    return np.clip(2.0 * frac - 1.0, -1.0, 1.0)


def normalized_to_physical(
    action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map normalized action in [-1, 1] to physical action."""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)

    if action.size != low.size:
        action = action[: low.size]
    action = np.clip(action, -1.0, 1.0)
    return low + 0.5 * (action + 1.0) * (high - low)


def obs_labels(n_sensors: int) -> list[str]:
    """Observation labels after converting scaled sensor terms to physical units."""
    return [
        *[f"w_sensor_{i + 1} (m)" for i in range(n_sensors)],
        *[f"wdot_sensor_{i + 1} (m/s)" for i in range(n_sensors)],
        "cutter_x/L1",
        "cutter_y/L2",
    ]


def modal_labels(state_dim: int) -> list[str]:
    """Labels for modal state [eta1, eta1_dot, eta2, eta2_dot, ...]."""
    labels: list[str] = []
    for dim in range(state_dim):
        mode_index = dim // 2 + 1
        labels.append(f"eta{mode_index}" if dim % 2 == 0 else f"eta{mode_index}_dot")
    return labels


def _as_2d_or_none(values: list[np.ndarray]) -> np.ndarray | None:
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def run_rollout(
    env: gym.Env,
    max_steps: int,
    seed: int,
    fixed_action: np.ndarray | None,
    reset_options: dict[str, Any] | None = None,
) -> dict[str, np.ndarray | list[dict[str, Any]] | None]:
    """Run one random/fixed-policy rollout and collect diagnostic arrays."""
    obs, info = env.reset(seed=seed, options=reset_options)
    plant = env.unwrapped.plant
    action_low, action_high = _physical_action_bounds(plant)

    times: list[float] = []
    observations: list[np.ndarray] = []
    x_modal_hist: list[np.ndarray] = []
    w_sensor_hist: list[np.ndarray] = []
    wdot_sensor_hist: list[np.ndarray] = []
    actions_norm: list[np.ndarray] = []
    actions_phys: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    reward_terms_hist: list[dict[str, Any]] = []

    print("Reset info:", info)

    for step in range(max_steps):
        action = fixed_action.copy() if fixed_action is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)

        times.append(float(info.get("t", (step + 1) * getattr(env.unwrapped, "dt", 1.0))))
        observations.append(obs_arr.copy())
        actions_norm.append(action_arr.copy())
        actions_phys.append(normalized_to_physical(action_arr, action_low, action_high))
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))

        if "x_modal" in info:
            x_modal_hist.append(np.asarray(info["x_modal"], dtype=np.float64).reshape(-1))
        if "w_sensor" in info:
            w_sensor_hist.append(np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1))
        if "wdot_sensor" in info:
            wdot_sensor_hist.append(np.asarray(info["wdot_sensor"], dtype=np.float64).reshape(-1))
        if "reward_terms" in info:
            reward_terms_hist.append(dict(info["reward_terms"]))

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

    return {
        "times": np.asarray(times, dtype=np.float64),
        "observations": np.asarray(observations, dtype=np.float64),
        "x_modal": _as_2d_or_none(x_modal_hist),
        "w_sensor": _as_2d_or_none(w_sensor_hist),
        "wdot_sensor": _as_2d_or_none(wdot_sensor_hist),
        "actions_norm": np.asarray(actions_norm, dtype=np.float64),
        "actions_phys": np.asarray(actions_phys, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "terminated_flags": np.asarray(terminated_flags, dtype=bool),
        "truncated_flags": np.asarray(truncated_flags, dtype=bool),
        "reward_terms": reward_terms_hist,
    }


def plot_rollout(
    rollout: dict[str, np.ndarray | list[dict[str, Any]] | None],
    *,
    plant: Any,
    out_dir: Path,
    title_suffix: str,
) -> None:
    """Save simple diagnostic plots for one rollout."""
    out_dir.mkdir(parents=True, exist_ok=True)

    times = np.asarray(rollout["times"], dtype=np.float64)
    observations = np.asarray(rollout["observations"], dtype=np.float64)
    x_modal = rollout["x_modal"]
    actions_norm = np.asarray(rollout["actions_norm"], dtype=np.float64)
    actions_phys = np.asarray(rollout["actions_phys"], dtype=np.float64)
    rewards = np.asarray(rollout["rewards"], dtype=np.float64)
    terminated_flags = np.asarray(rollout["terminated_flags"], dtype=bool)
    truncated_flags = np.asarray(rollout["truncated_flags"], dtype=bool)

    if times.size == 0:
        print("No rollout data to plot.")
        return

    n_sensors = int(getattr(plant, "n_sensors", observations.shape[1] // 2))
    w_obs_scale = float(getattr(plant, "w_obs_scale", 1.0))
    wdot_obs_scale = float(getattr(plant, "wdot_obs_scale", 1.0))
    w_limit = getattr(plant, "w_limit", None)

    # Physical sensor observations from scaled observation vector.
    obs_physical = observations.copy()
    if observations.shape[1] >= 2 * n_sensors:
        obs_physical[:, :n_sensors] *= w_obs_scale
        obs_physical[:, n_sensors : 2 * n_sensors] *= wdot_obs_scale

    fig, axes = plt.subplots(
        int(np.ceil(obs_physical.shape[1] / 2)),
        2,
        figsize=(10, 3.0 * int(np.ceil(obs_physical.shape[1] / 2))),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(-1)
    labels = obs_labels(n_sensors)
    for dim in range(obs_physical.shape[1]):
        ax = axes[dim]
        ax.plot(times, obs_physical[:, dim], lw=1.2)
        if w_limit is not None and np.isfinite(w_limit) and dim < n_sensors:
            ax.axhline(+w_limit, linestyle="--", linewidth=1)
            ax.axhline(-w_limit, linestyle="--", linewidth=1)
        ax.set_ylabel(labels[dim] if dim < len(labels) else f"obs{dim}")
        ax.grid(True, alpha=0.3)
    for ax in axes[obs_physical.shape[1] :]:
        ax.axis("off")
    axes[min(obs_physical.shape[1] - 1, len(axes) - 1)].set_xlabel("Time (s)")
    fig.suptitle(f"Physical sensor and path observations ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_physical_sensor_response.png", dpi=150)
    plt.close(fig)

    # Modal states for debugging the modal-coordinate dynamics.
    if isinstance(x_modal, np.ndarray) and x_modal.size > 0:
        n_align = min(times.size, x_modal.shape[0])
        x_modal = x_modal[:n_align]
        times_modal = times[:n_align]
        modal_dim = x_modal.shape[1]
        fig, axes = plt.subplots(
            int(np.ceil(modal_dim / 2)),
            2,
            figsize=(10, 3.0 * int(np.ceil(modal_dim / 2))),
            sharex=True,
        )
        axes = np.asarray(axes).reshape(-1)
        labels_m = modal_labels(modal_dim)
        for dim in range(modal_dim):
            axes[dim].plot(times_modal, x_modal[:, dim], lw=1.0)
            axes[dim].set_ylabel(labels_m[dim])
            axes[dim].grid(True, alpha=0.3)
        for ax in axes[modal_dim:]:
            ax.axis("off")
        axes[min(modal_dim - 1, len(axes) - 1)].set_xlabel("Time (s)")
        fig.suptitle(f"Modal state ({title_suffix})")
        fig.tight_layout()
        fig.savefig(out_dir / "pretrain_modal_states_debug.png", dpi=150)
        plt.close(fig)

    # Physical actions: omega and ap, with optional ae if control_ae=True.
    action_names = ["Spindle speed (rpm)", "Axial depth of cut ap (mm)"]
    action_series = [omega_to_rpm(actions_phys[:, 0]), actions_phys[:, 1]]
    if actions_phys.shape[1] >= 3:
        action_names.append("Radial immersion ae (mm)")
        action_series.append(actions_phys[:, 2])

    fig, axes = plt.subplots(len(action_series), 1, figsize=(8, 2.6 * len(action_series)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, series, name in zip(axes, action_series, action_names):
        ax.plot(times, series, lw=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Physical control inputs ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_actions_physical.png", dpi=150)
    plt.close(fig)

    # Normalized actions.
    action_labels = ["u_omega", "u_ap"] + (["u_ae"] if actions_norm.shape[1] >= 3 else [])
    fig, axes = plt.subplots(actions_norm.shape[1], 1, figsize=(8, 2.6 * actions_norm.shape[1]), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for dim, ax in enumerate(axes):
        ax.plot(times, actions_norm[:, dim], lw=1.5)
        ax.axhline(-1.0, linestyle="--", linewidth=0.8)
        ax.axhline(+1.0, linestyle="--", linewidth=0.8)
        ax.set_ylabel(action_labels[dim] if dim < len(action_labels) else f"u_{dim}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Normalized policy actions ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_actions_normalized.png", dpi=150)
    plt.close(fig)

    # Reward and termination flags.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, rewards, lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Step reward ({title_suffix})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_rewards.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, np.cumsum(rewards), lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative reward")
    ax.set_title(f"Cumulative reward ({title_suffix})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_cumulative_reward.png", dpi=150)
    plt.close(fig)

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

    print(f"Saved diagnostic plots to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple face-milling plate env smoke test.")
    parser.add_argument("--reward", default="dense", choices=["dense", "sparse", "quadratic"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Env step limit (0 = auto from pass duration).",
    )
    parser.add_argument("--out-dir", default=str(Path(DEFAULT_PLOT_DIR) / "pretrain"))
    parser.add_argument("--fixed-action", action="store_true", help="Use constant omega/ap.")
    parser.add_argument("--rpm", type=float, default=1000.0, help="Spindle speed [rpm] for --fixed-action.")
    parser.add_argument("--omega", type=float, default=None, help="Optional spindle speed [rad/s] instead of --rpm.")
    parser.add_argument("--ap", type=float, default=0, help="Axial depth of cut ap [mm] for --fixed-action.")
    parser.add_argument("--ae", type=float, default=None, help="Optional radial immersion ae [mm] if control_ae=True.")
    parser.add_argument("--control-ae", action="store_true", help="Use 3D action [omega, ap, ae].")
    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=0.0,
        help="Gaussian disturbance on modal accelerations in dynamics [0=off].",
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample milling line y0 on each reset.",
    )
    parser.add_argument("--y0", type=float, default=None, help="Fixed milling line y0 [m].")
    args = parser.parse_args()

    register_envs()

    env_kwargs: dict[str, Any] = {
        "reward_id": args.reward,
        "dt": args.dt,
        "n_substeps": 1,
        "dynamics_uncertainty_std": args.dynamics_uncertainty_std,
        "randomize_y0": args.randomize_y0,
        "control_ae": args.control_ae,
    }
    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(ENV_ID, **env_kwargs)
    plant = env.unwrapped.plant
    action_low, action_high = _physical_action_bounds(plant)

    omega_phys = float(args.omega) if args.omega is not None else float(rpm_to_omega(args.rpm))
    fixed_action = None
    if args.fixed_action:
        if action_low.size >= 3:
            ae_phys = float(args.ae) if args.ae is not None else float(getattr(plant, "ae_default", action_low[2]))
            u_phys = np.array([omega_phys, args.ap, ae_phys], dtype=np.float64)
        else:
            u_phys = np.array([omega_phys, args.ap], dtype=np.float64)
        fixed_action = physical_to_normalized(u_phys, action_low, action_high)

    print("Environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(
        f"Spindle speed range: {RPM_MIN:.0f} - {RPM_MAX:.0f} rpm "
        f"({plant.omega_min:.2f} - {plant.omega_max:.2f} rad/s)"
    )
    ap_min, ap_max = _depth_bounds_from_plant(plant)
    print(f"Axial depth ap range: {ap_min:.4g} - {ap_max:.4g} mm")
    if action_low.size >= 3:
        print(f"Radial immersion ae range: {action_low[2]:.4g} - {action_high[2]:.4g} mm")
    print(f"Dynamics uncertainty std: {plant.dynamics_uncertainty_std}")

    if fixed_action is not None:
        print(
            f"Fixed physical action: rpm={omega_to_rpm(omega_phys):.1f}, "
            f"ap={args.ap} mm"
            + (f", ae={u_phys[2]} mm" if action_low.size >= 3 else "")
        )
        print("Fixed normalized action:", fixed_action)

    reset_options = {"y0": args.y0} if args.y0 is not None else None
    rollout = run_rollout(
        env,
        max_steps=args.steps,
        seed=args.seed,
        fixed_action=fixed_action,
        reset_options=reset_options,
    )

    observations = np.asarray(rollout["observations"], dtype=np.float64)
    rewards = np.asarray(rollout["rewards"], dtype=np.float64)
    assert observations.ndim == 2 and observations.shape[1] == env.observation_space.shape[0]
    assert np.all(np.isfinite(observations))
    assert np.all(np.isfinite(rewards))

    title_suffix = (
        f"fixed rpm={omega_to_rpm(omega_phys):.0f}, ap={args.ap} mm"
        if fixed_action is not None
        else "random policy"
    )
    plot_rollout(rollout, plant=plant, out_dir=Path(args.out_dir), title_suffix=title_suffix)

    env.close()

    print("Rollout test finished successfully.")
    print("Total reward:", float(np.sum(rewards)))
    if rollout["reward_terms"]:
        print("Last-step reward terms:", rollout["reward_terms"][-1])

    x_modal = rollout["x_modal"]
    if isinstance(x_modal, np.ndarray) and x_modal.size > 0:
        print("Final modal displacements:", x_modal[-1, 0::2])
        print("Final modal velocities:", x_modal[-1, 1::2])

    w_sensor = rollout["w_sensor"]
    wdot_sensor = rollout["wdot_sensor"]
    if isinstance(w_sensor, np.ndarray) and w_sensor.size > 0:
        print("Final sensor displacement (m):", w_sensor[-1])
    if isinstance(wdot_sensor, np.ndarray) and wdot_sensor.size > 0:
        print("Final sensor velocity (m/s):", wdot_sensor[-1])


if __name__ == "__main__":
    main()
