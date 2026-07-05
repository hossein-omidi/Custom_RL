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
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from custom_rl import DEFAULT_PLOT_DIR, register_envs
from custom_rl.plants import f_nonlinear2_face_milling as fm
from custom_rl.plants.plate import RPM_MAX, RPM_MIN, omega_to_rpm, rpm_to_omega

ENV_ID = "CustomODEPlate-v0"


def make_registered_plate_env(**env_kwargs: Any) -> gym.Env:
    """Create the registered plate environment using the same workflow as training."""
    register_envs()
    return gym.make(ENV_ID, **env_kwargs)


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

    if action.size < low.size:
        padded = np.zeros(low.size, dtype=np.float64)
        padded[: action.size] = action
        action = padded
    elif action.size > low.size:
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


def _finite_or_nan(value: Any) -> float:
    """Convert scalar-like values to finite float, otherwise NaN."""
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _last_force_diagnostics() -> dict[str, Any]:
    """Return one compact regenerative-force diagnostic snapshot.

    The face-milling force module computes the regenerative chip thickness from
    the delayed state x(t-tau). These diagnostics are the direct evidence that
    the delayed path is active: after one tooth period, delta_rel_mm should be
    nonzero whenever the plate has nonzero vibration.
    """
    fi = fm.get_last_force_info() if hasattr(fm, "get_last_force_info") else None

    state_hist = getattr(fm, "_state_history", [])
    kin_hist = getattr(fm, "_kinematic_history", [])

    out: dict[str, Any] = {
        "state_history_len": int(len(state_hist)) if isinstance(state_hist, list) else 0,
        "kinematic_history_len": int(len(kin_hist)) if isinstance(kin_hist, list) else 0,
        "tau_s": float("nan"),
        "omega_rpm": float("nan"),
        "ap_mm": float("nan"),
        "ae_mm": float("nan"),
        "engaged_teeth": 0,
        "delta_rel_x_mm": float("nan"),
        "delta_rel_y_mm": float("nan"),
        "delta_rel_z_mm": float("nan"),
        "delta_rel_norm_mm": float("nan"),
        "rel_now_z_mm": float("nan"),
        "rel_delay_z_mm": float("nan"),
        "max_chip_raw_mm": float("nan"),
        "max_chip_eff_mm": float("nan"),
        "mean_chip_eff_mm": float("nan"),
        "Fz_total_N": float("nan"),
        "F_total_norm_N": float("nan"),
    }

    if fi is None:
        return out

    delta_rel = np.asarray(getattr(fi, "delta_rel_mm", np.full(3, np.nan)), dtype=np.float64).reshape(-1)
    rel_now = np.asarray(getattr(fi, "rel_now_mm", np.full(3, np.nan)), dtype=np.float64).reshape(-1)
    rel_delay = np.asarray(getattr(fi, "rel_delay_mm", np.full(3, np.nan)), dtype=np.float64).reshape(-1)
    chip_raw = np.asarray(getattr(fi, "chip_raw_mm", []), dtype=np.float64).reshape(-1)
    chip_eff = np.asarray(getattr(fi, "chip_eff_mm", []), dtype=np.float64).reshape(-1)
    engagement = np.asarray(getattr(fi, "engagement", []), dtype=bool).reshape(-1)
    F_total = np.asarray(getattr(fi, "F_total_N", np.full(3, np.nan)), dtype=np.float64).reshape(-1)

    if delta_rel.size >= 3:
        out["delta_rel_x_mm"] = _finite_or_nan(delta_rel[0])
        out["delta_rel_y_mm"] = _finite_or_nan(delta_rel[1])
        out["delta_rel_z_mm"] = _finite_or_nan(delta_rel[2])
        out["delta_rel_norm_mm"] = _finite_or_nan(np.linalg.norm(delta_rel[:3]))
    if rel_now.size >= 3:
        out["rel_now_z_mm"] = _finite_or_nan(rel_now[2])
    if rel_delay.size >= 3:
        out["rel_delay_z_mm"] = _finite_or_nan(rel_delay[2])
    if chip_raw.size:
        out["max_chip_raw_mm"] = _finite_or_nan(np.max(chip_raw))
    if chip_eff.size:
        out["max_chip_eff_mm"] = _finite_or_nan(np.max(chip_eff))
        out["mean_chip_eff_mm"] = _finite_or_nan(np.mean(chip_eff))
    if F_total.size >= 3:
        out["Fz_total_N"] = _finite_or_nan(F_total[2])
        out["F_total_norm_N"] = _finite_or_nan(np.linalg.norm(F_total[:3]))

    out["tau_s"] = _finite_or_nan(getattr(fi, "tau", np.nan))
    out["omega_rpm"] = _finite_or_nan(getattr(fi, "n_rpm", np.nan))
    out["ap_mm"] = _finite_or_nan(getattr(fi, "ap_mm", np.nan))
    out["ae_mm"] = _finite_or_nan(getattr(fi, "ae_mm", np.nan))
    out["engaged_teeth"] = int(np.count_nonzero(engagement))
    return out


def _diag_series(diags: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_finite_or_nan(d.get(key, np.nan)) for d in diags], dtype=np.float64)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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
    force_diagnostics_hist: list[dict[str, Any]] = []

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

        diag = _last_force_diagnostics()
        diag["t_env_s"] = float(info.get("t", times[-1] if times else np.nan))
        force_diagnostics_hist.append(diag)

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
        "force_diagnostics": force_diagnostics_hist,
    }


def plot_regeneration_diagnostics(
    rollout: dict[str, np.ndarray | list[dict[str, Any]] | None],
    *,
    plant: Any,
    out_dir: Path,
    title_suffix: str,
) -> dict[str, Any]:
    """Save plots and JSON summary proving delay/regeneration activity."""
    out_dir.mkdir(parents=True, exist_ok=True)

    times = np.asarray(rollout["times"], dtype=np.float64)
    w_sensor = rollout.get("w_sensor")
    diags_raw = rollout.get("force_diagnostics")
    diags = diags_raw if isinstance(diags_raw, list) else []

    summary: dict[str, Any] = {
        "steps_recorded": int(times.size),
        "final_time_s": float(times[-1]) if times.size else float("nan"),
        "regeneration_diagnostics_available": bool(len(diags) > 0),
    }

    if times.size == 0 or not diags:
        with open(out_dir / "pretrain_regeneration_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2)
        return summary

    n = min(times.size, len(diags))
    times = times[:n]
    diags = diags[:n]

    tau = _diag_series(diags, "tau_s")
    delta_x = _diag_series(diags, "delta_rel_x_mm")
    delta_y = _diag_series(diags, "delta_rel_y_mm")
    delta_z = _diag_series(diags, "delta_rel_z_mm")
    delta_norm = _diag_series(diags, "delta_rel_norm_mm")
    max_chip_eff = _diag_series(diags, "max_chip_eff_mm")
    max_chip_raw = _diag_series(diags, "max_chip_raw_mm")
    Fz = _diag_series(diags, "Fz_total_N")
    engaged = _diag_series(diags, "engaged_teeth")
    state_hist_len = _diag_series(diags, "state_history_len")
    kin_hist_len = _diag_series(diags, "kinematic_history_len")

    max_abs_w = np.full(n, np.nan, dtype=np.float64)
    if isinstance(w_sensor, np.ndarray) and w_sensor.size > 0:
        w_arr = np.asarray(w_sensor, dtype=np.float64)[:n]
        if w_arr.ndim == 2 and w_arr.shape[0] > 0:
            max_abs_w[: w_arr.shape[0]] = np.max(np.abs(w_arr), axis=1)

    finite_delta = np.isfinite(delta_norm)
    finite_tau = np.isfinite(tau)
    # A practical proof that regenerative displacement was actually evaluated:
    # after one tooth period, the current-minus-delayed relative displacement is nonzero.
    regen_active = finite_delta & finite_tau & (times >= tau) & (delta_norm > 1e-12)
    first_regen_time = float(times[np.argmax(regen_active)]) if np.any(regen_active) else float("nan")

    w_limit = getattr(plant, "w_limit", None)
    unstable_idx = None
    if w_limit is not None and np.isfinite(w_limit) and np.any(np.isfinite(max_abs_w)):
        idxs = np.where(max_abs_w > float(w_limit))[0]
        unstable_idx = int(idxs[0]) if idxs.size else None

    summary.update({
        "max_abs_w_sensor_m": _finite_or_nan(np.nanmax(max_abs_w)) if np.any(np.isfinite(max_abs_w)) else float("nan"),
        "w_limit_m": _finite_or_nan(w_limit),
        "max_delta_rel_norm_mm": _finite_or_nan(np.nanmax(delta_norm)) if np.any(np.isfinite(delta_norm)) else float("nan"),
        "max_abs_delta_rel_z_mm": _finite_or_nan(np.nanmax(np.abs(delta_z))) if np.any(np.isfinite(delta_z)) else float("nan"),
        "max_chip_eff_mm": _finite_or_nan(np.nanmax(max_chip_eff)) if np.any(np.isfinite(max_chip_eff)) else float("nan"),
        "max_chip_raw_mm": _finite_or_nan(np.nanmax(max_chip_raw)) if np.any(np.isfinite(max_chip_raw)) else float("nan"),
        "max_abs_Fz_total_N": _finite_or_nan(np.nanmax(np.abs(Fz))) if np.any(np.isfinite(Fz)) else float("nan"),
        "regeneration_active_steps": int(np.count_nonzero(regen_active)),
        "first_regeneration_time_s": first_regen_time,
        "final_state_history_len": int(state_hist_len[-1]) if state_hist_len.size and np.isfinite(state_hist_len[-1]) else 0,
        "final_kinematic_history_len": int(kin_hist_len[-1]) if kin_hist_len.size and np.isfinite(kin_hist_len[-1]) else 0,
        "first_instability_time_s": float(times[unstable_idx]) if unstable_idx is not None else float("nan"),
        "instability_detected_from_w_limit": bool(unstable_idx is not None),
    })

    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(times, max_abs_w, lw=1.4, label="max |w_sensor|")
    if w_limit is not None and np.isfinite(w_limit):
        axes[0].axhline(float(w_limit), linestyle="--", linewidth=1.0, label="w_limit")
    axes[0].set_ylabel("max |w| (m)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, delta_x, lw=1.0, label="Δrel x")
    axes[1].plot(times, delta_y, lw=1.0, label="Δrel y")
    axes[1].plot(times, delta_z, lw=1.2, label="Δrel z")
    axes[1].plot(times, delta_norm, lw=1.2, label="||Δrel||")
    axes[1].set_ylabel("regen disp. (mm)")
    axes[1].legend(loc="best", ncol=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, max_chip_raw, lw=1.2, label="max raw chip h")
    axes[2].plot(times, max_chip_eff, lw=1.2, label="max effective chip h_eff")
    axes[2].set_ylabel("chip (mm)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, Fz, lw=1.2, label="Fz total")
    axes[3].set_ylabel("Fz (N)")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(times, tau, lw=1.2, label="tooth delay τ")
    axes[4].plot(times, engaged, lw=1.0, label="engaged teeth")
    axes[4].plot(times, state_hist_len, lw=1.0, label="state history length")
    axes[4].plot(times, kin_hist_len, lw=1.0, label="kinematic history length")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("diagnostics")
    axes[4].legend(loc="best", ncol=2)
    axes[4].grid(True, alpha=0.3)

    if np.isfinite(first_regen_time):
        for ax in axes:
            ax.axvline(first_regen_time, linestyle=":", linewidth=1.0)
    if unstable_idx is not None:
        for ax in axes:
            ax.axvline(times[unstable_idx], linestyle="--", linewidth=1.0)

    fig.suptitle(f"Regenerative-delay diagnostics ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_regeneration_diagnostics.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "pretrain_regeneration_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    return summary


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
    w_sensor = rollout.get("w_sensor")
    wdot_sensor = rollout.get("wdot_sensor")
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

    # Physical sensor observations. Prefer un-clipped physical values from
    # ``info``. Fall back to scaled observations only if older environments do
    # not provide ``w_sensor``/``wdot_sensor``. This avoids hiding excessive
    # displacement behind Gym observation clipping.
    obs_physical = observations.copy()
    if observations.shape[1] >= 2 * n_sensors:
        if isinstance(w_sensor, np.ndarray) and w_sensor.size > 0:
            n_align = min(obs_physical.shape[0], w_sensor.shape[0])
            obs_physical[:n_align, :n_sensors] = w_sensor[:n_align, :n_sensors]
            if n_align < obs_physical.shape[0]:
                obs_physical[n_align:, :n_sensors] *= w_obs_scale
        else:
            obs_physical[:, :n_sensors] *= w_obs_scale

        if isinstance(wdot_sensor, np.ndarray) and wdot_sensor.size > 0:
            n_align = min(obs_physical.shape[0], wdot_sensor.shape[0])
            obs_physical[:n_align, n_sensors : 2 * n_sensors] = wdot_sensor[
                :n_align, :n_sensors
            ]
            if n_align < obs_physical.shape[0]:
                obs_physical[n_align:, n_sensors : 2 * n_sensors] *= wdot_obs_scale
        else:
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
    parser.add_argument("--reward", default="dense", choices=["dense", "productive", "sparse", "quadratic"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    # Defaults match the train/eval convention (1e-4 s substep x 10 = 1 ms
    # control step). This keeps the RK4 substep below the minimum one-tooth
    # regenerative delay at omega_max by default, so a random policy sampling
    # high spindle speed does not crash with a regenerative-delay RuntimeError.
    parser.add_argument("--dt", type=float, default=0.0001, help="RK4 integration substep [s].")
    parser.add_argument("--n-substeps", type=int, default=10, help="Number of RK4 substeps per environment/control step.")
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
        default=None,
        help="Sample milling line y0 on each reset (default: registration setting).",
    )
    parser.add_argument("--y0", type=float, default=None, help="Fixed milling line y0 [m].")
    parser.add_argument("--rpm-min", type=float, default=None, help="Override plant minimum spindle speed [rpm].")
    parser.add_argument("--rpm-max", type=float, default=None, help="Override plant maximum spindle speed [rpm].")
    parser.add_argument("--ap-min", type=float, default=None, help="Override plant minimum axial depth [mm].")
    parser.add_argument("--ap-max", type=float, default=None, help="Override plant maximum axial depth [mm].")
    parser.add_argument("--ae-default", type=float, default=None, help="Override default radial immersion ae [mm].")
    parser.add_argument("--feed-per-tooth-mm", type=float, default=None, help="Override feed per tooth [mm/tooth].")
    parser.add_argument("--w-limit", type=float, default=None, help="Override physical displacement termination limit [m].")
    parser.add_argument("--w-obs-scale", type=float, default=None, help="Override observation displacement scale [m].")
    parser.add_argument("--wdot-limit", type=float, default=None, help="Override velocity observation/diagnostic limit [m/s].")
    parser.add_argument("--wdot-obs-scale", type=float, default=None, help="Override velocity observation scale [m/s].")
    parser.add_argument("--modal-damping-ratio", type=float, default=None, help="Override modal damping ratio.")
    parser.add_argument("--initial-eta-std", type=float, default=None, help="Random initial modal displacement std [m] (default: registration setting).")
    parser.add_argument("--initial-etad-std", type=float, default=None, help="Random initial modal velocity std [m/s] (default: registration setting).")
    args = parser.parse_args()

    register_envs()

    env_kwargs: dict[str, Any] = {
        "reward_id": args.reward,
        "dt": args.dt,
        "n_substeps": args.n_substeps,
        "dynamics_uncertainty_std": args.dynamics_uncertainty_std,
        "control_ae": args.control_ae,
    }
    if args.randomize_y0 is not None:
        env_kwargs["randomize_y0"] = args.randomize_y0
    optional_env_overrides = {
        "omega_min": None if args.rpm_min is None else float(rpm_to_omega(args.rpm_min)),
        "omega_max": None if args.rpm_max is None else float(rpm_to_omega(args.rpm_max)),
        "ap_min": args.ap_min,
        "ap_max": args.ap_max,
        "ae_default": args.ae_default,
        "feed_per_tooth_mm": args.feed_per_tooth_mm,
        "w_limit": args.w_limit,
        "w_obs_scale": args.w_obs_scale,
        "wdot_limit": args.wdot_limit,
        "wdot_obs_scale": args.wdot_obs_scale,
        "modal_damping_ratio": args.modal_damping_ratio,
        "initial_eta_std": args.initial_eta_std,
        "initial_etad_std": args.initial_etad_std,
    }
    for key, value in optional_env_overrides.items():
        if value is not None:
            env_kwargs[key] = value

    if args.max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    if args.n_substeps <= 0:
        raise ValueError("--n-substeps must be positive.")

    env = make_registered_plate_env(**env_kwargs)
    plant = env.unwrapped.plant
    action_low, action_high = _physical_action_bounds(plant)

    omega_phys = float(args.omega) if args.omega is not None else float(rpm_to_omega(args.rpm))
    fixed_action = None
    u_phys_requested = None
    u_phys_applied = None
    if args.fixed_action:
        if action_low.size >= 3:
            ae_phys = float(args.ae) if args.ae is not None else float(getattr(plant, "ae_default", action_low[2]))
            u_phys_requested = np.array([omega_phys, args.ap, ae_phys], dtype=np.float64)
        else:
            u_phys_requested = np.array([omega_phys, args.ap], dtype=np.float64)

        u_phys_applied = np.clip(u_phys_requested, action_low, action_high)
        fixed_action = physical_to_normalized(u_phys_applied, action_low, action_high)

    print("Environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(
        f"Spindle speed range: {omega_to_rpm(plant.omega_min):.0f} - "
        f"{omega_to_rpm(plant.omega_max):.0f} rpm "
        f"({plant.omega_min:.2f} - {plant.omega_max:.2f} rad/s)"
    )
    print(
        f"Integrator: RK4 dt={args.dt:g} s, n_substeps={args.n_substeps}, "
        f"environment/control step={args.dt * args.n_substeps:g} s"
    )
    tau_min = 2.0 * np.pi / (max(int(getattr(plant, "N", 1)), 1) * max(float(plant.omega_max), 1e-12))
    print(f"Minimum one-tooth regenerative delay at omega_max: {tau_min:.6g} s")
    if args.dt >= tau_min:
        print(
            "WARNING: dt is not smaller than the minimum tooth delay. "
            "High-rpm regenerative checks may fail or be invalid. "
            f"Use dt <= {0.25 * tau_min:.3g} s for a safer check."
        )
    ap_min, ap_max = _depth_bounds_from_plant(plant)
    print(f"Axial depth ap range: {ap_min:.4g} - {ap_max:.4g} mm")
    if action_low.size >= 3:
        print(f"Radial immersion ae range: {action_low[2]:.4g} - {action_high[2]:.4g} mm")
    print(f"Dynamics uncertainty std: {plant.dynamics_uncertainty_std}")

    if fixed_action is not None:
        if u_phys_requested is not None and u_phys_applied is not None:
            if not np.allclose(u_phys_requested, u_phys_applied, rtol=0.0, atol=1e-12):
                print("Requested fixed physical action was outside plant bounds and was clipped:")
                print("  requested:", u_phys_requested)
                print("  applied:  ", u_phys_applied)

            print(
                f"Fixed physical action applied: rpm={omega_to_rpm(u_phys_applied[0]):.1f}, "
                f"ap={u_phys_applied[1]} mm"
                + (f", ae={u_phys_applied[2]} mm" if action_low.size >= 3 else "")
            )
        print("Fixed normalized action:", fixed_action)

    reset_options = {"y0": args.y0} if args.y0 is not None else None
    try:
        rollout = run_rollout(
            env,
            max_steps=args.steps,
            seed=args.seed,
            fixed_action=fixed_action,
            reset_options=reset_options,
        )
    except RuntimeError as exc:
        if "Regenerative delay time falls inside the current RK substep" in str(exc):
            print("\nRegenerative-delay resolution error detected.")
            print("This is not a random failure: the tooth delay is smaller than your RK4 dt.")
            print(f"Current dt: {args.dt:g} s")
            print(f"Suggested dt for this plant upper speed: <= {0.25 * tau_min:.3g} s")
            print("Example for high-speed checks: --dt 0.0001 --n-substeps 10")
        env.close()
        raise

    observations = np.asarray(rollout["observations"], dtype=np.float64)
    rewards = np.asarray(rollout["rewards"], dtype=np.float64)
    assert observations.ndim == 2 and observations.shape[1] == env.observation_space.shape[0]
    assert np.all(np.isfinite(observations))
    assert np.all(np.isfinite(rewards))

    for key in ("x_modal", "w_sensor", "wdot_sensor", "actions_phys"):
        value = rollout.get(key)
        if isinstance(value, np.ndarray) and value.size > 0:
            assert np.all(np.isfinite(value)), f"Non-finite values found in {key}."

    title_suffix = (
        f"fixed rpm={omega_to_rpm(u_phys_applied[0]):.0f}, ap={u_phys_applied[1]} mm"
        if fixed_action is not None and u_phys_applied is not None
        else "random policy"
    )
    out_dir = Path(args.out_dir)
    plot_rollout(rollout, plant=plant, out_dir=out_dir, title_suffix=title_suffix)
    regen_summary = plot_regeneration_diagnostics(
        rollout,
        plant=plant,
        out_dir=out_dir,
        title_suffix=title_suffix,
    )

    env.close()

    print("Rollout test finished successfully.")
    print("Total reward:", float(np.sum(rewards)))
    print("Regeneration diagnostic summary:")
    print("  active steps:", regen_summary.get("regeneration_active_steps"))
    print("  first regeneration time [s]:", regen_summary.get("first_regeneration_time_s"))
    print("  max ||delta_rel|| [mm]:", regen_summary.get("max_delta_rel_norm_mm"))
    print("  max |w_sensor| [m]:", regen_summary.get("max_abs_w_sensor_m"))
    print("  instability detected by w_limit:", regen_summary.get("instability_detected_from_w_limit"))
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
