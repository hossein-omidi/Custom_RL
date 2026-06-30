"""Shared training / evaluation / plotting configuration for CustomODEPlate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from custom_rl.plants.plate import RPM_MAX, RPM_MIN, omega_to_rpm


def plate_env_kwargs(
    *,
    reward_id: str = "productive",
    dt: float = 0.002,
    n_substeps: int = 1,
    max_episode_steps: int | None = None,
    dynamics_uncertainty_std: float = 0.0,
    randomize_y0: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """
    Build consistent gym.make() kwargs for train, eval, and smoke tests.

    Physics uses rad/s internally; spindle operating range is 50-4000 rpm.
    """
    kwargs: dict[str, Any] = {
        "reward_id": reward_id,
        "dt": dt,
        "n_substeps": n_substeps,
        "randomize_y0": randomize_y0,
        "dynamics_uncertainty_std": float(max(dynamics_uncertainty_std, 0.0)),
    }
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    kwargs.update(extra)
    return kwargs


def plant_plot_metadata(env) -> dict[str, Any]:
    """Collect metadata shared by eval JSON and plot_results."""
    base_env = env.unwrapped
    plant = base_env.plant

    metadata: dict[str, Any] = {
        "env_id": "CustomODEPlate-v0",
        "dt": float(base_env.dt),
        "n_substeps": int(base_env.n_substeps),
        "step_dt": float(base_env._step_dt),
        "max_episode_steps": int(base_env.max_episode_steps),
        "observation_type": "scaled_physical_sensor_disp_vel",
        "n_sensors": int(getattr(plant, "n_sensors", 0)),
        "sensor_points": np.asarray(
            getattr(plant, "sensor_points", []),
            dtype=np.float64,
        ).tolist(),
        "w_limit": float(getattr(plant, "w_limit", np.nan)),
        "w_obs_scale": float(getattr(plant, "w_obs_scale", np.nan)),
        "wdot_obs_scale": float(getattr(plant, "wdot_obs_scale", np.nan)),
        "dynamics_uncertainty_std": float(
            getattr(plant, "dynamics_uncertainty_std", 0.0)
        ),
        "randomize_y0": bool(getattr(plant, "randomize_y0", False)),
        "y0_min_m": float(getattr(plant, "y0_min", np.nan)),
        "y0_max_m": float(getattr(plant, "y0_max", np.nan)),
        "rpm_min": float(RPM_MIN),
        "rpm_max": float(RPM_MAX),
        "physical_actions_are_rad_s_mm": True,
    }

    if hasattr(plant, "u_phys_low") and hasattr(plant, "u_phys_high"):
        metadata["physical_action_low"] = np.asarray(
            plant.u_phys_low, dtype=np.float64
        ).tolist()
        metadata["physical_action_high"] = np.asarray(
            plant.u_phys_high, dtype=np.float64
        ).tolist()
        metadata["physical_action_units"] = ["rad/s", "mm"]
        metadata["omega_min_rad_s"] = float(plant.omega_min)
        metadata["omega_max_rad_s"] = float(plant.omega_max)
        metadata["rpm_min"] = float(omega_to_rpm(plant.omega_min))
        metadata["rpm_max"] = float(omega_to_rpm(plant.omega_max))

    return metadata


def primary_episode_record(ep: dict) -> dict:
    """Return the primary rollout dict (first MC run or episode itself)."""
    runs = ep.get("runs")
    if runs:
        return runs[0]
    return ep


def actions_are_physical(ep: dict, metadata: dict) -> bool:
    """True when saved actions are physical [omega rad/s, ac mm]."""
    record = primary_episode_record(ep)
    if "physical_actions" in ep or "physical_actions" in record:
        return True
    return bool(metadata.get("physical_actions_are_rad_s_mm", False))


def resolve_ppo_model_path(
    model_dir: Path,
    seed: int,
    *,
    prefer_final: bool = False,
) -> Path | None:
    """
    Locate a trained PPO checkpoint for a seed.

    Training (train_sb3_ppo.py) writes:
      - {model_dir}/best_{seed}/best_model.zip  (EvalCallback best, updated during training)
      - {model_dir}/final_{seed}.zip            (end of training only)
    """
    root = Path(model_dir).resolve()
    if prefer_final:
        candidates = (
            root / f"final_{seed}.zip",
            root / f"best_{seed}" / "best_model.zip",
            root / f"best_{seed}.zip",
        )
    else:
        candidates = (
            root / f"best_{seed}" / "best_model.zip",
            root / f"final_{seed}.zip",
            root / f"best_{seed}.zip",
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def discover_model_seeds(model_dir: Path) -> list[int]:
    """Return sorted seeds with at least one checkpoint under model_dir."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        return []

    seeds: set[int] = set()
    for path in root.glob("best_*/best_model.zip"):
        try:
            seeds.add(int(path.parent.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    for path in root.glob("final_*.zip"):
        try:
            seeds.add(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(seeds)


def discover_trajectory_seeds(traj_dir: Path) -> list[int]:
    """Return sorted seeds with trajectories_seed{N}.json files."""
    root = Path(traj_dir).resolve()
    if not root.is_dir():
        return []

    seeds: list[int] = []
    for path in sorted(root.glob("trajectories_seed*.json")):
        stem = path.stem.replace("trajectories_seed", "")
        try:
            seeds.append(int(stem))
        except ValueError:
            continue
    return seeds


def discover_log_seeds(log_dir: Path) -> list[int]:
    """Return sorted seeds with SB3 monitor CSVs under log_dir/seed_{N}/."""
    root = Path(log_dir).resolve()
    if not root.is_dir():
        return []

    seeds: list[int] = []
    for path in sorted(root.glob("seed_*")):
        if not path.is_dir():
            continue
        try:
            seed = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        has_monitor = any(path.glob("*.monitor.csv")) or (path / "monitor.csv").is_file()
        if has_monitor:
            seeds.append(seed)
    return seeds
