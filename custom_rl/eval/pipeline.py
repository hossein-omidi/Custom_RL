"""Shared training / evaluation / plotting configuration for CustomODEPlate.

This version is aligned with the face-milling plate plant.

Default control convention
--------------------------
The default normalized plant action is two-dimensional::

    u = [u_omega, u_ap] in [-1, 1]^2

After plant scaling, the physical action is::

    [omega_rad_s, ap_mm]

where ``omega`` is spindle speed [rad/s] and ``ap`` is axial depth of cut [mm].

The radial immersion/depth ``ae`` is a fixed process parameter by default.  In
our current face-milling plant this is ``plant.ae_default``; with the default
D=50 mm and ae_default=25 mm, this gives ae/D = 0.5 half-immersion milling.
Only when the plant is created with ``control_ae=True`` does the action become::

    [omega_rad_s, ap_mm, ae_mm]

This module therefore records both the physical action bounds and the fixed
face-milling process parameters so train/eval/plot scripts use consistent units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from custom_rl.plants.plate import RPM_MAX, RPM_MIN, omega_to_rpm


# ---------------------------------------------------------------------------
# Small compatibility helpers
# ---------------------------------------------------------------------------

def _translate_legacy_depth_kwargs(kwargs: dict[str, Any]) -> None:
    """
    Translate old peripheral-milling ``ac_*`` kwargs to face-milling ``ap_*``.

    This keeps old experiment configs from breaking, but the physical meaning in
    the current plant is axial depth of cut ``ap`` [mm].
    """
    if "ac_min" in kwargs and "ap_min" not in kwargs:
        kwargs["ap_min"] = kwargs.pop("ac_min")
    else:
        kwargs.pop("ac_min", None)

    if "ac_max" in kwargs and "ap_max" not in kwargs:
        kwargs["ap_max"] = kwargs.pop("ac_max")
    else:
        kwargs.pop("ac_max", None)

    if "ac_productive_target" in kwargs and "ap_productive_target" not in kwargs:
        kwargs["ap_productive_target"] = kwargs.pop("ac_productive_target")


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Return value as float, or default if conversion is not possible."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Return a robust bool for metadata fields."""
    if value is None:
        return bool(default)
    return bool(value)


def _deg_from_rad(value: Any) -> float:
    """Convert radians to degrees for human-readable metadata."""
    val = _safe_float(value)
    if not np.isfinite(val):
        return float(np.nan)
    return float(np.rad2deg(val))


def _physical_action_names_from_plant(plant: Any) -> list[str]:
    """Return semantic names for the physical action vector used by the plant."""
    action_dim = int(getattr(plant, "action_dim", 2))
    names = ["omega_rad_s", "ap_mm"]
    if action_dim >= 3 or _safe_bool(getattr(plant, "control_ae", False)):
        names.append("ae_mm")
    return names[: max(action_dim, 2)]


def _physical_action_units_from_names(names: list[str]) -> list[str]:
    """Return units corresponding to physical action names."""
    units: list[str] = []
    for name in names:
        if name == "omega_rad_s":
            units.append("rad/s")
        elif name.endswith("_mm"):
            units.append("mm")
        else:
            units.append("")
    return units


def _physical_action_labels_from_names(names: list[str]) -> list[str]:
    """Human-readable labels for plots/tables."""
    label_map = {
        "omega_rad_s": "Spindle speed omega [rad/s]",
        "ap_mm": "Axial depth of cut ap [mm]",
        "ae_mm": "Radial immersion/depth ae [mm]",
    }
    return [label_map.get(name, name) for name in names]


def _ae_over_d(ae_mm: float, d_mm: float) -> float:
    """Return radial immersion ratio ae/D when both values are available."""
    if not (np.isfinite(ae_mm) and np.isfinite(d_mm)) or d_mm <= 0.0:
        return float(np.nan)
    return float(ae_mm / d_mm)


# ---------------------------------------------------------------------------
# Environment kwargs
# ---------------------------------------------------------------------------

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
    Build consistent ``gym.make()`` kwargs for train, eval, and smoke tests.

    Physics uses rad/s internally.  The default physical action after plant
    scaling is ``[omega_rad_s, ap_mm]``.

    The radial immersion/depth ``ae`` is a fixed plant parameter by default:
    ``ae_default``.  Pass ``control_ae=True`` only if you intentionally want the
    RL action to control ``ae`` as a third action component.
    """
    extra = dict(extra)
    _translate_legacy_depth_kwargs(extra)

    kwargs: dict[str, Any] = {
        "reward_id": reward_id,
        "dt": float(dt),
        "n_substeps": int(n_substeps),
        "randomize_y0": bool(randomize_y0),
        "dynamics_uncertainty_std": float(max(dynamics_uncertainty_std, 0.0)),
    }
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    kwargs.update(extra)
    return kwargs


# ---------------------------------------------------------------------------
# Metadata for evaluation JSON / plotting
# ---------------------------------------------------------------------------

def plant_plot_metadata(env) -> dict[str, Any]:
    """Collect metadata shared by evaluation JSON files and plot scripts."""
    base_env = env.unwrapped
    plant = base_env.plant

    action_names = _physical_action_names_from_plant(plant)
    action_units = _physical_action_units_from_names(action_names)
    action_labels = _physical_action_labels_from_names(action_names)

    d_mm = _safe_float(getattr(plant, "D_mm", np.nan))
    ae_default = _safe_float(getattr(plant, "ae_default", np.nan))
    ae_ratio = _ae_over_d(ae_default, d_mm)

    metadata: dict[str, Any] = {
        # Environment/time configuration
        "env_id": "CustomODEPlate-v0",
        "dt": float(base_env.dt),
        "n_substeps": int(base_env.n_substeps),
        "step_dt": float(base_env._step_dt),
        "max_episode_steps": int(base_env.max_episode_steps),
        # Observation convention
        "observation_type": "scaled_physical_sensor_disp_vel",
        "observation_description": "[w_sensor/w_obs_scale, wdot_sensor/wdot_obs_scale]",
        "n_sensors": int(getattr(plant, "n_sensors", 0)),
        "sensor_points_m": np.asarray(
            getattr(plant, "sensor_points", []),
            dtype=np.float64,
        ).tolist(),
        # Keep old key for backward compatibility with older plotting code.
        "sensor_points": np.asarray(
            getattr(plant, "sensor_points", []),
            dtype=np.float64,
        ).tolist(),
        "w_limit_m": _safe_float(getattr(plant, "w_limit", np.nan)),
        "w_limit": _safe_float(getattr(plant, "w_limit", np.nan)),
        "wdot_limit_m_s": _safe_float(getattr(plant, "wdot_limit", np.nan)),
        "w_obs_scale_m": _safe_float(getattr(plant, "w_obs_scale", np.nan)),
        "w_obs_scale": _safe_float(getattr(plant, "w_obs_scale", np.nan)),
        "wdot_obs_scale_m_s": _safe_float(getattr(plant, "wdot_obs_scale", np.nan)),
        "wdot_obs_scale": _safe_float(getattr(plant, "wdot_obs_scale", np.nan)),
        # Plant/path convention
        "mode_clamped_axis": str(getattr(plant, "mode_clamped_axis", "x")),
        "clamped_side": "x=0",
        "free_side": "x=L1",
        "path_direction": "x=L1 free side -> x=0 clamped side",
        "L1_m": _safe_float(getattr(plant, "L1", np.nan)),
        "L2_m": _safe_float(getattr(plant, "L2", np.nan)),
        "h_m": _safe_float(getattr(plant, "h", np.nan)),
        "rho_input": _safe_float(getattr(plant, "rho", np.nan)),
        "rho_type": str(getattr(plant, "rho_type", "auto")),
        "rho_areal_kg_m2": _safe_float(getattr(plant, "rho_areal", np.nan)),
        # Face-milling process parameters
        "milling_model": "face_milling",
        "force_projection_mode": str(getattr(plant, "force_projection_mode", "z")),
        "n_teeth": int(getattr(plant, "N", 0)),
        "D_mm": d_mm,
        "feed_per_tooth_mm": _safe_float(getattr(plant, "feed_per_tooth_mm", np.nan)),
        "milling_mode": str(getattr(plant, "milling_mode", "up")),
        "gamma_L_deg": _deg_from_rad(getattr(plant, "gamma_L", np.nan)),
        "gamma_r_deg": _deg_from_rad(getattr(plant, "gamma_r", np.nan)),
        "gamma_a_deg": _deg_from_rad(getattr(plant, "gamma_a", np.nan)),
        "Kt_N_mm2": _safe_float(getattr(plant, "Kt", np.nan)),
        "Kr_N_mm2": _safe_float(getattr(plant, "Kr", np.nan)),
        "Ka_N_mm2": _safe_float(getattr(plant, "Ka", np.nan)),
        "Kte_N_mm": _safe_float(getattr(plant, "Kte", np.nan)),
        "Kre_N_mm": _safe_float(getattr(plant, "Kre", np.nan)),
        "Kae_N_mm": _safe_float(getattr(plant, "Kae", np.nan)),
        # ae convention: fixed by default, controlled only if control_ae=True.
        "control_ae": _safe_bool(getattr(plant, "control_ae", False)),
        "ae_default_mm": ae_default,
        "ae_min_mm": _safe_float(getattr(plant, "ae_min", np.nan)),
        "ae_max_mm": _safe_float(getattr(plant, "ae_max", np.nan)),
        "ae_over_D": ae_ratio,
        "ae_is_controlled": _safe_bool(getattr(plant, "control_ae", False)),
        "ae_note": (
            "ae is the third RL action because control_ae=True"
            if _safe_bool(getattr(plant, "control_ae", False))
            else "ae is a fixed process parameter equal to ae_default_mm"
        ),
        # Action convention
        "action_convention": "normalized_action_in_minus1_plus1_scaled_by_plant",
        "physical_action_names": action_names,
        "physical_action_units": action_units,
        "physical_action_labels": action_labels,
        "physical_actions_are_rad_s_mm": True,
        "physical_actions_are_omega_ap": True,
        # Stochasticity / reset options
        "dynamics_uncertainty_std": _safe_float(
            getattr(plant, "dynamics_uncertainty_std", 0.0), 0.0
        ),
        "randomize_y0": _safe_bool(getattr(plant, "randomize_y0", False)),
        "y0_min_m": _safe_float(getattr(plant, "y0_min", np.nan)),
        "y0_max_m": _safe_float(getattr(plant, "y0_max", np.nan)),
        "y_cutter_m": _safe_float(getattr(plant, "y_cutter", np.nan)),
        # Speed range, both original rpm constants and actual plant bounds.
        "rpm_min": float(RPM_MIN),
        "rpm_max": float(RPM_MAX),
    }

    if hasattr(plant, "u_phys_low") and hasattr(plant, "u_phys_high"):
        u_low = np.asarray(plant.u_phys_low, dtype=np.float64).reshape(-1)
        u_high = np.asarray(plant.u_phys_high, dtype=np.float64).reshape(-1)
        metadata["physical_action_low"] = u_low.tolist()
        metadata["physical_action_high"] = u_high.tolist()
        metadata["action_dim"] = int(u_low.size)
    else:
        metadata["action_dim"] = len(action_names)

    if hasattr(plant, "omega_min") and hasattr(plant, "omega_max"):
        metadata["omega_min_rad_s"] = _safe_float(plant.omega_min)
        metadata["omega_max_rad_s"] = _safe_float(plant.omega_max)
        metadata["rpm_min"] = float(omega_to_rpm(plant.omega_min))
        metadata["rpm_max"] = float(omega_to_rpm(plant.omega_max))

    if hasattr(plant, "ap_min") and hasattr(plant, "ap_max"):
        metadata["ap_min_mm"] = _safe_float(plant.ap_min)
        metadata["ap_max_mm"] = _safe_float(plant.ap_max)

    return metadata


# ---------------------------------------------------------------------------
# Saved trajectory helpers
# ---------------------------------------------------------------------------

def primary_episode_record(ep: dict) -> dict:
    """Return the primary rollout dict: first MC run if present, else episode."""
    runs = ep.get("runs")
    if runs:
        return runs[0]
    return ep


def actions_are_physical(ep: dict, metadata: dict) -> bool:
    """
    True when saved actions are physical, not normalized.

    Face-milling physical actions are ``[omega_rad_s, ap_mm]`` by default, or
    ``[omega_rad_s, ap_mm, ae_mm]`` when ``control_ae=True``.
    """
    record = primary_episode_record(ep)
    if "physical_actions" in ep or "physical_actions" in record:
        return True
    if "actions_are_physical" in ep:
        return bool(ep["actions_are_physical"])
    if "actions_are_physical" in record:
        return bool(record["actions_are_physical"])
    return bool(metadata.get("physical_actions_are_rad_s_mm", False))


def physical_action_names(metadata: dict, n_dims: int | None = None) -> list[str]:
    """Return physical action names from metadata with a robust fallback."""
    names = list(metadata.get("physical_action_names") or ["omega_rad_s", "ap_mm"])
    if n_dims is not None:
        if n_dims > len(names):
            fallback = ["omega_rad_s", "ap_mm", "ae_mm"]
            names = (names + fallback[len(names):])[:n_dims]
        else:
            names = names[:n_dims]
    return names


def physical_action_units(metadata: dict, n_dims: int | None = None) -> list[str]:
    """Return physical action units from metadata with a robust fallback."""
    names = physical_action_names(metadata, n_dims=n_dims)
    units = list(metadata.get("physical_action_units") or [])
    if len(units) < len(names):
        units = _physical_action_units_from_names(names)
    return units[: len(names)]


# ---------------------------------------------------------------------------
# Checkpoint / output discovery helpers
# ---------------------------------------------------------------------------

def resolve_ppo_model_path(
    model_dir: Path,
    seed: int,
    *,
    prefer_final: bool = False,
) -> Path | None:
    """
    Locate a trained PPO checkpoint for a seed.

    Training writes:
      - ``{model_dir}/best_{seed}/best_model.zip``  (EvalCallback best)
      - ``{model_dir}/final_{seed}.zip``            (end of training)
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
    """Return sorted seeds with ``trajectories_seed{N}.json`` files."""
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
    """Return sorted seeds with SB3 monitor CSVs under ``log_dir/seed_{N}/``."""
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


__all__ = [
    "plate_env_kwargs",
    "plant_plot_metadata",
    "primary_episode_record",
    "actions_are_physical",
    "physical_action_names",
    "physical_action_units",
    "resolve_ppo_model_path",
    "discover_model_seeds",
    "discover_trajectory_seeds",
    "discover_log_seeds",
]
