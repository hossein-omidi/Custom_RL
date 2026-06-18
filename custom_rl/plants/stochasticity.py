"""
Structured uncertainty model for the plate milling environment.

Uncertainty taxonomy (research-oriented)
----------------------------------------

1. **Episode-constant geometry / material** Θ_geom
   Sampled once at ``reset()``, held fixed until episode termination:
       Θ_geom = (L1, L2, h, E, ρ)
   Models plate-to-plate manufacturing tolerance and material scatter.
   Modal frequencies ω_k(Θ_geom), stiffness λ_k(Θ_geom), and mode shapes W_k
   are recomputed from Θ_geom each episode.

2. **Per-reset layout** ξ_layout | Θ_geom
   Re-sampled every ``reset()`` but constant within the episode:
   - straight pass line index / x-position on the sampled plate width L1
   - sensor coordinates as relative positions (x_s/L1, y_s/L2) with jitter

3. **Process noise** w_k (per integration step)
   Discrete-time unmodeled disturbance after deterministic RK4:
       x_{k+1} = Φ(x_k, u_k; Θ_geom) + w_k,   w_k ~ N(0, Q(Δt))
   This is the standard *post-integration* injection used when the
   continuous-time plant ẋ = f(x, u; Θ) is integrated deterministically and
   stochasticity is treated as bounded exogenous disturbance (not altering f).

   Time-step consistency (Euler–Maruyama scaling):
       σ_step = σ_cont * sqrt(Δt)
   so that Var(Σ w_k) grows approximately linearly in elapsed time.

4. **Observation noise** (handled in ODEControlEnv, separate from plant)
   ε_k ~ N(0, σ_obs² I) on normalized sensor readings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometryNominal:
    """Nominal plate geometry and material parameters."""

    L1: float = 1.0
    L2: float = 0.5
    h: float = 0.02
    E: float = 70e9
    nu: float = 0.3
    rho: float = 7850.0


@dataclass(frozen=True)
class GeometryUncertaintyConfig:
    """
    Multiplicative log-normal-like perturbations around nominal geometry.

    Each parameter θ is sampled as:
        θ_ep = θ_nom * (1 + δ),   δ ~ clip(N(0, rel_std²), ±max_rel_deviation)
    """

    enable: bool = True
    rel_std_L1: float = 0.02
    rel_std_L2: float = 0.02
    rel_std_h: float = 0.03
    rel_std_E: float = 0.01
    rel_std_rho: float = 0.01
    max_rel_deviation: float = 0.05


@dataclass(frozen=True)
class SensorUncertaintyConfig:
    """
    Sensor placement as relative plate coordinates (x_s/L1, y_s/L2).

    Jitter is applied in relative space then mapped to physical coordinates
    for the sampled episode geometry.
    """

    enable: bool = True
    rel_positions: tuple[tuple[float, float], ...] = ((0.83, 0.40), (0.17, 0.40))
    rel_jitter_std: float = 0.015
    edge_margin_frac: float = 0.03


@dataclass(frozen=True)
class ProcessNoiseConfig:
    """
    Diagonal modal process noise applied after RK4 integration.

    Continuous-time intensity σ_cont (per sqrt(second)) is scaled by sqrt(Δt)
    at each env step for consistent diffusion scaling when Δt changes.
    """

    enable: bool = True
    eta_std_per_sqrt_s: float = 5.0e-7
    eta_dot_std_per_sqrt_s: float = 5.0e-6
    clip_sigma: float = 3.0


def _relative_perturbation(
    rng: np.random.Generator,
    rel_std: float,
    max_rel_deviation: float,
) -> float:
    delta = float(rng.normal(0.0, rel_std))
    return float(np.clip(delta, -max_rel_deviation, max_rel_deviation))


def sample_episode_geometry(
    rng: np.random.Generator,
    nominal: GeometryNominal,
    config: GeometryUncertaintyConfig,
) -> dict[str, float]:
    """
    Sample episode-constant geometry Θ_geom.

    Returns dict with keys L1, L2, h, E, nu, rho and relative perturbations.
    """
    if not config.enable:
        return {
            "L1": float(nominal.L1),
            "L2": float(nominal.L2),
            "h": float(nominal.h),
            "E": float(nominal.E),
            "nu": float(nominal.nu),
            "rho": float(nominal.rho),
            "geometry_uncertainty_enabled": False,
            "delta_L1": 0.0,
            "delta_L2": 0.0,
            "delta_h": 0.0,
            "delta_E": 0.0,
            "delta_rho": 0.0,
        }

    deltas = {
        "delta_L1": _relative_perturbation(rng, config.rel_std_L1, config.max_rel_deviation),
        "delta_L2": _relative_perturbation(rng, config.rel_std_L2, config.max_rel_deviation),
        "delta_h": _relative_perturbation(rng, config.rel_std_h, config.max_rel_deviation),
        "delta_E": _relative_perturbation(rng, config.rel_std_E, config.max_rel_deviation),
        "delta_rho": _relative_perturbation(rng, config.rel_std_rho, config.max_rel_deviation),
    }

    return {
        "L1": float(nominal.L1 * (1.0 + deltas["delta_L1"])),
        "L2": float(nominal.L2 * (1.0 + deltas["delta_L2"])),
        "h": float(nominal.h * (1.0 + deltas["delta_h"])),
        "E": float(nominal.E * (1.0 + deltas["delta_E"])),
        "nu": float(nominal.nu),
        "rho": float(nominal.rho * (1.0 + deltas["delta_rho"])),
        "geometry_uncertainty_enabled": True,
        **deltas,
    }


def absolute_to_relative_sensor_coords(
    sensor_coords: np.ndarray,
    L1: float,
    L2: float,
) -> tuple[tuple[float, float], ...]:
    """Convert absolute sensor coordinates to (x/L1, y/L2)."""
    coords = np.asarray(sensor_coords, dtype=np.float64).reshape(-1, 2)
    rel = coords / np.array([L1, L2], dtype=np.float64)
    return tuple((float(r[0]), float(r[1])) for r in rel)


def sample_sensor_coords(
    rng: np.random.Generator,
    L1: float,
    L2: float,
    config: SensorUncertaintyConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Sample physical sensor coordinates for the current episode geometry.

    Returns (coords, info) with shape (n_sensors, 2).
    """
    rel_positions = config.rel_positions
    n_sensors = len(rel_positions)

    if not config.enable:
        coords = np.array(
            [[p[0] * L1, p[1] * L2] for p in rel_positions],
            dtype=np.float64,
        )
        return coords, {
            "sensor_uncertainty_enabled": False,
            "sensor_rel_positions": [list(p) for p in rel_positions],
            "sensor_rel_jitter": [[0.0, 0.0] for _ in rel_positions],
        }

    margin_x = config.edge_margin_frac * L1
    margin_y = config.edge_margin_frac * L2
    jitter_std = config.rel_jitter_std

    physical = []
    rel_used = []
    jitter_used = []

    for x_rel_nom, y_rel_nom in rel_positions:
        jx = float(rng.normal(0.0, jitter_std))
        jy = float(rng.normal(0.0, jitter_std))
        x_rel = float(np.clip(x_rel_nom + jx, config.edge_margin_frac, 1.0 - config.edge_margin_frac))
        y_rel = float(np.clip(y_rel_nom + jy, config.edge_margin_frac, 1.0 - config.edge_margin_frac))

        xs = float(np.clip(x_rel * L1, margin_x, L1 - margin_x))
        ys = float(np.clip(y_rel * L2, margin_y, L2 - margin_y))

        physical.append([xs, ys])
        rel_used.append([x_rel, y_rel])
        jitter_used.append([jx, jy])

    coords = np.asarray(physical, dtype=np.float64)
    return coords, {
        "sensor_uncertainty_enabled": True,
        "sensor_rel_positions": rel_used,
        "sensor_rel_jitter": jitter_used,
    }


def build_process_noise_vector(
    rng: np.random.Generator,
    state_dim: int,
    K: int,
    step_dt: float,
    config: ProcessNoiseConfig,
) -> np.ndarray:
    """
    Build diagonal modal process noise w with Euler–Maruyama sqrt(Δt) scaling.

    State layout: [η_1, η̇_1, η_2, η̇_2, ...].
    """
    if not config.enable or step_dt <= 0.0:
        return np.zeros(state_dim, dtype=np.float64)

    scale = float(np.sqrt(step_dt))
    eta_std = config.eta_std_per_sqrt_s * scale
    eta_dot_std = config.eta_dot_std_per_sqrt_s * scale

    noise = np.zeros(state_dim, dtype=np.float64)
    n_subsystems = state_dim // (2 * K)
    for sub in range(n_subsystems):
        base = sub * 2 * K
        for k in range(K):
            noise[base + 2 * k] = float(rng.normal(0.0, eta_std))
            noise[base + 2 * k + 1] = float(rng.normal(0.0, eta_dot_std))

    if config.clip_sigma > 0.0:
        eta_lim = config.clip_sigma * eta_std
        eta_dot_lim = config.clip_sigma * eta_dot_std
        for sub in range(n_subsystems):
            base = sub * 2 * K
            for k in range(K):
                noise[base + 2 * k] = float(np.clip(noise[base + 2 * k], -eta_lim, eta_lim))
                noise[base + 2 * k + 1] = float(
                    np.clip(noise[base + 2 * k + 1], -eta_dot_lim, eta_dot_lim)
                )

    return noise


def process_noise_info(config: ProcessNoiseConfig, step_dt: float) -> dict[str, Any]:
    """Diagnostic metadata for logging / plotting."""
    scale = float(np.sqrt(step_dt)) if step_dt > 0.0 else 0.0
    return {
        "process_noise_enabled": bool(config.enable),
        "process_noise_eta_std_step": float(config.eta_std_per_sqrt_s * scale),
        "process_noise_eta_dot_std_step": float(config.eta_dot_std_per_sqrt_s * scale),
        "process_noise_clip_sigma": float(config.clip_sigma),
        "step_dt": float(step_dt),
    }
