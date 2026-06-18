"""Tests for structured environment uncertainty."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from custom_rl import register_envs
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.stochasticity import (
    GeometryUncertaintyConfig,
    ProcessNoiseConfig,
    SensorUncertaintyConfig,
    sample_episode_geometry,
)


def _deterministic_plant() -> PlatePlant:
    return PlatePlant(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        pass_sampling="sequential",
        n_pass_lines=5,
    )


def test_geometry_constant_within_episode() -> None:
    plant = _deterministic_plant()
    plant.geometry_uncertainty = GeometryUncertaintyConfig(enable=True, rel_std_L1=0.05)
    plant.sensor_uncertainty = SensorUncertaintyConfig(enable=False)
    plant.process_noise = ProcessNoiseConfig(enable=False)

    rng = np.random.default_rng(42)
    _, info1 = plant.reset(rng)
    L1_ep = plant.L1
    L2_ep = plant.L2

    for _ in range(20):
        x = np.zeros(plant.state_dim)
        u = np.zeros(2)
        x_dot = plant.dynamics(0.0, x, u)
        assert plant.L1 == L1_ep
        assert plant.L2 == L2_ep
        assert np.all(np.isfinite(x_dot))

    _, info2 = plant.reset(rng)
    assert info2["geometry_uncertainty_enabled"] is True


def test_geometry_resampled_each_reset() -> None:
    plant = _deterministic_plant()
    plant.geometry_uncertainty = GeometryUncertaintyConfig(enable=True, rel_std_L1=0.04)
    plant.sensor_uncertainty = SensorUncertaintyConfig(enable=False)
    plant.process_noise = ProcessNoiseConfig(enable=False)

    rng = np.random.default_rng(0)
    L1_values = []
    for _ in range(30):
        plant.reset(rng)
        L1_values.append(plant.L1)

    assert len(set(np.round(L1_values, 6))) > 1


def test_sensor_and_pass_vary_per_reset() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=True,
        enable_process_noise=False,
        pass_sampling="random",
        n_pass_lines=20,
        trajectory_mode="pass_grid",
    )
    plant = env.unwrapped.plant

    coords_seen = []
    pass_seen = []
    for seed in range(15):
        _, info = env.reset(seed=seed)
        coords_seen.append(tuple(map(tuple, info["sensor_coords"])))
        pass_seen.append(int(info["pass_line_index"]))

    assert len(set(pass_seen)) > 1
    assert len(set(coords_seen)) > 1

    env.close()


def test_process_noise_sqrt_dt_scaling() -> None:
    plant = _deterministic_plant()
    plant.process_noise = ProcessNoiseConfig(enable=True, eta_std_per_sqrt_s=1e-5, clip_sigma=0.0)

    rng = np.random.default_rng(1)
    x = np.zeros(plant.state_dim)
    n = 5000
    dt_small = 0.001
    dt_large = 0.01

    noise_small = np.array(
        [plant.apply_process_noise(x, rng, dt_small) - x for _ in range(n)]
    )
    noise_large = np.array(
        [plant.apply_process_noise(x, rng, dt_large) - x for _ in range(n)]
    )

    std_small = np.std(noise_small[:, 0])
    std_large = np.std(noise_large[:, 0])
    ratio = std_large / std_small
    expected = np.sqrt(dt_large / dt_small)
    assert np.isclose(ratio, expected, rtol=0.15)


def test_random_pass_line_default() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        pass_sampling="random",
        n_pass_lines=50,
        trajectory_mode="pass_grid",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
    )

    indices = set()
    for seed in range(40):
        _, info = env.reset(seed=seed)
        indices.add(int(info["pass_line_index"]))

    assert len(indices) > 5
    env.close()


def test_sample_geometry_bounds() -> None:
    from custom_rl.plants.stochasticity import GeometryNominal

    rng = np.random.default_rng(99)
    nom = GeometryNominal()
    cfg = GeometryUncertaintyConfig(enable=True, max_rel_deviation=0.05)
    for _ in range(100):
        g = sample_episode_geometry(rng, nom, cfg)
        assert abs(g["delta_L1"]) <= 0.05 + 1e-12
        assert g["L1"] > 0 and g["L2"] > 0 and g["h"] > 0
