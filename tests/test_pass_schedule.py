"""Tests for straight-line pass episodes and stationary reset."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from custom_rl import register_envs
from custom_rl.plants.pass_schedule import build_straight_pass_trajectory, pass_line_x_positions


def test_pass_line_grid_and_trajectory_geometry() -> None:
    xs = pass_line_x_positions(100, L1=1.0, margin=0.01)
    assert xs.shape == (100,)
    assert np.isclose(xs[0], 0.01)
    assert np.isclose(xs[-1], 0.99)

    t, x, y, duration = build_straight_pass_trajectory(
        x_line=0.42,
        L2=0.5,
        feed_speed=0.1,
        dt=0.02,
    )
    assert duration == 5.0
    assert np.allclose(x, 0.42)
    assert np.isclose(y[0], 0.5)
    assert np.isclose(y[-1], 0.0)
    assert t[-1] <= duration + 1e-12


def test_reset_is_stationary_and_selects_pass_line() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        n_pass_lines=10,
        feed_speed=0.5,
        pass_sampling="sequential",
        trajectory_mode="pass_grid",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
    )
    plant = env.unwrapped.plant

    obs, info = env.reset(seed=0)
    assert np.allclose(env.unwrapped._state, 0.0)
    assert info["pass_line_index"] == 0
    assert "pass_x" in info
    assert info["pass_duration"] > 0.0

    obs2, info2 = env.reset(seed=0)
    assert info2["pass_line_index"] == 1
    assert info2["pass_x"] != info["pass_x"] or plant.n_pass_lines == 1

    step_dt = env.unwrapped._step_dt
    expected_steps = plant.recommended_max_episode_steps(step_dt)
    assert env.unwrapped.max_episode_steps == expected_steps

    env.close()


def test_pass_complete_truncation_not_failure() -> None:
    register_envs()
    env = gym.make("CustomODEPlate-v0", feed_speed=1.0)
    plant = env.unwrapped.plant

    x = np.zeros(plant.state_dim, dtype=np.float64)
    terminated, truncated, info = plant.termination(plant.pass_duration, x)
    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "pass_complete"

    env.close()


def test_excessive_sensor_velocity_terminates() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        velocity_failure_limit=1e-6,
        displacement_failure_limit=1e9,
    )
    plant = env.unwrapped.plant

    x = np.zeros(plant.state_dim, dtype=np.float64)
    x[1::2] = 1.0  # large modal velocities -> large sensor velocities

    terminated, truncated, info = plant.termination(0.0, x)
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "excessive_sensor_velocity"

    env.close()
