"""End-to-end mathematical consistency of agent ↔ environment interface."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from custom_rl import register_envs
from custom_rl.rewards.plate_rewards import SensorProductivePlateReward


def _make_env(**kwargs):
    register_envs()
    defaults = {
        "enable_geometry_uncertainty": False,
        "enable_sensor_uncertainty": False,
        "enable_process_noise": False,
        "max_episode_steps": 100,
    }
    defaults.update(kwargs)
    return gym.make("CustomODEPlate-v0", **defaults)


def test_action_normalization_round_trip() -> None:
    env = _make_env()
    plant = env.unwrapped.plant

    for u_norm in (
        np.array([-1.0, -1.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.3, -0.7]),
    ):
        u_phys = plant.normalized_to_physical_action(u_norm)
        u_back = plant.physical_to_normalized_action(u_phys)
        assert np.allclose(u_back, np.clip(u_norm, -1.0, 1.0), atol=1e-12)

    env.close()


def test_sensor_normalization_matches_state_reconstruction() -> None:
    env = _make_env(obs_clip=None)
    plant = env.unwrapped.plant

    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1e-5, size=plant.state_dim)

    w, w_dot = plant.state_to_sensor_signals(x)
    obs_norm = plant.state_to_sensor_obs_norm(x, clip_for_observation=False)
    w2, w_dot2 = plant.sensor_obs_norm_to_physical(obs_norm)

    assert np.allclose(w, w2, rtol=1e-10, atol=1e-14)
    assert np.allclose(w_dot, w_dot2, rtol=1e-10, atol=1e-14)

    env.close()


def test_env_info_sensor_w_matches_modal_reconstruction() -> None:
    env = _make_env()
    obs, _ = env.reset(seed=3)
    action = np.array([0.2, 0.4], dtype=np.float64)

    obs2, _, _, _, info = env.step(action)
    plant = env.unwrapped.plant

    w_info = np.asarray(info["sensor_w"], dtype=np.float64)
    w_dot_info = np.asarray(info["sensor_w_dot"], dtype=np.float64)

    obs_norm = np.asarray(info["sensor_obs_norm"], dtype=np.float64)
    w_from_norm, w_dot_from_norm = plant.sensor_obs_norm_to_physical(obs_norm)

    assert np.allclose(w_info, w_from_norm, rtol=1e-10, atol=1e-14)
    assert np.allclose(w_dot_info, w_dot_from_norm, rtol=1e-10, atol=1e-14)

    # PPO observation = [w_s_norm..., w_dot_s_norm..., prev_u_omega, prev_u_ac].
    assert np.allclose(obs2[-2:], action, atol=1e-12)

    env.close()


def test_reward_uses_same_sensor_normalization_as_plant() -> None:
    env = _make_env()
    plant = env.unwrapped.plant
    reward_fn = SensorProductivePlateReward()

    x = np.zeros(plant.state_dim, dtype=np.float64)
    x[0] = 2e-4
    x[2] = -1e-4
    x[1] = 3e-3
    x[3] = -2e-3

    u = np.array([0.1, 0.5], dtype=np.float64)
    sensor_obs_norm = plant.state_to_sensor_obs_norm(x, clip_for_observation=False)
    action_phys = plant.normalized_to_physical_action(u)

    info = {
        "sensor_obs_norm": sensor_obs_norm,
        "action_phys": action_phys,
        "action_phys_low": plant.u_phys_low,
        "action_phys_high": plant.u_phys_high,
        "action_norm": u,
        "prev_action_norm": np.zeros(2),
    }

    r = reward_fn(0.0, x, u, x, False, False, info)
    w_norm, w_dot_norm = sensor_obs_norm[: plant.n_sensors], sensor_obs_norm[plant.n_sensors :]
    expected_vib = 1.0 * np.mean(w_norm**2) + 0.1 * np.mean(w_dot_norm**2)
    assert info["reward_components"]["vibration_cost"] == float(expected_vib)
    assert np.isfinite(r)

    env.close()


def test_ppo_observation_excludes_modal_coordinates() -> None:
    env = _make_env()
    plant = env.unwrapped.plant

    obs, _ = env.reset(seed=1)
    assert obs.shape[0] == 2 * plant.n_sensors + 2
    assert obs.shape[0] != plant.state_dim

    env.close()


def test_cutting_force_projection_moves_with_tool() -> None:
    """b_vec at t=0 and t>0 must differ when the tool travels along y."""
    env = _make_env(feed_speed=0.1)
    plant = env.unwrapped.plant
    env.reset(seed=0)

    from custom_rl.plants import f_nonlinear2 as fmods

    u_phys = plant.normalized_to_physical_action(np.array([0.5, 0.5], dtype=np.float64))
    x0 = np.zeros(plant.state_dim)

    _ = fmods.f_nonlinear2(0.0, x0, u_phys)
    b0 = fmods.cache["b_vec_series"][:, 0].copy()

    t_mid = 0.5 * float(plant.pass_duration)
    _ = fmods.f_nonlinear2(t_mid, x0, u_phys)
    idx = int(np.argmin(np.abs(fmods.cache["time_discrete"] - t_mid)))
    b_mid = fmods.cache["b_vec_series"][:, idx]

    x0_tool, y0_tool = plant.tool_position_at(0.0)
    x_mid_tool, y_mid_tool = plant.tool_position_at(t_mid)
    assert y_mid_tool < y0_tool
    assert not np.allclose(b0, b_mid), "Force projection must vary with tool position"

    env.close()
