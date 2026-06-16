"""Sanity checks for sensor-based plate RL interface."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from custom_rl import register_envs
from custom_rl.plants.mode_ordering import mode_index
from custom_rl.rewards.plate_rewards import SensorProductivePlateReward


def test_plate_observation_is_sensor_based_not_modal() -> None:
    register_envs()
    env = gym.make("CustomODEPlate-v0", reward_id="dense", max_episode_steps=50)
    plant = env.unwrapped.plant

    obs, _ = env.reset(seed=0)
    expected_dim = 2 * plant.n_sensors + 2
    assert obs.shape == (expected_dim,)
    assert env.observation_space.shape == (expected_dim,)

    # PPO observation must not equal raw modal state dimension.
    assert obs.shape != (plant.state_dim,)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    assert obs2.shape == (expected_dim,)
    assert np.isfinite(reward)
    assert "sensor_w" in info
    assert "sensor_w_dot" in info
    assert "reward_components" in info
    assert env.observation_space.contains(obs2)

    env.close()


def test_S_disp_shape_and_matrix_definition() -> None:
    register_envs()
    env = gym.make("CustomODEPlate-v0")
    plant = env.unwrapped.plant

    assert plant.S_disp.shape == (plant.n_sensors, plant.K)

    for k, m, n in plant.mode_index_map:
        for s in range(plant.n_sensors):
            xs = float(plant.sensor_coords[s, 0])
            ys = float(plant.sensor_coords[s, 1])
            expected = float(plant.W_mn[m][n](xs, ys))
            assert np.isclose(plant.S_disp[s, k], expected, rtol=1e-10, atol=1e-12)

    env.close()


def test_obs_sensor_part_matches_unclipped_normalization() -> None:
    register_envs()
    env = gym.make("CustomODEPlate-v0", obs_clip=None)
    plant = env.unwrapped.plant

    x, _ = plant.reset(np.random.default_rng(1))
    w, w_dot = plant.state_to_sensor_signals(x)

    unclipped = plant.state_to_sensor_obs_norm(x, clip_for_observation=False)
    assert np.allclose(unclipped[: plant.n_sensors], w / plant.disp_norm_scale)
    assert np.allclose(unclipped[plant.n_sensors :], w_dot / plant.vel_norm_scale)

    # Matrix form must match direct reconstruction.
    eta = x[0::2]
    eta_dot = x[1::2]
    assert np.allclose(w, plant.S_disp @ eta)
    assert np.allclose(w_dot, plant.S_disp @ eta_dot)

    env.close()


def test_reward_depends_on_sensor_obs_norm_not_modal_state() -> None:
    reward_fn = SensorProductivePlateReward()

    u = np.array([0.5, -0.2], dtype=np.float64)
    x_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x_b = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    info = {
        "sensor_obs_norm": np.array([0.2, 0.1, 0.3, 0.4], dtype=np.float64),
        "action_phys": np.array([500.0, 5.0], dtype=np.float64),
        "action_phys_low": np.array([50.0, 0.0], dtype=np.float64),
        "action_phys_high": np.array([2000.0, 10.0], dtype=np.float64),
        "action_norm": u,
        "prev_action_norm": np.zeros(2, dtype=np.float64),
    }

    r_a = reward_fn(0.0, x_a, u, x_a, False, False, dict(info))
    r_b = reward_fn(0.0, x_b, u, x_b, False, False, dict(info))
    assert r_a == r_b

    info_high_vib = dict(info)
    info_high_vib["sensor_obs_norm"] = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    r_low = reward_fn(0.0, x_a, u, x_a, False, False, dict(info))
    r_high = reward_fn(0.0, x_a, u, x_a, False, False, info_high_vib)
    assert r_high < r_low


def test_sensor_coordinates_are_valid_and_matrix_matches_mode_order() -> None:
    register_envs()
    env = gym.make("CustomODEPlate-v0")
    plant = env.unwrapped.plant

    assert plant.sensor_coords.shape == (2, 2)
    assert np.all(plant.sensor_coords[:, 0] >= 0.0)
    assert np.all(plant.sensor_coords[:, 0] <= plant.L1)
    assert np.all(plant.sensor_coords[:, 1] >= 0.0)
    assert np.all(plant.sensor_coords[:, 1] <= plant.L2)

    x0, _ = plant.reset(np.random.default_rng(0))
    w, w_dot = plant.state_to_sensor_signals(x0)
    assert w.shape == (plant.n_sensors,)
    assert w_dot.shape == (plant.n_sensors,)

    # Manual reconstruction for one sensor to confirm ordering.
    eta = x0[0::2]
    manual = 0.0
    cnt = 0
    for m in range(plant.m_max):
        for n in range(plant.n_max):
            xs = float(plant.sensor_coords[0, 0])
            ys = float(plant.sensor_coords[0, 1])
            manual += float(plant.W_mn[m][n](xs, ys)) * float(eta[cnt])
            cnt += 1
    assert np.isclose(w[0], manual, rtol=1e-10, atol=1e-12)

    # Explicit k <-> (m, n) mapping checks for omega_vec ordering.
    # Assumes k is flattened in m-major / n-minor order:
    #   k = m * n_max + n

    # Check omega_vec ordering against compute_natural_frequencies output.
    from custom_rl.plants.compute_natural_frequencies import compute_natural_frequencies

    omega_mn_expected = compute_natural_frequencies(
        plant.E,
        plant.nu,
        plant.rho,
        plant.h,
        plant.L1,
        plant.L2,
        plant.m_max,
        plant.n_max,
    )
    for m in range(plant.m_max):
        for n in range(plant.n_max):
            k = mode_index(m, n, plant.n_max)
            assert (k, m, n) in plant.mode_index_map
            assert np.isclose(
                plant.omega_vec[k],
                float(omega_mn_expected[m, n]),
                rtol=1e-10,
                atol=1e-12,
            ), f"omega_vec ordering mismatch at k={k} (m={m}, n={n})"

    # Check lambda_vec ordering against compute_nonlinear_stiffness output.
    # (This is the cubic nonlinearity coefficient used in f_nonlinear2.)
    from custom_rl.plants.compute_nonlinear_stiffness import compute_nonlinear_stiffness

    lambda_mn_expected, _lambda_prime_dummy = compute_nonlinear_stiffness(
        plant.E,
        plant.nu,
        plant.h,
        plant.L1,
        plant.L2,
        plant.W_mn,
        plant.V_mn,
        plant.m_max,
        plant.n_max,
    )
    for m in range(plant.m_max):
        for n in range(plant.n_max):
            k = mode_index(m, n, plant.n_max)
            assert (k, m, n) in plant.mode_index_map
            assert np.isclose(
                plant.lambda_vec[k],
                float(lambda_mn_expected[m, n]),
                rtol=1e-10,
                atol=1e-12,
            ), f"lambda_vec ordering mismatch at k={k} (m={m}, n={n})"

    # Compare b_vec at the first force-grid time (≥ tooth period) against mode-shape projection.
    x_tool0, y_tool0 = plant.tool_position_at(0.0)
    assert np.isclose(x_tool0, float(plant.x_traj[0]))
    assert np.isclose(y_tool0, float(plant.y_traj[0]))

    from custom_rl.plants import f_nonlinear2 as fmods

    u_phys = plant.normalized_to_physical_action(np.array([0.0, 0.5], dtype=np.float64))
    _ = fmods.f_nonlinear2(0.0, x0, u_phys)

    b_vec_series = fmods.cache.get("b_vec_series", None)
    assert b_vec_series is not None, "f_nonlinear2.cache['b_vec_series'] was not populated"

    t_j = float(fmods.cache["time_discrete"][0])
    x_tool, y_tool = plant.tool_position_at(t_j)

    for m in range(plant.m_max):
        for n in range(plant.n_max):
            k = mode_index(m, n, plant.n_max)
            b_expected = float(plant.W_mn[m][n](x_tool, y_tool)) / float(plant.M_modal)
            assert np.isclose(
                b_vec_series[k, 0],
                b_expected,
                rtol=1e-8,
                atol=1e-10,
            ), f"b_vec_series ordering mismatch at k={k} (m={m}, n={n})"
    env.close()


def test_termination_uses_physical_sensor_displacement_limit() -> None:
    register_envs()
    env = gym.make("CustomODEPlate-v0", displacement_failure_limit=1e-3)
    plant = env.unwrapped.plant

    x = np.zeros(plant.state_dim, dtype=np.float64)
    # Force a huge physical sensor displacement by inflating modal coords.
    x[0::2] = 1.0

    terminated, _, info = plant.termination(0.0, x)
    assert terminated is True
    assert info.get("termination_reason") == "excessive_sensor_displacement"

    # Termination must use raw physical displacement, not normalized/clipped values.
    w, _ = plant.state_to_sensor_signals(x)
    assert np.max(np.abs(w)) > plant.displacement_failure_limit

    # If we only normalized/clipped, a tiny physical displacement would not terminate.
    x_small = np.zeros(plant.state_dim, dtype=np.float64)
    x_small[0] = plant.displacement_failure_limit / max(abs(plant.S_disp[0, 0]), 1e-12) * 0.5
    w_small, _ = plant.state_to_sensor_signals(x_small)
    assert np.max(np.abs(w_small)) < plant.displacement_failure_limit
    terminated_small, _, _ = plant.termination(0.0, x_small)
    assert terminated_small is False

    env.close()
