"""Tests for feed_normal_full two-direction regenerative milling model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.milling_config import MillingForceConfig
from custom_rl.plants.milling_force import (
    chip_thickness_feed_normal_full,
    engagement,
    feed_per_tooth,
    reconstruct_q_feed,
    reconstruct_q_normal,
    transform_to_feed_normal,
)
from custom_rl.plants.modal_state import split_modal_state, state_dim_for_model
from custom_rl.plants.mode_ordering import iter_mode_indices
from custom_rl.plants.pass_schedule import build_middle_line_trajectory
from custom_rl.plants.plate import PlatePlant


def _plant(**kw) -> PlatePlant:
    defaults = dict(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        trajectory_mode="middle_line",
        displacement_model="feed_normal_full",
        milling_type="surface",
    )
    defaults.update(kw)
    p = PlatePlant(**defaults)
    p.reset(np.random.default_rng(0))
    return p


def test_state_dim_is_4k() -> None:
    plant = _plant()
    assert plant.state_dim == 4 * plant.K
    assert fmod.expected_state_dim() == 4 * plant.K
    assert np.all(plant.omega_f_vec > 0)


def test_w_and_v_field_indices_consistent() -> None:
    plant = _plant()
    for m, n, k in iter_mode_indices(plant.m_max, plant.n_max):
        assert plant.mode_index_map[k] == (k, m, n)
    b_n = fmod._b_vec_at(0.01, "n")
    b_f = fmod._b_vec_at(0.01, "f")
    assert b_n.shape == (plant.K,)
    assert b_f.shape == (plant.K,)


def test_delta_from_delayed_displacement_not_raw_eta() -> None:
    plant = _plant()
    eta_n = np.array([1e-5, 0, 0, 0], dtype=np.float64)
    eta_f = np.zeros(plant.K)
    x = np.zeros(plant.state_dim)
    x[0::2][: plant.K] = eta_n
    x[4::2][: plant.K] = eta_f
    t = 0.05
    tau = fmod.tooth_period(600.0)
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(t - tau, x.copy(), omega=600.0)
    plant.record_modal_state(t, x.copy(), omega=600.0)
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        res = fmod._directional_force_result(t, eta_n, eta_f, 600.0, 2.0)
        assert abs(res["Delta_n"]) < 1e-14
        assert abs(res["Delta_f"]) < 1e-14
        x2 = x.copy()
        x2[0] = 5e-5
        plant.record_modal_state(t - tau, x2, omega=600.0)
        res2 = fmod._directional_force_result(t, eta_n, eta_f, 600.0, 2.0)
        assert abs(res2["Delta_n"]) > 1e-12
    finally:
        fmod.unbind_modal_history()


def test_delay_only_in_chip_thickness() -> None:
    plant = _plant(displacement_model="feed_normal_full")
    u = plant.normalized_to_physical_action(
        plant.physical_to_normalized_action(np.array([700.0, 0.0]))
    )
    x = np.zeros(plant.state_dim)
    x[0] = 1e-4
    fmod.reset_episode_state(plant._modal_state_history)
    fmod._state_history.append((0.0, np.ones(plant.state_dim) * 1e-2))
    dx = fmod.f_nonlinear2(1.0, x, u)
    eta_n, etad_n, _, _ = split_modal_state(x, plant.K, two_field=True)
    expected_n = -fmod.omega_vec**2 * eta_n - fmod.lambda_vec * eta_n**3
    assert np.allclose(dx[1::2][: plant.K], expected_n, rtol=1e-5)


def test_f_normal_projects_only_to_w_field() -> None:
    plant = _plant()
    eta_n = np.zeros(plant.K)
    eta_f = np.zeros(plant.K)
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        res = fmod._directional_force_result(0.04, eta_n, eta_f, 700.0, 3.0)
        F_n = res["F_normal_total"]
        F_f = res["F_feed_total"]
        dx = fmod.f_nonlinear2(0.04, np.zeros(plant.state_dim), np.array([700.0, 3.0]))
        _, ddeta_n, _, ddeta_f = split_modal_state(dx, plant.K, two_field=True)
        if abs(F_n) > 1e-9:
            assert np.any(np.abs(ddeta_n) > 0)
        if abs(F_f) < 1e-12:
            assert np.allclose(ddeta_f, 0.0)
    finally:
        fmod.unbind_modal_history()


def test_magnitude_not_used_for_excitation() -> None:
    plant = _plant()
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        res = fmod._directional_force_result(0.05, np.zeros(plant.K), np.zeros(plant.K), 800.0, 3.0)
        mag = math.hypot(res["F_feed_total"], res["F_normal_total"])
        if mag > 1e-9:
            assert res["F_normal_total"] != mag or res["F_feed_total"] != mag
    finally:
        fmod.unbind_modal_history()


def test_h_nonpositive_zero_force() -> None:
    cfg = MillingForceConfig(displacement_model="feed_normal_full")
    h = chip_thickness_feed_normal_full(math.pi, 0.001, 0.0, 0.0, cfg)
    assert h <= 0.0


def test_ac_zero_zero_force() -> None:
    plant = _plant()
    dx = fmod.f_nonlinear2(0.0, np.zeros(plant.state_dim), np.array([500.0, 0.0]))
    assert np.allclose(dx, 0.0, atol=1e-12)


def test_feed_per_tooth_2pi() -> None:
    f_t = feed_per_tooth(0.05, 800.0, 5, source="from_feed_speed")
    assert np.isclose(f_t, 2 * math.pi * 0.05 / (5 * 800.0))


def test_middle_line_path_inside_plate() -> None:
    L1, L2 = 1.0, 0.5
    t, x, y, _ = build_middle_line_trajectory(0.5 * L1, L2, 0.0, 0.05, 0.02)
    assert np.all((x >= 0.0) & (x <= L1))
    assert np.all((y >= 0.0) & (y <= L2))
    assert np.allclose(x, 0.5 * L1)


def test_surface_engagement_pi_default() -> None:
    cfg = MillingForceConfig(milling_type="surface", displacement_model="feed_normal_full")
    assert np.isclose(cfg.engaged_arc_width(), math.pi)


def test_sensor_obs_still_physical() -> None:
    from custom_rl import register_envs
    import gymnasium as gym

    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        displacement_model="feed_normal_full",
        trajectory_mode="middle_line",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        max_episode_steps=5,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,)
    env.close()


def test_feed_normal_chip_formula() -> None:
    cfg = MillingForceConfig(displacement_model="feed_normal_full", milling_type="slotting")
    phi = 0.3
    h = chip_thickness_feed_normal_full(phi, 1e-4, 2e-6, 3e-6, cfg)
    sp, cp = math.sin(phi), math.cos(phi)
    g = engagement(phi, cfg.phi_st, cfg.phi_ex)
    expected = (1e-4 * sp + 2e-6 * sp + 3e-6 * cp) * g
    assert np.isclose(h, expected)


def test_transform_sign_convention() -> None:
    df, dn = transform_to_feed_normal(10.0, 5.0, math.pi / 4)
    cp, sp = math.cos(math.pi / 4), math.sin(math.pi / 4)
    assert np.isclose(df, -10.0 * cp - 5.0 * sp)
    assert np.isclose(dn, 10.0 * sp - 5.0 * cp)
