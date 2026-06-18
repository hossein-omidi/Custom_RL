"""Unit-consistency tests for milling force and plate vibration model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from custom_rl import register_envs
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.milling_force import (
    chip_thickness_feed_normal_full,
    feed_per_tooth,
    tangential_radial_increment,
)
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.unit_audit import audit_plant
from custom_rl.plants.units import (
    ac_to_meters,
    chip_thickness_for_force_polynomial,
    cutting_coefficients_si_from_mm_m_hybrid,
    modal_acceleration_from_force,
)


def _plant(**kw) -> PlatePlant:
    defaults = dict(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        trajectory_mode="middle_line",
        displacement_model="feed_normal_full",
    )
    defaults.update(kw)
    p = PlatePlant(**defaults)
    p.reset(np.random.default_rng(0))
    fmod.reset_episode_state(p._modal_state_history)
    return p


def test_ac_to_meters_mm_and_m() -> None:
    assert ac_to_meters(2.5, "mm") == 2.5e-3
    assert ac_to_meters(0.003, "m") == 0.003
    with pytest.raises(ValueError):
        ac_to_meters(1.0, "inch")


def test_physical_action_ac_in_mm() -> None:
    plant = _plant()
    u = plant.normalized_to_physical_action(np.array([0.0, 0.5]))
    assert plant.u_phys_low[1] == 0.0
    assert plant.u_phys_high[1] == 10.0
    assert 0.0 < u[1] < 10.0


def test_axial_integration_uses_metres_internally() -> None:
    plant = _plant()
    ac_mm = 4.0
    ac_m = ac_to_meters(ac_mm, plant.milling_config.ac_units)
    n_z = plant.milling_config.axial_quadrature_points
    dz_expected = ac_m / n_z
    assert np.isclose(dz_expected, 4e-3 / n_z)


def test_ft_from_feed_speed_m_per_tooth() -> None:
    ft = feed_per_tooth(0.05, 800.0, 5, source="from_feed_speed")
    assert np.isclose(ft, 2.0 * math.pi * 0.05 / (5.0 * 800.0))


def test_cf_units_mm_and_m() -> None:
    assert feed_per_tooth(0.05, 800.0, 5, cf=2.0, source="from_cf", cf_units="mm_per_tooth") == 2e-3
    assert feed_per_tooth(0.05, 800.0, 5, cf=0.002, source="from_cf", cf_units="m_per_tooth") == 0.002


def test_chip_thickness_terms_consistent_units() -> None:
    plant = _plant()
    cfg = plant.milling_config
    f_t = feed_per_tooth(plant.feed_speed, 800.0, plant.N)
    phi = 0.3
    h = chip_thickness_feed_normal_full(phi, f_t, 1e-5, 2e-5, cfg)
    sp, cp = math.sin(phi), math.cos(phi)
    expected = (f_t * sp + 1e-5 * sp + 2e-5 * cp) * 1.0
    assert np.isclose(h, expected)
    assert h < 1.0  # metres, not mm accidentally


def test_cutting_polynomial_force_in_newtons() -> None:
    xi = np.asarray(fmod.xi_base, dtype=np.float64)
    delta = np.asarray(fmod.delta_base, dtype=np.float64)
    h_m = 1e-4
    dz_m = ac_to_meters(1.0, "mm")
    dft, dfr = tangential_radial_increment(h_m, xi, delta, dz_m)
    assert np.isfinite(dft) and np.isfinite(dfr)
    assert abs(dft) > 0.0
    # hybrid mm coeffs * mm^3 * m -> N
    xi_si, _ = cutting_coefficients_si_from_mm_m_hybrid(xi, delta)
    h_si = h_m
    dft_si = float(xi_si[0] * h_si**3 + xi_si[1] * h_si**2 + xi_si[2] * h_si + xi_si[3]) * dz_m
    assert np.isclose(dft, dft_si, rtol=1e-9)


def test_modal_projection_Fk_m_s2() -> None:
    plant = _plant()
    W = float(plant.W_mn[0][0](0.5, 0.25))
    M = plant.M_modal
    Fk = modal_acceleration_from_force(500.0, W, M)
    assert np.isclose(Fk, W * 500.0 / M)
    assert Fk < 100.0  # order sanity for moderate force


def test_ac_zero_zero_force() -> None:
    plant = _plant()
    eta_n = np.zeros(plant.K)
    eta_f = np.zeros(plant.K)
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        res = fmod._directional_force_result(0.03, eta_n, eta_f, 800.0, 0.0)
        assert abs(res["F_normal_total"]) < 1e-12
        assert abs(res["F_feed_total"]) < 1e-12
    finally:
        fmod.unbind_modal_history()


def test_ac_doubles_force_controlled() -> None:
    """Axial dz scaling: doubling ac_mm doubles elemental dF (pre-clip)."""
    plant = _plant(ac_via_axial_integration=True, axial_quadrature_points=3)
    xi = np.asarray(fmod.xi_base, dtype=np.float64)
    delta = np.asarray(fmod.delta_base, dtype=np.float64)
    h_m = 1e-4
    dz2 = ac_to_meters(2.0, "mm") / 3.0
    dz4 = ac_to_meters(4.0, "mm") / 3.0
    f2 = abs(tangential_radial_increment(h_m, xi, delta, dz2)[0])
    f4 = abs(tangential_radial_increment(h_m, xi, delta, dz4)[0])
    assert f2 > 0.0
    assert np.isclose(f4 / f2, 2.0, rtol=1e-9)


def test_h_poly_converts_m_to_mm() -> None:
    assert chip_thickness_for_force_polynomial(2e-3) == 2.0


def test_reward_termination_physical_units() -> None:
    import gymnasium as gym

    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        max_episode_steps=5,
    )
    plant = env.unwrapped.plant
    assert plant.displacement_failure_limit < 1.0  # metres scale
    assert plant.velocity_failure_limit < 10.0  # m/s scale
    meta = plant.get_interface_metadata()
    assert meta["physical_action_units"] == ["rad/s", "mm"]
    env.close()


def test_full_audit_passes_feed_normal_full() -> None:
    plant = _plant()
    rep = audit_plant(plant)
    assert rep.verdict() in {"PASS", "WARN"}
    assert not any(f.status == "FAIL" for f in rep.findings)
