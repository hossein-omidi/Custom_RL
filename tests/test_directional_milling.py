"""Tests for directional regenerative milling force (surface-normal reduced model)."""

from __future__ import annotations

import math

import numpy as np

from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.milling_config import MillingForceConfig
from custom_rl.plants.milling_force import (
    ac_to_meters,
    chip_thickness_surface_normal_reduced,
    engagement,
    feed_per_tooth,
    reconstruct_q_surface_normal,
    surface_normal_force_component,
    tangential_radial_increment,
)
from custom_rl.plants.mode_ordering import iter_mode_indices
from custom_rl.plants.plate import PlatePlant


def _plant(**kw) -> PlatePlant:
    defaults = dict(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        pass_sampling="sequential",
        n_pass_lines=5,
        milling_type="slotting",
        displacement_model="surface_normal_reduced",
        trajectory_mode="pass_grid",
    )
    defaults.update(kw)
    p = PlatePlant(**defaults)
    p.reset(np.random.default_rng(0))
    return p


def _force(t, eta, omega, ac, plant):
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        return fmod._directional_force_result(t, eta, None, omega, ac)
    finally:
        fmod.unbind_modal_history()


def test_f_projected_equals_surface_normal_total() -> None:
    plant = _plant()
    res = _force(0.05, np.zeros(plant.K), 800.0, 3.0, plant)
    assert np.isclose(res["F_projected"], res["F_surface_normal_total"])


def test_in_plane_forces_not_projected_into_modal() -> None:
    plant = _plant()
    res = _force(0.05, np.zeros(plant.K), 800.0, 3.0, plant)
    f_proj = res["F_projected"]
    if abs(res["F_feed_in_plane_total"]) > 1e-9:
        assert f_proj != res["F_feed_in_plane_total"]


def test_magnitude_never_used_for_modal_excitation() -> None:
    plant = _plant()
    res = _force(0.05, np.zeros(plant.K), 800.0, 3.0, plant)
    f_proj = res["F_projected"]
    mag = math.hypot(res["F_feed_in_plane_total"], res["F_normal_in_plane_total"])
    assert np.isclose(f_proj, res["F_surface_normal_total"])
    if mag > 1e-9 and abs(res["F_feed_in_plane_total"]) > 1e-9:
        assert not np.isclose(abs(f_proj), mag)


def test_delta_q_zero_when_q_c_equal() -> None:
    plant = _plant()
    eta = np.array([1e-5, -2e-6, 3e-6, 0.0], dtype=np.float64)
    omega = 600.0
    t = 0.05
    delay = fmod.delay_time(t, omega)
    assert delay is not None and delay > 0.0
    t_d = t - delay
    x_state = np.zeros(plant.state_dim)
    x_state[0::2] = eta
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(t_d, x_state, omega=omega)
    plant.record_modal_state(t, x_state, omega=omega)
    res = _force(t, eta, omega, 2.0, plant)
    assert abs(res["Delta_q"]) < 1e-12


def test_delay_only_in_chip_not_structure() -> None:
    plant = _plant()
    u = plant.normalized_to_physical_action(
        plant.physical_to_normalized_action(np.array([700.0, 0.0]))
    )
    x = np.zeros(plant.state_dim)
    x[0] = 1e-4
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(0.0, np.ones(plant.state_dim) * 1e-2, omega=700.0)
    dx = fmod.f_nonlinear2(1.0, x, u)
    expected = -fmod.omega_vec**2 * x[0::2] - fmod.lambda_vec * x[0::2] ** 3
    assert np.allclose(dx[1::2], expected, rtol=1e-5)


def test_structural_current_eta_only() -> None:
    test_delay_only_in_chip_not_structure()


def test_feed_per_tooth_2pi_factor() -> None:
    vf, om, n = 0.05, 800.0, 5
    f_t = feed_per_tooth(vf, om, n, source="from_feed_speed")
    assert np.isclose(f_t, 2 * math.pi * vf / (n * om))
    assert not np.isclose(f_t, vf / (n * om))


def test_cf_mm_per_tooth_conversion() -> None:
    f_m = feed_per_tooth(0.05, 800.0, 5, cf=2.0, source="from_cf", cf_units="mm_per_tooth")
    assert f_m == 2.0e-3
    f_m2 = feed_per_tooth(0.05, 800.0, 5, cf=0.002, source="from_cf", cf_units="m_per_tooth")
    assert f_m2 == 0.002


def test_ac_doubles_force_axial_integration() -> None:
    plant = _plant(ac_via_axial_integration=True, axial_quadrature_points=3)
    eta = np.zeros(plant.K)
    # Use ac [mm] in linear range after coefficient calibration.
    f1 = abs(_force(0.03, eta, 900.0, 1.0, plant)["F_projected"])
    f2 = abs(_force(0.03, eta, 900.0, 2.0, plant)["F_projected"])
    if f1 > 1e-9:
        assert 1.5 < f2 / f1 < 2.5


def test_ac_units_mm_default_and_si_conversion() -> None:
    plant = _plant()
    assert plant.milling_config.ac_units == "mm"
    assert plant.u_phys_high[1] == 10.0
    assert ac_to_meters(2.0, "mm") == 2.0e-3
    assert ac_to_meters(0.002, "m") == 0.002
    meta = plant.get_interface_metadata()
    assert meta["physical_action_units"] == ["rad/s", "mm"]


def test_surface_default_engagement_pi() -> None:
    cfg = MillingForceConfig(milling_type="surface", displacement_model="surface_normal_reduced")
    assert np.isclose(cfg.engaged_arc_width(), math.pi)
    cfg2 = MillingForceConfig(milling_type="surface", phi_st=0.2, phi_ex=1.0)
    assert np.isclose(cfg2.engaged_arc_width(), 0.8)


def test_metadata_reports_convention_fields() -> None:
    cfg = MillingForceConfig(milling_type="face", displacement_model="surface_normal_reduced")
    meta = cfg.to_dict()
    assert meta["displacement_model"] == "surface_normal_reduced"
    assert meta["is_surface_normal_reduced"] is True


def test_h_nonpositive_zero_force() -> None:
    xi = np.array([1.0, 0, 0, 0])
    delta = np.array([1.0, 0, 0, 0])
    assert tangential_radial_increment(-1e-6, xi, delta, 1.0) == (0.0, 0.0)


def test_surface_normal_reduced_delta_f_zero() -> None:
    plant = _plant(displacement_model="surface_normal_reduced")
    res = _force(0.04, np.zeros(plant.K), 700.0, 2.0, plant)
    assert res["Delta_f"] == 0.0
    assert res["is_surface_normal_reduced"]


def test_delta_q_affects_chip_via_cos() -> None:
    cfg = MillingForceConfig(milling_type="slotting", displacement_model="surface_normal_reduced")
    phi = 0.0
    h0 = chip_thickness_surface_normal_reduced(phi, 0.0, 0.0, cfg)
    h1 = chip_thickness_surface_normal_reduced(phi, 0.0, 1e-5, cfg)
    if engagement(phi, cfg.phi_st, cfg.phi_ex) > 0:
        assert h1 != h0


def test_surface_normal_component_from_ft_fr() -> None:
    d_sn = surface_normal_force_component(10.0, 5.0, math.pi / 4)
    expected = 10.0 * math.sin(math.pi / 4) + 5.0 * math.cos(math.pi / 4)
    assert np.isclose(d_sn, expected)


def test_wrap_around_engagement() -> None:
    cfg = MillingForceConfig(phi_st=5.0, phi_ex=1.0)
    assert engagement(5.5, cfg.phi_st, cfg.phi_ex) == 1.0
    assert engagement(0.5, cfg.phi_st, cfg.phi_ex) == 1.0
    assert engagement(3.0, cfg.phi_st, cfg.phi_ex) == 0.0


def test_mode_ordering_in_b_vec() -> None:
    plant = _plant()
    b = fmod._b_vec_at(0.3, "n")
    assert b.shape == (plant.K,)
    for m, n, k in iter_mode_indices(plant.m_max, plant.n_max):
        assert np.isfinite(b[k])
