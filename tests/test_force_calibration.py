"""Force calibration for paper-like regenerative oscillation (no training clip)."""

from __future__ import annotations

import numpy as np

from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.milling_force import compute_directional_forces, tangential_radial_increment
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.units import (
    FORCE_COEFFICIENT_SCALE,
    HIGH_ORDER_FORCE_SCALE,
    MAX_FORCE_SAFETY_N,
    ac_to_meters,
)


def _plant() -> PlatePlant:
    p = PlatePlant(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        displacement_model="feed_normal_full",
        trajectory_mode="middle_line",
    )
    p.reset(np.random.default_rng(0))
    fmod.reset_episode_state(p._modal_state_history)
    return p


def test_force_scale_applied() -> None:
    plant = _plant()
    assert plant.force_coefficient_scale == FORCE_COEFFICIENT_SCALE
    raw_xi, raw_delta = (
        np.array([6765e9, -4910e6, 2840e3, 132]) / 2.5,
        np.array([12740e9, -7452e6, 1674e3, 246]) / 2.5,
    )
    exp_xi, exp_delta = raw_xi.copy(), raw_delta.copy()
    exp_xi[:2] *= FORCE_COEFFICIENT_SCALE * HIGH_ORDER_FORCE_SCALE
    exp_xi[2:] *= FORCE_COEFFICIENT_SCALE
    exp_delta[:2] *= FORCE_COEFFICIENT_SCALE * HIGH_ORDER_FORCE_SCALE
    exp_delta[2:] *= FORCE_COEFFICIENT_SCALE
    assert np.allclose(plant.xi_base, exp_xi)
    assert np.allclose(plant.delta_base, exp_delta)


def _directional_force(plant: PlatePlant, t: float, omega: float, ac: float) -> dict:
    eta_n = np.zeros(plant.K)
    eta_f = np.zeros(plant.K)
    return compute_directional_forces(
        t,
        omega,
        ac,
        plant.milling_config,
        eta_n=eta_n,
        eta_f=eta_f,
        n_teeth=plant.N,
        xi_base=plant.xi_base,
        delta_base=plant.delta_base,
        w_mn=plant.W_mn,
        v_mn=plant.V_mn,
        m_max=plant.m_max,
        n_max=plant.n_max,
        feed_speed=plant.feed_speed,
        theta_at=fmod.theta_at,
        tool_position=plant.tool_position_at,
        delay_time_fn=fmod.delay_time,
        modal_state_at_delay=fmod._get_modal_state_at_delay,
        z_contact=plant.z_contact,
        M_n=plant.M_modal,
        M_f=plant.M_modal_f,
        split_delayed_state=fmod._split_delayed,
        max_force=MAX_FORCE_SAFETY_N,
    )


def test_no_clip_at_typical_depths() -> None:
    plant = _plant()
    for ac in (0.5, 1.0, 3.0, 6.0, 10.0):
        r = _directional_force(plant, 0.04, 800.0, ac)
        assert not r["force_clipped"], f"clipped at ac={ac} mm"
        assert abs(r["F_normal_raw"]) < 0.5 * MAX_FORCE_SAFETY_N


def test_force_scales_linearly_with_ac() -> None:
    plant = _plant()

    def fn(ac: float) -> float:
        return abs(_directional_force(plant, 0.04, 800.0, ac)["F_normal_raw"])

    f1, f2 = fn(1.0), fn(2.0)
    assert f1 > 10.0
    assert 1.8 < f2 / f1 < 2.2


def test_elemental_dz_doubles_with_ac() -> None:
    plant = _plant()
    xi, delta = fmod.xi_base, fmod.delta_base
    h_m = 1e-4
    dz1 = ac_to_meters(1.0, "mm") / 5
    dz2 = ac_to_meters(2.0, "mm") / 5
    f1 = abs(tangential_radial_increment(h_m, xi, delta, dz1)[0])
    f2 = abs(tangential_radial_increment(h_m, xi, delta, dz2)[0])
    assert np.isclose(f2 / f1, 2.0, rtol=1e-9)
