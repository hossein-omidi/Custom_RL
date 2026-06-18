"""DDE-RK4 integration audit tests."""

from __future__ import annotations

import numpy as np

from custom_rl.integration.dde_rk4 import integrate_dde_rk4, rk4_step_dde
from custom_rl.integration.rk4 import integrate as integrate_rk4
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.milling_force import reconstruct_q_feed, reconstruct_q_normal
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.time_scales import recommend_integration_dt


def _plant() -> PlatePlant:
    p = PlatePlant(
        displacement_model="feed_normal_full",
        trajectory_mode="middle_line",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
    )
    p.reset(np.random.default_rng(0))
    return p


def test_delay_none_before_tau() -> None:
    plant = _plant()
    fmod.reset_episode_state(plant._modal_state_history)
    omega = 800.0
    tau = fmod.tooth_period(omega)
    plant.record_modal_state(0.0, np.zeros(plant.state_dim))
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        assert fmod._get_modal_state_at_delay(0.25 * tau, tau) is None
        plant.record_modal_state(tau, np.ones(plant.state_dim) * 1e-9)
        assert fmod._get_modal_state_at_delay(1.5 * tau, tau) is not None
    finally:
        fmod.unbind_modal_history()


def test_history_strictly_increasing() -> None:
    plant = _plant()
    hist: list = []
    fmod.reset_episode_state(hist)
    x = np.zeros(plant.state_dim)
    for i, t in enumerate([0.0, 0.001, 0.002, 0.0035]):
        fmod.record_modal_state(t, x + i * 1e-9, history=hist)
    times = [h[0] for h in hist]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_delay_interpolation_linear() -> None:
    plant = _plant()
    hist: list = []
    fmod.reset_episode_state(hist)
    dim = plant.state_dim
    fmod.record_modal_state(0.0, np.zeros(dim), history=hist)
    fmod.record_modal_state(0.01, np.ones(dim), history=hist)
    fmod.bind_modal_history(hist)
    try:
        x_mid = fmod._get_modal_state_at_delay(0.015, 0.01)
    finally:
        fmod.unbind_modal_history()
    assert x_mid is not None
    assert np.allclose(x_mid, 0.5 * np.ones(dim))


def test_rk_scratch_not_in_global_history() -> None:
    plant = _plant()
    fmod.reset_episode_state(plant._modal_state_history)
    x0 = np.zeros(plant.state_dim)
    u = plant.physical_to_normalized_action(np.array([800.0, 1.0]))
    t0 = 0.01
    plant.record_modal_state(t0, x0.copy(), omega=800.0)
    n_before = len(plant._modal_state_history)
    fmod.bind_modal_history(plant._modal_state_history)
    scratch: list = []
    fmod.set_rk4_scratch(scratch)
    try:
        rk4_step_dde(plant.dynamics, t0, x0, u, 0.0005, scratch)
    finally:
        fmod.set_rk4_scratch(None)
        fmod.unbind_modal_history()
    assert len(plant._modal_state_history) == n_before


def test_mode_shape_basis_matches_callables() -> None:
    """Fast ModeShapeBasis must match legacy callable mode shapes."""
    from custom_rl.plants.compute_mode_shapes import ModeShapeBasis, compute_mode_shapes

    L1, L2, h = 0.1, 0.08, 0.003
    m_max, n_max = 2, 2
    basis = ModeShapeBasis.build(L1, L2, h, m_max, n_max)
    W_mn, V_mn = compute_mode_shapes(L1, L2, h, m_max, n_max)
    pts = [(0.05, 0.02), (0.05, 0.06), (0.04, 0.04)]
    z_c = 0.0015
    for x_c, y_c in pts:
        w_fast = basis.w_values(x_c, y_c)
        v_fast = basis.v_values(z_c, y_c)
        for m, n, k in __import__(
            "custom_rl.plants.mode_ordering", fromlist=["iter_mode_indices"]
        ).iter_mode_indices(m_max, n_max):
            w_ref = float(W_mn[m][n](x_c, y_c))
            v_ref = float(V_mn[m][n](z_c, y_c))
            assert np.isclose(w_fast[k], w_ref, rtol=1e-10, atol=1e-12)
            assert np.isclose(v_fast[k], v_ref, rtol=1e-10, atol=1e-12)


def test_omega_history_coalesces_constant_speed() -> None:
    plant = _plant()
    fmod.reset_episode_state(plant._modal_state_history)
    omega = 800.0
    for i in range(500):
        fmod.record_omega(1e-5 * i, omega)
    assert len(fmod._current_omega_history()) == 1
    assert np.isclose(fmod.theta_at(0.005), omega * 0.005)


def test_reconstruct_q_matches_b_projection() -> None:
    """Cached b_vec path must match explicit mode-shape reconstruction."""
    plant = _plant()
    eta = np.array([1e-5, -2e-6, 3e-6, 1e-6], dtype=np.float64)
    t = 0.04
    x_c, y_c = plant.tool_position_at(t)
    z_c = plant.z_contact
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        b_n = fmod._b_vec_at(t, "n")
        b_f = fmod._b_vec_at(t, "f")
        q_n_direct = reconstruct_q_normal(
            eta, x_c, y_c, plant.W_mn, plant.m_max, plant.n_max
        )
        q_n_fast = reconstruct_q_normal(
            eta,
            x_c,
            y_c,
            plant.W_mn,
            plant.m_max,
            plant.n_max,
            b_vec=b_n,
            M_n=plant.M_modal,
        )
        q_f_direct = reconstruct_q_feed(
            eta, z_c, y_c, plant.V_mn, plant.m_max, plant.n_max
        )
        q_f_fast = reconstruct_q_feed(
            eta,
            z_c,
            y_c,
            plant.V_mn,
            plant.m_max,
            plant.n_max,
            b_vec=b_f,
            M_f=plant.M_modal_f,
        )
    finally:
        fmod.unbind_modal_history()
    assert np.isclose(q_n_direct, q_n_fast, rtol=1e-10, atol=1e-15)
    assert np.isclose(q_f_direct, q_f_fast, rtol=1e-10, atol=1e-15)


def test_dde_rk4_finite_with_recommended_dt() -> None:
    plant = _plant()
    scales = recommend_integration_dt(plant, macro_dt=0.002, training_mode=True)
    sub_dt = scales["dt_recommended_training_substep_s"]
    omega, ac = 800.0, 2.0
    u = plant.physical_to_normalized_action(np.array([omega, ac]))
    x = np.zeros(plant.state_dim)
    x[0] = 1e-6
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(0.0, x.copy(), omega=omega)
    t = 0.0
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        for _ in range(200):
            x = integrate_dde_rk4(plant.dynamics, t, x, u, sub_dt, n_steps=1)
            t += sub_dt
            plant.record_modal_state(t, x, omega=omega)
    finally:
        fmod.unbind_modal_history()
    assert np.all(np.isfinite(x))


def test_tau_changes_with_omega() -> None:
    t1 = fmod.tooth_period(400.0)
    t2 = fmod.tooth_period(1200.0)
    assert t2 < t1
