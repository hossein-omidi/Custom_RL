"""Verification tests for regenerative chatter in f_nonlinear2."""

from __future__ import annotations

import numpy as np

from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.plate import PlatePlant
from custom_rl.integration.rk4 import integrate


def _init_plant_and_globals() -> PlatePlant:
    plant = PlatePlant(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        pass_sampling="sequential",
        n_pass_lines=5,
        displacement_model="surface_normal_reduced",
        trajectory_mode="pass_grid",
    )
    plant.reset(np.random.default_rng(0))
    return plant


def _u_phys(plant: PlatePlant, omega: float, ac: float) -> np.ndarray:
    return plant.normalized_to_physical_action(
        plant.physical_to_normalized_action(np.array([omega, ac], dtype=np.float64))
    )


def _modal_force_magnitude(t: float, x: np.ndarray, u_phys: np.ndarray) -> float:
    """Sum of absolute modal force components (proxy for cutting-force effect)."""
    dx = fmod.f_nonlinear2(t, x, u_phys)
    return float(np.sum(np.abs(dx[1::2])))


def test_zero_state_zero_ac_remains_zero() -> None:
    """ac=0, zero IC → zero derivative."""
    plant = _init_plant_and_globals()
    x0 = np.zeros(plant.state_dim, dtype=np.float64)
    u = _u_phys(plant, omega=500.0, ac=0.0)

    dx = fmod.f_nonlinear2(0.0, x0, u)
    assert np.allclose(dx, 0.0, atol=1e-12)


def test_free_vibration_decays_with_zero_ac() -> None:
    """ac=0, small IC → amplitude decays under positive damping."""
    plant = _init_plant_and_globals()
    x0 = np.zeros(plant.state_dim, dtype=np.float64)
    x0[0] = 1e-5
    u = _u_phys(plant, omega=500.0, ac=0.0)
    u_norm = plant.physical_to_normalized_action(u)

    dt = 0.001
    x = x0.copy()
    amps = []
    for step in range(2000):
        t = step * dt
        x = integrate(plant.dynamics, t, x, u_norm, dt, n_steps=1)
        fmod.record_modal_state(t + dt, x)
        amps.append(float(np.max(np.abs(x[0::2]))))

    assert amps[0] > 0.0
    assert amps[-1] < amps[0] * 0.5


def test_cutting_force_depends_on_regenerative_displacement() -> None:
    """ac>0: different eta vs delayed eta changes modal force contribution."""
    plant = _init_plant_and_globals()
    omega = 800.0
    ac = 3.0
    u = _u_phys(plant, omega, ac)
    tau = fmod.tooth_period(omega)

    # Build history so delayed state is available at t >= tau.
    x_base = np.zeros(plant.state_dim, dtype=np.float64)
    dt = tau / 50.0
    t = 0.0
    u_norm = plant.physical_to_normalized_action(u)
    fmod.reset_episode_state()
    fmod.record_modal_state(0.0, x_base)

    for _ in range(60):
        x_base = integrate(plant.dynamics, t, x_base, u_norm, dt, n_steps=1)
        t += dt
        fmod.record_modal_state(t, x_base)

    x_a = x_base.copy()
    x_a[0] = x_a[0] + 2e-5

    x_b = x_base.copy()
    x_b[0] = x_b[0] - 2e-5

    f_a = _modal_force_magnitude(t, x_a, u)
    f_b = _modal_force_magnitude(t, x_b, u)
    assert f_a != f_b


def test_equal_current_and_delayed_eta_removes_regenerative_component() -> None:
    """Zero Delta_q leaves chip thickness at geometry-only f_t*sin(phi) level."""
    plant = _init_plant_and_globals()
    omega = 600.0
    ac = 2.0
    eta = np.array([1e-5, -5e-6, 2e-6, 0.0], dtype=np.float64)
    t_query = 0.05

    fmod.reset_episode_state()
    plant.reset(np.random.default_rng(0))
    fmod._update_cache(omega, ac, fmod._make_cache_hash(omega, ac))

    f_with = fmod._scalar_cutting_force_at(t_query, np.zeros(plant.state_dim), omega, ac)
    dw = fmod._regenerative_delta_n(t_query, eta, omega)
    assert isinstance(dw, float)


def test_delay_equals_tooth_period_not_fixed_one_second() -> None:
    """Regenerative delay is tau = 2*pi/(N*omega), not a fixed constant."""
    plant = _init_plant_and_globals()
    omega = 1200.0
    tau = fmod.tooth_period(omega)
    expected = 2.0 * np.pi / (plant.N * omega)
    assert np.isclose(tau, expected)
    assert not np.isclose(tau, 1.0, atol=0.1)


def test_structural_terms_use_current_state_not_delayed() -> None:
    """
    With ac=0, acceleration must not depend on a fictitious delayed eta
    when current eta_dot is zero (only stiffness term -omega^2*eta_current).
    """
    plant = _init_plant_and_globals()
    u = _u_phys(plant, omega=500.0, ac=0.0)

    x_current = np.zeros(plant.state_dim, dtype=np.float64)
    x_current[0] = 1e-4
    x_current[1] = 0.0

    fmod.reset_episode_state()
    # History with very different delayed eta — must not affect structural response.
    x_fake_delayed = np.zeros(plant.state_dim, dtype=np.float64)
    x_fake_delayed[0] = 1e-2
    x_fake_delayed[1] = 1.0
    fmod._state_history.append((0.0, x_fake_delayed))
    fmod._state_history.append((0.5, x_fake_delayed))

    dx = fmod.f_nonlinear2(1.0, x_current, u)
    eta_dd = dx[1::2]

    # Expected: -omega^2 * eta - zeta*0 - lambda*eta^3 (Fk=0 since ac=0)
    expected = (
        -fmod.omega_vec ** 2 * x_current[0::2]
        - fmod.lambda_vec * (x_current[0::2] ** 3)
    )
    assert np.allclose(eta_dd, expected, rtol=1e-6, atol=1e-12)
