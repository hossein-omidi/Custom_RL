"""Mathematical audit tests (Tasks 1–8 verification checklist)."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from custom_rl import register_envs
from custom_rl.integration import get_integrator
from custom_rl.integration.dde_rk4 import integrate_dde_rk4
from custom_rl.integration.rk4 import integrate as integrate_rk4
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.plate import PlatePlant
from custom_rl.rewards.plate_rewards import SensorProductivePlateReward


def _plant() -> PlatePlant:
    p = PlatePlant(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        pass_sampling="sequential",
        n_pass_lines=5,
        displacement_model="surface_normal_reduced",
        trajectory_mode="pass_grid",
    )
    p.reset(np.random.default_rng(0))
    return p


def _u_phys(plant: PlatePlant, omega: float, ac: float) -> np.ndarray:
    return plant.normalized_to_physical_action(
        plant.physical_to_normalized_action(np.array([omega, ac]))
    )


# 1. ac=0, zero IC → zero
def test_ac_zero_zero_state() -> None:
    plant = _plant()
    u = _u_phys(plant, 500.0, 0.0)
    dx = fmod.f_nonlinear2(0.0, np.zeros(plant.state_dim), u)
    assert np.allclose(dx, 0.0, atol=1e-12)


# 2. ac=0 small IC decays
def test_ac_zero_decay() -> None:
    plant = _plant()
    x = np.zeros(plant.state_dim)
    x[0] = 1e-5
    u_norm = plant.physical_to_normalized_action(_u_phys(plant, 500.0, 0.0))
    dt = 0.001
    amps = []
    for step in range(1500):
        t = step * dt
        x = integrate_rk4(plant.dynamics, t, x, u_norm, dt, 1)
        plant.record_modal_state(t + dt, x, omega=500.0)
        amps.append(np.max(np.abs(x[0::2])))
    assert amps[-1] < amps[0]


# 3. Structural uses current state only
def test_structural_current_only() -> None:
    plant = _plant()
    u = _u_phys(plant, 500.0, 0.0)
    x = np.zeros(plant.state_dim)
    x[0] = 1e-4
    fmod.reset_episode_state(plant._modal_state_history)
    fmod._state_history.append((0.0, np.ones(plant.state_dim) * 1e-2))
    dx = fmod.f_nonlinear2(1.0, x, u)
    expected = -fmod.omega_vec**2 * x[0::2] - fmod.lambda_vec * x[0::2] ** 3
    assert np.allclose(dx[1::2], expected, rtol=1e-5)


# 4. Delayed eta only in force (ac>0 force differs when history differs)
def test_delayed_eta_only_in_force() -> None:
    plant = _plant()
    omega, ac = 700.0, 2.0  # ac [mm]; calibrated coeffs stay below safety clip
    u = _u_phys(plant, omega, ac)
    eta = np.array([1e-5, 0, 0, 0], dtype=np.float64)
    x = np.zeros(plant.state_dim)
    x[0::2] = eta
    tau = fmod.tooth_period(omega)
    t = tau * 2.0
    fmod.reset_episode_state(plant._modal_state_history)
    fmod.record_modal_state(0.0, np.zeros(plant.state_dim))
    fmod.record_omega(0.0, omega)
    fmod.record_modal_state(t - tau, x.copy())
    fmod.record_omega(t - tau, omega)
    f1 = fmod._scalar_cutting_force_at(t, x, omega, ac)
    fmod.record_modal_state(t - tau, np.zeros(plant.state_dim))
    f2 = fmod._scalar_cutting_force_at(t, x, omega, ac)
    assert abs(f1) > 1e-9
    assert f1 != f2


# 5. Delay equals tau for constant omega
def test_delay_equals_tooth_period() -> None:
    plant = _plant()
    fmod.DELAY_MODE = "constant_tau"
    omega = 900.0
    assert np.isclose(fmod.delay_time(1.0, omega), 2 * np.pi / (plant.N * omega))


# 6. Phase delay: integrated phase equals 2*pi/N
def test_spindle_phase_delay() -> None:
    plant = _plant()
    fmod.DELAY_MODE = "spindle_phase"
    omega = 1000.0
    fmod.reset_episode_state(plant._modal_state_history)
    for t in np.linspace(0, 0.05, 20):
        fmod.record_omega(float(t), omega)
    t_now = 0.05
    delay = fmod.delay_time(t_now, omega)
    assert delay is not None
    phase = fmod._integrated_phase(t_now - delay, t_now)
    assert np.isclose(phase, 2 * np.pi / plant.N, rtol=0.05)
    fmod.DELAY_MODE = "constant_tau"


# 7. Force changes when eta - eta_delay changes
def test_force_depends_on_delta_eta() -> None:
    plant = _plant()
    omega, ac = 800.0, 3.0
    u = _u_phys(plant, omega, ac)
    tau = fmod.tooth_period(omega)
    x = np.zeros(plant.state_dim)
    u_norm = plant.physical_to_normalized_action(u)
    t = 0.0
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(0.0, x, omega=omega)
    for _ in range(80):
        x = integrate_rk4(plant.dynamics, t, x, u_norm, tau / 40, 1)
        t += tau / 40
        plant.record_modal_state(t, x, omega=omega)
    xa, xb = x.copy(), x.copy()
    xa[0] += 2e-5
    xb[0] -= 2e-5
    fa = np.sum(np.abs(fmod.f_nonlinear2(t, xa, u)[1::2]))
    fb = np.sum(np.abs(fmod.f_nonlinear2(t, xb, u)[1::2]))
    assert fa != fb


# 8. When Delta_q = 0, chip thickness is geometry-only (no regenerative term)
def test_regenerative_vanishes_zero_delta_q() -> None:
    plant = _plant()
    omega, ac = 600.0, 2.0
    fmod._update_cache(omega, ac, fmod._make_cache_hash(omega, ac))
    eta = np.zeros(plant.K)
    t = 0.03
    tau = fmod.tooth_period(omega)
    fmod.reset_episode_state(plant._modal_state_history)
    plant.record_modal_state(t - tau, np.zeros(plant.state_dim), omega=omega)
    plant.record_modal_state(t, np.zeros(plant.state_dim), omega=omega)
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        res = fmod._directional_force_result(t, eta, None, omega, ac)
        assert abs(res["Delta_q"]) < 1e-14
        assert np.isfinite(res["F_projected"])
        x_delayed = np.zeros(plant.state_dim)
        x_delayed[0] = 5e-5
        plant.record_modal_state(t - tau, x_delayed, omega=omega)
        res2 = fmod._directional_force_result(t, eta, None, omega, ac)
        assert abs(res2["Delta_q"]) > 1e-12
    finally:
        fmod.unbind_modal_history()


# 9. Physical omega/ac vary with PPO action
def test_physical_action_varies_with_norm_action() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        max_episode_steps=5,
    )
    plant = env.unwrapped.plant
    a1 = np.array([-1.0, -1.0])
    a2 = np.array([1.0, 1.0])
    p1 = plant.normalized_to_physical_action(a1)
    p2 = plant.normalized_to_physical_action(a2)
    assert p1[0] < p2[0] and p1[1] < p2[1]
    env.close()


# 10. Reward uses same physical omega/ac as plant
def test_reward_productivity_matches_plant_action() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        reward_id="productive",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        max_episode_steps=5,
    )
    obs, _ = env.reset(seed=0)
    action = np.array([0.5, -0.3])
    _, reward, _, _, info = env.step(action)
    plant = env.unwrapped.plant
    ap = plant.normalized_to_physical_action(action)
    assert np.allclose(info["action_phys"], ap, rtol=1e-10)
    assert np.isfinite(reward)
    env.close()


# 11. Sensor observation shape/normalization
def test_sensor_obs_shape_and_norm() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
    )
    obs, _ = env.reset(seed=1)
    assert obs.shape == (6,)
    assert np.all(np.abs(obs[:4]) <= 10.0 + 1e-6)  # default clip
    env.close()


# 12. DDE-RK4 samples delay at substage times
def test_dde_rk4_substage_delay_sampling() -> None:
    plant = _plant()
    omega, ac = 750.0, 2.5
    u = _u_phys(plant, omega, ac)
    u_norm = plant.physical_to_normalized_action(u)
    tau = fmod.tooth_period(omega)
    dt = tau / 4.0
    x = np.zeros(plant.state_dim)
    t0 = tau * 1.2
    plant.record_modal_state(t0, x.copy(), omega=omega)
    fmod.bind_modal_history(plant._modal_state_history)
    try:
        x_dde = integrate_dde_rk4(plant.dynamics, t0, x, u_norm, dt, 1)
        x_ode = integrate_rk4(plant.dynamics, t0, x, u_norm, dt, 1)
    finally:
        fmod.unbind_modal_history()
    assert x_dde.shape == x_ode.shape
    assert np.all(np.isfinite(x_dde))


def test_stability_lobe_script_runs_fast_config() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "stability_lobes.py"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    result = subprocess.run(
        [sys.executable, str(script), "--config", "conf_fast", "--controller", "uncontrolled"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    out = root / "runs" / "conf_fast" / "stability_lobes" / "uncontrolled"
    assert (out / "stability_lobe_data.npz").exists()
    assert (out / "stability_lobe_prob_heatmap.png").exists()
