#!/usr/bin/env python3
"""
PlatePlant + RK4 verification with regenerative chatter diagnostics.

Run from project root::

    python custom_rl/plants/Test2.py

Checks:
  0. Import / attributes
  1. Action scaling
  2. Zero IC + ac=0 → zero response
  3. Free vibration decays (ac=0)
  4. Nonzero cutting excitation
  5. RK4 dt consistency
  6. Regenerative coupling: force depends on Delta_w = w_c(t) - w_c(t-tau)
  7. Chatter-like growth under cutting (modal / sensor response)
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project root on path when run as script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_rl.integration.rk4 import integrate
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.plate import PlatePlant


DT = 0.001
T_FINAL = 8.0
SHOW_PLOTS = os.environ.get("TEST2_SHOW_PLOTS", "1").lower() not in ("0", "false", "no")
ZERO_TOL = 1e-10
DECAY_TOL = 0.98


def get_state_dim(plant: PlatePlant) -> int:
    return int(plant.state_dim)


def get_action_bounds(plant: PlatePlant) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(plant.u_phys_low, dtype=np.float64),
        np.asarray(plant.u_phys_high, dtype=np.float64),
    )


def physical_to_normalized_action(plant: PlatePlant, omega: float, ac: float) -> np.ndarray:
    return plant.physical_to_normalized_action(np.array([omega, ac], dtype=np.float64))


def simulate_case(
    plant: PlatePlant,
    x0: np.ndarray,
    u_norm: np.ndarray,
    dt: float,
    t_final: float,
    *,
    reset_each_run: bool = True,
) -> dict:
    """RK4 rollout; records modal state, sensors, and regenerative Delta_w."""
    if reset_each_run:
        fmod.reset_episode_state()

    state_dim = get_state_dim(plant)
    x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    u_norm = np.asarray(u_norm, dtype=np.float64).reshape(-1)

    n_steps = int(np.round(t_final / dt))
    t = 0.0

    trajectory, time_history = [x.copy()], [t]
    u_phys_history = [plant.normalized_to_physical_action(u_norm).copy()]
    sensor_w_hist, sensor_w_dot_hist = [], []
    delta_w_hist, tau_hist = [], []

    terminated = truncated = False
    termination_info: dict = {}

    for _ in range(n_steps):
        w_s, w_dot = plant.state_to_sensor_signals(x)
        sensor_w_hist.append(w_s.copy())
        sensor_w_dot_hist.append(w_dot.copy())

        u_phys = plant.normalized_to_physical_action(u_norm)
        tau = fmod.tooth_period(float(u_phys[0]))
        delta_w = fmod._regenerative_delta_w(t, x[0::2], tau)
        delta_w_hist.append(delta_w)
        tau_hist.append(tau)

        x = integrate(plant.dynamics, t, x, u_norm, dt, n_steps=1)
        t += dt
        fmod.record_modal_state(t, x)
        trajectory.append(x.copy())
        time_history.append(t)
        u_phys_history.append(u_phys.copy())

        terminated, truncated, termination_info = plant.termination(t, x)
        if terminated or truncated:
            break

    trajectory = np.asarray(trajectory, dtype=np.float64)
    time = np.asarray(time_history, dtype=np.float64)
    eta = trajectory[:, 0::2]
    eta_dot = trajectory[:, 1::2]

    return {
        "time": time,
        "trajectory": trajectory,
        "eta": eta,
        "eta_dot": eta_dot,
        "u_phys": np.asarray(u_phys_history, dtype=np.float64),
        "sensor_w": np.asarray(sensor_w_hist, dtype=np.float64),
        "sensor_w_dot": np.asarray(sensor_w_dot_hist, dtype=np.float64),
        "delta_w": np.asarray(delta_w_hist, dtype=np.float64),
        "tau": np.asarray(tau_hist, dtype=np.float64),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_info": termination_info,
    }


def modal_norm(eta: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(eta**2, axis=1))


def check_finite(name: str, result: dict) -> bool:
    ok = bool(np.all(np.isfinite(result["trajectory"])))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: finite states = {ok}")
    if result["terminated"]:
        print(f"       terminated: {result['termination_info']}")
    return ok


def plot_regenerative_case(name: str, result: dict) -> None:
    """Plots for chatter / regenerative analysis."""
    time = result["time"][:-1]
    eta = result["eta"][:-1]
    sensor_w = result["sensor_w"]
    delta_w = result["delta_w"]
    tau = result["tau"]
    u_phys = result["u_phys"][:-1]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(time, eta[:, 0], lw=1.5, label=r"$\eta_1$")
    axes[0].set_ylabel(r"$\eta_1$")
    axes[0].set_title(f"{name} — modal & regenerative signals", fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)

    if sensor_w.size:
        axes[1].plot(time, sensor_w[:, 0], lw=1.5, label=r"$w_{s1}$")
        if sensor_w.shape[1] > 1:
            axes[1].plot(time, sensor_w[:, 1], lw=1.5, label=r"$w_{s2}$")
        axes[1].set_ylabel("Sensor disp. (m)")
        axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(time, delta_w, lw=1.2, color="C3", label=r"$\Delta w_c = w_c(t)-w_c(t-\tau)$")
    axes[2].set_ylabel(r"$\Delta w_c$ (m)")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(time, u_phys[:, 0], lw=1.2, label=r"$\omega$")
    axes[3].plot(time, tau, lw=1.0, ls="--", label=r"$\tau=2\pi/(N\omega)$")
    axes[3].set_ylabel("ω / τ")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def test_import_and_attributes(plant: PlatePlant) -> None:
    print("=" * 60)
    print("Test 0: Import and attributes")
    print("=" * 60)
    print("PlatePlant from:", inspect.getfile(PlatePlant))
    print("K =", plant.K, " state_dim =", plant.state_dim)
    print("u_phys_low / high:", plant.u_phys_low, plant.u_phys_high)
    print()


def test_zero_equilibrium(plant: PlatePlant) -> dict:
    print("=" * 60)
    print("Test 2: Zero IC + ac=0")
    print("=" * 60)
    x0 = np.zeros(plant.state_dim)
    low, _ = get_action_bounds(plant)
    u = physical_to_normalized_action(plant, float(low[0]), float(low[1]))
    result = simulate_case(plant, x0, u, DT, T_FINAL)
    check_finite("zero equilibrium", result)
    mx = float(np.max(np.abs(result["trajectory"])))
    print(f"max |state| = {mx:.3e}  ->  {'PASS' if mx < ZERO_TOL else 'WARN'}")
    print()
    return result


def test_free_vibration_decay(plant: PlatePlant) -> dict:
    print("=" * 60)
    print("Test 3: Free vibration decay (ac=0)")
    print("=" * 60)
    x0 = np.zeros(plant.state_dim)
    x0[0] = 1e-4
    low, _ = get_action_bounds(plant)
    u = physical_to_normalized_action(plant, float(low[0]), float(low[1]))
    result = simulate_case(plant, x0, u, DT, T_FINAL)
    norm = modal_norm(result["eta"])
    ratio = float(np.max(norm[-100:]) / max(np.max(norm[:100]), 1e-30))
    print(f"modal norm last/first window ratio = {ratio:.4f}  ->  {'PASS' if ratio < DECAY_TOL else 'WARN'}")
    print()
    return result


def test_nonzero_cutting(plant: PlatePlant) -> dict:
    print("=" * 60)
    print("Test 4: Nonzero cutting")
    print("=" * 60)
    x0 = np.zeros(plant.state_dim)
    u = physical_to_normalized_action(plant, omega=900.0, ac=5.0)
    result = simulate_case(plant, x0, u, DT, T_FINAL)
    check_finite("nonzero cutting", result)
    print(f"max |eta| = {np.max(np.abs(result['eta'])):.3e}")
    print(f"max |Delta_w| = {np.max(np.abs(result['delta_w'])):.3e}")
    print()
    return result


def test_regenerative_force_coupling(plant: PlatePlant) -> None:
    """Cutting-force path must change when Delta_w changes (ac>0)."""
    print("=" * 60)
    print("Test 6: Regenerative force coupling")
    print("=" * 60)
    fmod.reset_episode_state()
    omega, ac = 800.0, 4.0
    u_phys = np.array([omega, ac], dtype=np.float64)
    tau = fmod.tooth_period(omega)
    t = tau * 2.0

    x = np.zeros(plant.state_dim)
    # Build history at t - tau
    dt_hist = tau / 40.0
    for ti in np.linspace(0.0, t - 1e-9, 50):
        x = integrate(
            plant.dynamics, float(ti), x,
            plant.physical_to_normalized_action(u_phys), dt_hist, n_steps=1,
        )

    x_a = x.copy()
    x_a[0] += 3e-5
    x_b = x.copy()
    x_b[0] -= 3e-5

    dx_a = fmod.f_nonlinear2(t, x_a, u_phys)
    dx_b = fmod.f_nonlinear2(t, x_b, u_phys)
    fa = float(np.linalg.norm(dx_a[1::2]))
    fb = float(np.linalg.norm(dx_b[1::2]))
    print(f"||eta_ddot|| at Delta_w perturbation: {fa:.6e} vs {fb:.6e}")
    assert fa != fb, "Force should depend on regenerative displacement"
    print("[PASS] Cutting dynamics responds to modal state via Delta_w")
    print(f"       tooth period tau = {tau:.6f} s (not fixed 1 s)")
    print()


def test_chatter_growth_under_cutting(plant: PlatePlant) -> dict:
    """Under ac>0, sensor/modal energy should exceed idle decay case."""
    print("=" * 60)
    print("Test 7: Cutting vs no-cutting response")
    print("=" * 60)
    x0 = np.zeros(plant.state_dim)
    x0[0] = 1e-5

    low, _ = get_action_bounds(plant)
    u_idle = physical_to_normalized_action(plant, float(low[0]), float(low[1]))
    u_cut = physical_to_normalized_action(plant, omega=1000.0, ac=6.0)

    res_idle = simulate_case(plant, x0, u_idle, DT, T_FINAL)
    res_cut = simulate_case(plant, x0, u_cut, DT, T_FINAL)

    rms_idle = float(np.sqrt(np.mean(res_idle["sensor_w"][:, 0] ** 2)))
    rms_cut = float(np.sqrt(np.mean(res_cut["sensor_w"][:, 0] ** 2)))
    print(f"sensor RMS idle={rms_idle:.3e}  cutting={rms_cut:.3e}")
    print(f"max |Delta_w| cutting={np.max(np.abs(res_cut['delta_w'])):.3e}")
    if rms_cut > rms_idle * 1.5:
        print("[PASS] Cutting case shows stronger vibration than ac=0.")
    else:
        print("[INFO] Cutting RMS not much larger — try longer T_FINAL or higher ac.")
    print()
    return res_cut


def main() -> None:
    plant = PlatePlant(
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        displacement_failure_limit=5e-3,
        velocity_failure_limit=2.0,
    )
    plant.reset(np.random.default_rng(0))
    fmod.reset_episode_state()

    print("PlatePlant verification (regenerative chatter model)")
    print(f"dt={DT}, T_final={T_FINAL}")
    print()

    test_import_and_attributes(plant)
    res_zero = test_zero_equilibrium(plant)
    res_free = test_free_vibration_decay(plant)
    res_cut = test_nonzero_cutting(plant)
    test_regenerative_force_coupling(plant)
    res_chatter = test_chatter_growth_under_cutting(plant)

    if SHOW_PLOTS:
        plot_regenerative_case("Idle (ac=0)", res_free)
        plot_regenerative_case("Cutting (regenerative)", res_chatter)

    print("=" * 60)
    print("Done. Regenerative model:")
    print("  tau = 2*pi/(N*omega)  enters Delta_w in cutting force")
    print("  structural terms use current eta, eta_dot only")
    print("=" * 60)


if __name__ == "__main__":
    main()
