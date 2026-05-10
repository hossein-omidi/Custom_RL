# -*- coding: utf-8 -*-
"""
Verification script for PlatePlant + RK4 simulation.

This script checks:

1. Which PlatePlant class is actually imported.
2. Whether required attributes exist.
3. Zero initial condition + zero depth of cut.
4. Free vibration decay with zero depth of cut.
5. Nonzero cutting response.
6. RK4 time-step consistency.

Important:
Your state format is interleaved:

    [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]

Therefore:

    eta     = trajectory[:, 0::2]
    eta_dot = trajectory[:, 1::2]

Also:
PlatePlant.dynamics() expects normalized action:

    u_norm = [u_omega, u_ac] in [-1, 1]

The physical action is:

    [omega, ac]
"""

from __future__ import annotations

import inspect
import numpy as np
import matplotlib.pyplot as plt

from plate import PlatePlant
from rk4 import integrate


# ============================================================
# Configuration
# ============================================================

DT = 0.001
T_FINAL = 5.0
SHOW_PLOTS = True

ZERO_TOL = 1e-10
DECAY_TOL = 0.98
CONVERGENCE_REL_TOL = 0.10


# ============================================================
# Robust plant helpers
# ============================================================

def patch_missing_attributes(plant: PlatePlant) -> None:
    """
    Patch missing attributes if the imported PlatePlant is an older version.

    Your pasted class defines:
        self.K
        self.state_dim = 2 * self.K

    But your runtime error shows state_dim is missing.
    This function fixes that and prints diagnostics.
    """
    if not hasattr(plant, "K"):
        raise AttributeError(
            "The imported PlatePlant does not have attribute K. "
            "Please check that PlatePlant1.py contains the expected class."
        )

    if not hasattr(plant, "state_dim"):
        plant.state_dim = 2 * int(plant.K)
        print("[PATCH] plant.state_dim was missing. Set state_dim = 2 * K =", plant.state_dim)

    if not hasattr(plant, "u_phys_low") or not hasattr(plant, "u_phys_high"):
        required = ["omega_min", "omega_max", "ac_min", "ac_max"]
        missing = [name for name in required if not hasattr(plant, name)]

        if missing:
            raise AttributeError(
                "Cannot create physical action bounds because these attributes are missing: "
                f"{missing}"
            )

        plant.u_phys_low = np.array([plant.omega_min, plant.ac_min], dtype=np.float64)
        plant.u_phys_high = np.array([plant.omega_max, plant.ac_max], dtype=np.float64)

        print("[PATCH] u_phys_low/u_phys_high were missing. Created them from bounds.")

    if not hasattr(plant, "eta_limit"):
        plant.eta_limit = np.inf
        print("[PATCH] eta_limit was missing. Set eta_limit = inf.")


def get_state_dim(plant: PlatePlant) -> int:
    """
    Return state dimension safely.
    """
    if hasattr(plant, "state_dim"):
        return int(plant.state_dim)

    if hasattr(plant, "K"):
        return 2 * int(plant.K)

    raise AttributeError("Cannot determine state dimension. plant.K is missing.")


def get_action_bounds(plant: PlatePlant) -> tuple[np.ndarray, np.ndarray]:
    """
    Return physical action bounds [omega, ac].
    """
    if hasattr(plant, "u_phys_low") and hasattr(plant, "u_phys_high"):
        low = np.asarray(plant.u_phys_low, dtype=np.float64)
        high = np.asarray(plant.u_phys_high, dtype=np.float64)
        return low, high

    low = np.array([plant.omega_min, plant.ac_min], dtype=np.float64)
    high = np.array([plant.omega_max, plant.ac_max], dtype=np.float64)

    return low, high


def scale_action_fallback(plant: PlatePlant, u_norm: np.ndarray) -> np.ndarray:
    """
    Convert normalized action to physical [omega, ac].

    Uses plant._scale_action() if available. Otherwise uses u_phys_low/u_phys_high.
    """
    u_norm = np.asarray(u_norm, dtype=np.float64).reshape(-1)

    if u_norm.size != 2:
        raise ValueError(f"Expected normalized action shape (2,), got {u_norm.shape}")

    u_norm = np.clip(u_norm, -1.0, 1.0)

    if hasattr(plant, "_scale_action"):
        return np.asarray(plant._scale_action(u_norm), dtype=np.float64)

    low, high = get_action_bounds(plant)

    return low + 0.5 * (u_norm + 1.0) * (high - low)


def physical_to_normalized_action(
    plant: PlatePlant,
    omega: float,
    ac: float,
) -> np.ndarray:
    """
    Convert physical action [omega, ac] to normalized action [-1, 1]^2.
    """
    low, high = get_action_bounds(plant)

    u_phys = np.array([omega, ac], dtype=np.float64)

    u_norm = 2.0 * (u_phys - low) / (high - low) - 1.0

    return np.clip(u_norm, -1.0, 1.0)


# ============================================================
# Simulation utilities
# ============================================================

def simulate_case(
    plant: PlatePlant,
    x0: np.ndarray,
    u_norm: np.ndarray,
    dt: float,
    t_final: float,
) -> dict:
    """
    Simulate one fixed-input case using RK4.
    """
    state_dim = get_state_dim(plant)

    x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()

    if x.size != state_dim:
        raise ValueError(f"x0 has size {x.size}, but expected {state_dim}")

    u_norm = np.asarray(u_norm, dtype=np.float64).reshape(-1)

    if u_norm.size != 2:
        raise ValueError(f"u_norm must have shape (2,), got {u_norm.shape}")

    n_steps = int(np.round(t_final / dt))

    t = 0.0

    trajectory = [x.copy()]
    time_history = [t]
    u_norm_history = [u_norm.copy()]
    u_phys_history = [scale_action_fallback(plant, u_norm).copy()]

    terminated = False
    truncated = False
    termination_info = {}

    for _ in range(n_steps):
        x = integrate(
            dynamics=plant.dynamics,
            t0=t,
            x0=x,
            u=u_norm,
            dt=dt,
            n_steps=1,
        )

        t += dt

        trajectory.append(x.copy())
        time_history.append(t)
        u_norm_history.append(u_norm.copy())
        u_phys_history.append(scale_action_fallback(plant, u_norm).copy())

        if hasattr(plant, "termination"):
            terminated, truncated, info = plant.termination(t, x)
        else:
            invalid_state = not np.all(np.isfinite(x))
            eta = x[0::2]
            excessive_displacement = np.any(np.abs(eta) > plant.eta_limit)
            terminated = bool(invalid_state or excessive_displacement)
            truncated = False
            info = {}

            if invalid_state:
                info["termination_reason"] = "invalid_state"

            if excessive_displacement:
                info["termination_reason"] = "excessive_modal_displacement"

        if terminated or truncated:
            termination_info = info
            break

    trajectory = np.asarray(trajectory, dtype=np.float64)
    time = np.asarray(time_history, dtype=np.float64)
    u_norm_history = np.asarray(u_norm_history, dtype=np.float64)
    u_phys_history = np.asarray(u_phys_history, dtype=np.float64)

    eta = trajectory[:, 0::2]
    eta_dot = trajectory[:, 1::2]

    return {
        "time": time,
        "trajectory": trajectory,
        "eta": eta,
        "eta_dot": eta_dot,
        "u_norm": u_norm_history,
        "u_phys": u_phys_history,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_info": termination_info,
    }


def modal_norm(eta: np.ndarray) -> np.ndarray:
    """
    Compute modal displacement norm over time.
    """
    return np.sqrt(np.sum(eta**2, axis=1))


def check_finite(name: str, result: dict) -> bool:
    """
    Check that trajectory contains no NaN/Inf.
    """
    finite = np.all(np.isfinite(result["trajectory"]))

    if finite:
        print(f"[PASS] {name}: all states are finite.")
    else:
        print(f"[FAIL] {name}: NaN or Inf detected.")

    if result["terminated"]:
        print(f"[INFO] {name}: simulation terminated.")
        print("       reason:", result["termination_info"])

    if result["truncated"]:
        print(f"[INFO] {name}: simulation truncated.")

    return bool(finite)


# ============================================================
# Plotting
# ============================================================

def plot_case(name: str, result: dict, max_modes_to_plot: int = 4) -> None:
    """
    Plot first mode, phase portrait, inputs, and selected modal states.
    """
    time = result["time"]
    eta = result["eta"]
    eta_dot = result["eta_dot"]
    u_phys = result["u_phys"]

    K = eta.shape[1]
    n_plot = min(K, max_modes_to_plot)

    eta1 = eta[:, 0]
    eta1_dot = eta_dot[:, 0]

    omega = u_phys[:, 0]
    ac = u_phys[:, 1]

    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })

    # First mode and phase portrait
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    ax[0].plot(time, eta1, linewidth=2)
    ax[0].set_title(f"{name} - First Mode", fontsize=14, fontweight="bold")
    ax[0].set_ylabel(r"$\eta_1$")

    ax[1].plot(time, eta1_dot, linewidth=2)
    ax[1].set_ylabel(r"$\dot{\eta}_1$")
    ax[1].set_xlabel("Time (s)")

    ax[2].plot(eta1, eta1_dot, linewidth=1.6)
    ax[2].scatter(eta1[0], eta1_dot[0], s=60, label="Start", zorder=3)
    ax[2].scatter(eta1[-1], eta1_dot[-1], s=60, label="End", zorder=3)
    ax[2].set_title("Phase Portrait")
    ax[2].set_xlabel(r"$\eta_1$")
    ax[2].set_ylabel(r"$\dot{\eta}_1$")
    ax[2].legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # Inputs
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax[0].plot(time, omega, linewidth=2)
    ax[0].set_title(f"{name} - Physical Inputs", fontsize=14, fontweight="bold")
    ax[0].set_ylabel(r"$\Omega$")

    ax[1].plot(time, ac, linewidth=2)
    ax[1].set_ylabel(r"$a_c$")
    ax[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

    # Selected modal states
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for i in range(n_plot):
        ax[0].plot(time, eta[:, i], linewidth=1.4, label=fr"$\eta_{i+1}$")

    ax[0].set_title(f"{name} - Modal Displacements", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Modal displacement")
    ax[0].legend(ncol=min(n_plot, 4), frameon=False)

    for i in range(n_plot):
        ax[1].plot(time, eta_dot[:, i], linewidth=1.4, label=fr"$\dot{{\eta}}_{i+1}$")

    ax[1].set_title("Modal Velocities")
    ax[1].set_ylabel("Modal velocity")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(ncol=min(n_plot, 4), frameon=False)

    plt.tight_layout()
    plt.show()


# ============================================================
# Tests
# ============================================================

def test_import_and_attributes(plant: PlatePlant) -> None:
    """
    Print import source and important attributes.
    """
    print("============================================================")
    print("Test 0: Import and attribute check")
    print("============================================================")

    try:
        print("PlatePlant imported from:", inspect.getfile(PlatePlant))
    except Exception as exc:
        print("Could not determine import file:", exc)

    print("PlatePlant class module:", getattr(PlatePlant, "__module__", "unknown"))

    print("Has K:", hasattr(plant, "K"))
    print("Has state_dim:", hasattr(plant, "state_dim"))
    print("Has u_phys_low:", hasattr(plant, "u_phys_low"))
    print("Has u_phys_high:", hasattr(plant, "u_phys_high"))
    print("Has _scale_action:", hasattr(plant, "_scale_action"))
    print("Has dynamics:", hasattr(plant, "dynamics"))
    print("Has termination:", hasattr(plant, "termination"))

    print()
    print("K:", getattr(plant, "K", None))
    print("state_dim:", get_state_dim(plant))
    print("u_phys_low [omega, ac]:", get_action_bounds(plant)[0])
    print("u_phys_high [omega, ac]:", get_action_bounds(plant)[1])
    print()


def test_action_scaling(plant: PlatePlant) -> None:
    """
    Verify normalized action mapping.
    """
    print("============================================================")
    print("Test 1: Action scaling")
    print("============================================================")

    test_actions = {
        "low": np.array([-1.0, -1.0]),
        "middle": np.array([0.0, 0.0]),
        "high": np.array([1.0, 1.0]),
    }

    for label, u_norm in test_actions.items():
        u_phys = scale_action_fallback(plant, u_norm)
        print(f"{label:>6} normalized {u_norm} -> physical [omega, ac] {u_phys}")

    print()
    print("Important:")
    print("  normalized [0, 0] means middle of action range, not zero physical input.")
    print("  physical action order is [omega, ac].")
    print()


def test_zero_equilibrium(plant: PlatePlant) -> dict:
    """
    Zero IC + zero depth of cut.
    """
    print("============================================================")
    print("Test 2: Zero initial condition + zero depth of cut")
    print("============================================================")

    state_dim = get_state_dim(plant)

    x0 = np.zeros(state_dim, dtype=np.float64)

    omega_no_cut = float(get_action_bounds(plant)[0][0])
    ac_no_cut = float(get_action_bounds(plant)[0][1])

    u_no_cut = physical_to_normalized_action(
        plant,
        omega=omega_no_cut,
        ac=ac_no_cut,
    )

    print("Physical input [omega, ac]:", scale_action_fallback(plant, u_no_cut))
    print("Normalized input:", u_no_cut)

    result = simulate_case(
        plant=plant,
        x0=x0,
        u_norm=u_no_cut,
        dt=DT,
        t_final=T_FINAL,
    )

    check_finite("Zero equilibrium", result)

    max_abs_state = float(np.max(np.abs(result["trajectory"])))
    print("Maximum absolute state:", max_abs_state)

    if max_abs_state < ZERO_TOL:
        print("[PASS] Zero state remains approximately zero.")
    else:
        print("[WARNING] Zero state did not remain zero.")
        print("          This may mean residual forcing exists even when ac = 0.")
    print()

    return result


def test_free_vibration_decay(plant: PlatePlant) -> dict:
    """
    Small initial modal displacement + zero depth of cut.
    """
    print("============================================================")
    print("Test 3: Free vibration decay")
    print("============================================================")

    state_dim = get_state_dim(plant)

    x0 = np.zeros(state_dim, dtype=np.float64)
    x0[0] = 1e-4

    omega_no_cut = float(get_action_bounds(plant)[0][0])
    ac_no_cut = float(get_action_bounds(plant)[0][1])

    u_no_cut = physical_to_normalized_action(
        plant,
        omega=omega_no_cut,
        ac=ac_no_cut,
    )

    print("Initial eta_1:", x0[0])
    print("Physical input [omega, ac]:", scale_action_fallback(plant, u_no_cut))
    print("Normalized input:", u_no_cut)

    result = simulate_case(
        plant=plant,
        x0=x0,
        u_norm=u_no_cut,
        dt=DT,
        t_final=T_FINAL,
    )

    check_finite("Free vibration decay", result)

    norm_eta = modal_norm(result["eta"])

    n = len(norm_eta)
    window = max(10, n // 10)

    first_peak = float(np.max(norm_eta[:window]))
    last_peak = float(np.max(norm_eta[-window:]))

    ratio = last_peak / max(first_peak, 1e-30)

    print("First-window peak modal norm:", first_peak)
    print("Last-window peak modal norm:", last_peak)
    print("Last / first ratio:", ratio)

    if ratio < DECAY_TOL:
        print("[PASS] Vibration decays with zero depth of cut.")
    else:
        print("[WARNING] Vibration did not clearly decay.")
        print("          Check damping, time step, or residual forcing.")
    print()

    return result


def test_nonzero_cutting(plant: PlatePlant) -> dict:
    """
    Nonzero cutting condition.
    """
    print("============================================================")
    print("Test 4: Nonzero cutting condition")
    print("============================================================")

    state_dim = get_state_dim(plant)
    low, high = get_action_bounds(plant)

    x0 = np.zeros(state_dim, dtype=np.float64)
    x0[0] = 1e-5

    omega_test = 0.5 * (low[0] + high[0])
    ac_test = min(max(2.5, low[1]), high[1])

    u_cut = physical_to_normalized_action(
        plant,
        omega=omega_test,
        ac=ac_test,
    )

    print("Initial eta_1:", x0[0])
    print("Physical input [omega, ac]:", scale_action_fallback(plant, u_cut))
    print("Normalized input:", u_cut)

    result = simulate_case(
        plant=plant,
        x0=x0,
        u_norm=u_cut,
        dt=DT,
        t_final=T_FINAL,
    )

    check_finite("Nonzero cutting", result)

    max_eta = float(np.max(np.abs(result["eta"])))
    max_eta_dot = float(np.max(np.abs(result["eta_dot"])))

    print("Maximum |eta|:", max_eta)
    print("Maximum |eta_dot|:", max_eta_dot)

    if result["terminated"]:
        print("[INFO] Simulation terminated. This may indicate instability or unsafe response.")
    else:
        print("[PASS] Nonzero cutting simulation completed without termination.")

    print()

    return result


def test_rk4_timestep_consistency(plant: PlatePlant) -> None:
    """
    Compare free vibration using dt and dt/2.
    """
    print("============================================================")
    print("Test 5: RK4 time-step consistency")
    print("============================================================")

    state_dim = get_state_dim(plant)

    x0 = np.zeros(state_dim, dtype=np.float64)
    x0[0] = 1e-4

    omega_no_cut = float(get_action_bounds(plant)[0][0])
    ac_no_cut = float(get_action_bounds(plant)[0][1])

    u_no_cut = physical_to_normalized_action(
        plant,
        omega=omega_no_cut,
        ac=ac_no_cut,
    )

    result_dt = simulate_case(
        plant=plant,
        x0=x0,
        u_norm=u_no_cut,
        dt=DT,
        t_final=T_FINAL,
    )

    result_half_dt = simulate_case(
        plant=plant,
        x0=x0,
        u_norm=u_no_cut,
        dt=DT / 2.0,
        t_final=T_FINAL,
    )

    eta1_dt_final = float(result_dt["eta"][-1, 0])
    eta1_half_dt_final = float(result_half_dt["eta"][-1, 0])

    abs_diff = abs(eta1_dt_final - eta1_half_dt_final)
    scale = max(abs(eta1_half_dt_final), 1e-12)
    rel_diff = abs_diff / scale

    print("Final eta_1 with dt:", eta1_dt_final)
    print("Final eta_1 with dt/2:", eta1_half_dt_final)
    print("Absolute difference:", abs_diff)
    print("Relative difference:", rel_diff)

    if rel_diff < 0.10:
        print("[PASS] RK4 response is reasonably consistent between dt and dt/2.")
    else:
        print("[WARNING] RK4 response differs noticeably between dt and dt/2.")
        print("          Consider reducing dt or checking force discontinuities.")

    print()


# ============================================================
# Main
# ============================================================

def main() -> None:
    plant = PlatePlant()

    patch_missing_attributes(plant)

    print("============================================================")
    print("PlatePlant verification started")
    print("============================================================")
    print("Number of modes K:", plant.K)
    print("State dimension:", get_state_dim(plant))
    print("Physical action low  [omega, ac]:", get_action_bounds(plant)[0])
    print("Physical action high [omega, ac]:", get_action_bounds(plant)[1])
    print("eta_limit:", getattr(plant, "eta_limit", None))
    print("dt:", DT)
    print("t_final:", T_FINAL)
    print()

    test_import_and_attributes(plant)
    test_action_scaling(plant)

    result_zero = test_zero_equilibrium(plant)
    result_free = test_free_vibration_decay(plant)
    result_cut = test_nonzero_cutting(plant)
    test_rk4_timestep_consistency(plant)

    if SHOW_PLOTS:
        plot_case("Test 2: Zero IC + Zero Depth of Cut", result_zero)
        plot_case("Test 3: Free Vibration Decay", result_free)
        plot_case("Test 4: Nonzero Cutting", result_cut)

    print("============================================================")
    print("Verification completed")
    print("============================================================")
    print("Interpretation:")
    print("1. Passing these tests means the ODE/RK4 simulator is numerically consistent.")
    print("2. These tests do not prove that the model contains regenerative chatter.")
    print("3. To verify regenerative chatter, the cutting force should depend on")
    print("   current and delayed vibration states, such as eta(t) - eta(t - tau).")
    print("4. If the import path printed above is not the file you expect, update your")
    print("   working directory, import statement, or file name.")


if __name__ == "__main__":
    main()