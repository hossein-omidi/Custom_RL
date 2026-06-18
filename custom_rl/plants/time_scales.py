"""Time-scale analysis for regenerative milling integration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TWO_PI = 2.0 * math.pi


def tooth_period(omega_rad_s: float, n_teeth: int) -> float:
    """Tooth-passing delay tau = 2*pi / (N * omega) [s]."""
    omega_rad_s = max(float(omega_rad_s), 1e-9)
    return TWO_PI / (float(n_teeth) * omega_rad_s)


def revolution_period(omega_rad_s: float) -> float:
    """Spindle revolution period 2*pi/omega [s]."""
    return TWO_PI / max(float(omega_rad_s), 1e-9)


def milling_time_scales(
    *,
    omega_min: float,
    omega_max: float,
    n_teeth: int,
    omega_n_rad_s: np.ndarray,
    omega_f_rad_s: np.ndarray | None = None,
    macro_dt: float | None = None,
    rk4_stability_factor: float = 0.45,
    samples_per_tau: float = 6.0,
    samples_per_tooth_pass: float = 10.0,
    training_mode: bool = False,
) -> dict[str, Any]:
    """
    Compute characteristic time scales and recommended integration steps.

    Heuristics (explicit RK4 + tooth delay):
    - modal: dt * omega_max < ~2  -> dt < rk4_stability_factor * 2 / omega_max
    - delay: >= samples_per_tau points per shortest tooth delay
    - tooth excitation: resolve one tooth immersion per revolution
    """
    if training_mode:
        rk4_stability_factor = max(rk4_stability_factor, 0.5)
        samples_per_tau = min(samples_per_tau, 4.0)
        samples_per_tooth_pass = min(samples_per_tooth_pass, 6.0)

    omega_min = max(float(omega_min), 1e-9)
    omega_max = max(float(omega_max), omega_min)

    tau_min = tooth_period(omega_max, n_teeth)
    tau_max = tooth_period(omega_min, n_teeth)
    T_rev_min = revolution_period(omega_max)
    T_rev_max = revolution_period(omega_min)
    T_tooth_min = T_rev_min / float(n_teeth)

    om_n = np.asarray(omega_n_rad_s, dtype=np.float64).reshape(-1)
    om_f = (
        np.asarray(omega_f_rad_s, dtype=np.float64).reshape(-1)
        if omega_f_rad_s is not None
        else np.array([], dtype=np.float64)
    )
    omega_modal_max = float(np.max(np.concatenate([om_n, om_f])) if om_f.size else np.max(om_n))
    omega_modal_min = float(np.min(np.concatenate([om_n, om_f])) if om_f.size else np.min(om_n))
    T_modal_min = TWO_PI / omega_modal_max
    T_modal_max = TWO_PI / omega_modal_min

    dt_modal_rk4 = rk4_stability_factor * 2.0 / omega_modal_max
    dt_delay = tau_min / samples_per_tau
    dt_tooth = T_tooth_min / samples_per_tooth_pass

    dt_verification = min(dt_modal_rk4, dt_delay, dt_tooth)
    dt_training = dt_verification

    if macro_dt is not None and macro_dt > 0.0:
        n_substeps = max(1, int(math.ceil(float(macro_dt) / dt_training)))
        dt_training = float(macro_dt) / n_substeps
    else:
        n_substeps = 1
        macro_dt = dt_training

    return {
        "tau_min_s": tau_min,
        "tau_max_s": tau_max,
        "T_rev_min_s": T_rev_min,
        "T_rev_max_s": T_rev_max,
        "T_tooth_min_s": T_tooth_min,
        "omega_modal_max_rad_s": omega_modal_max,
        "omega_modal_min_rad_s": omega_modal_min,
        "T_modal_min_s": T_modal_min,
        "T_modal_max_s": T_modal_max,
        "dt_modal_rk4_s": dt_modal_rk4,
        "dt_delay_s": dt_delay,
        "dt_tooth_s": dt_tooth,
        "dt_recommended_verification_s": dt_verification,
        "dt_recommended_training_substep_s": dt_training,
        "macro_dt_s": float(macro_dt),
        "n_substeps": n_substeps,
        "rk4_omega_dt_product": dt_training * omega_modal_max,
        "delay_dt_ratio": tau_min / dt_training,
    }


def recommend_integration_dt(
    plant,
    macro_dt: float | None = None,
    *,
    training_mode: bool = False,
) -> dict[str, Any]:
    """Time scales from a configured :class:`PlatePlant`."""
    two_field = plant.milling_config.is_feed_normal_full()
    return milling_time_scales(
        omega_min=plant.omega_min,
        omega_max=plant.omega_max,
        n_teeth=plant.N,
        omega_n_rad_s=plant.omega_vec,
        omega_f_rad_s=plant.omega_f_vec if two_field else None,
        macro_dt=macro_dt,
        training_mode=training_mode,
    )


def format_time_scales_report(scales: dict[str, Any]) -> str:
    """Human-readable summary for scripts and Test2/Test3."""
    lines = [
        "Time scales:",
        f"  tau (tooth delay): [{scales['tau_min_s']:.4e}, {scales['tau_max_s']:.4e}] s",
        f"  modal period:      [{scales['T_modal_min_s']:.4e}, {scales['T_modal_max_s']:.4e}] s",
        f"  omega_modal max:   {scales['omega_modal_max_rad_s']:.2f} rad/s",
        f"  tooth-pass scale:  {scales['T_tooth_min_s']:.4e} s",
        "Recommended dt:",
        f"  verification:  {scales['dt_recommended_verification_s']:.4e} s",
        f"  training sub: {scales['dt_recommended_training_substep_s']:.4e} s "
        f"({scales['n_substeps']} substeps @ macro {scales['macro_dt_s']:.4e} s)",
        f"  dt*omega_max: {scales['rk4_omega_dt_product']:.3f}",
        f"  tau_min/dt:   {scales['delay_dt_ratio']:.1f} samples per shortest delay",
    ]
    return "\n".join(lines)
