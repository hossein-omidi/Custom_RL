"""
Physical unit conventions for the plate milling plant.

Target convention
-----------------
* omega: rad/s
* RL physical action a_c: mm
* Internal axial z, dz, a_c integration: m
* Plate L1, L2, h: m
* Modal eta, q, w, chip h_j, Delta_f, Delta_n: m
* Modal velocity / acceleration: m/s, m/s^2
* f_t: m/tooth
* Cutting forces: N
* Modal excitation F_k in eta_ddot: m/s^2  (F_k = W_k/M * F_scalar)

Cutting coefficients (MATLAB port)
----------------------------------
``xi`` and ``delta`` are stored in **mm–m hybrid** form for the polynomial

    dF = (xi1*h_mm^3 + xi2*h_mm^2 + xi3*h_mm + xi4) * dz_m

with ``h_mm = 1000 * h_m`` and ``dz_m`` in metres.  This matches the legacy
coefficient table in ``plate.py`` (values / 2.5 from the MATLAB source).

``FORCE_COEFFICIENT_SCALE`` calibrates the legacy polynomial so typical surface
milling at ``ac ~ 1–10 mm`` produces O(10^2–10^4) N total directional force
without hitting the numerical safety clip (regenerative oscillation, not clipping).

High-order terms (h^3, h^2) receive an extra ``HIGH_ORDER_FORCE_SCALE`` because the
legacy MATLAB table is stiff in mm–m hybrid form: small regenerative chip variations
must not trigger explosive force growth before chatter physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MM_PER_M = 1000.0
M_PER_MM = 1e-3

# Legacy MATLAB coeffs are /2.5 in plate; additional scale for ac in mm, dz in m.
MATLAB_COEFF_DIVISOR = 2.5
# Calibrated: |F_n| ~ 1e2–1e3 N at ac=1 mm, omega=800 rad/s (zero vibration); ~linear in ac.
FORCE_COEFFICIENT_SCALE = 1e-2
# Extra attenuation on h^3/h^2 terms (legacy hybrid coeffs are overly stiff in h).
HIGH_ORDER_FORCE_SCALE = 1e-2

# Numerical safety only — not active during normal stable cuts after calibration.
MAX_FORCE_SAFETY_N = 1.0e6

# Empirical polynomial term units for the hybrid (h_mm, dz_m) convention.
CUTTING_COEFF_TERM_UNITS_MM_M = (
    "N/(mm^3·m)",  # xi1 on h^3
    "N/(mm^2·m)",  # xi2 on h^2
    "N/(mm·m)",    # xi3 on h
    "N/m",         # xi4 constant term
)


@dataclass(frozen=True)
class CuttingCoefficientUnits:
    """Document original vs SI-equivalent cutting polynomial coefficients."""

    term: str
    original_unit: str
    h_in_polynomial: str
    dz_in_integration: str
    si_equivalent_scale: float
    si_unit: str


def ac_to_meters(ac: float, units: str = "mm") -> float:
    """Convert axial depth of cut to metres for internal force integration."""
    val = float(ac)
    if units == "mm":
        return val * M_PER_MM
    if units == "m":
        return val
    raise ValueError(f"Unknown ac_units: {units!r}. Use 'mm' or 'm'.")


def meters_to_mm(x_m: float) -> float:
    return float(x_m) * MM_PER_M


def chip_thickness_for_force_polynomial(h_m: float) -> float:
    """Convert geometric chip thickness [m] to mm for xi/delta polynomial."""
    return meters_to_mm(h_m)


def cutting_coefficients_si_from_mm_m_hybrid(
    xi: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert hybrid (h_mm, dz_m) coeffs to pure SI (h_m, dz_m) polynomial.

    dF = poly_SI(h_m) * dz_m  with  poly_SI(h_m) = sum_i xi_SI[i] * h_m^(3-i)
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(4)
    delta = np.asarray(delta, dtype=np.float64).reshape(4)
    scales = np.array([MM_PER_M**3, MM_PER_M**2, MM_PER_M, 1.0], dtype=np.float64)
    return xi * scales, delta * scales


def cutting_coefficient_unit_table(
    xi: np.ndarray,
    delta: np.ndarray,
) -> list[CuttingCoefficientUnits]:
    """Build audit table for coefficient units."""
    xi = np.asarray(xi, dtype=np.float64).reshape(4)
    powers = (3, 2, 1, 0)
    rows: list[CuttingCoefficientUnits] = []
    for i, p in enumerate(powers):
        si_scale = MM_PER_M ** (3 - p)
        rows.append(
            CuttingCoefficientUnits(
                term=f"h^{p}" if p else "const",
                original_unit=CUTTING_COEFF_TERM_UNITS_MM_M[i],
                h_in_polynomial="mm",
                dz_in_integration="m",
                si_equivalent_scale=float(si_scale),
                si_unit=f"N/m^{4-p}" if p < 4 else "N/m",
            )
        )
    return rows


def format_coefficient_table(xi: np.ndarray, delta: np.ndarray) -> str:
    """Human-readable coefficient unit table for reports."""
    lines = [
        "Cutting coefficient unit table (tangential xi, radial delta):",
        f"{'term':<8} {'orig unit':<16} {'h_poly':<6} {'dz':<4} {'SI scale':<12} {'SI unit':<10} xi value",
    ]
    for row, xv, dv in zip(
        cutting_coefficient_unit_table(xi, delta),
        np.asarray(xi).reshape(4),
        np.asarray(delta).reshape(4),
    ):
        lines.append(
            f"{row.term:<8} {row.original_unit:<16} {row.h_in_polynomial:<6} {row.dz_in_integration:<4} "
            f"{row.si_equivalent_scale:<12.3e} {row.si_unit:<10} {xv:.3e}"
        )
    return "\n".join(lines)


def scale_cutting_coefficients(
    xi: np.ndarray,
    delta: np.ndarray,
    *,
    scale: float = FORCE_COEFFICIENT_SCALE,
    high_order_scale: float = HIGH_ORDER_FORCE_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply calibration scale to legacy xi/delta arrays."""
    xi = np.asarray(xi, dtype=np.float64).reshape(4).copy()
    delta = np.asarray(delta, dtype=np.float64).reshape(4).copy()
    xi[:2] *= float(scale) * float(high_order_scale)
    xi[2:] *= float(scale)
    delta[:2] *= float(scale) * float(high_order_scale)
    delta[2:] *= float(scale)
    return xi, delta


def legacy_matlab_cutting_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Raw MATLAB-port tangential/radial polynomial coefficients (/2.5 only)."""
    xi = np.array([6765e9, -4910e6, 2840e3, 132], dtype=np.float64) / MATLAB_COEFF_DIVISOR
    delta = np.array([12740e9, -7452e6, 1674e3, 246], dtype=np.float64) / MATLAB_COEFF_DIVISOR
    return xi, delta


def calibrated_cutting_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Production xi/delta for feed_normal_full surface milling."""
    xi, delta = legacy_matlab_cutting_coefficients()
    return scale_cutting_coefficients(xi, delta)


def modal_acceleration_from_force(F_scalar_n: float, W_k: float, M_kg: float) -> float:
    """F_k = W_k/M * F_scalar  [m/s^2]."""
    return float(W_k) * float(F_scalar_n) / float(M_kg)


def dimensional_check_modal_ode(
    *,
    eta_m: float,
    eta_dot_m_s: float,
    omega_rad_s: float,
    zeta: float,
    lambda_k: float,
    F_k_m_s2: float,
) -> dict[str, float]:
    """Return term magnitudes [m/s^2] for the modal ODE."""
    lin = omega_rad_s**2 * eta_m
    damp = 2.0 * zeta * omega_rad_s * eta_dot_m_s
    nl = lambda_k * eta_m**3
    return {
        "omega2_eta": lin,
        "2zeta_omega_etad": damp,
        "lambda_eta3": nl,
        "F_k": F_k_m_s2,
    }
