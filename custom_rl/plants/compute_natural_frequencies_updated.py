# -*- coding: utf-8 -*-
"""Natural frequencies for the flexible plate modal model.

This module is structural. It does not contain any milling-force model.
Therefore, changing the cutting process from peripheral milling to face milling
normally does not change this file. The natural frequencies depend on plate
material, geometry, boundary-condition coefficients, and mass per unit area.

The function keeps the original public name and positional arguments:

    compute_natural_frequencies(E, nu, rho, h, a, b, m_max, n_max)

Units:
    E      : Pa = N/m^2
    nu     : dimensionless
    rho    : kg/m^3 if rho_type='volumetric', kg/m^2 if rho_type='areal'
             With rho_type='auto', values larger than 500 are treated as
             volumetric density and are multiplied by h.
    h      : m
    a, b   : m, plate dimensions used in the frequency formula
    output : rad/s

Important mass convention:
    The plate frequency formula uses areal mass rho*h [kg/m^2]. If the caller
    passes material density rho [kg/m^3], the function must multiply by h.
    If the caller passes areal mass directly, the function must not multiply by h.
"""

from __future__ import annotations

import numpy as np


# Dimensionless frequency coefficients. These values are kept from the original
# MATLAB/Python port. The first two columns identify the mode-index pair used by
# the original code; the remaining columns correspond to aspect ratios below.
_EXP_DATA = np.array(
    [
        [0, 0, 3.50, 3.50, 3.50, 3.45],
        [0, 1, 21.7, 21.6, 21.5, 21.1],
        [0, 2, 60.5, 60.4, 59.8, 59.3],
        [0, 3, 118.7, 117.5, 116.5, 115.2],
        [0, 4, 196.0, 195.0, 190.0, 188.0],
        [0, 5, 292.0, 285.0, 283.0, 281.0],
        [1, 0, 14.5, 17.3, 22.5, 32.0],
        [1, 1, 48.1, 54.8, 69.6, 98.0],
        [1, 2, 92.3, 101.5, 125.0, 169.0],
        [1, 3, 154.0, 170.0, 187.0, 248.0],
        [1, 4, 228.0, np.nan, np.nan, np.nan],
        [1, 5, np.nan, np.nan, np.nan, np.nan],
        [2, 0, 92.8, 139.1, 246.0, 321.0],
        [2, 1, 125.1, np.nan, np.nan, np.nan],
        [2, 2, 176.0, np.nan, np.nan, np.nan],
        [2, 3, 244.0, np.nan, np.nan, np.nan],
        [3, 0, 246.0, np.nan, np.nan, np.nan],
        [3, 1, 274.0, np.nan, np.nan, np.nan],
        [3, 2, np.nan, np.nan, np.nan, np.nan],
    ],
    dtype=np.float64,
)

_ASPECT_RATIOS = np.array([2.00, 2.50, 3.33, 5.00], dtype=np.float64)

# Square-plate coefficients for the first six x-y-plane modes reported for the
# AL7075 cantilever-plate reference used by this project:
# [107.4, 258.8, 654.6, 837.4, 946.0, 1654.2] rad/s.
#
# The mapping is zero-based and follows compute_mode_shapes_updated:
#   W[m][n] = phi_m(clamped-free direction) * psi_n(width/free-free direction)
# so the first-six reference set is represented by:
#   (0,0), (0,1), (1,0), (0,2), (1,1), (1,2).
#
# These dimensionless coefficients are omega_k /
# sqrt(D/(rho_areal*L1^4)) for E=71.7e9 Pa, nu=0.33,
# rho_areal=56.2 kg/m^2, L1=L2=1 m, h=0.02 m.  They prevent invalid
# extrapolation of the older table, whose aspect-ratio data start at a/b=2.
_SQUARE_ASPECT_MODE_COEFFS = {
    (0, 0): 3.4763357259909563,
    (0, 1): 8.376868598725245,
    (1, 0): 21.18816919712854,
    (0, 2): 27.105060934097756,
    (1, 1): 30.62023840479446,
    (1, 2): 53.54333864954204,
}


def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return value


def _linear_interp_extrap(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """1D linear interpolation with true linear extrapolation and NaN handling."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)

    if not np.any(valid):
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]

    if x_valid.size == 1:
        return float(y_valid[0])

    xq = float(xq)

    # np.interp clamps outside the data range, so explicit extrapolation is used.
    if xq < x_valid[0]:
        slope = (y_valid[1] - y_valid[0]) / (x_valid[1] - x_valid[0])
        return float(y_valid[0] + slope * (xq - x_valid[0]))

    if xq > x_valid[-1]:
        slope = (y_valid[-1] - y_valid[-2]) / (x_valid[-1] - x_valid[-2])
        return float(y_valid[-1] + slope * (xq - x_valid[-1]))

    return float(np.interp(xq, x_valid, y_valid))


def _resolve_areal_density(rho: float, h: float, rho_type: str) -> tuple[float, str]:
    """Return areal mass [kg/m^2] and the interpreted density type."""
    rho = _validate_positive("rho", rho)
    h = _validate_positive("h", h)

    rho_type = str(rho_type).lower().strip()

    if rho_type in {"volumetric", "volume", "density", "kg/m3", "kg/m^3"}:
        return rho * h, "volumetric"

    if rho_type in {"areal", "surface", "area", "kg/m2", "kg/m^2"}:
        return rho, "areal"

    if rho_type == "auto":
        # Engineering heuristic: metallic volumetric densities are usually
        # thousands of kg/m^3; areal masses for thin plates are usually tens.
        if rho > 500.0:
            return rho * h, "volumetric(auto)"
        return rho, "areal(auto)"

    raise ValueError(
        "rho_type must be 'auto', 'volumetric', or 'areal'. "
        f"Received {rho_type!r}."
    )


def _frequency_coefficient_table(aspect_ratio: float) -> np.ndarray:
    """Interpolate/extrapolate all dimensionless frequency coefficients."""
    values = _EXP_DATA[:, 2:6]
    coefficients = np.empty(values.shape[0], dtype=np.float64)

    for row_idx, y_row in enumerate(values):
        coefficients[row_idx] = _linear_interp_extrap(
            _ASPECT_RATIOS,
            y_row,
            aspect_ratio,
        )

    return coefficients


def compute_natural_frequencies(
    E,
    nu,
    rho,
    h,
    a,
    b,
    m_max,
    n_max,
    *,
    rho_type: str = "auto",
    fallback_coefficient: float = 500.0,
    return_info: bool = False,
):
    """Compute natural angular frequencies of the flexible plate.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio. Must satisfy -1 < nu < 0.5 for this plate formula.
    rho : float
        Either volumetric density [kg/m^3] or areal mass [kg/m^2]. Controlled
        by ``rho_type``. The default ``rho_type='auto'`` treats large values
        such as 2810 as volumetric density and values such as 56.2 as areal
        mass.
    h : float
        Plate thickness [m].
    a, b : float
        Plate dimensions [m]. The aspect ratio is a / b. The frequency scale
        follows the original MATLAB model, sqrt(D / (rho_areal * a^4)).
    m_max, n_max : int
        Number of modes in each index direction.
    rho_type : {'auto', 'volumetric', 'areal'}, optional
        How to interpret ``rho``.
    fallback_coefficient : float, optional
        Dimensionless coefficient used when the table has no valid coefficient
        for a requested mode pair.
    return_info : bool, optional
        If True, returns ``(omega_mn, info)``.

    Returns
    -------
    omega_mn : ndarray, shape (m_max, n_max)
        Natural angular frequencies [rad/s].
    info : dict, optional
        Returned only when ``return_info=True``. Contains density convention and
        plate rigidity used in the calculation.

    Notes
    -----
    This module is independent of the milling-force model. It is valid for the
    structural modal model used by both peripheral and face milling, as long as
    the same plate geometry/boundary assumptions are intended.
    """

    E = _validate_positive("E", E)
    h = _validate_positive("h", h)
    a = _validate_positive("a", a)
    b = _validate_positive("b", b)

    nu = float(nu)
    if not np.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise ValueError(f"nu must satisfy -1 < nu < 0.5, got {nu!r}.")

    m_max = int(m_max)
    n_max = int(n_max)
    if m_max <= 0 or n_max <= 0:
        raise ValueError("m_max and n_max must be positive integers.")

    rho_areal, interpreted_rho_type = _resolve_areal_density(rho, h, rho_type)

    # Plate bending rigidity [N*m].
    bending_rigidity = E * h**3 / (12.0 * (1.0 - nu**2))

    aspect_ratio = a / b
    coeffs = _frequency_coefficient_table(aspect_ratio)

    fallback_coefficient = float(fallback_coefficient)
    if not np.isfinite(fallback_coefficient) or fallback_coefficient <= 0.0:
        raise ValueError("fallback_coefficient must be positive and finite.")

    omega_mn = np.zeros((m_max, n_max), dtype=np.float64)
    scale = np.sqrt(bending_rigidity / (rho_areal * a**4))

    for m in range(m_max):
        for n in range(n_max):
            if abs(aspect_ratio - 1.0) < 1e-10 and (m, n) in _SQUARE_ASPECT_MODE_COEFFS:
                coeff = _SQUARE_ASPECT_MODE_COEFFS[(m, n)]
            else:
                # Keep the original indexing convention:
                # table column 0 is matched to n, table column 1 is matched to m.
                mask = (_EXP_DATA[:, 0] == float(n)) & (_EXP_DATA[:, 1] == float(m))
                idx = np.where(mask)[0]

                if idx.size == 0:
                    coeff = fallback_coefficient
                else:
                    coeff = coeffs[idx[0]]
                    if not np.isfinite(coeff):
                        coeff = fallback_coefficient

            omega_mn[m, n] = scale * coeff

    if return_info:
        info = {
            "bending_rigidity_Nm": bending_rigidity,
            "rho_areal_kg_m2": rho_areal,
            "interpreted_rho_type": interpreted_rho_type,
            "aspect_ratio": aspect_ratio,
            "frequency_scale_rad_s": scale,
            "uses_square_reference_coefficients": bool(abs(aspect_ratio - 1.0) < 1e-10),
            "units": "rad/s",
        }
        return omega_mn, info

    return omega_mn


# Alias for readability in face-milling projects. It intentionally calls the
# same structural calculation; face milling changes the force model, not the
# natural-frequency calculation.
compute_natural_frequencies_for_face_milling = compute_natural_frequencies
