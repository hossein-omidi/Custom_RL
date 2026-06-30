# -*- coding: utf-8 -*-
"""
Mode-shape functions for the flexible plate model.

This module is structural, not cutting-force-specific. Changing the milling force
model from peripheral milling to face milling does not require changing the
mode-shape definition itself. The face-milling force module should use these
mode shapes to project the global force, usually the axial/transverse component
Fz, into modal coordinates.

Public function kept compatible with the original code:
    W_mn, V_mn = compute_mode_shapes(L1, L2, h, m_max, n_max)

Default behaviour matches the previous MATLAB/Python port:
    W_mn[m][n](x, y) = phi_m(x/L1) * psi_n(y/L2)
    V_mn[m][n](z, y) = phi_m(z/h)  * psi_n(y/L2)

Additional optional keyword:
    clamped_axis="x"  -> original behaviour
    clamped_axis="y"  -> W_mn[m][n](x, y) = psi_n(x/L1) * phi_m(y/L2)

Use clamped_axis="y" only if the physical plate boundary condition is clamped
along the y-direction. The default is intentionally unchanged for backward
compatibility with the existing code base.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

ModeFunction = Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
ModeShapeGrid = List[List[ModeFunction]]


# More accurate roots than the rounded values in the original file.
# alpha: clamped-free Euler-Bernoulli beam roots.
# beta: free-free non-rigid roots. The first two free-free modes are treated
# explicitly as rigid/linear shape functions, so beta[0] is used for n == 2.
ALPHA_CLAMPED_FREE = np.array(
    [
        1.875104068711961,
        4.694091132974174,
        7.854757438237613,
        10.99554073487547,
        14.13716839104647,
        17.27875953208824,
    ],
    dtype=float,
)

BETA_FREE_FREE = np.array(
    [
        4.730040744862704,
        7.853204624095838,
        10.99560783800167,
        14.137165491257,
        17.2787596573995,
        20.4203522456261,
    ],
    dtype=float,
)


def _as_positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value. Got {value!r}.")
    return value


def _as_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {value!r}.")
    return value


def _to_array(value):
    """Convert scalar/array input to float ndarray without changing shape."""
    return np.asarray(value, dtype=float)


def _clamped_free_shape(alpha: float) -> Callable[[np.ndarray | float], np.ndarray]:
    """
    Return the clamped-free one-dimensional shape function.

    This preserves the algebraic form used in the original MATLAB/Python port,
    but uses more accurate roots and safer closures.
    """
    mu = (np.cosh(alpha) + np.cos(alpha)) / (np.sinh(alpha) * np.sin(alpha))
    v = (np.sinh(alpha) - np.sin(alpha)) / (np.sinh(alpha) * np.sin(alpha))

    def phi(s_normalized):
        s = _to_array(s_normalized)
        return (
            mu * (np.cosh(alpha * s) - np.cos(alpha * s))
            - v * (np.sinh(alpha * s) - np.sin(alpha * s))
        )

    return phi


def _free_free_shape(index: int) -> Callable[[np.ndarray | float], np.ndarray]:
    """
    Return free-free one-dimensional shape function for index n.

    n == 0: constant mode
    n == 1: linear mode
    n >= 2: flexible free-free beam-like mode using beta[n-2]
    """
    if index == 0:
        def psi0(s_normalized):
            s = _to_array(s_normalized)
            return np.ones_like(s, dtype=float)
        return psi0

    if index == 1:
        def psi1(s_normalized):
            s = _to_array(s_normalized)
            return np.sqrt(3.0) * (2.0 * s - 1.0)
        return psi1

    beta_idx = index - 2
    if beta_idx >= len(BETA_FREE_FREE):
        raise ValueError(
            f"n_max requests free-free mode index {index}, but only "
            f"{len(BETA_FREE_FREE) + 2} modes are available in this implementation."
        )

    beta = BETA_FREE_FREE[beta_idx]
    mu = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) * np.sin(beta))
    v = (np.sinh(beta) + np.sin(beta)) / (np.sinh(beta) * np.sin(beta))

    def psi(s_normalized):
        s = _to_array(s_normalized)
        return (
            mu * (np.cosh(beta * s) + np.cos(beta * s))
            - v * (np.sinh(beta * s) + np.sin(beta * s))
        )

    return psi


def _make_W_original(phi_m, psi_n, L1: float, L2: float) -> ModeFunction:
    """Original W_mn(x,y) = phi_m(x/L1) * psi_n(y/L2)."""
    def W_func(x, y):
        x_arr = _to_array(x)
        y_arr = _to_array(y)
        return phi_m(x_arr / L1) * psi_n(y_arr / L2)
    return W_func


def _make_W_clamped_y(phi_m, psi_n, L1: float, L2: float) -> ModeFunction:
    """Alternative W_mn(x,y) = psi_n(x/L1) * phi_m(y/L2)."""
    def W_func(x, y):
        x_arr = _to_array(x)
        y_arr = _to_array(y)
        return psi_n(x_arr / L1) * phi_m(y_arr / L2)
    return W_func


def _make_V_legacy(phi_m, psi_n, h: float, L2: float) -> ModeFunction:
    """Legacy V_mn(z,y) = phi_m(z/h) * psi_n(y/L2)."""
    def V_func(z, y):
        z_arr = _to_array(z)
        y_arr = _to_array(y)
        return phi_m(z_arr / h) * psi_n(y_arr / L2)
    return V_func


def compute_mode_shapes(
    L1: float,
    L2: float,
    h: float,
    m_max: int,
    n_max: int,
    *,
    clamped_axis: str = "x",
) -> Tuple[ModeShapeGrid, ModeShapeGrid]:
    """
    Compute mode-shape function grids W_mn and V_mn.

    Parameters
    ----------
    L1, L2 : float
        Plate dimensions in meters. W_mn uses x in [0, L1] and y in [0, L2].
    h : float
        Plate thickness in meters. V_mn uses z in [0, h] and y in [0, L2].
    m_max, n_max : int
        Number of mode functions in each direction.
    clamped_axis : {"x", "y"}, optional
        "x" preserves the original code's mode definition:
            W(x,y) = phi(x/L1) * psi(y/L2)
        "y" swaps the one-dimensional factors for W:
            W(x,y) = psi(x/L1) * phi(y/L2)
        V_mn is kept in the legacy z-y definition for backward compatibility.

    Returns
    -------
    W_mn, V_mn : list[list[callable]]
        W_mn[m][n](x, y) and V_mn[m][n](z, y).

    Notes
    -----
    This function is independent of whether the cutting process is peripheral
    milling or face milling. For the face-milling force model, W_mn is used to
    project the axial/transverse force component, typically Fz, into modal space:
        Q_k(t) = W_k(x_c, y_c) * Fz(t) / M_k
    """
    L1 = _as_positive_float(L1, "L1")
    L2 = _as_positive_float(L2, "L2")
    h = _as_positive_float(h, "h")
    m_max = _as_positive_int(m_max, "m_max")
    n_max = _as_positive_int(n_max, "n_max")

    clamped_axis = str(clamped_axis).lower().strip()
    if clamped_axis not in {"x", "y"}:
        raise ValueError("clamped_axis must be either 'x' or 'y'.")

    if m_max > len(ALPHA_CLAMPED_FREE):
        raise ValueError(
            f"m_max={m_max} exceeds available clamped-free roots "
            f"({len(ALPHA_CLAMPED_FREE)}). Add more ALPHA_CLAMPED_FREE roots."
        )

    if n_max > len(BETA_FREE_FREE) + 2:
        raise ValueError(
            f"n_max={n_max} exceeds available free-free modes "
            f"({len(BETA_FREE_FREE) + 2}). Add more BETA_FREE_FREE roots."
        )

    W_mn: ModeShapeGrid = [[None for _ in range(n_max)] for _ in range(m_max)]  # type: ignore[list-item]
    V_mn: ModeShapeGrid = [[None for _ in range(n_max)] for _ in range(m_max)]  # type: ignore[list-item]

    for m in range(m_max):
        phi_m = _clamped_free_shape(ALPHA_CLAMPED_FREE[m])

        for n in range(n_max):
            psi_n = _free_free_shape(n)

            if clamped_axis == "x":
                W_mn[m][n] = _make_W_original(phi_m, psi_n, L1, L2)
            else:
                W_mn[m][n] = _make_W_clamped_y(phi_m, psi_n, L1, L2)

            V_mn[m][n] = _make_V_legacy(phi_m, psi_n, h, L2)

    return W_mn, V_mn


def build_mode_shape_matrix(
    W_mn: ModeShapeGrid,
    sensor_points: Sequence[tuple[float, float]],
    m_max: int | None = None,
    n_max: int | None = None,
) -> np.ndarray:
    """
    Build a physical-location mode-shape matrix Phi.

    This is useful for wrappers and force projection:
        Phi[i, k] = W_k(x_i, y_i)

    Parameters
    ----------
    W_mn : list[list[callable]]
        Mode-shape function grid returned by compute_mode_shapes.
    sensor_points : sequence of (x, y)
        Physical points in meters.
    m_max, n_max : int, optional
        If omitted, inferred from W_mn.

    Returns
    -------
    Phi : ndarray, shape (len(sensor_points), m_max*n_max)
    """
    if m_max is None:
        m_max = len(W_mn)
    if n_max is None:
        n_max = len(W_mn[0]) if len(W_mn) > 0 else 0

    m_max = _as_positive_int(m_max, "m_max")
    n_max = _as_positive_int(n_max, "n_max")

    points = list(sensor_points)
    Phi = np.zeros((len(points), m_max * n_max), dtype=float)

    k = 0
    for m in range(m_max):
        for n in range(n_max):
            W = W_mn[m][n]
            for i, (x, y) in enumerate(points):
                Phi[i, k] = float(np.asarray(W(float(x), float(y))).reshape(()))
            k += 1

    return Phi


__all__ = [
    "compute_mode_shapes",
    "build_mode_shape_matrix",
    "ALPHA_CLAMPED_FREE",
    "BETA_FREE_FREE",
]
