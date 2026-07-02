# -*- coding: utf-8 -*-
"""
Numerical nonlinear stiffness coefficients for modal plate models.

This module is independent of the milling force model. Therefore, changing the
cutting-force model from peripheral milling to face milling does not require a
new stiffness derivation here. The modal stiffness coefficients are still based
on the selected plate/beam mode shapes. In the face-milling model, the axial
cutting force Fz should be projected onto the transverse mode shapes W_mn(x, y)
in the dynamics module; this file only computes the nonlinear stiffness terms.

Units:
    E   : Pa = N/m^2
    nu  : dimensionless
    h   : m
    L1  : m
    L2  : m
    W_mn(x, y) : mode shape over the x-y plate surface
    V_mn(z, y) : optional mode shape over the z-y side plane

Public functions:
    compute_second_derivatives(...)
    compute_nonlinear_stiffness(...)
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np


ModeFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _trapezoid(values: np.ndarray, grid: np.ndarray, axis: int) -> np.ndarray:
    """Compatibility wrapper for NumPy trapezoidal integration."""
    trapz = getattr(np, "trapezoid", np.trapz)
    return trapz(values, grid, axis=axis)


def _validate_positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar, got {value!r}.")
    return value


def _validate_mode_grid(name: str, values: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ensure mode-shape values are numeric and have the mesh-grid shape."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 0:
        # Some simple mode functions may return a scalar. Broadcast safely.
        values = np.full_like(X, float(values), dtype=np.float64)

    if values.shape != X.shape or values.shape != Y.shape:
        raise ValueError(
            f"{name} must return an array with shape {X.shape}; got {values.shape}."
        )

    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def compute_second_derivatives(
    f_values: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute second derivatives of a mode-shape field on a rectangular grid.

    The grid convention is the same as ``np.meshgrid(x, y, indexing="xy")``:
        - axis 0 corresponds to y rows
        - axis 1 corresponds to x columns

    Parameters
    ----------
    f_values:
        Function values f(x, y) on the grid, shape (ny, nx).
    X, Y:
        Mesh-grid arrays with the same shape as f_values.
    Lx, Ly:
        Domain lengths in the x-like and y-like directions.

    Returns
    -------
    f_xx, f_yy:
        Numerical approximations of d²f/dx² and d²f/dy².
    """
    f_values = np.asarray(f_values, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if f_values.shape != X.shape or f_values.shape != Y.shape:
        raise ValueError(
            "f_values, X, and Y must have the same shape. "
            f"Got {f_values.shape}, {X.shape}, and {Y.shape}."
        )

    if f_values.ndim != 2:
        raise ValueError(f"f_values must be 2D, got ndim={f_values.ndim}.")

    ny, nx = f_values.shape
    if nx < 3 or ny < 3:
        raise ValueError(
            "At least 3 grid points in both directions are required for second "
            f"derivatives; got nx={nx}, ny={ny}."
        )

    Lx = _validate_positive_scalar("Lx", Lx)
    Ly = _validate_positive_scalar("Ly", Ly)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # np.gradient returns derivatives in axis order: (d/dy, d/dx)
    edge_order = 2 if min(nx, ny) >= 3 else 1

    f_y, f_x = np.gradient(f_values, dy, dx, edge_order=edge_order)
    f_yy, _ = np.gradient(f_y, dy, dx, edge_order=edge_order)
    _, f_xx = np.gradient(f_x, dy, dx, edge_order=edge_order)

    f_xx = np.nan_to_num(f_xx, nan=0.0, posinf=0.0, neginf=0.0)
    f_yy = np.nan_to_num(f_yy, nan=0.0, posinf=0.0, neginf=0.0)

    return f_xx, f_yy


def _compute_single_family_stiffness(
    mode_family: Sequence[Sequence[ModeFunction]],
    coord_1: np.ndarray,
    coord_2: np.ndarray,
    L_coord_1: float,
    L_coord_2: float,
    prefactor: float,
    m_max: int,
    n_max: int,
    family_name: str,
) -> np.ndarray:
    """
    Compute stiffness coefficients for one 2D mode-shape family.

    The first coordinate corresponds to columns, and the second coordinate
    corresponds to rows, following ``meshgrid(coord_1, coord_2, indexing='xy')``.
    """
    C1, C2 = np.meshgrid(coord_1, coord_2, indexing="xy")
    coeffs = np.zeros((m_max, n_max), dtype=np.float64)

    for m in range(m_max):
        for n in range(n_max):
            try:
                mode_func = mode_family[m][n]
            except Exception as exc:
                raise ValueError(
                    f"{family_name}[{m}][{n}] is missing or not indexable."
                ) from exc

            if not callable(mode_func):
                raise TypeError(f"{family_name}[{m}][{n}] must be callable.")

            values = _validate_mode_grid(
                f"{family_name}[{m}][{n}]", mode_func(C1, C2), C1, C2
            )

            f_11, f_22 = compute_second_derivatives(
                values,
                C1,
                C2,
                L_coord_1,
                L_coord_2,
            )

            # Curvature-like nonlinear stiffness contribution used by the
            # original flexible-plate model: integral((d2f/dq1^2 + d2f/dq2^2)^2 dA)
            integrand = (f_11 + f_22) ** 2
            inner = _trapezoid(integrand, coord_1, axis=1)
            coeffs[m, n] = prefactor * float(_trapezoid(inner, coord_2, axis=0))

    return coeffs


def compute_nonlinear_stiffness(
    E: float,
    nu: float,
    h: float,
    L1: float,
    L2: float,
    W_mn: Sequence[Sequence[ModeFunction]],
    V_mn: Optional[Sequence[Sequence[ModeFunction]]],
    m_max: int,
    n_max: int,
    grid_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nonlinear modal stiffness coefficients for the flexible plate model.

    Parameters
    ----------
    E:
        Young's modulus [Pa].
    nu:
        Poisson's ratio [-]. Must satisfy -1 < nu < 0.5 for ordinary isotropic materials.
    h:
        Plate thickness [m].
    L1, L2:
        Plate dimensions [m].
    W_mn:
        Main transverse mode-shape functions W_mn(x, y). For face milling, the
        axial force Fz is normally projected onto these modes if W describes the
        out-of-plane displacement of the machined x-y surface.
    V_mn:
        Optional side-plane mode-shape functions V_mn(z, y). Pass None if the
        face-milling model does not use the z-y side-plane modes.
    m_max, n_max:
        Number of modes in the two modal indices.
    grid_points:
        Number of points per coordinate direction for numerical differentiation
        and integration. Use >= 50 for routine runs; increase for verification.

    Returns
    -------
    lambda_mn, lambda_prime_mn:
        Arrays with shape (m_max, n_max). If V_mn is None, lambda_prime_mn is
        returned as zeros for backward compatibility.

    Notes
    -----
    This module does not contain peripheral- or face-milling force equations.
    It only computes structural modal nonlinear stiffness coefficients. The
    force-model replacement should be performed in the dynamics/force module.
    """
    E = _validate_positive_scalar("E", E)
    h = _validate_positive_scalar("h", h)
    L1 = _validate_positive_scalar("L1", L1)
    L2 = _validate_positive_scalar("L2", L2)

    nu = float(nu)
    if not np.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise ValueError(f"nu must be finite and satisfy -1 < nu < 0.5, got {nu!r}.")

    m_max = int(m_max)
    n_max = int(n_max)
    grid_points = int(grid_points)

    if m_max <= 0 or n_max <= 0:
        raise ValueError(f"m_max and n_max must be positive; got {m_max}, {n_max}.")
    if grid_points < 3:
        raise ValueError(f"grid_points must be at least 3, got {grid_points}.")

    prefactor = (E * h) / (2.0 * (1.0 - nu**2))

    # Main x-y surface modes W(x, y)
    x = np.linspace(0.0, L1, grid_points, dtype=np.float64)
    y = np.linspace(0.0, L2, grid_points, dtype=np.float64)

    lambda_mn = _compute_single_family_stiffness(
        mode_family=W_mn,
        coord_1=x,
        coord_2=y,
        L_coord_1=L1,
        L_coord_2=L2,
        prefactor=prefactor,
        m_max=m_max,
        n_max=n_max,
        family_name="W_mn",
    )

    # Optional z-y side-plane modes V(z, y). For many face-milling setups only
    # W_mn is needed; keep this output for compatibility with older code.
    lambda_prime_mn = np.zeros((m_max, n_max), dtype=np.float64)
    if V_mn is not None:
        z = np.linspace(0.0, h, grid_points, dtype=np.float64)
        lambda_prime_mn = _compute_single_family_stiffness(
            mode_family=V_mn,
            coord_1=z,
            coord_2=y,
            L_coord_1=h,
            L_coord_2=L2,
            prefactor=prefactor,
            m_max=m_max,
            n_max=n_max,
            family_name="V_mn",
        )

    lambda_mn = np.nan_to_num(lambda_mn, nan=0.0, posinf=0.0, neginf=0.0)
    lambda_prime_mn = np.nan_to_num(lambda_prime_mn, nan=0.0, posinf=0.0, neginf=0.0)

    return lambda_mn, lambda_prime_mn
