"""Shared modal index ordering utilities for the plate plant.

Convention (m-major / n-minor):
    k = m * n_max + n

This ordering is used consistently for:
    - modal state components eta[k], eta_dot[k]
    - flattened omega_vec[k], lambda_vec[k]
    - force projection b_vec[k]
    - sensor matrix columns S[:, k]
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def mode_index(m: int, n: int, n_max: int) -> int:
    """Return flat modal index k for (m, n) under m-major / n-minor ordering."""
    return m * n_max + n


def iter_mode_indices(m_max: int, n_max: int) -> Iterator[tuple[int, int, int]]:
    """Yield (m, n, k) triples in the canonical modal order."""
    for m in range(m_max):
        for n in range(n_max):
            yield m, n, mode_index(m, n, n_max)


def flatten_mode_matrix(mat: np.ndarray, m_max: int, n_max: int) -> np.ndarray:
    """
    Flatten an (m_max, n_max) modal matrix to length K using m-major / n-minor order.

    Equivalent to mat.reshape(m_max * n_max) for C-contiguous mat[m, n].
    """
    K = m_max * n_max
    vec = np.zeros(K, dtype=np.float64)
    for m, n, k in iter_mode_indices(m_max, n_max):
        vec[k] = float(mat[m, n])
    return vec


def build_sensor_displacement_matrix(
    W_mn,
    sensor_coords: np.ndarray,
    m_max: int,
    n_max: int,
) -> np.ndarray:
    """
  Build S_disp with shape (n_sensors, K) such that:
        w_s = S_disp @ eta
        w_dot_s = S_disp @ eta_dot

    Entry definition:
        S[i, k] = W_k(x_s_i, y_s_i)
    """
    sensor_coords = np.asarray(sensor_coords, dtype=np.float64)
    n_sensors = int(sensor_coords.shape[0])
    K = m_max * n_max

    S = np.zeros((n_sensors, K), dtype=np.float64)
    for m, n, k in iter_mode_indices(m_max, n_max):
        Wk = W_mn[m][n]
        for i in range(n_sensors):
            xs = float(sensor_coords[i, 0])
            ys = float(sensor_coords[i, 1])
            S[i, k] = float(np.asarray(Wk(xs, ys), dtype=np.float64).item())
    return S


def mode_index_map(m_max: int, n_max: int) -> list[tuple[int, int, int]]:
    """Return [(k, m, n), ...] for diagnostics and tests."""
    return [(k, m, n) for m, n, k in iter_mode_indices(m_max, n_max)]
