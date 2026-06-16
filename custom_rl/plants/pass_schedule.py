"""Straight-line milling pass schedule for full-surface plate coverage.

Each RL episode corresponds to one vertical pass:
    tool starts at the upper plate edge (y = L2),
    moves straight down to the lower edge (y = 0),
    at a fixed x position selected from a grid of parallel lines.
"""

from __future__ import annotations

import numpy as np


def pass_line_x_positions(
    n_lines: int,
    L1: float,
    margin: float = 0.0,
) -> np.ndarray:
    """
    Return x-coordinates for n_lines parallel passes across the plate width.

    Lines are evenly spaced along x in [margin, L1 - margin].
    """
    if n_lines < 1:
        raise ValueError("n_lines must be >= 1.")
    if margin < 0.0 or 2.0 * margin >= L1:
        raise ValueError("margin must satisfy 0 <= 2*margin < L1.")

    if n_lines == 1:
        return np.array([0.5 * L1], dtype=np.float64)

    return np.linspace(margin, L1 - margin, n_lines, dtype=np.float64)


def build_straight_pass_trajectory(
    x_line: float,
    L2: float,
    feed_speed: float,
    dt: float,
    *,
    y_start: float | None = None,
    y_end: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build a straight top-to-bottom pass trajectory.

    Args:
        x_line: fixed tool x position for this pass (m)
        L2: plate span in y (m); upper edge defaults to y = L2
        feed_speed: tool feed speed along the pass (m/s), must be > 0
        dt: trajectory sample period (s)
        y_start: pass start y (default L2, upper edge)
        y_end: pass end y (default 0, lower edge)

    Returns:
        t_original, x_traj, y_traj, pass_duration
    """
    if feed_speed <= 0.0:
        raise ValueError("feed_speed must be > 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    y_start_val = float(L2 if y_start is None else y_start)
    y_end_val = float(y_end)
    travel = abs(y_start_val - y_end_val)

    if travel <= 0.0:
        t_original = np.array([0.0], dtype=np.float64)
        x_traj = np.array([x_line], dtype=np.float64)
        y_traj = np.array([y_start_val], dtype=np.float64)
        return t_original, x_traj, y_traj, 0.0

    pass_duration = travel / feed_speed
    t_original = np.arange(0.0, pass_duration + dt, dt, dtype=np.float64)
    t_original = t_original[t_original <= pass_duration + 1e-12]
    if t_original.size == 0:
        t_original = np.array([0.0], dtype=np.float64)

    # Linear interpolation from upper to lower edge.
    alpha = t_original / pass_duration
    x_traj = np.full(t_original.shape, float(x_line), dtype=np.float64)
    y_traj = y_start_val + (y_end_val - y_start_val) * alpha

    return t_original, x_traj, y_traj, float(pass_duration)
