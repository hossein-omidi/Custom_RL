"""Critical depth extraction and start-position sensitivity."""

from __future__ import annotations

import numpy as np


def critical_ac_from_probability(
    omega_grid: np.ndarray,
    ac_grid: np.ndarray,
    p_instability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate critical ac [mm] where p_instability crosses threshold.

    Scans ac from low to high; assumes instability probability is non-decreasing
    in ac for conventional lobe (warn if not monotonic).
    """
    n_omega = len(omega_grid)
    critical = np.full(n_omega, np.nan, dtype=np.float64)
    uncertainty = np.full(n_omega, np.nan, dtype=np.float64)

    for i in range(n_omega):
        row = p_instability[i, :]
        cross = np.where(row >= threshold)[0]
        if not cross.size:
            if np.all(row < threshold):
                critical[i] = float(ac_grid[-1])
                uncertainty[i] = float(ac_grid[-1] - ac_grid[0]) * 0.0
            continue
        j = int(cross[0])
        if j == 0:
            critical[i] = np.nan
            uncertainty[i] = float(ac_grid[0])
        else:
            p0, p1 = float(row[j - 1]), float(row[j])
            denom = max(p1 - p0, 1e-12)
            t = (threshold - p0) / denom
            critical[i] = float(ac_grid[j - 1] + t * (ac_grid[j] - ac_grid[j - 1]))
            uncertainty[i] = float(ac_grid[j] - ac_grid[j - 1]) * 0.5
    return critical, uncertainty


def critical_ac_from_class_grid(
    ac_grid: np.ndarray,
    class_row: np.ndarray,
) -> float:
    """
    Last stable/bounded ac before first chatter-like label (low-to-high ac scan).

    Returns nan if chatter at the shallowest ac or row is all failed.
    """
    ac_grid = np.asarray(ac_grid, dtype=np.float64).reshape(-1)
    last_ok = np.nan
    for ac, klass in zip(ac_grid, class_row):
        k = str(klass)
        if k in ("stable", "bounded"):
            last_ok = float(ac)
        elif k == "chatter-like":
            return last_ok if np.isfinite(last_ok) else np.nan
        elif k == "failed":
            return np.nan
    return last_ok


def enforce_monotone_increasing(p_row: np.ndarray) -> np.ndarray:
    """Non-decreasing envelope along ac (conventional lobe assumption)."""
    return np.maximum.accumulate(np.asarray(p_row, dtype=np.float64).reshape(-1))


def critical_ac_from_rollout_labels(
    ac_grid: np.ndarray,
    unstable_flags: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Critical ac for one omega from binary instability flags along ac."""
    ac_grid = np.asarray(ac_grid, dtype=np.float64).reshape(-1)
    flags = np.asarray(unstable_flags, dtype=np.float64).reshape(-1)
    if flags.size != ac_grid.size:
        raise ValueError("unstable_flags must match ac_grid length")
    p_mono = enforce_monotone_increasing(flags)
    crit, _ = critical_ac_from_probability(
        np.array([0.0]), ac_grid, p_mono.reshape(1, -1), threshold,
    )
    return float(crit[0])


def critical_ac_curve_with_mc_std(
    omega_grid: np.ndarray,
    ac_grid: np.ndarray,
    instability_labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Conventional lobe from Monte Carlo instability labels.

    instability_labels : (n_omega, n_ac, n_rollouts) bool — True if chatter-like.
    """
    n_omega, _, n_rollouts = instability_labels.shape
    p_inst = np.mean(instability_labels, axis=2)
    p_mono = np.zeros_like(p_inst)
    for i in range(n_omega):
        p_mono[i, :] = enforce_monotone_increasing(p_inst[i, :])

    critical_mean, critical_unc = critical_ac_from_probability(
        omega_grid, ac_grid, p_mono, threshold,
    )

    per_rollout = np.full((n_omega, n_rollouts), np.nan, dtype=np.float64)
    for i in range(n_omega):
        for r in range(n_rollouts):
            per_rollout[i, r] = critical_ac_from_rollout_labels(
                ac_grid, instability_labels[i, :, r], threshold=threshold,
            )
    critical_std = np.nanstd(per_rollout, axis=1)
    return critical_mean, critical_unc, critical_std, p_mono


def critical_ac_curve_from_classes(
    omega_grid: np.ndarray,
    ac_grid: np.ndarray,
    class_grid: np.ndarray,
) -> np.ndarray:
    """Per-omega critical ac from majority class labels on the ac scan."""
    n_omega = len(omega_grid)
    out = np.full(n_omega, np.nan, dtype=np.float64)
    for i in range(n_omega):
        out[i] = critical_ac_from_class_grid(ac_grid, class_grid[i, :])
    return out


def monotonicity_violations(p_instability: np.ndarray) -> int:
    """Count ac-scan violations of non-decreasing instability probability."""
    violations = 0
    for row in p_instability:
        violations += int(np.sum(np.diff(row) < -0.15))
    return violations


def start_position_sensitivity(
    records: list[dict],
    omega_grid: np.ndarray,
    ac_grid: np.ndarray,
) -> dict:
    """
    Assess whether path_y_start shifts the instability label at each grid point.

    Returns summary statistics; recommends 3D lobe only if effect is strong.
    """
    if not records:
        return {"recommend_3d_lobe": False, "mean_start_effect_mm": 0.0}

    effects: list[float] = []
    for i, om in enumerate(omega_grid):
        for j, ac in enumerate(ac_grid):
            subset = [
                r for r in records
                if abs(r.get("omega_rad_s", r.get("omega", 0.0)) - om) < 1e-9
                and abs(r.get("ac_mm", 0.0) - ac) < 1e-9
                and r.get("path_y_start_m") is not None
            ]
            if len(subset) < 2:
                continue
            ys = np.array([r["path_y_start_m"] for r in subset], dtype=np.float64)
            unstable = np.array([r["class"] == "chatter-like" for r in subset], dtype=float)
            if np.std(ys) < 1e-9:
                continue
            corr = float(np.corrcoef(ys, unstable)[0, 1]) if len(subset) > 2 else 0.0
            effects.append(abs(corr))

    mean_effect = float(np.mean(effects)) if effects else 0.0
    return {
        "recommend_3d_lobe": mean_effect > 0.5,
        "mean_start_correlation": mean_effect,
        "n_points_compared": len(effects),
    }
