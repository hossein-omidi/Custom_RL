"""Tests for stability-lobe classification and critical-ac extraction."""

from __future__ import annotations

import numpy as np

from custom_rl.analysis.response_classification import (
    RolloutMetrics,
    classify_milling_response,
)
from custom_rl.analysis.stability_lobe import (
    critical_ac_curve_with_mc_std,
    critical_ac_from_class_grid,
    critical_ac_from_probability,
    enforce_monotone_increasing,
)


def test_classify_failed_on_clip() -> None:
    m = RolloutMetrics(
        omega_rad_s=800.0, ac_mm=2.0, finite=True, clip_pct=10.0,
        max_w_m=1e-5, max_delta_f_m=0.0, max_delta_n_m=0.0,
        max_F_normal_N=100.0, max_F_feed_N=50.0,
        rms_w_m=1e-6, rms_w_post_m=1e-6,
        terminated=False, term_reason="",
    )
    assert classify_milling_response(m) == "failed"


def test_classify_chatter_growth() -> None:
    t = np.linspace(0, 1, 100)
    sw = 1e-7 + 2e-4 * t**2
    m = RolloutMetrics(
        omega_rad_s=800.0, ac_mm=2.0, finite=True, clip_pct=0.0,
        max_w_m=float(np.max(sw)), max_delta_f_m=1e-5, max_delta_n_m=1e-6,
        max_F_normal_N=100.0, max_F_feed_N=50.0,
        rms_w_m=float(np.sqrt(np.mean(sw**2))),
        rms_w_post_m=float(np.sqrt(np.mean(sw[50:]**2))),
        terminated=False, term_reason="",
        sensor_w_series=sw,
    )
    assert classify_milling_response(m, growth_factor=3.0) == "chatter-like"


def test_critical_ac_from_class_grid() -> None:
    ag = np.array([0.5, 1.0, 2.0, 4.0, 6.0])
    row = np.array(["stable", "bounded", "bounded", "chatter-like", "chatter-like"])
    assert critical_ac_from_class_grid(ag, row) == 2.0


def test_classify_bounded_after_startup_transient() -> None:
    """Startup quiet period must not be mistaken for post-transient growth."""
    sw = np.zeros(200, dtype=np.float64)
    sw[40:] = 8e-5 * (1.0 + 0.05 * np.sin(np.linspace(0, 40 * np.pi, 160)))
    m = RolloutMetrics(
        omega_rad_s=800.0, ac_mm=2.0, finite=True, clip_pct=0.0,
        max_w_m=float(np.max(sw)), max_delta_f_m=1e-6, max_delta_n_m=1e-6,
        max_F_normal_N=10.0, max_F_feed_N=5.0,
        rms_w_m=float(np.sqrt(np.mean(sw**2))),
        rms_w_post_m=float(np.sqrt(np.mean(sw[80:]**2))),
        terminated=False, term_reason="",
        sensor_w_series=sw,
    )
    assert classify_milling_response(m, growth_factor=3.0) in ("stable", "bounded")


def test_enforce_monotone_increasing() -> None:
    row = np.array([0.1, 0.8, 0.3, 0.9])
    mono = enforce_monotone_increasing(row)
    assert np.allclose(mono, [0.1, 0.8, 0.8, 0.9])


def test_critical_ac_mc_std() -> None:
    ag = np.array([1.0, 2.0, 3.0, 4.0])
    og = np.array([500.0, 1000.0])
    labels = np.zeros((2, 4, 4), dtype=bool)
    labels[0, 2:, :] = True
    labels[1, 3:, :] = True
    mean, unc, std, _ = critical_ac_curve_with_mc_std(og, ag, labels, threshold=0.5)
    assert 2.0 < mean[0] < 3.0
    assert mean[1] > 2.5


def test_critical_ac_nan_when_unstable_at_shallowest_ac() -> None:
    og = np.array([800.0])
    ag = np.array([1.0, 2.0, 3.0])
    p = np.array([[1.0, 0.5, 0.0]])
    crit, unc = critical_ac_from_probability(og, ag, p, 0.5)
    assert np.isnan(crit[0])


def test_critical_ac_interpolation() -> None:
    og = np.array([400.0, 800.0])
    ag = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([
        [0.0, 0.2, 0.8, 1.0],
        [0.0, 0.0, 0.3, 0.9],
    ])
    crit, unc = critical_ac_from_probability(og, ag, p, 0.5)
    assert 2.0 < crit[0] <= 2.5
    assert crit[1] > 3.0
