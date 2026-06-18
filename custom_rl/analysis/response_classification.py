"""Response classification for regenerative milling rollouts (Test2 philosophy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ResponseClass = Literal["stable", "bounded", "chatter-like", "failed"]


@dataclass
class RolloutMetrics:
    """Physical metrics collected from one omega/ac rollout."""

    omega_rad_s: float
    ac_mm: float
    finite: bool
    clip_pct: float
    max_w_m: float
    max_delta_f_m: float
    max_delta_n_m: float
    max_F_normal_N: float
    max_F_feed_N: float
    rms_w_m: float
    rms_w_post_m: float
    terminated: bool
    term_reason: str
    path_y_start_m: float | None = None
    sensor_w_series: np.ndarray | None = None

    @property
    def max_w_mm(self) -> float:
        return self.max_w_m * 1e3

    @property
    def delta_f_mm(self) -> float:
        return self.max_delta_f_m * 1e3


def sensor_rms_envelope(sensor_w: np.ndarray, transient_fraction: float = 0.25) -> tuple[float, float]:
    """RMS of |w| over full series and post-transient tail."""
    sw = np.asarray(sensor_w, dtype=np.float64).reshape(-1)
    if sw.size == 0:
        return 0.0, 0.0
    mag = np.abs(sw)
    rms_all = float(np.sqrt(np.mean(mag**2)))
    start = int(np.floor(transient_fraction * mag.size))
    tail = mag[start:] if start < mag.size else mag[-1:]
    rms_post = float(np.sqrt(np.mean(tail**2))) if tail.size else rms_all
    return rms_all, rms_post


def _sustained_growth(sensor_w: np.ndarray, transient_fraction: float, growth_factor: float) -> bool:
    """Detect post-transient RMS or envelope growth (regenerative chatter signature)."""
    sw = np.abs(np.asarray(sensor_w, dtype=np.float64).reshape(-1))
    if sw.size < 16:
        return False
    start = max(int(np.floor(transient_fraction * sw.size)), 1)
    post = sw[start:]
    if post.size < 12:
        return False

    mid = post.size // 2
    first_half = post[:mid]
    second_half = post[mid:]
    if first_half.size < 4 or second_half.size < 4:
        return False

    rms_first = float(np.sqrt(np.mean(first_half**2)) + 1e-15)
    rms_second = float(np.sqrt(np.mean(second_half**2)))
    if rms_second > growth_factor * rms_first and rms_second > 5e-5:
        return True

    q = max(second_half.size // 4, 2)
    early = float(np.max(second_half[:q]))
    late = float(np.max(second_half[-q:]))
    return late > growth_factor * max(early, 1e-12) and late > 1e-4


def classify_milling_response(
    metrics: RolloutMetrics,
    *,
    transient_fraction: float = 0.25,
    displacement_limit_m: float = 1e-3,
    growth_factor: float = 3.0,
    delta_f_fail_m: float = 1.0,
    clip_fail_pct: float = 5.0,
) -> ResponseClass:
    """
    Label rollout: stable, bounded, chatter-like, or failed.

    Failed: non-finite state, force-clip dominated, hidden modal blow-up (|Delta_f| huge).
    Chatter-like: sustained post-transient growth, large vibration, or failure termination
    without numerical pathology.
    """
    if not metrics.finite:
        return "failed"
    if metrics.clip_pct > clip_fail_pct:
        return "failed"
    if metrics.max_delta_f_m > delta_f_fail_m:
        return "failed"

    if metrics.terminated and metrics.term_reason.startswith("excessive"):
        if metrics.clip_pct > 1.0:
            return "failed"
        if metrics.max_w_m > displacement_limit_m * 0.1:
            return "chatter-like"
        return "bounded"

    if metrics.ac_mm <= 0.0:
        return "stable" if metrics.max_w_m < 1e-8 else "bounded"

    if metrics.sensor_w_series is not None and _sustained_growth(
        metrics.sensor_w_series, transient_fraction, growth_factor
    ):
        return "chatter-like"

    if metrics.max_w_m > displacement_limit_m * 0.8:
        return "chatter-like"
    if metrics.max_w_m > 2e-4 or metrics.delta_f_mm > 0.5:
        return "chatter-like"
    if metrics.max_w_m > 5e-5:
        return "bounded"
    return "stable"


def is_instability_label(klass: ResponseClass) -> bool:
    """True for regenerative chatter-like instability (excludes failed)."""
    return klass == "chatter-like"


def metrics_from_rollout_dict(d: dict) -> RolloutMetrics:
    sw = d.get("sensor_w_series")
    rms_all, rms_post = sensor_rms_envelope(
        sw if sw is not None else np.array([d.get("max_w_m", 0.0)]),
        float(d.get("transient_fraction", 0.25)),
    )
    return RolloutMetrics(
        omega_rad_s=float(d["omega"]),
        ac_mm=float(d["ac_mm"]),
        finite=bool(d.get("finite", True)),
        clip_pct=float(d.get("clip_pct", 0.0)),
        max_w_m=float(d.get("max_w_m", 0.0)),
        max_delta_f_m=float(d.get("max_delta_f_m", 0.0)),
        max_delta_n_m=float(d.get("max_delta_n_m", 0.0)),
        max_F_normal_N=float(d.get("max_F_normal_N", 0.0)),
        max_F_feed_N=float(d.get("max_F_feed_N", 0.0)),
        rms_w_m=rms_all,
        rms_w_post_m=rms_post,
        terminated=bool(d.get("terminated", False)),
        term_reason=str(d.get("term_reason", "")),
        path_y_start_m=d.get("path_y_start_m"),
        sensor_w_series=np.asarray(sw) if sw is not None else None,
    )
