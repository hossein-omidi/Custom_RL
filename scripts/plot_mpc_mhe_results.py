#!/usr/bin/env python3
r"""Standalone plotting and reporting for phase-domain NMPC + MHE trajectories.

The script reads the JSON files produced by ``mpc_mhe_dompc.py``:

    trajectories_seed0.json
    trajectories_seed1.json
    ...

It does not import, load, or compare any RL policy or RL trajectory.

Plotting note: the overview uses twin y-axes for spindle speed and axial depth.
The axial-depth curve is dashed so it cannot be mistaken for a duplicate speed trace.

Outputs
-------
For every episode, the script creates a dedicated directory containing:

01. overview
02. vibration sensors and limits
03. vibration velocity sensors
04. controls versus time
05. controls and vibration versus pass progress
06. control rates
07. cutting forces
08. chip load
09. cutter path and pass progress
10. modal states
11. reward and cumulative reward
12. MPC solver diagnostics
13. MHE diagnostics
14. phase-synchronous execution diagnostics
15. vibration PSD
16. vibration spectrogram
17. operating-point map
18. displacement phase portrait
19. episode summary card

It also writes:

- one multipage PDF report per episode;
- ``episode_summary.csv`` and ``episode_summary.json``;
- aggregate plots and an aggregate PDF when several episodes are present.

Required packages
-----------------
    pip install numpy matplotlib

Example
-------
    python scripts/plot_mpc_mhe_results.py \
        --input-dir mpc_phase_trajectories \
        --output-dir plots/mpc_only

Windows PowerShell
------------------
    python scripts/plot_mpc_mhe_results.py `
        --input-dir .\mpc_phase_trajectories `
        --output-dir .\plots\mpc_only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import EngFormatter, MaxNLocator, PercentFormatter


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

SEED_RE = re.compile(r"trajectories_seed(?P<seed>-?\d+)\.json$", re.IGNORECASE)
PASS_REASONS = {"pass_completed", "pass_completed_90percent"}
EPS = np.finfo(float).eps


@dataclass(frozen=True)
class EpisodeRef:
    seed: int
    episode: int
    source: Path
    data: dict[str, Any]

    @property
    def label(self) -> str:
        return f"seed {self.seed}, episode {self.episode}"

    @property
    def slug(self) -> str:
        return f"seed{self.seed}_ep{self.episode}"


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def _arr1(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)
    return array


def _arr2(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=float)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 0), dtype=float)
    if array.size == 0:
        return np.empty((0, 0), dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _nested_get(mapping: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _first_finite(values: Iterable[Any], default: float = float("nan")) -> float:
    for value in values:
        out = _float(value)
        if np.isfinite(out):
            return out
    return default


def _finite(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _nan_stat(func: Callable[[np.ndarray], float], array: np.ndarray,
              default: float = float("nan")) -> float:
    good = _finite(array)
    if good.size == 0:
        return default
    return float(func(good))


def _common_length(*arrays: np.ndarray) -> int:
    lengths = [len(a) for a in arrays if np.asarray(a).ndim >= 1 and len(a) > 0]
    return min(lengths) if lengths else 0


def _safe_time(ep: dict[str, Any], n_hint: int = 0) -> np.ndarray:
    t = _arr1(ep.get("times"))
    if t.size:
        t = t - t[0]
        # Repair occasional non-finite or non-monotonic samples conservatively.
        good = np.isfinite(t)
        if not np.all(good):
            idx = np.arange(t.size)
            if np.count_nonzero(good) >= 2:
                t = np.interp(idx, idx[good], t[good])
            else:
                t = idx.astype(float)
        if np.any(np.diff(t) <= 0):
            dt = _nan_stat(np.median, np.diff(t)[np.diff(t) > 0], 1.0)
            t = np.arange(t.size, dtype=float) * max(dt, EPS)
        return t

    n = n_hint or int(ep.get("length", 0) or 0)
    if n <= 0:
        return np.empty(0)
    dt = _first_finite([
        _nested_get(ep, ("metadata", "dt_env")),
        _nested_get(ep, ("metadata", "dt")),
        _nested_get(ep, ("reset_info", "dt")),
    ], 1.0)
    return np.arange(n, dtype=float) * max(dt, EPS)


def _process(ep: dict[str, Any], key: str) -> np.ndarray:
    process = ep.get("process", {})
    return _arr1(process.get(key) if isinstance(process, dict) else None)


def _w_limit(ep: dict[str, Any]) -> float:
    meta = ep.get("metadata", {})
    return _first_finite([
        _nested_get(meta, ("mpc", "w_limit")),
        _nested_get(meta, ("w_limit",)),
        _nested_get(meta, ("plant", "w_limit")),
        _nested_get(ep, ("reset_info", "w_limit")),
    ])


def _w_inner_limit(ep: dict[str, Any]) -> float:
    meta = ep.get("metadata", {})
    return _first_finite([
        _nested_get(meta, ("mpc", "w_inner_limit")),
        _nested_get(meta, ("w_inner_limit",)),
    ])


def _episode_status(ep: dict[str, Any]) -> tuple[bool, str]:
    reason = ep.get("termination_reason")
    reason_text = "None" if reason in (None, "") else str(reason)
    passed = _bool(ep.get("pass_completed")) or reason_text in PASS_REASONS
    if not passed and reason_text == "None" and int(ep.get("length", 0) or 0) > 0:
        reason_text = "max_env_steps or external stop"
    return passed, reason_text


def _seed_from_path(path: Path) -> int | None:
    match = SEED_RE.search(path.name)
    return int(match.group("seed")) if match else None


def discover_files(input_dir: Path, patterns: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(p.resolve() for p in input_dir.glob(pattern) if p.is_file())
    return sorted(files, key=lambda p: (_seed_from_path(p) is None,
                                        _seed_from_path(p) or 0, p.name))


def load_episodes(files: Sequence[Path], seeds: set[int] | None,
                  episodes: set[int] | None) -> list[EpisodeRef]:
    found: list[EpisodeRef] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(f"Skipping {path}: {exc}")
            continue

        rows = payload if isinstance(payload, list) else [payload]
        file_seed = _seed_from_path(path)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                warnings.warn(f"Skipping non-object episode {index} in {path}")
                continue
            seed = int(row.get("seed", row.get("scenario_seed", file_seed if file_seed is not None else 0)))
            episode = int(row.get("episode", index))
            if seeds is not None and seed not in seeds:
                continue
            if episodes is not None and episode not in episodes:
                continue
            found.append(EpisodeRef(seed=seed, episode=episode, source=path, data=row))
    return sorted(found, key=lambda e: (e.seed, e.episode, e.source.name))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_axes(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _finite_limits(values: np.ndarray, pad: float = 0.05) -> tuple[float, float] | None:
    good = _finite(values)
    if good.size == 0:
        return None
    lo = float(np.min(good)); hi = float(np.max(good))
    if math.isclose(lo, hi):
        delta = max(abs(lo) * pad, 1.0)
    else:
        delta = (hi - lo) * pad
    return lo - delta, hi + delta


def _peak_preserving_indices(y: np.ndarray, max_points: int) -> np.ndarray:
    """Return sorted indices that preserve local minima/maxima for long signals."""
    arr = np.asarray(y, dtype=float)
    n = arr.shape[0]
    if n <= max_points or max_points < 8:
        return np.arange(n)

    if arr.ndim == 1:
        score = np.abs(arr)
    else:
        with np.errstate(invalid="ignore"):
            score = np.nanmax(np.abs(arr), axis=1)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    buckets = max(2, max_points // 2)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    selected: list[int] = [0, n - 1]
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        segment = score[start:stop]
        selected.append(start + int(np.argmin(segment)))
        selected.append(start + int(np.argmax(segment)))
    idx = np.unique(np.asarray(selected, dtype=int))
    if idx.size > max_points:
        idx = idx[np.linspace(0, idx.size - 1, max_points, dtype=int)]
    return idx


def _plot_series(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *,
                 labels: Sequence[str] | None = None, max_points: int = 15000,
                 linewidth: float = 0.9, alpha: float = 0.9) -> None:
    y2 = _arr2(y)
    n = _common_length(x, y2)
    if n == 0:
        _empty_axis(ax, "No data")
        return
    x = x[:n]; y2 = y2[:n]
    idx = _peak_preserving_indices(y2, max_points)
    labels = list(labels or [])
    for col in range(y2.shape[1]):
        label = labels[col] if col < len(labels) else f"channel {col + 1}"
        ax.plot(x[idx], y2[idx, col], label=label, linewidth=linewidth, alpha=alpha)


def _empty_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_legend(ax: plt.Axes, max_items: int = 10, **kwargs: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if len(handles) > max_items:
        handles = handles[:max_items]
        labels = labels[:max_items]
    ax.legend(handles, labels, frameon=False, ncols=kwargs.pop("ncols", 1), **kwargs)


def _format_time_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Time [s]")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))


def _save_figure(fig: plt.Figure, stem: Path, formats: Sequence[str], dpi: int,
                 pdf: PdfPages | None) -> list[Path]:
    fig.tight_layout()
    outputs: list[Path] = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _control_time(ep: dict[str, Any], t: np.ndarray, n_ctrl: int) -> np.ndarray:
    diag = ep.get("mpc_diagnostics", {})
    n_hold = _arr1(diag.get("n_hold") if isinstance(diag, dict) else None)
    if t.size and n_hold.size:
        idx = np.clip(np.cumsum(n_hold.astype(int)) - 1, 0, t.size - 1)
        idx = idx[:n_ctrl]
        if idx.size == n_ctrl:
            return t[idx]
    if t.size and n_ctrl > 0:
        return np.linspace(t[0], t[-1], n_ctrl)
    return np.arange(n_ctrl, dtype=float)


def _progress_axis(ep: dict[str, Any], n: int) -> tuple[np.ndarray, str]:
    progress = _process(ep, "feed_progress")
    if progress.size >= n and np.any(np.isfinite(progress[:n])):
        return 100.0 * progress[:n], "Feed progress [%]"
    distance = _process(ep, "feed_distance_m")
    if distance.size >= n and np.any(np.isfinite(distance[:n])):
        return 1000.0 * distance[:n], "Feed distance [mm]"
    return np.arange(n, dtype=float), "Environment step"


def _sensor_labels(n: int, prefix: str = "sensor") -> list[str]:
    return [f"{prefix} {i + 1}" for i in range(n)]


def _modal_labels(n: int, prefix: str = "mode") -> list[str]:
    return [f"{prefix} {i + 1}" for i in range(n)]


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

def _uniform_sample_rate(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return 1.0 / float(np.median(dt))


def _welch_psd(signal: np.ndarray, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 16 or not np.isfinite(fs) or fs <= 0:
        return np.empty(0), np.empty(0)
    x = x - np.mean(x)
    if nperseg is None:
        target = min(8192, x.size)
        nperseg = 2 ** int(np.floor(np.log2(max(16, target))))
    nperseg = int(max(16, min(nperseg, x.size)))
    if nperseg % 2:
        nperseg -= 1
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    scale = fs * np.sum(window ** 2)
    starts = np.arange(0, x.size - nperseg + 1, step)
    if starts.size == 0:
        starts = np.array([0])
        nperseg = x.size
        window = np.hanning(nperseg)
        scale = fs * np.sum(window ** 2)
    acc = None
    for start in starts:
        segment = x[start:start + nperseg]
        if segment.size != nperseg:
            continue
        spec = np.fft.rfft(segment * window)
        pxx = (np.abs(spec) ** 2) / max(scale, EPS)
        if pxx.size > 2:
            pxx[1:-1] *= 2.0
        acc = pxx if acc is None else acc + pxx
    if acc is None:
        return np.empty(0), np.empty(0)
    acc /= max(len(starts), 1)
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return f, acc


def _dominant_frequency(signal: np.ndarray, fs: float) -> float:
    f, pxx = _welch_psd(signal, fs)
    if f.size <= 1:
        return float("nan")
    idx = 1 + int(np.argmax(pxx[1:]))
    return float(f[idx])


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------

def summarize_episode(ref: EpisodeRef) -> dict[str, Any]:
    ep = ref.data
    w = _arr2(ep.get("w_sensor"))
    wd = _arr2(ep.get("wdot_sensor"))
    actions = _arr2(ep.get("physical_actions"))
    t = _safe_time(ep, max(len(w), len(actions), int(ep.get("length", 0) or 0)))
    rewards = _arr1(ep.get("rewards"))
    diag = ep.get("mpc_diagnostics", {}) if isinstance(ep.get("mpc_diagnostics"), dict) else {}
    summary = ep.get("summary", {}) if isinstance(ep.get("summary"), dict) else {}
    passed, reason = _episode_status(ep)

    omega = actions[:, 0] if actions.shape[1] >= 1 else _process(ep, "omega_rad_s")
    ap = actions[:, 1] if actions.shape[1] >= 2 else _process(ep, "ap_mm")
    force_mag = _process(ep, "F_mag_N")
    progress = _process(ep, "feed_progress")
    solve = _arr1(diag.get("solve_time_s"))
    mhe_solve = _arr1(diag.get("mhe_solve_time_s"))
    feasible = _arr1(diag.get("feasible"))
    mhe_ok = _arr1(diag.get("mhe_solve_ok"))
    fs = _uniform_sample_rate(t)

    max_w = _nan_stat(np.max, np.abs(w))
    rms_w = float(np.sqrt(np.nanmean(w ** 2))) if w.size else float("nan")
    max_wd = _nan_stat(np.max, np.abs(wd))
    return_value = _first_finite([ep.get("return"), summary.get("return")],
                                 float(np.nansum(rewards)) if rewards.size else float("nan"))

    row: dict[str, Any] = {
        "seed": ref.seed,
        "episode": ref.episode,
        "source": str(ref.source),
        "controller": ep.get("controller", "mpc"),
        "estimator": ep.get("estimator", "mhe-phase"),
        "pass_completed": passed,
        "termination_reason": reason,
        "environment_steps": int(ep.get("length", len(rewards)) or len(rewards)),
        "duration_s": float(t[-1] - t[0]) if t.size >= 2 else float("nan"),
        "return": return_value,
        "final_feed_progress": _nan_stat(lambda x: x[-1], progress),
        "max_abs_w_m": _first_finite([summary.get("max_abs_w_m"), max_w]),
        "rms_w_m": _first_finite([summary.get("rms_w_m"), rms_w]),
        "max_abs_wdot_m_s": max_wd,
        "w_limit_m": _w_limit(ep),
        "omega_mean_rad_s": _nan_stat(np.mean, omega),
        "omega_min_rad_s": _nan_stat(np.min, omega),
        "omega_max_rad_s": _nan_stat(np.max, omega),
        "omega_mean_rpm": _nan_stat(np.mean, omega) * 60.0 / (2.0 * np.pi),
        "ap_mean_mm": _nan_stat(np.mean, ap),
        "ap_min_mm": _nan_stat(np.min, ap),
        "ap_max_mm": _nan_stat(np.max, ap),
        "force_max_N": _nan_stat(np.max, force_mag),
        "force_rms_N": float(np.sqrt(np.nanmean(force_mag ** 2))) if force_mag.size else float("nan"),
        "mpc_feasible_frac": _first_finite([summary.get("mpc_solve_feasible_frac"),
                                             _nan_stat(np.mean, feasible)]),
        "mpc_fallback_frac": _first_finite([summary.get("mpc_fallback_frac"),
                                             _nan_stat(np.mean, _arr1(diag.get("fallback_used")))]),
        "mpc_mean_solve_time_s": _first_finite([summary.get("mpc_mean_solve_time_s"),
                                                 _nan_stat(np.mean, solve)]),
        "mpc_p95_solve_time_s": _nan_stat(lambda x: np.percentile(x, 95), solve),
        "mhe_ok_frac": _first_finite([summary.get("mhe_solve_ok_frac"),
                                       _nan_stat(np.mean, mhe_ok)]),
        "mhe_mean_solve_time_s": _nan_stat(np.mean, mhe_solve),
        "mhe_p95_solve_time_s": _nan_stat(lambda x: np.percentile(x, 95), mhe_solve),
        "mhe_mean_output_err": _first_finite([summary.get("mhe_mean_output_err"),
                                               _nan_stat(np.mean, _arr1(diag.get("mhe_output_err")))]),
        "mhe_final_modal_err": _first_finite([_nested_get(ep, ("mhe", "final_modal_err")),
                                               _nan_stat(lambda x: x[-1], _arr1(diag.get("mhe_modal_err")))]),
        "dtheta_nominal_rad": _first_finite([summary.get("dtheta_nominal_rad"),
                                              _nested_get(ep, ("mhe", "dtheta_rad"))]),
        "dtheta_realized_mean_rad": _first_finite([summary.get("dtheta_realized_mean_rad"),
                                                    _nan_stat(np.mean, _arr1(diag.get("dtheta_realized")))]),
        "dtheta_max_rel_dev": _first_finite([summary.get("dtheta_max_rel_dev")]),
        "rate_clip_active_frac": _first_finite([summary.get("rate_clip_active_frac"),
                                                 _nan_stat(np.mean, _arr1(diag.get("rate_clip_active")))]),
        "dominant_vibration_frequency_hz": (_dominant_frequency(w[:, 0], fs)
                                              if w.shape[1] else float("nan")),
        "sample_rate_hz": fs,
    }
    if np.isfinite(row["max_abs_w_m"]) and np.isfinite(row["w_limit_m"]) and row["w_limit_m"] > 0:
        row["peak_w_over_limit"] = row["max_abs_w_m"] / row["w_limit_m"]
    else:
        row["peak_w_over_limit"] = float("nan")
    return row


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_summary(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    csv_path = output_dir / "episode_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    json_path = output_dir / "episode_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(list(rows)), handle, indent=2)


# ---------------------------------------------------------------------------
# Episode plot builders
# ---------------------------------------------------------------------------

def plot_overview(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    w = _arr2(ep.get("w_sensor"))
    actions = _arr2(ep.get("physical_actions"))
    rewards = _arr1(ep.get("rewards"))
    n = max(len(w), len(actions), len(rewards), int(ep.get("length", 0) or 0))
    t = _safe_time(ep, n)
    process = ep.get("process", {}) if isinstance(ep.get("process"), dict) else {}
    force = _arr1(process.get("F_mag_N"))
    progress = _arr1(process.get("feed_progress"))
    passed, reason = _episode_status(ep)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    fig.suptitle(f"MPC + MHE overview — {ref.label}\n"
                 f"pass={passed}, reason={reason}", fontsize=14)

    ax = axes[0, 0]
    if w.size:
        _plot_series(ax, t, 1e6 * w, labels=_sensor_labels(w.shape[1]), max_points=max_points)
        limit = _w_limit(ep)
        inner = _w_inner_limit(ep)
        if np.isfinite(limit):
            ax.axhline(1e6 * limit, linestyle="--", linewidth=1.0, label="hard limit")
            ax.axhline(-1e6 * limit, linestyle="--", linewidth=1.0)
        if np.isfinite(inner):
            ax.axhline(1e6 * inner, linestyle=":", linewidth=1.0, label="MPC inner limit")
            ax.axhline(-1e6 * inner, linestyle=":", linewidth=1.0)
        _add_legend(ax, max_items=8, ncols=2, fontsize=8)
    else:
        _empty_axis(ax, "No vibration data")
    ax.set_ylabel("Displacement [µm]"); ax.set_title("Measured vibration")
    _style_axes(ax)

    ax = axes[0, 1]
    if actions.shape[1] >= 1:
        nn = _common_length(t, actions)
        idx = _peak_preserving_indices(actions[:nn], max_points)
        ax.plot(t[:nn][idx], actions[:nn, 0][idx] * 60.0 / (2.0 * np.pi), label="spindle speed", linewidth=1.1)
        ax.set_ylabel("Spindle speed [rpm]")
        ax2 = ax.twinx()
        if actions.shape[1] >= 2:
            ax2.plot(t[:nn][idx], actions[:nn, 1][idx], label="axial depth", linestyle="--", linewidth=1.1)
            ax2.set_ylabel("Axial depth $a_p$ [mm]")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, frameon=False, loc="best")
        ax2.spines["top"].set_visible(False)
    else:
        _empty_axis(ax, "No physical action data")
    ax.set_title("Control commands"); _style_axes(ax)

    ax = axes[1, 0]
    if force.size:
        nn = _common_length(t, force)
        idx = _peak_preserving_indices(force[:nn], max_points)
        ax.plot(t[:nn][idx], force[:nn][idx], linewidth=0.55, alpha=0.75)
    else:
        _empty_axis(ax, "No force data")
    ax.set_ylabel("Force magnitude [N]"); ax.set_title("Cutting force"); _style_axes(ax)

    ax = axes[1, 1]
    if progress.size:
        nn = _common_length(t, progress)
        idx = _peak_preserving_indices(progress[:nn], max_points)
        ax.plot(t[:nn][idx], 100.0 * progress[:nn][idx])
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    else:
        _empty_axis(ax, "No progress data")
    ax.set_ylabel("Pass progress [%]"); ax.set_title("Machining progress"); _style_axes(ax)

    ax = axes[2, 0]
    if rewards.size:
        nn = _common_length(t, rewards)
        idx = _peak_preserving_indices(rewards[:nn], max_points)
        ax.plot(t[:nn][idx], rewards[:nn][idx], linewidth=0.8)
    else:
        _empty_axis(ax, "No reward data")
    ax.set_ylabel("Reward / step"); ax.set_title("Instantaneous environment reward")
    _format_time_axis(ax); _style_axes(ax)

    ax = axes[2, 1]
    if rewards.size:
        nn = _common_length(t, rewards)
        cumulative = np.nancumsum(np.nan_to_num(rewards[:nn]))
        idx = _peak_preserving_indices(cumulative, max_points)
        ax.plot(t[:nn][idx], cumulative[idx])
    else:
        _empty_axis(ax, "No reward data")
    ax.set_ylabel("Cumulative reward"); ax.set_title("Cumulative environment reward")
    _format_time_axis(ax); _style_axes(ax)
    return fig


def plot_vibration(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    w = _arr2(ep.get("w_sensor"))
    t = _safe_time(ep, len(w))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Measured vibration — {ref.label}")

    ax = axes[0]
    if w.size:
        _plot_series(ax, t, 1e6 * w, labels=_sensor_labels(w.shape[1]), max_points=max_points)
        limit = _w_limit(ep); inner = _w_inner_limit(ep)
        if np.isfinite(limit):
            ax.axhline(1e6 * limit, linestyle="--", linewidth=1.0, label="hard limit")
            ax.axhline(-1e6 * limit, linestyle="--", linewidth=1.0)
        if np.isfinite(inner):
            ax.axhline(1e6 * inner, linestyle=":", linewidth=1.0, label="MPC inner limit")
            ax.axhline(-1e6 * inner, linestyle=":", linewidth=1.0)
        _add_legend(ax, max_items=12, ncols=3, fontsize=8)
    else:
        _empty_axis(ax, "No vibration sensor data")
    ax.set_ylabel("Displacement [µm]"); ax.set_title("Individual sensors")
    _style_axes(ax)

    ax = axes[1]
    if w.size:
        envelope = np.nanmax(np.abs(w), axis=1)
        rms = np.sqrt(np.nanmean(w ** 2, axis=1))
        nn = _common_length(t, envelope)
        idx = _peak_preserving_indices(np.column_stack([envelope[:nn], rms[:nn]]), max_points)
        ax.plot(t[:nn][idx], 1e6 * envelope[:nn][idx], label="max absolute sensor")
        ax.plot(t[:nn][idx], 1e6 * rms[:nn][idx], label="sensor RMS")
        limit = _w_limit(ep); inner = _w_inner_limit(ep)
        if np.isfinite(limit):
            ax.axhline(1e6 * limit, linestyle="--", linewidth=1.0, label="hard limit")
        if np.isfinite(inner):
            ax.axhline(1e6 * inner, linestyle=":", linewidth=1.0, label="MPC inner limit")
        _add_legend(ax, ncols=4, fontsize=8)
    else:
        _empty_axis(ax, "No vibration sensor data")
    ax.set_ylabel("Amplitude [µm]"); ax.set_title("Vibration envelope")
    _format_time_axis(ax); _style_axes(ax)
    return fig


def plot_velocity(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    wd = _arr2(ep.get("wdot_sensor"))
    t = _safe_time(ep, len(wd))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Measured vibration velocity — {ref.label}")

    ax = axes[0]
    if wd.size:
        _plot_series(ax, t, 1000.0 * wd, labels=_sensor_labels(wd.shape[1]), max_points=max_points)
        _add_legend(ax, max_items=12, ncols=3, fontsize=8)
    else:
        _empty_axis(ax, "No velocity sensor data")
    ax.set_ylabel("Velocity [mm/s]"); ax.set_title("Individual sensors"); _style_axes(ax)

    ax = axes[1]
    if wd.size:
        envelope = np.nanmax(np.abs(wd), axis=1)
        rms = np.sqrt(np.nanmean(wd ** 2, axis=1))
        nn = _common_length(t, envelope)
        idx = _peak_preserving_indices(np.column_stack([envelope[:nn], rms[:nn]]), max_points)
        ax.plot(t[:nn][idx], 1000.0 * envelope[:nn][idx], label="max absolute sensor")
        ax.plot(t[:nn][idx], 1000.0 * rms[:nn][idx], label="sensor RMS")
        _add_legend(ax, ncols=2)
    else:
        _empty_axis(ax, "No velocity sensor data")
    ax.set_ylabel("Velocity [mm/s]"); ax.set_title("Velocity envelope")
    _format_time_axis(ax); _style_axes(ax)
    return fig


def plot_controls_time(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    u = _arr2(ep.get("physical_actions"))
    t = _safe_time(ep, len(u))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Applied MPC commands — {ref.label}")

    if u.shape[1] >= 1:
        nn = _common_length(t, u)
        idx = _peak_preserving_indices(u[:nn], max_points)
        axes[0].plot(t[:nn][idx], u[:nn, 0][idx])
        axes[0].set_ylabel("ω [rad/s]")
        axes[1].plot(t[:nn][idx], u[:nn, 0][idx] * 60.0 / (2.0 * np.pi))
        axes[1].set_ylabel("Speed [rpm]")
        if u.shape[1] >= 2:
            axes[2].plot(t[:nn][idx], u[:nn, 1][idx])
            axes[2].set_ylabel("aₚ [mm]")
        else:
            _empty_axis(axes[2], "No axial-depth command")
    else:
        for ax in axes:
            _empty_axis(ax, "No physical action data")
    axes[0].set_title("Spindle angular speed")
    axes[1].set_title("Spindle rotational speed")
    axes[2].set_title("Axial depth of cut")
    for ax in axes:
        _style_axes(ax)
    _format_time_axis(axes[-1])
    return fig


def plot_controls_progress(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    u = _arr2(ep.get("physical_actions"))
    w = _arr2(ep.get("w_sensor"))
    n = min([x for x in (len(u), len(w)) if x > 0], default=max(len(u), len(w)))
    x, xlabel = _progress_axis(ep, n)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Controller behavior along the machining pass — {ref.label}")

    if n and u.shape[1] >= 1:
        idx = _peak_preserving_indices(u[:n], max_points)
        axes[0].plot(x[idx], u[:n, 0][idx] * 60.0 / (2.0 * np.pi))
    else:
        _empty_axis(axes[0], "No spindle command")
    axes[0].set_ylabel("Speed [rpm]"); axes[0].set_title("Spindle speed")

    if n and u.shape[1] >= 2:
        idx = _peak_preserving_indices(u[:n], max_points)
        axes[1].plot(x[idx], u[:n, 1][idx])
    else:
        _empty_axis(axes[1], "No axial-depth command")
    axes[1].set_ylabel("aₚ [mm]"); axes[1].set_title("Axial depth")

    if n and w.size:
        envelope = np.nanmax(np.abs(w[:n]), axis=1)
        idx = _peak_preserving_indices(envelope, max_points)
        axes[2].plot(x[idx], 1e6 * envelope[idx])
        limit = _w_limit(ep); inner = _w_inner_limit(ep)
        if np.isfinite(limit):
            axes[2].axhline(1e6 * limit, linestyle="--", linewidth=1.0, label="hard limit")
        if np.isfinite(inner):
            axes[2].axhline(1e6 * inner, linestyle=":", linewidth=1.0, label="MPC inner limit")
        _add_legend(axes[2])
    else:
        _empty_axis(axes[2], "No vibration data")
    axes[2].set_ylabel("Peak |w| [µm]"); axes[2].set_title("Vibration envelope")
    axes[2].set_xlabel(xlabel)
    if "[%]" in xlabel:
        axes[2].xaxis.set_major_formatter(PercentFormatter(100))
    for ax in axes:
        _style_axes(ax)
    return fig


def plot_control_rates(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    u = _arr2(ep.get("physical_actions"))
    t = _safe_time(ep, len(u))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Realized control rates — {ref.label}")
    n = _common_length(t, u)
    if n >= 2 and u.shape[1] >= 1:
        dt = np.diff(t[:n])
        valid = dt > 0
        tm = t[:n][1:]
        domega = np.full(n - 1, np.nan)
        domega[valid] = np.diff(u[:n, 0])[valid] / dt[valid]
        idx = _peak_preserving_indices(domega, max_points)
        axes[0].plot(tm[idx], domega[idx])
        axes[0].axhline(0.0, linewidth=0.8)
    else:
        _empty_axis(axes[0], "Insufficient spindle command data")
    axes[0].set_ylabel("dω/dt [rad/s²]"); axes[0].set_title("Spindle acceleration")

    if n >= 2 and u.shape[1] >= 2:
        dt = np.diff(t[:n]); valid = dt > 0; tm = t[:n][1:]
        dap = np.full(n - 1, np.nan)
        dap[valid] = np.diff(u[:n, 1])[valid] / dt[valid]
        idx = _peak_preserving_indices(dap, max_points)
        axes[1].plot(tm[idx], dap[idx])
        axes[1].axhline(0.0, linewidth=0.8)
    else:
        _empty_axis(axes[1], "Insufficient axial-depth command data")
    axes[1].set_ylabel("daₚ/dt [mm/s]"); axes[1].set_title("Axial-depth feed rate")
    _format_time_axis(axes[1])
    for ax in axes:
        _style_axes(ax)
    return fig


def plot_forces(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    t = _safe_time(ep, int(ep.get("length", 0) or 0))
    fx = _process(ep, "Fx_N"); fy = _process(ep, "Fy_N")
    fz = _process(ep, "Fz_N"); fm = _process(ep, "F_mag_N")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Cutting forces — {ref.label}")

    ax = axes[0]
    components = [(fx, "Fx"), (fy, "Fy"), (fz, "Fz")]
    plotted = False
    for arr, label in components:
        n = _common_length(t, arr)
        if n:
            idx = _peak_preserving_indices(arr[:n], max_points)
            ax.plot(t[:n][idx], arr[:n][idx], label=label, linewidth=0.85)
            plotted = True
    if plotted:
        _add_legend(ax, ncols=3)
    else:
        _empty_axis(ax, "No force component data")
    ax.set_ylabel("Force [N]"); ax.set_title("Force components"); _style_axes(ax)

    ax = axes[1]
    n = _common_length(t, fm)
    if n:
        idx = _peak_preserving_indices(fm[:n], max_points)
        ax.plot(t[:n][idx], fm[:n][idx], label="|F|")
        rms = float(np.sqrt(np.nanmean(fm[:n] ** 2)))
        peak = _nan_stat(np.max, fm[:n])
        if np.isfinite(rms):
            ax.axhline(rms, linestyle=":", linewidth=1.0, label=f"RMS {rms:.3g} N")
        if np.isfinite(peak):
            ax.axhline(peak, linestyle="--", linewidth=1.0, label=f"peak {peak:.3g} N")
        _add_legend(ax)
    else:
        _empty_axis(ax, "No force magnitude data")
    ax.set_ylabel("Force magnitude [N]"); ax.set_title("Resultant force")
    _format_time_axis(ax); _style_axes(ax)
    return fig


def plot_chip(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    mean_chip = _process(ep, "mean_chip_mm")
    max_chip = _process(ep, "max_chip_mm")
    t = _safe_time(ep, max(len(mean_chip), len(max_chip)))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Chip-load history — {ref.label}")

    for ax, arr, title in [(axes[0], mean_chip, "Mean effective chip thickness"),
                           (axes[1], max_chip, "Maximum effective chip thickness")]:
        n = _common_length(t, arr)
        if n:
            idx = _peak_preserving_indices(arr[:n], max_points)
            ax.plot(t[:n][idx], arr[:n][idx])
        else:
            _empty_axis(ax, "No chip-thickness data")
        ax.set_ylabel("Chip thickness [mm]"); ax.set_title(title); _style_axes(ax)
    _format_time_axis(axes[-1])
    return fig


def plot_path_progress(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    x = _process(ep, "cutter_x"); y = _process(ep, "cutter_y")
    progress = _process(ep, "feed_progress"); distance = _process(ep, "feed_distance_m")
    n = max(len(x), len(y), len(progress), len(distance))
    t = _safe_time(ep, n)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Tool path and pass progression — {ref.label}")

    ax = axes[0, 0]
    nn = _common_length(x, y)
    if nn:
        idx = _peak_preserving_indices(np.column_stack([x[:nn], y[:nn]]), max_points)
        ax.plot(1000.0 * x[:nn][idx], 1000.0 * y[:nn][idx])
        ax.scatter([1000.0 * x[0]], [1000.0 * y[0]], marker="o", label="start")
        ax.scatter([1000.0 * x[nn - 1]], [1000.0 * y[nn - 1]], marker="x", label="end")
        _add_legend(ax)
    else:
        _empty_axis(ax, "No cutter coordinates")
    ax.set_xlabel("Cutter x [mm]"); ax.set_ylabel("Cutter y [mm]"); ax.set_title("Cutter path")
    ax.set_aspect("equal", adjustable="datalim"); _style_axes(ax)

    ax = axes[0, 1]
    nn = _common_length(t, x)
    if nn:
        idx = _peak_preserving_indices(x[:nn], max_points)
        ax.plot(t[:nn][idx], 1000.0 * x[:nn][idx])
    else:
        _empty_axis(ax, "No cutter x data")
    ax.set_ylabel("Cutter x [mm]"); ax.set_title("Cutter position"); _format_time_axis(ax); _style_axes(ax)

    ax = axes[1, 0]
    nn = _common_length(t, progress)
    if nn:
        idx = _peak_preserving_indices(progress[:nn], max_points)
        ax.plot(t[:nn][idx], 100.0 * progress[:nn][idx])
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    else:
        _empty_axis(ax, "No feed progress data")
    ax.set_ylabel("Progress [%]"); ax.set_title("Normalized pass progress")
    _format_time_axis(ax); _style_axes(ax)

    ax = axes[1, 1]
    nn = _common_length(t, distance)
    if nn:
        idx = _peak_preserving_indices(distance[:nn], max_points)
        ax.plot(t[:nn][idx], 1000.0 * distance[:nn][idx])
    else:
        _empty_axis(ax, "No feed distance data")
    ax.set_ylabel("Feed distance [mm]"); ax.set_title("Physical feed distance")
    _format_time_axis(ax); _style_axes(ax)
    return fig


def plot_modal_states(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    xm = _arr2(ep.get("x_modal"))
    t = _safe_time(ep, len(xm))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Plant modal states — {ref.label}")

    if xm.size and xm.shape[1] >= 2:
        eta = xm[:, 0::2]
        vel = xm[:, 1::2]
        _plot_series(axes[0], t, 1e6 * eta, labels=_modal_labels(eta.shape[1]), max_points=max_points)
        _plot_series(axes[1], t, 1000.0 * vel, labels=_modal_labels(vel.shape[1]), max_points=max_points)
        _add_legend(axes[0], max_items=12, ncols=3, fontsize=8)
        _add_legend(axes[1], max_items=12, ncols=3, fontsize=8)
    else:
        _empty_axis(axes[0], "No modal-state data")
        _empty_axis(axes[1], "No modal-state data")
    axes[0].set_ylabel("Modal displacement [µm]"); axes[0].set_title("Modal displacements")
    axes[1].set_ylabel("Modal velocity [mm/s]"); axes[1].set_title("Modal velocities")
    _format_time_axis(axes[-1])
    for ax in axes:
        _style_axes(ax)
    return fig


def plot_reward(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    r = _arr1(ep.get("rewards"))
    t = _safe_time(ep, len(r))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Environment reward diagnostics — {ref.label}")

    ax = axes[0, 0]
    n = _common_length(t, r)
    if n:
        idx = _peak_preserving_indices(r[:n], max_points)
        ax.plot(t[:n][idx], r[:n][idx], linewidth=0.8)
    else:
        _empty_axis(ax, "No reward data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Reward / step"); ax.set_title("Instantaneous reward"); _style_axes(ax)

    ax = axes[0, 1]
    if n:
        cum = np.nancumsum(np.nan_to_num(r[:n]))
        idx = _peak_preserving_indices(cum, max_points)
        ax.plot(t[:n][idx], cum[idx])
    else:
        _empty_axis(ax, "No reward data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Cumulative reward"); ax.set_title("Cumulative reward"); _style_axes(ax)

    ax = axes[1, 0]
    good = _finite(r)
    if good.size:
        ax.hist(good, bins=min(100, max(10, int(np.sqrt(good.size)))))
    else:
        _empty_axis(ax, "No finite reward data")
    ax.set_xlabel("Reward / step"); ax.set_ylabel("Count"); ax.set_title("Reward distribution"); _style_axes(ax)

    ax = axes[1, 1]
    if n:
        window = max(1, min(1000, n // 100 if n >= 100 else 1))
        if window > 1:
            kernel = np.ones(window) / window
            smooth = np.convolve(np.nan_to_num(r[:n]), kernel, mode="valid")
            ts = t[:n][window - 1:]
        else:
            smooth = r[:n]; ts = t[:n]
        idx = _peak_preserving_indices(smooth, max_points)
        ax.plot(ts[idx], smooth[idx])
        ax.set_title(f"Moving-average reward ({window} steps)")
    else:
        _empty_axis(ax, "No reward data")
        ax.set_title("Moving-average reward")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Mean reward / step"); _style_axes(ax)
    return fig


def plot_mpc_solver(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    diag = ep.get("mpc_diagnostics", {}) if isinstance(ep.get("mpc_diagnostics"), dict) else {}
    solve = _arr1(diag.get("solve_time_s")); feasible = _arr1(diag.get("feasible"))
    fallback = _arr1(diag.get("fallback_used")); clip = _arr1(diag.get("rate_clip_active"))
    t_env = _safe_time(ep, int(ep.get("length", 0) or 0))
    n_ctrl = max(len(solve), len(feasible), len(fallback), len(clip))
    tc = _control_time(ep, t_env, n_ctrl)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"MPC numerical diagnostics — {ref.label}")

    ax = axes[0, 0]
    n = _common_length(tc, solve)
    if n:
        idx = _peak_preserving_indices(solve[:n], max_points)
        ax.plot(tc[:n][idx], 1000.0 * solve[:n][idx])
        p95 = _nan_stat(lambda x: np.percentile(x, 95), solve[:n])
        if np.isfinite(p95):
            ax.axhline(1000.0 * p95, linestyle="--", linewidth=1.0, label=f"p95 {1000*p95:.1f} ms")
            _add_legend(ax)
    else:
        _empty_axis(ax, "No MPC solve-time data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Solve time [ms]"); ax.set_title("Solve time per control step"); _style_axes(ax)

    ax = axes[0, 1]
    good = 1000.0 * _finite(solve)
    if good.size:
        ax.hist(good, bins=min(80, max(10, int(np.sqrt(good.size)))))
    else:
        _empty_axis(ax, "No MPC solve-time data")
    ax.set_xlabel("Solve time [ms]"); ax.set_ylabel("Count"); ax.set_title("Solve-time distribution"); _style_axes(ax)

    ax = axes[1, 0]
    signals: list[tuple[np.ndarray, str]] = [(feasible, "feasible"), (1.0 - fallback, "no fallback")]
    plotted = False
    for arr, label in signals:
        n = _common_length(tc, arr)
        if n:
            ax.step(tc[:n], arr[:n], where="post", label=label, linewidth=1.0)
            plotted = True
    if plotted:
        ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"])
        _add_legend(ax)
    else:
        _empty_axis(ax, "No feasibility/fallback flags")
    ax.set_xlabel("Time [s]"); ax.set_title("Optimization status"); _style_axes(ax)

    ax = axes[1, 1]
    n = _common_length(tc, clip)
    if n:
        ax.step(tc[:n], clip[:n], where="post")
        ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 1]); ax.set_yticklabels(["inactive", "active"])
    else:
        _empty_axis(ax, "No rate-clip diagnostic")
    ax.set_xlabel("Time [s]"); ax.set_title("External rate clip activation"); _style_axes(ax)
    return fig


def plot_mhe(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    diag = ep.get("mpc_diagnostics", {}) if isinstance(ep.get("mpc_diagnostics"), dict) else {}
    solve = _arr1(diag.get("mhe_solve_time_s")); ok = _arr1(diag.get("mhe_solve_ok"))
    output_err = _arr1(diag.get("mhe_output_err")); modal_err = _arr1(diag.get("mhe_modal_err"))
    t_env = _safe_time(ep, int(ep.get("length", 0) or 0))
    n_ctrl = max(len(solve), len(ok), len(output_err), len(modal_err))
    tc = _control_time(ep, t_env, n_ctrl)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"MHE estimation diagnostics — {ref.label}")

    ax = axes[0, 0]
    n = _common_length(tc, solve)
    if n:
        idx = _peak_preserving_indices(solve[:n], max_points)
        ax.plot(tc[:n][idx], 1000.0 * solve[:n][idx])
        p95 = _nan_stat(lambda x: np.percentile(x, 95), solve[:n])
        if np.isfinite(p95):
            ax.axhline(1000.0 * p95, linestyle="--", linewidth=1.0, label=f"p95 {1000*p95:.1f} ms")
            _add_legend(ax)
    else:
        _empty_axis(ax, "No MHE solve-time data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Solve time [ms]"); ax.set_title("MHE solve time"); _style_axes(ax)

    ax = axes[0, 1]
    n = _common_length(tc, ok)
    if n:
        ax.step(tc[:n], ok[:n], where="post")
        ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 1]); ax.set_yticklabels(["not accepted", "accepted"])
    else:
        _empty_axis(ax, "No MHE acceptance data")
    ax.set_xlabel("Time [s]"); ax.set_title("MHE solution accepted after warm-up"); _style_axes(ax)

    ax = axes[1, 0]
    n = _common_length(tc, output_err)
    if n:
        idx = _peak_preserving_indices(output_err[:n], max_points)
        ax.semilogy(tc[:n][idx], np.maximum(output_err[:n][idx], 1e-16))
    else:
        _empty_axis(ax, "No output-error data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Relative output error"); ax.set_title("Sensor-output reconstruction error"); _style_axes(ax)

    ax = axes[1, 1]
    n = _common_length(tc, modal_err)
    if n:
        idx = _peak_preserving_indices(modal_err[:n], max_points)
        ax.semilogy(tc[:n][idx], np.maximum(modal_err[:n][idx], 1e-16))
    else:
        _empty_axis(ax, "No modal-error data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Relative modal error"); ax.set_title("Retained modal-state error"); _style_axes(ax)
    return fig


def plot_phase_sync(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    diag = ep.get("mpc_diagnostics", {}) if isinstance(ep.get("mpc_diagnostics"), dict) else {}
    dtheta = _arr1(diag.get("dtheta_realized")); n_hold = _arr1(diag.get("n_hold"))
    nominal = _first_finite([_nested_get(ep, ("summary", "dtheta_nominal_rad")),
                             _nested_get(ep, ("mhe", "dtheta_rad")),
                             _nested_get(ep, ("metadata", "mpc", "dtheta_rad"))])
    t_env = _safe_time(ep, int(ep.get("length", 0) or 0))
    n_ctrl = max(len(dtheta), len(n_hold))
    tc = _control_time(ep, t_env, n_ctrl)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Phase-synchronous execution — {ref.label}")

    ax = axes[0, 0]
    n = _common_length(tc, dtheta)
    if n:
        idx = _peak_preserving_indices(dtheta[:n], max_points)
        ax.plot(tc[:n][idx], dtheta[:n][idx])
        if np.isfinite(nominal):
            ax.axhline(nominal, linestyle="--", linewidth=1.0, label="nominal")
            _add_legend(ax)
    else:
        _empty_axis(ax, "No realized phase-step data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Δθ [rad]"); ax.set_title("Realized phase increment"); _style_axes(ax)

    ax = axes[0, 1]
    if n and np.isfinite(nominal) and nominal != 0:
        rel = 100.0 * (dtheta[:n] - nominal) / nominal
        idx = _peak_preserving_indices(rel, max_points)
        ax.plot(tc[:n][idx], rel[idx])
        ax.axhline(0.0, linewidth=0.8)
    else:
        _empty_axis(ax, "Nominal Δθ unavailable")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Relative error [%]"); ax.set_title("Phase-step realization error"); _style_axes(ax)

    ax = axes[1, 0]
    n2 = _common_length(tc, n_hold)
    if n2:
        idx = _peak_preserving_indices(n_hold[:n2], max_points)
        ax.step(tc[:n2][idx], n_hold[:n2][idx], where="post")
    else:
        _empty_axis(ax, "No hold-length data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Environment steps"); ax.set_title("Integer hold length per control interval"); _style_axes(ax)

    ax = axes[1, 1]
    if n:
        good = _finite(dtheta[:n])
        ax.hist(good, bins=min(60, max(10, int(np.sqrt(max(good.size, 1))))))
        if np.isfinite(nominal):
            ax.axvline(nominal, linestyle="--", linewidth=1.0, label="nominal")
            _add_legend(ax)
    else:
        _empty_axis(ax, "No realized phase-step data")
    ax.set_xlabel("Realized Δθ [rad]"); ax.set_ylabel("Count"); ax.set_title("Phase-step distribution"); _style_axes(ax)
    return fig


def plot_psd(ref: EpisodeRef, max_frequency: float | None) -> plt.Figure:
    ep = ref.data
    w = _arr2(ep.get("w_sensor")); t = _safe_time(ep, len(w)); fs = _uniform_sample_rate(t)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"Vibration frequency content — {ref.label}")

    ax = axes[0]
    if w.size and np.isfinite(fs):
        for col in range(w.shape[1]):
            f, pxx = _welch_psd(w[:, col], fs)
            if f.size:
                mask = f > 0
                if max_frequency is not None:
                    mask &= f <= max_frequency
                ax.semilogy(f[mask], np.maximum(pxx[mask], 1e-30), label=f"sensor {col + 1}")
        _add_legend(ax, max_items=12, ncols=3, fontsize=8)
    else:
        _empty_axis(ax, "Frequency analysis requires vibration and time data")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("PSD [m²/Hz]"); ax.set_title("Welch power spectral density")
    _style_axes(ax)

    ax = axes[1]
    if w.size and np.isfinite(fs):
        rows = []
        for col in range(w.shape[1]):
            f, pxx = _welch_psd(w[:, col], fs)
            if f.size > 1:
                idx = 1 + int(np.argmax(pxx[1:]))
                rows.append((col + 1, f[idx], math.sqrt(max(float(np.trapezoid(pxx, f)), 0.0))))
        if rows:
            sensors = np.arange(len(rows))
            peaks = np.asarray([r[1] for r in rows])
            ax.bar(sensors, peaks)
            ax.set_xticks(sensors, [f"S{r[0]}" for r in rows])
            for i, val in enumerate(peaks):
                ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        else:
            _empty_axis(ax, "No resolvable spectral peak")
    else:
        _empty_axis(ax, "Frequency analysis requires vibration and time data")
    ax.set_xlabel("Sensor"); ax.set_ylabel("Dominant frequency [Hz]"); ax.set_title("Dominant vibration frequency")
    _style_axes(ax)
    return fig


def plot_spectrogram(ref: EpisodeRef, max_frequency: float | None) -> plt.Figure:
    ep = ref.data
    w = _arr2(ep.get("w_sensor")); t = _safe_time(ep, len(w)); fs = _uniform_sample_rate(t)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"Vibration spectrogram — {ref.label}")
    if w.size and np.isfinite(fs) and len(w) >= 64:
        rms = np.sqrt(np.nanmean(w ** 2, axis=0))
        sensor = int(np.nanargmax(rms))
        signal = np.nan_to_num(w[:, sensor] - np.nanmean(w[:, sensor]))
        nfft = min(4096, 2 ** int(np.floor(np.log2(max(64, min(len(signal), 4096))))))
        noverlap = int(0.75 * nfft)
        _, freqs, bins, image = ax.specgram(signal, NFFT=nfft, Fs=fs, noverlap=noverlap,
                                             scale="dB", mode="psd")
        if max_frequency is not None:
            ax.set_ylim(0, max_frequency)
        fig.colorbar(image, ax=ax, label="PSD [dB/Hz]")
        ax.set_title(f"Sensor {sensor + 1} (largest RMS amplitude)")
    else:
        _empty_axis(ax, "Spectrogram requires at least 64 vibration samples and valid time data")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]"); _style_axes(ax)
    return fig


def plot_operating_map(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    u = _arr2(ep.get("physical_actions")); w = _arr2(ep.get("w_sensor"))
    n = _common_length(u, w)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Operating-point map — {ref.label}")
    if n and u.shape[1] >= 2:
        rpm = u[:n, 0] * 60.0 / (2.0 * np.pi)
        ap = u[:n, 1]
        amp = 1e6 * np.nanmax(np.abs(w[:n]), axis=1)
        idx = _peak_preserving_indices(np.column_stack([rpm, ap, amp]), max_points)
        scatter = axes[0].scatter(rpm[idx], ap[idx], c=amp[idx], s=12, alpha=0.7)
        fig.colorbar(scatter, ax=axes[0], label="Peak |w| [µm]")
        axes[0].set_xlabel("Spindle speed [rpm]"); axes[0].set_ylabel("Axial depth [mm]")
        axes[0].set_title("Visited control operating points")

        axes[1].scatter(rpm[idx], amp[idx], s=12, alpha=0.7, label="speed")
        ax2 = axes[1].twiny()
        ax2.scatter(ap[idx], amp[idx], s=12, alpha=0.35, label="axial depth")
        axes[1].set_xlabel("Spindle speed [rpm]")
        ax2.set_xlabel("Axial depth [mm]")
        axes[1].set_ylabel("Peak |w| [µm]")
        axes[1].set_title("Vibration versus commanded operating point")
        ax2.spines["top"].set_visible(True)
    else:
        _empty_axis(axes[0], "Operating map requires controls and vibration")
        _empty_axis(axes[1], "Operating map requires controls and vibration")
    for ax in axes:
        _style_axes(ax)
    return fig


def plot_phase_portrait(ref: EpisodeRef, max_points: int) -> plt.Figure:
    ep = ref.data
    w = _arr2(ep.get("w_sensor")); wd = _arr2(ep.get("wdot_sensor"))
    n = _common_length(w, wd)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Measured displacement–velocity phase portrait — {ref.label}")
    if n:
        channels = min(w.shape[1], wd.shape[1])
        idx = _peak_preserving_indices(np.column_stack([w[:n, :channels], wd[:n, :channels]]), max_points)
        for col in range(channels):
            axes[0].plot(1e6 * w[:n, col][idx], 1000.0 * wd[:n, col][idx],
                         linewidth=0.75, label=f"sensor {col + 1}")
        _add_legend(axes[0], max_items=12, ncols=2, fontsize=8)
        axes[0].set_xlabel("Displacement [µm]"); axes[0].set_ylabel("Velocity [mm/s]")
        axes[0].set_title("All sensor trajectories")

        sensor = int(np.nanargmax(np.sqrt(np.nanmean(w[:n, :channels] ** 2, axis=0))))
        axes[1].plot(1e6 * w[:n, sensor][idx], 1000.0 * wd[:n, sensor][idx], linewidth=0.8)
        axes[1].scatter([1e6 * w[0, sensor]], [1000.0 * wd[0, sensor]], marker="o", label="start")
        axes[1].scatter([1e6 * w[n - 1, sensor]], [1000.0 * wd[n - 1, sensor]], marker="x", label="end")
        _add_legend(axes[1])
        axes[1].set_xlabel("Displacement [µm]"); axes[1].set_ylabel("Velocity [mm/s]")
        axes[1].set_title(f"Sensor {sensor + 1} (largest displacement RMS)")
    else:
        _empty_axis(axes[0], "Phase portrait requires displacement and velocity")
        _empty_axis(axes[1], "Phase portrait requires displacement and velocity")
    for ax in axes:
        _style_axes(ax)
    return fig


def plot_summary_card(ref: EpisodeRef, summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    passed = summary["pass_completed"]
    status = "PASS COMPLETED" if passed else "PASS NOT COMPLETED"
    lines = [
        ("Episode", f"seed {summary['seed']}, episode {summary['episode']}"),
        ("Status", status),
        ("Termination", str(summary["termination_reason"])),
        ("Duration", _fmt(summary["duration_s"], " s")),
        ("Environment steps", f"{summary['environment_steps']:,}"),
        ("Final feed progress", _fmt(100.0 * summary["final_feed_progress"], " %")),
        ("Return", _fmt(summary["return"])),
        ("Peak |w|", _fmt(1e6 * summary["max_abs_w_m"], " µm")),
        ("RMS w", _fmt(1e6 * summary["rms_w_m"], " µm")),
        ("Peak / hard limit", _fmt(100.0 * summary["peak_w_over_limit"], " %")),
        ("Mean spindle speed", _fmt(summary["omega_mean_rpm"], " rpm")),
        ("Mean axial depth", _fmt(summary["ap_mean_mm"], " mm")),
        ("Peak force", _fmt(summary["force_max_N"], " N")),
        ("MPC feasible", _fmt(100.0 * summary["mpc_feasible_frac"], " %")),
        ("MPC solve mean / p95", f"{_fmt(1000*summary['mpc_mean_solve_time_s'], ' ms')} / "
                                  f"{_fmt(1000*summary['mpc_p95_solve_time_s'], ' ms')}"),
        ("MHE accepted", _fmt(100.0 * summary["mhe_ok_frac"], " %")),
        ("MHE solve mean / p95", f"{_fmt(1000*summary['mhe_mean_solve_time_s'], ' ms')} / "
                                  f"{_fmt(1000*summary['mhe_p95_solve_time_s'], ' ms')}"),
        ("Dominant vibration frequency", _fmt(summary["dominant_vibration_frequency_hz"], " Hz")),
    ]
    ax.text(0.02, 0.97, "MPC + MHE episode summary", fontsize=20, weight="bold", va="top")
    ax.text(0.02, 0.91, ref.label, fontsize=13, va="top")
    y = 0.84
    for key, value in lines:
        ax.text(0.04, y, key, fontsize=11, weight="bold", va="top")
        ax.text(0.42, y, value, fontsize=11, va="top")
        y -= 0.043
    ax.text(0.02, 0.03, f"Source: {ref.source}", fontsize=8, va="bottom")
    return fig


def _fmt(value: Any, suffix: str = "") -> str:
    val = _float(value)
    if not np.isfinite(val):
        return "n/a"
    if val == 0:
        return f"0{suffix}"
    if abs(val) >= 1000 or abs(val) < 1e-3:
        return f"{val:.3e}{suffix}"
    return f"{val:.4g}{suffix}"


PLOT_BUILDERS: list[tuple[str, Callable[..., plt.Figure]]] = [
    ("01_overview", plot_overview),
    ("02_vibration", plot_vibration),
    ("03_vibration_velocity", plot_velocity),
    ("04_controls_time", plot_controls_time),
    ("05_controls_vs_progress", plot_controls_progress),
    ("06_control_rates", plot_control_rates),
    ("07_forces", plot_forces),
    ("08_chip_load", plot_chip),
    ("09_tool_path_progress", plot_path_progress),
    ("10_modal_states", plot_modal_states),
    ("11_reward", plot_reward),
    ("12_mpc_solver", plot_mpc_solver),
    ("13_mhe_diagnostics", plot_mhe),
    ("14_phase_sync", plot_phase_sync),
    ("17_operating_map", plot_operating_map),
    ("18_phase_portrait", plot_phase_portrait),
]


# ---------------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------------

def plot_aggregate_metrics(rows: Sequence[dict[str, Any]]) -> plt.Figure:
    labels = [f"s{r['seed']}e{r['episode']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("MPC + MHE episode summary")

    metrics = [
        ("max_abs_w_m", 1e6, "Peak |w| [µm]", "Peak measured displacement"),
        ("final_feed_progress", 100.0, "Progress [%]", "Final pass progress"),
        ("mpc_feasible_frac", 100.0, "Feasible solves [%]", "MPC numerical feasibility"),
        ("mpc_mean_solve_time_s", 1000.0, "Mean solve time [ms]", "MPC computational cost"),
    ]
    for ax, (key, scale, ylabel, title) in zip(axes.flat, metrics):
        values = np.asarray([_float(r.get(key)) * scale for r in rows])
        ax.bar(x, values)
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title); _style_axes(ax)
        if key in {"final_feed_progress", "mpc_feasible_frac"}:
            ax.set_ylim(0, max(100, _nan_stat(np.max, values, 100) * 1.1))
    return fig


def plot_aggregate_reliability(rows: Sequence[dict[str, Any]]) -> plt.Figure:
    labels = [f"s{r['seed']}e{r['episode']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Controller and estimator reliability across episodes")

    feasible = 100.0 * np.asarray([_float(r.get("mpc_feasible_frac")) for r in rows])
    mhe = 100.0 * np.asarray([_float(r.get("mhe_ok_frac")) for r in rows])
    axes[0].bar(x - 0.2, feasible, width=0.4, label="MPC feasible")
    axes[0].bar(x + 0.2, mhe, width=0.4, label="MHE accepted")
    axes[0].set_ylabel("Successful steps [%]"); axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False); axes[0].set_title("Numerical success fractions"); _style_axes(axes[0])

    mpc = 1000.0 * np.asarray([_float(r.get("mpc_mean_solve_time_s")) for r in rows])
    mhe_t = 1000.0 * np.asarray([_float(r.get("mhe_mean_solve_time_s")) for r in rows])
    axes[1].bar(x - 0.2, mpc, width=0.4, label="MPC")
    axes[1].bar(x + 0.2, mhe_t, width=0.4, label="MHE")
    axes[1].set_ylabel("Mean solve time [ms]"); axes[1].set_title("Mean computation time")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].legend(frameon=False); _style_axes(axes[1])
    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_episode_report(ref: EpisodeRef, output_dir: Path, formats: Sequence[str],
                            dpi: int, max_points: int, include_spectrogram: bool,
                            max_frequency: float | None) -> tuple[dict[str, Any], list[Path]]:
    ep_dir = output_dir / ref.slug
    ep_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_episode(ref)
    generated: list[Path] = []

    pdf_path = ep_dir / f"mpc_mhe_report_{ref.slug}.pdf"
    with PdfPages(pdf_path) as pdf:
        for stem, builder in PLOT_BUILDERS:
            try:
                fig = builder(ref, max_points)
            except Exception as exc:  # Keep the rest of the report usable.
                warnings.warn(f"{ref.label}: failed to build {stem}: {exc}")
                fig, ax = plt.subplots(figsize=(12, 5))
                _empty_axis(ax, f"{stem} failed\n{type(exc).__name__}: {exc}")
                fig.suptitle(f"{stem} — {ref.label}")
            generated.extend(_save_figure(fig, ep_dir / stem, formats, dpi, pdf))

        try:
            fig = plot_psd(ref, max_frequency)
        except Exception as exc:
            warnings.warn(f"{ref.label}: failed to build PSD: {exc}")
            fig, ax = plt.subplots(figsize=(12, 5)); _empty_axis(ax, str(exc))
        generated.extend(_save_figure(fig, ep_dir / "15_vibration_psd", formats, dpi, pdf))

        if include_spectrogram:
            try:
                fig = plot_spectrogram(ref, max_frequency)
            except Exception as exc:
                warnings.warn(f"{ref.label}: failed to build spectrogram: {exc}")
                fig, ax = plt.subplots(figsize=(12, 5)); _empty_axis(ax, str(exc))
            generated.extend(_save_figure(fig, ep_dir / "16_vibration_spectrogram", formats, dpi, pdf))

        fig = plot_summary_card(ref, summary)
        generated.extend(_save_figure(fig, ep_dir / "19_episode_summary", formats, dpi, pdf))

    generated.append(pdf_path)
    with (ep_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)
    generated.append(ep_dir / "summary.json")
    return summary, generated


def generate_aggregate_report(rows: Sequence[dict[str, Any]], output_dir: Path,
                              formats: Sequence[str], dpi: int) -> list[Path]:
    if not rows:
        return []
    generated: list[Path] = []
    pdf_path = output_dir / "mpc_mhe_aggregate_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plot_aggregate_metrics(rows)
        generated.extend(_save_figure(fig, output_dir / "aggregate_episode_metrics", formats, dpi, pdf))
        fig = plot_aggregate_reliability(rows)
        generated.extend(_save_figure(fig, output_dir / "aggregate_reliability", formats, dpi, pdf))
    generated.append(pdf_path)
    return generated


def write_manifest(output_dir: Path, episodes: Sequence[EpisodeRef], generated: Sequence[Path],
                   args: argparse.Namespace) -> None:
    payload = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "episodes": [{"seed": e.seed, "episode": e.episode, "source": str(e.source)} for e in episodes],
        "generated_files": [str(path.resolve()) for path in generated],
        "settings": {
            "formats": list(args.formats),
            "dpi": args.dpi,
            "max_points": args.max_points,
            "spectrogram": args.spectrogram,
            "max_frequency": args.max_frequency,
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create standalone MPC/MHE plots from trajectories_seed*.json files."
    )
    parser.add_argument("--input-dir", default="mpc_phase_trajectories",
                        help="Directory containing trajectories_seed*.json files.")
    parser.add_argument("--output-dir", default="plots/mpc_only",
                        help="Directory receiving figures and PDF reports.")
    parser.add_argument("--patterns", nargs="+", default=["trajectories_seed*.json"],
                        help="Input filename glob pattern(s).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Optional seed filter, e.g. --seeds 0 1 2.")
    parser.add_argument("--episodes", nargs="+", type=int, default=None,
                        help="Optional episode filter, e.g. --episodes 0.")
    parser.add_argument("--formats", nargs="+", choices=["png", "svg"], default=["png"],
                        help="Individual figure formats. A multipage PDF is always generated.")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Raster output resolution.")
    parser.add_argument("--max-points", type=int, default=15000,
                        help="Maximum plotted points per curve after peak-preserving downsampling.")
    parser.add_argument("--spectrogram", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate a spectrogram of the highest-RMS vibration sensor.")
    parser.add_argument("--max-frequency", type=float, default=None,
                        help="Optional PSD/spectrogram upper frequency [Hz].")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    files = discover_files(input_dir, args.patterns)
    if not files:
        print(f"ERROR: no files matched {args.patterns} in {input_dir}", file=sys.stderr)
        return 2

    episodes = load_episodes(files,
                             set(args.seeds) if args.seeds is not None else None,
                             set(args.episodes) if args.episodes is not None else None)
    if not episodes:
        print("ERROR: no episodes remained after applying filters.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} trajectory file(s) and {len(episodes)} episode(s).")
    print(f"Output directory: {output_dir}")

    rows: list[dict[str, Any]] = []
    generated: list[Path] = []
    for ref in episodes:
        print(f"Plotting {ref.label} from {ref.source.name} ...", flush=True)
        summary, paths = generate_episode_report(
            ref=ref,
            output_dir=output_dir,
            formats=args.formats,
            dpi=max(72, int(args.dpi)),
            max_points=max(1000, int(args.max_points)),
            include_spectrogram=bool(args.spectrogram),
            max_frequency=args.max_frequency,
        )
        rows.append(summary)
        generated.extend(paths)
        print(f"  created {len(paths)} output file(s) in {output_dir / ref.slug}")

    write_summary(rows, output_dir)
    generated.extend([output_dir / "episode_summary.csv", output_dir / "episode_summary.json"])
    generated.extend(generate_aggregate_report(rows, output_dir, args.formats, args.dpi))
    write_manifest(output_dir, episodes, generated, args)
    generated.append(output_dir / "manifest.json")

    print("\nDone.")
    print(f"Episode summary: {output_dir / 'episode_summary.csv'}")
    print(f"Aggregate report: {output_dir / 'mpc_mhe_aggregate_report.pdf'}")
    for ref in episodes:
        print(f"Episode report: {output_dir / ref.slug / f'mpc_mhe_report_{ref.slug}.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
