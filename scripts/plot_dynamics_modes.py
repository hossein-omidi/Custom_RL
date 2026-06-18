#!/usr/bin/env python3
"""
Visualize milling dynamic modes: stable, bounded, chatter-like.

Runs preset (omega, ac) rollouts via Test2 ``simulate`` (rich diagnostics) and
plots time-domain, spectral, phase-space, and envelope views.

Usage (from project root)::

    python scripts/plot_dynamics_modes.py
    python scripts/plot_dynamics_modes.py --quick
    python scripts/plot_dynamics_modes.py --omega 800 --ac 0.5 2 8 --t-final 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_rl.analysis.response_classification import (
    classify_milling_response,
    metrics_from_rollout_dict,
)
from custom_rl.plants.Test2 import make_plant, simulate
from custom_rl.plants.time_scales import tooth_period

PLOT_DIR = _ROOT / "plots" / "dynamics_modes"
MACRO_DT = 0.002

CLASS_COLORS = {
    "stable": "#2ca02c",
    "bounded": "#ffbf00",
    "chatter-like": "#d62728",
    "failed": "#111111",
}


@dataclass
class DynamicsCase:
    case_id: str
    label: str
    omega: float
    ac_mm: float
    color: str


DEFAULT_CASES = [
    DynamicsCase("idle", "Idle — no cut (ac=0)", 800.0, 0.0, "#7f7f7f"),
    DynamicsCase("stable_400", "Stable cutting (ω=400, ac=2 mm)", 400.0, 2.0, "#2ca02c"),
    DynamicsCase("bounded_800", "Bounded vibration (ω=800, ac=8 mm)", 800.0, 8.0, "#ffbf00"),
    DynamicsCase("bounded_mid", "Moderate cut (ω=800, ac=2 mm)", 800.0, 2.0, "#17becf"),
    DynamicsCase("chatter_shallow", "Large vibration (ω=800, ac=0.5 mm)", 800.0, 0.5, "#d62728"),
]

QUICK_CASES = [
    DEFAULT_CASES[0],  # idle
    DEFAULT_CASES[1],  # stable_400
    DEFAULT_CASES[3],  # bounded_mid
    DEFAULT_CASES[4],  # chatter_shallow
]


def _sensor_mag_series(res: dict) -> np.ndarray:
    sw = np.asarray(res["sensor_w"], dtype=np.float64)
    if sw.ndim > 1:
        return np.max(np.abs(sw), axis=1)
    return np.abs(sw)


def _metrics_from_sim(res: dict, omega: float, ac_mm: float, plant) -> tuple:
    sw_mag = _sensor_mag_series(res)
    term_info = res.get("termination_info", {})
    raw = {
        "omega": omega,
        "ac_mm": ac_mm,
        "finite": bool(np.all(np.isfinite(res["trajectory"]))),
        "clip_pct": 100.0 * float(np.mean(res.get("force_clipped", [False]))),
        "max_w_m": float(np.max(sw_mag)) if sw_mag.size else 0.0,
        "max_delta_f_m": float(np.max(np.abs(res["delta_f"]))) if len(res["delta_f"]) else 0.0,
        "max_delta_n_m": float(np.max(np.abs(res["delta_n"]))) if len(res["delta_n"]) else 0.0,
        "max_F_normal_N": float(np.max(np.abs(res["f_normal_raw"]))) if len(res["f_normal_raw"]) else 0.0,
        "max_F_feed_N": float(np.max(np.abs(res["f_feed_raw"]))) if len(res["f_feed_raw"]) else 0.0,
        "sensor_w_series": sw_mag,
        "terminated": bool(res.get("terminated", False)),
        "term_reason": str(term_info.get("termination_reason", "")),
        "transient_fraction": 0.35,
    }
    metrics = metrics_from_rollout_dict(raw)
    klass = classify_milling_response(
        metrics,
        displacement_limit_m=plant.displacement_failure_limit,
        transient_fraction=0.35,
        growth_factor=3.0,
    )
    return metrics, klass


def run_case(plant, case: DynamicsCase, t_final: float, perturb: float) -> dict:
    plant.reset(np.random.default_rng(0))
    x0 = np.zeros(plant.state_dim, dtype=np.float64)
    if case.ac_mm > 0 and perturb > 0:
        x0[0] = perturb
        if plant.milling_config.is_feed_normal_full():
            x0[2 * plant.K] = perturb * 0.1

    u_norm = plant.physical_to_normalized_action(
        np.array([case.omega, max(case.ac_mm, plant.u_phys_low[1])])
    )
    res = simulate(plant, x0, u_norm, MACRO_DT, t_final)
    metrics, klass = _metrics_from_sim(res, case.omega, case.ac_mm, plant)
    n = len(res["sensor_w"])
    time = np.asarray(res["time"][:n], dtype=np.float64)
    sw_mag = _sensor_mag_series(res)

    return {
        "case": case,
        "res": res,
        "time": time,
        "sw_mag": sw_mag,
        "metrics": metrics,
        "class": klass,
        "tau": tooth_period(case.omega, plant.N),
    }


def _rolling_rms(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < win:
        return np.full_like(x, np.sqrt(np.mean(x**2)))
    c = np.cumsum(np.insert(x**2, 0, 0.0))
    out = np.sqrt((c[win:] - c[:-win]) / win)
    pad = np.full(win - 1, np.nan)
    return np.concatenate([pad, out])


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import hilbert
        return np.abs(hilbert(x))
    except ImportError:
        return np.abs(x)


def _welch_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
        nperseg = min(512, max(64, len(x) // 4))
        return welch(x, fs=fs, nperseg=nperseg)
    except ImportError:
        n = len(x)
        win = np.hanning(n)
        spec = np.fft.rfft(x * win)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = (np.abs(spec) ** 2) / (fs * n)
        return freqs, psd


def plot_overview(runs: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for run in runs:
        c = run["case"]
        t = run["time"]
        w_um = run["sw_mag"] * 1e6
        ax.plot(t, w_um, lw=0.9, color=c.color, label=f"{c.label} [{run['class']}]")
    ax.set_ylabel("|w| max sensors [µm]")
    ax.set_xlabel("time [s]")
    ax.set_title("Displacement — all modes")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for run in runs:
        c = run["case"]
        res = run["res"]
        n = len(run["time"])
        fn = np.asarray(res["f_normal_raw"][:n])
        ax.plot(run["time"], fn, lw=0.8, color=c.color, alpha=0.85, label=c.case_id)
    ax.set_ylabel("F_normal [N]")
    ax.set_xlabel("time [s]")
    ax.set_title("Normal cutting force")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for run in runs:
        c = run["case"]
        env = _analytic_envelope(run["sw_mag"]) * 1e6
        ax.plot(run["time"], env, lw=0.9, color=c.color, alpha=0.85, label=c.case_id)
    ax.set_ylabel("envelope |w| [µm]")
    ax.set_xlabel("time [s]")
    ax.set_title("Analytic signal envelope")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Case summary", "─" * 42]
    for run in runs:
        c = run["case"]
        m = run["metrics"]
        lines.append(
            f"{c.case_id:14s} ω={c.omega:.0f} ac={c.ac_mm:.1f}mm  "
            f"class={run['class']:12s}"
        )
        lines.append(
            f"               max|w|={m.max_w_mm*1e3:6.1f}µm  "
            f"rms_post={m.rms_w_post_m*1e6:5.1f}µm  τ={run['tau']*1e3:.1f}ms"
        )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    patches = [mpatches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8, title="classes")

    fig.suptitle("Milling dynamic modes — overview", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "dynamics_overview.png", dpi=150)
    plt.close(fig)


def plot_envelope_growth(runs: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.4 * len(runs)), sharex=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        c = run["case"]
        t = run["time"]
        w = run["sw_mag"]
        win = max(int(0.05 * len(w)), 10)
        rms = _rolling_rms(w, win) * 1e6
        ax.plot(t, w * 1e6, color=c.color, lw=0.5, alpha=0.45, label="|w|")
        ax.plot(t, rms, color="k", lw=1.4, label=f"rolling RMS (win={win})")
        ax.axvline(0.35 * t[-1], color="gray", ls="--", lw=0.8, alpha=0.7, label="35% transient")
        ax.set_ylabel("µm")
        ax.set_title(f"{c.label}  →  {run['class']}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Post-transient growth diagnostic (rolling RMS)")
    fig.tight_layout()
    fig.savefig(out_dir / "envelope_growth.png", dpi=150)
    plt.close(fig)


def plot_spectra(runs: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fs = 1.0 / MACRO_DT

    ax = axes[0]
    for run in runs:
        c = run["case"]
        w = run["sw_mag"]
        if w.size < 32:
            continue
        f, psd = _welch_psd(w, fs)
        ax.semilogy(f, psd + 1e-30, color=c.color, lw=1.2, label=f"{c.case_id} [{run['class']}]")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [|w|²]")
    ax.set_title("Welch PSD — chatter frequency content")
    ax.set_xlim(left=0.0)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for run in runs:
        c = run["case"]
        tau = run["tau"]
        if tau <= 0:
            continue
        f_tooth = 1.0 / tau
        ax.axvline(f_tooth, color=c.color, ls="--", lw=1.2, alpha=0.8,
                   label=f"{c.case_id} 1/τ={f_tooth:.0f}Hz")
    ax.set_xlim(0, 500)
    ax.set_xlabel("frequency [Hz]")
    ax.set_title("Tooth-passing frequencies 1/τ")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "spectra_comparison.png", dpi=150)
    plt.close(fig)


def plot_phase_portraits(runs: list[dict], out_dir: Path) -> None:
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        c = run["case"]
        res = run["res"]
        n_pts = len(run["time"])
        w = np.asarray(res["sensor_w"][:n_pts, 0], dtype=np.float64)
        wd = np.asarray(res["sensor_w_dot"][:n_pts, 0], dtype=np.float64)
        start = int(0.35 * n_pts)
        ax.scatter(w[:start] * 1e6, wd[:start] * 1e3, s=3, c="gray", alpha=0.3, label="transient")
        ax.scatter(w[start:] * 1e6, wd[start:] * 1e3, s=4, c=c.color, alpha=0.6, label="post-transient")
        ax.set_xlabel("w [µm]")
        ax.set_ylabel("ẇ [mm/s]")
        ax.set_title(f"{c.case_id}\n{run['class']}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    fig.suptitle("Phase portraits — sensor 0")
    fig.tight_layout()
    fig.savefig(out_dir / "phase_portraits.png", dpi=150)
    plt.close(fig)


def plot_case_detail(run: dict, out_dir: Path) -> None:
    c = run["case"]
    res = run["res"]
    n = len(run["time"])
    time = run["time"]
    traj = res["trajectory"][:n]

    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)

    if res.get("eta_f_series") is not None:
        eta_n = traj[:, 0::2][:, 0]
        k_block = traj.shape[1] // 2
        eta_f = traj[:, k_block::2][:, 0]
        axes[0, 0].plot(time, eta_n * 1e6, color=c.color, lw=0.8)
        axes[0, 0].set_ylabel("η_n,0 [µm]")
        axes[0, 1].plot(time, eta_f * 1e6, color=c.color, lw=0.8)
        axes[0, 1].set_ylabel("η_f,0 [µm]")
    else:
        axes[0, 0].plot(time, traj[:, 0] * 1e6, color=c.color)
        axes[0, 0].set_ylabel("η [µm]")
        axes[0, 1].axis("off")

    axes[1, 0].plot(time, run["sw_mag"] * 1e6, color=c.color, lw=0.8)
    axes[1, 0].set_ylabel("|w| [µm]")
    axes[1, 1].plot(time, np.asarray(res["sensor_w_dot"][:n, 0]) * 1e3, color=c.color, lw=0.8)
    axes[1, 1].set_ylabel("ẇ [mm/s]")

    axes[2, 0].plot(time, res["f_normal_raw"][:n], color=c.color, lw=0.8)
    axes[2, 0].set_ylabel("F_n [N]")
    axes[2, 1].plot(time, res["f_feed_raw"][:n], color=c.color, lw=0.8)
    axes[2, 1].set_ylabel("F_f [N]")

    axes[3, 0].plot(time, np.asarray(res["delta_f"][:n]) * 1e3, color=c.color, lw=0.8)
    axes[3, 0].set_ylabel("Δf [mm]")
    axes[3, 1].plot(time, np.asarray(res["h_sample"][:n]) * 1e3, color=c.color, lw=0.8)
    axes[3, 1].set_ylabel("h [mm]")
    axes[3, 0].set_xlabel("time [s]")
    axes[3, 1].set_xlabel("time [s]")

    fig.suptitle(
        f"{c.label}  |  class={run['class']}  |  "
        f"max|w|={run['metrics'].max_w_mm*1e3:.1f}µm",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / f"detail_{c.case_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regenerative_panel(runs: list[dict], out_dir: Path) -> None:
    """Chip thickness vs displacement — regenerative loop view."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4))
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        c = run["case"]
        if c.ac_mm <= 0:
            ax.text(0.5, 0.5, "no cut", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(c.case_id)
            continue
        res = run["res"]
        n = len(run["time"])
        df = np.asarray(res["delta_f"][:n]) * 1e6
        w = run["sw_mag"] * 1e6
        sc = ax.scatter(df, w, c=run["time"], cmap="viridis", s=6, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="time [s]")
        ax.set_xlabel("Δf [µm]")
        ax.set_ylabel("|w| [µm]")
        ax.set_title(f"{c.case_id} [{run['class']}]")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Regenerative coupling: chip thickness Δf vs vibration")
    fig.tight_layout()
    fig.savefig(out_dir / "regenerative_scatter.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot milling dynamic modes")
    parser.add_argument("--quick", action="store_true", help="fewer cases, shorter t")
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--omega", type=float, nargs="+", default=None)
    parser.add_argument("--ac", type=float, nargs="+", default=None)
    parser.add_argument("--perturb", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path, default=PLOT_DIR)
    args = parser.parse_args()

    t_final = args.t_final if args.t_final is not None else (1.5 if args.quick else 3.0)

    if args.omega and args.ac:
        if len(args.omega) != len(args.ac):
            parser.error("--omega and --ac must have the same length")
        colors = plt.cm.tab10(np.linspace(0, 1, len(args.omega)))
        cases = [
            DynamicsCase(
                f"case_{i}", f"ω={o:.0f} ac={a:.1f}", o, a,
                plt.cm.colors.to_hex(colors[i]),
            )
            for i, (o, a) in enumerate(zip(args.omega, args.ac))
        ]
    else:
        cases = QUICK_CASES if args.quick else DEFAULT_CASES

    warnings.simplefilter("ignore", RuntimeWarning)
    plant = make_plant("feed_normal_full")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dynamics modes: {len(cases)} cases, t_final={t_final}s\n")
    runs: list[dict] = []
    for case in cases:
        run = run_case(plant, case, t_final, args.perturb)
        runs.append(run)
        m = run["metrics"]
        print(
            f"  {case.case_id:16s} omega={case.omega:5.0f} ac={case.ac_mm:4.1f}  "
            f"class={run['class']:12s}  max|w|={m.max_w_mm*1e3:6.1f}um"
        )

    plot_overview(runs, out_dir)
    plot_envelope_growth(runs, out_dir)
    plot_spectra(runs, out_dir)
    plot_phase_portraits(runs, out_dir)
    plot_regenerative_panel(runs, out_dir)
    for run in runs:
        plot_case_detail(run, out_dir)

    print(f"\nPlots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
