#!/usr/bin/env python3
"""
Simple regenerative milling response check.

Runs a short constant-(omega, ac) rollout and plots sensor displacement.
Uses time-scale analysis to pick a stable integration step (no hard-coded dt).

From project root::

    python custom_rl/plants/Test3.py
    python custom_rl/plants/Test3.py --omega 800 --ac 2 --t-final 2
    python custom_rl/plants/Test3.py --no-plot
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_rl.integration import get_integrator
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.plants.modal_state import split_modal_state
from custom_rl.plants.plate import PlatePlant
from custom_rl.plants.time_scales import format_time_scales_report, recommend_integration_dt

MACRO_DT = 1.0472e-04
PLOT_DIR = _ROOT / "plots" / "test3_response"


def make_plant() -> PlatePlant:
    return PlatePlant(
        displacement_model="feed_normal_full",
        trajectory_mode="pass_grid",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
    )


def rollout(
    plant: PlatePlant,
    omega: float,
    ac_mm: float,
    t_final: float,
    *,
    perturb: float = 1e-6,
    macro_dt: float | None = None,
) -> dict:
    """Integrate with recommended sub-step dt; return time series."""
    mdt = float(macro_dt if macro_dt is not None else MACRO_DT)
    scales = recommend_integration_dt(plant, macro_dt=mdt, training_mode=True)
    sub_dt = scales["dt_recommended_training_substep_s"]
    n_sub = scales["n_substeps"]
    integrate = get_integrator("dde_rk4")

    plant.reset(np.random.default_rng(0))
    fmod.reset_episode_state(plant._modal_state_history)
    fmod.DELAY_MODE = "constant_tau"

    x0 = np.zeros(plant.state_dim, dtype=np.float64)
    if ac_mm > 0.0 and perturb > 0.0:
        x0[0] = perturb
        x0[2 * plant.K] = perturb * 0.1

    u_norm = plant.physical_to_normalized_action(np.array([omega, max(ac_mm, plant.u_phys_low[1])]))
    fmod.bind_modal_history(plant._modal_state_history)
    plant.record_modal_state(0.0, x0.copy(), omega=omega)

    x = x0.copy()
    t = 0.0
    n_macro = max(int(round(t_final / mdt)), 1)

    time_hist = [0.0]
    w_hist: list[np.ndarray] = []
    fn_hist: list[float] = []
    ff_hist: list[float] = []
    df_hist: list[float] = []
    dn_hist: list[float] = []
    clip_hist: list[bool] = []

    for _ in range(n_macro):
        eta_n, _, eta_f, _ = split_modal_state(x, plant.K, two_field=True)
        fr = fmod._directional_force_result(t, eta_n, eta_f, omega, ac_mm)
        w_s, _ = plant.state_to_sensor_signals(x)
        w_hist.append(w_s.copy())
        fn_hist.append(float(fr.get("F_normal_raw", fr["F_normal_total"])))
        ff_hist.append(float(fr.get("F_feed_raw", fr["F_feed_total"])))
        df_hist.append(float(fr["Delta_f"]))
        dn_hist.append(float(fr["Delta_n"]))
        clip_hist.append(bool(fr.get("force_clipped", False)))

        for _ in range(n_sub):
            x = integrate(plant.dynamics, t, x, u_norm, sub_dt, n_steps=1)
            t += sub_dt
            plant.record_modal_state(t, x, omega=omega)

        time_hist.append(t)
        if not np.all(np.isfinite(x)):
            break
        term, trunc, _ = plant.termination(t, x)
        if term or trunc:
            break

    fmod.unbind_modal_history()

    w_arr = np.asarray(w_hist, dtype=np.float64)
    return {
        "time": np.asarray(time_hist[: len(w_hist) + 1], dtype=np.float64),
        "sensor_w": w_arr,
        "F_normal_raw": np.asarray(fn_hist, dtype=np.float64),
        "F_feed_raw": np.asarray(ff_hist, dtype=np.float64),
        "Delta_f": np.asarray(df_hist, dtype=np.float64),
        "Delta_n": np.asarray(dn_hist, dtype=np.float64),
        "force_clipped": np.asarray(clip_hist, dtype=bool),
        "scales": scales,
        "omega": omega,
        "ac_mm": ac_mm,
        "finite": bool(np.all(np.isfinite(x))),
        "max_w_um": float(np.max(np.abs(w_arr)) * 1e6) if w_arr.size else 0.0,
        "clip_pct": 100.0 * float(np.mean(clip_hist)) if clip_hist else 0.0,
    }


def plot_response(res: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = res["time"][: res["sensor_w"].shape[0]]
    w = res["sensor_w"][:, 0] * 1e6

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, w, "b-", lw=0.8)
    axes[0].set_ylabel("w_sensor [um]")
    axes[0].set_title(
        f"feed_normal_full  omega={res['omega']:.0f} rad/s  ac={res['ac_mm']:.2f} mm"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, res["F_normal_raw"], label="F_normal", lw=0.8)
    axes[1].plot(t, res["F_feed_raw"], label="F_feed", lw=0.8, alpha=0.8)
    axes[1].set_ylabel("force [N]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, res["Delta_f"] * 1e3, label="Delta_f", lw=0.8)
    axes[2].plot(t, res["Delta_n"] * 1e3, label="Delta_n", lw=0.8, alpha=0.8)
    axes[2].set_ylabel("Delta [mm]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "response.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple milling response check")
    parser.add_argument("--omega", type=float, default=800.0, help="spindle speed [rad/s]")
    parser.add_argument("--ac", type=float, default=2.0, help="axial depth [mm]")
    parser.add_argument("--t-final", type=float, default=2.0, help="simulation time [s]")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    import warnings

    warnings.simplefilter("ignore", RuntimeWarning)
    plant = make_plant()
    plant.reset(np.random.default_rng(0))
    scales = recommend_integration_dt(plant, macro_dt=MACRO_DT, training_mode=True)
    print(format_time_scales_report(scales))
    print()

    warnings.simplefilter("ignore", RuntimeWarning)
    res = rollout(plant, args.omega, args.ac, args.t_final)
    print(
        f"Result: finite={res['finite']}  max|w|={res['max_w_um']:.2f} um  "
        f"force_clip={res['clip_pct']:.1f}%  "
        f"max|Delta_f|={np.max(np.abs(res['Delta_f']))*1e3:.4f} mm"
    )

    if not res["finite"]:
        print("FAIL: non-finite state")
        return 1
    if res["clip_pct"] > 1.0:
        print("WARN: force safety clip active (>1% steps)")
    if res["max_w_um"] > 1000.0:
        print("WARN: large sensor displacement (check stability / ac)")

    if not args.no_plot:
        path = plot_response(res, PLOT_DIR)
        print(f"Plot: {path}")

    print("PASS: response simulation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
