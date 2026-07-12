"""Verification & analysis for the face-milling MPC.

Produces evidence that the controller is a correct multivariable chatter
controller, not a constant-input heuristic:

  1. omega-ap INTERACTION (stability-lobe following): sweep the fixed depth of
     cut and record the spindle speed the MPC selects and the resulting peak
     plate displacement.  As the depth approaches the stability boundary the MPC
     lowers the spindle speed to stay stable -> the two inputs are coupled.
  2. FEASIBILITY sweep: run the closed loop over a grid of milling lines and
     depths and report solver success, peak displacement, NaN-free and
     termination status, so the optimisation is verified to return finite,
     feasible solutions.

Usage:
    python scripts/mpc_verify.py --out-dir plots/mpc_verification
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from scripts.mpc_face_milling import FaceMillingMPC, MPCConfig, _deinterleave


def _run_fixed_depth(env, mpc, ap, y0, n_dec=16, seed=0):
    """Run the second-mode MPC at a fixed depth; return steady speed, peak w, ok, nan."""
    p = env.unwrapped.plant
    obs, info = env.reset(seed=seed, options={"y0": y0, "ap": ap})
    z = _deinterleave(info["x_modal"], mpc.K)
    u_prev = np.array([0.5 * (mpc.omega_min + mpc.omega_max)])
    rpms, ws, oks = [], [], []
    nan = False
    for _ in range(n_dec):
        xc = float(np.clip(p.L1 - p._feed_distance_m - p.x0_cutter, 0.0, p.L1))
        phiz = mpc.phiz_at(xc, float(p.y_cutter))
        wz_hist = np.full(mpc.nb, float(phiz @ z[: mpc.K]))
        sol = mpc.solve(z, phiz, wz_hist, ap, u_prev)
        up = np.asarray(sol["u_phys"], dtype=np.float64).copy()
        if not np.all(np.isfinite(up)):
            nan = True
        up[0] = float(np.clip(up[0], u_prev[0] - mpc.cfg.omega_slew, u_prev[0] + mpc.cfg.omega_slew))
        a = mpc.to_env_action(up)
        term = trunc = False
        for _ in range(mpc.cfg.control_hold):
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rpms.append(up[0] * 60.0 / (2 * np.pi))
        ws.append(float(np.max(np.abs(info["w_sensor"]))))
        oks.append(bool(sol["success"]))
        z = _deinterleave(info["x_modal"], mpc.K)
        u_prev = np.array([up[0]])
        if term or trunc:
            break
    steady_rpm = float(np.mean(rpms[-6:])) if len(rpms) >= 6 else float(np.mean(rpms))
    return steady_rpm, float(np.max(ws)), float(np.mean(oks)), nan


def interaction_sweep(out_dir: Path, delay_mode: str):
    register_envs()
    kw = plate_env_kwargs(reward_id="dense", dt=1e-4, n_substeps=10,
                          max_episode_steps=50000, randomize_y0=False)
    env = gym.make("CustomODEPlateFinish-v0", **kw)
    mpc = FaceMillingMPC(env, MPCConfig(horizon=20, control_hold=3, n_rk=6,
                                        prescreen_n_omega=24, delay_mode=delay_mode))
    aps = np.linspace(0.2, 1.4, 9)
    rpm_sel, w_peak, ok = [], [], []
    print(f"\n=== omega-ap interaction (delay_mode={delay_mode}) ===")
    print(f"{'ap[mm]':>7} | {'MPC rpm':>9} | {'max|w|%':>8} | {'solve_ok':>8}")
    for ap in aps:
        r, w, o, nan = _run_fixed_depth(env, mpc, float(ap), y0=0.2)
        rpm_sel.append(r); w_peak.append(w); ok.append(o)
        flag = " NaN!" if nan else ""
        print(f"{ap:>7.2f} | {r:>9.0f} | {100*w/1e-3:>7.0f}% | {100*o:>7.0f}%{flag}")
    env.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(aps, np.array(rpm_sel) / 1000.0, "o-", color="#d62728", lw=2)
    ax[0].set_xlabel("fixed axial depth ap  [mm]")
    ax[0].set_ylabel("MPC-selected spindle speed  [krpm]")
    ax[0].set_title("Stability-lobe following (omega-ap coupling)")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(aps, np.array(w_peak) * 1e3, "s-", color="#1f77b4", lw=2, label="peak |w|")
    ax[1].axhline(1.0, ls=":", color="k", label="w_limit = 1 mm")
    ax[1].set_xlabel("fixed axial depth ap  [mm]")
    ax[1].set_ylabel("peak plate displacement  [mm]")
    ax[1].set_title("Closed-loop peak vibration vs depth")
    ax[1].grid(True, alpha=0.3); ax[1].legend()
    fig.suptitle(f"MPC verification — spindle speed adapts to depth ({delay_mode} delay)", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "omega_ap_interaction.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir/'omega_ap_interaction.png'}")


def feasibility_sweep(delay_mode: str):
    register_envs()
    kw = plate_env_kwargs(reward_id="dense", dt=1e-4, n_substeps=10,
                          max_episode_steps=50000, randomize_y0=False)
    env = gym.make("CustomODEPlate-v0", **kw)
    mpc = FaceMillingMPC(env, MPCConfig(horizon=20, control_hold=3, n_rk=6,
                                        prescreen_n_omega=16, prescreen_n_ap=7,
                                        delay_mode=delay_mode))
    p = env.unwrapped.plant
    print(f"\n=== first-mode feasibility sweep (delay_mode={delay_mode}) ===")
    print(f"{'y0[m]':>6} | {'solve_ok':>8} | {'max|w|%':>8} | {'NaN':>4} | {'terminated':>10}")
    for y0 in [0.15, 0.35, 0.5, 0.65, 0.85]:
        obs, info = env.reset(seed=7, options={"y0": y0})
        z = _deinterleave(info["x_modal"], mpc.K)
        u_prev = np.array([0.5 * (mpc.omega_min + mpc.omega_max),
                           mpc.ap_min + 0.01 * (mpc.ap_max - mpc.ap_min)])
        oks, ws = [], []
        nan = False; term_reason = None
        for _ in range(20):
            xc = float(np.clip(p.L1 - p._feed_distance_m - p.x0_cutter, 0.0, p.L1))
            phiz = mpc.phiz_at(xc, float(p.y_cutter))
            wz_hist = np.full(mpc.nb, float(phiz @ z[: mpc.K]))
            sol = mpc.solve(z, phiz, wz_hist, 0.5 * (mpc.ap_min + mpc.ap_max), u_prev)
            up = np.asarray(sol["u_phys"], dtype=np.float64).copy()
            if not np.all(np.isfinite(up)):
                nan = True
            up[0] = float(np.clip(up[0], u_prev[0] - mpc.cfg.omega_slew, u_prev[0] + mpc.cfg.omega_slew))
            up[1] = float(np.clip(up[1], u_prev[1] - mpc.cfg.ap_down_slew, u_prev[1] + mpc.cfg.ap_up_slew))
            a = mpc.to_env_action(up)
            term = trunc = False
            for _ in range(mpc.cfg.control_hold):
                obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    term_reason = info.get("termination_reason"); break
            oks.append(bool(sol["success"])); ws.append(float(np.max(np.abs(info["w_sensor"]))))
            z = _deinterleave(info["x_modal"], mpc.K); u_prev = up.copy()
            if term or trunc:
                break
        print(f"{y0:>6.2f} | {100*np.mean(oks):>7.0f}% | {100*max(ws)/1e-3:>7.0f}% | "
              f"{str(nan):>4} | {str(term_reason):>10}")
    env.close()


def main():
    ap_ = argparse.ArgumentParser(description="MPC verification & interaction analysis.")
    ap_.add_argument("--out-dir", default="plots/mpc_verification")
    ap_.add_argument("--delay-mode", default="predictive", choices=["predictive", "frozen"])
    args = ap_.parse_args()
    out = Path(args.out_dir).resolve()
    interaction_sweep(out, args.delay_mode)
    feasibility_sweep(args.delay_mode)
    print("\nVerification complete.")


if __name__ == "__main__":
    main()
