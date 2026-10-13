#!/usr/bin/env python3
"""
Enhanced model fidelity validation: compare the Gymnasium plant against the
reduced phase-domain internal model used by NMPC-MHE.

This script uses safe default commands to avoid plant instability and handles
early termination gracefully.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import correlate

import gymnasium as gym

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from mpc_mhe_dompc import PhasePlant, build_model, cheb_lobatto_D, PhaseSchedule

try:
    import do_mpc
    from do_mpc.simulator import Simulator
except ImportError:
    print("do-mpc required. Install: pip install do-mpc casadi")
    sys.exit(1)


def _deinterleave_modal(x_modal, K_full):
    x = np.asarray(x_modal).reshape(-1)
    eta = x[0::2]
    v = x[1::2]
    return np.concatenate([eta, v])


def _extract_plant_force(info):
    F_total = info.get('F_total_N', np.zeros(3))
    return float(F_total[2]) if len(F_total) >= 3 else np.nan


def _select_delay_order(pp, omega_range, p, tol=1e-4, max_M=60):
    """Automatically select M based on transfer function error."""
    try:
        if hasattr(pp, 'omega_vec') and hasattr(pp, 'sel'):
            wn_full = np.asarray(pp.omega_vec, dtype=np.float64)
            wn_red = wn_full[pp.sel]
        else:
            wn_red = np.asarray(pp.wn)[:pp.K] if hasattr(pp, 'wn') else None
            if wn_red is None:
                print("Warning: cannot retrieve retained frequencies; using default M=20")
                return 20
    except Exception:
        print("Warning: cannot retrieve retained frequencies; using default M=20")
        return 20

    omega_vals = np.linspace(omega_range[0], omega_range[1], 10)
    freqs = []
    for om in omega_vals:
        om = max(om, 1e-12)
        nu = wn_red / om
        freqs.extend(np.abs(nu))
    freqs = np.unique(freqs)

    for M in range(2, max_M + 1, 2):
        xi, D = cheb_lobatto_D(M)
        D_II = D[1:, 1:]
        D_I0 = D[1:, 0]
        errors = []
        for nu in freqs:
            s = 1j * nu
            A = s * np.eye(M) + D_II / p
            b = -D_I0 / p
            try:
                x = np.linalg.solve(A, b)
                G_M = x[-1]
                G_exact = np.exp(-1j * nu * p)
                errors.append(abs(G_M - G_exact))
            except np.linalg.LinAlgError:
                errors.append(1.0)
        max_err = max(errors)
        if max_err < tol:
            return M
    return max_M


def run_validation(args):
    register_envs()

    # Build environment
    env_kwargs = plate_env_kwargs(
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=args.n_substeps,
        max_episode_steps=args.max_episode_steps,
        dynamics_uncertainty_std=0.0,
        randomize_y0=False,
    )
    env = gym.make(args.env_id, **env_kwargs)
    plant = env.unwrapped.plant

    pp = PhasePlant(env, n_modes=args.n_modes)
    K = pp.K
    p = pp.p

    # Delay order
    omega_range = (pp.omega_min, pp.omega_max)
    if args.n_delay is None:
        M = _select_delay_order(pp, omega_range, p, tol=1e-4)
        print(f"Auto-selected delay order M = {M}")
    else:
        M = args.n_delay

    dtheta = args.dtheta_frac * p

    # Build do-mpc model
    model, xi = build_model(pp, M, args.chip_eps, args.n_quad)
    sim = Simulator(model)
    sim.settings.t_step = dtheta
    sim.settings.integration_tool = 'idas'
    sim.settings.t_rel = 1e-6
    sim.settings.t_abs = 1e-8
    sim.settings.supress_exit_warning = True

    sched = PhaseSchedule(pp, dtheta, args.sharp, args.n_quad)

    tmpl = sim.get_tvp_template()
    def tvp_fun(t_now):
        phi, g, s = sched.at(0)
        tmpl['phi'] = phi.reshape(-1, 1)
        tmpl['gvec'] = g.reshape(-1, 1)
        tmpl['svec'] = s.reshape(-1, 1)
        return tmpl

    sim.set_tvp_fun(tvp_fun)
    sim.setup()

    env_dt = float(args.dt) * int(args.n_substeps)
    total_time = args.duration
    n_steps = int(total_time / env_dt)

    print("===== Validation Configuration =====")
    print(f"  env dt = {env_dt*1e3:.3f} ms, control dtheta = {dtheta:.4f} rad")
    print(f"  total steps: {n_steps} (≈ {n_steps*env_dt:.3f} s)")
    print(f"  retained modes: {K}, delay order: {M}")

    # ---- SAFE WARM-UP ----
    # Use very low speed and tiny depth to avoid instability
    warmup_omega = pp.omega_min * 1.1
    warmup_ap = 0.05  # mm – tiny
    warmup_ap = np.clip(warmup_ap, pp.ap_min, pp.ap_max)
    warmup_cmd = np.array([warmup_omega, warmup_ap], dtype=float)
    low = np.asarray(plant._ctrl_low, dtype=float)
    high = np.asarray(plant._ctrl_high, dtype=float)
    act_warm = 2.0 * (warmup_cmd[:len(low)] - low) / (high - low) - 1.0
    act_warm = np.clip(act_warm, -1.0, 1.0)

    min_steps = int(np.ceil(5 * p / (warmup_omega * env_dt))) + 20
    warmup_steps = max(300, min_steps)
    print(f"Warming up for {warmup_steps} env steps (≈ {warmup_steps*env_dt:.3f} s) ...")

    obs, info = env.reset(seed=args.seed)
    warmup_theta = []
    warmup_wc = []
    for _ in range(warmup_steps):
        obs, _, term, trunc, info = env.step(act_warm)
        if term or trunc:
            print(f"Warm-up terminated: {info.get('termination_reason')}")
            # Still record the last valid state (info contains current state)
            break
        theta = float(info.get('spindle_phase_rad', 0.0))
        xm = info.get('x_modal', np.zeros(2*pp.K_full))
        eta_full = xm[0::2]
        eta_sel = eta_full[pp.sel]
        phi = pp.phi_at(info.get('cutter_x', plant.x0_cutter), plant.y_cutter)
        wc = float(phi @ eta_sel)
        warmup_theta.append(theta)
        warmup_wc.append(wc)

    if len(warmup_theta) == 0:
        print("Warm-up failed to collect any data. Aborting.")
        env.close()
        return

    warmup_theta = np.array(warmup_theta)
    warmup_wc = np.array(warmup_wc)
    warmup_theta_unwrapped = np.unwrap(warmup_theta)

    # Current state
    xm_current = info.get('x_modal', np.zeros(2*pp.K_full))
    z_block_current = _deinterleave_modal(xm_current, pp.K_full)
    eta_cur = z_block_current[:pp.K_full][pp.sel]
    v_cur = z_block_current[pp.K_full:][pp.sel]
    theta_now = warmup_theta_unwrapped[-1]
    xc_now = float(info.get('cutter_x', plant.x0_cutter))
    yc_now = float(plant.y_cutter)

    # Seed delay state
    xi_full, _ = cheb_lobatto_D(M)
    xi_nodes = xi_full[1:]
    target_phases = theta_now - p * xi_nodes

    if len(warmup_theta_unwrapped) > 4:
        cs = CubicSpline(warmup_theta_unwrapped, warmup_wc, extrapolate=False)
        qh_init = np.zeros_like(target_phases)
        for idx, tp in enumerate(target_phases):
            if tp < warmup_theta_unwrapped[0] or tp > warmup_theta_unwrapped[-1]:
                nearest_idx = np.argmin(np.abs(warmup_theta_unwrapped - tp))
                qh_init[idx] = warmup_wc[nearest_idx]
            else:
                qh_init[idx] = cs(tp)
    else:
        qh_init = np.full(M, warmup_wc[-1])

    z0_model = np.concatenate([eta_cur, v_cur, qh_init, [0.0]]).reshape(-1, 1)
    sim.x0 = z0_model.copy()
    sim.set_initial_guess()
    sched.set_now(theta_now, xc_now, yc_now)

    # ---- MAIN COMMAND SEQUENCE ----
    # Use very low command amplitude to avoid instability
    if args.unsafe:
        omega_avg = args.omega_avg
        ap_avg = args.ap_avg
    else:
        omega_avg = args.omega_avg if args.omega_avg < 200 else 100.0
        ap_avg = args.ap_avg if args.ap_avg < 0.2 else 0.05

    omega_seq = []
    ap_seq = []
    for i in range(n_steps):
        t = i * env_dt
        if args.command_type == 'sine':
            omega = omega_avg + 0.2 * omega_avg * np.sin(2 * np.pi * args.sine_freq * t)
            omega = np.clip(omega, pp.omega_min, pp.omega_max)
            ap = ap_avg + 0.2 * ap_avg * np.sin(2 * np.pi * args.sine_freq * t)
            ap = np.clip(ap, pp.ap_min, pp.ap_max)
        else:
            omega = omega_avg
            ap = ap_avg
        omega_seq.append(omega)
        ap_seq.append(ap)

    # ---- MAIN LOOP ----
    times = []
    plant_w = []
    model_w = []
    plant_Fz = []
    model_Fz = []
    plant_eta = []
    model_eta = []
    model_times = []

    model_phase_accum = 0.0
    prev_theta = theta_now
    prev_theta_raw = info.get('spindle_phase_rad', 0.0)
    prev_theta_unwrapped = theta_now

    print("Running main validation ...")
    for i in range(n_steps):
        omega_cmd = omega_seq[i]
        ap_cmd = ap_seq[i]
        u_phys = np.array([omega_cmd, ap_cmd], dtype=float)
        act = 2.0 * (u_phys[:len(low)] - low) / (high - low) - 1.0
        act = np.clip(act, -1.0, 1.0)

        obs, _, term, trunc, info = env.step(act)

        t = i * env_dt
        times.append(t)
        w_sensor = info.get('w_sensor', np.zeros(pp.nS))
        plant_w.append(w_sensor)
        xm = info.get('x_modal', np.zeros(2*pp.K_full))
        eta_full = xm[0::2]
        plant_eta.append(eta_full[pp.sel])
        plant_Fz.append(_extract_plant_force(info))

        # Unwrap phase
        theta_raw = float(info.get('spindle_phase_rad', 0.0))
        if i == 0:
            theta_unwrapped = theta_raw
        else:
            delta = theta_raw - prev_theta_raw
            if delta < -np.pi:
                delta += 2*np.pi
            elif delta > np.pi:
                delta -= 2*np.pi
            theta_unwrapped = prev_theta_unwrapped + delta
        prev_theta_raw = theta_raw
        prev_theta_unwrapped = theta_unwrapped

        if term or trunc:
            print(f"Episode terminated at step {i}: {info.get('termination_reason')}")
            # We still have data up to this point; break after recording
            break

        dtheta_real = theta_unwrapped - prev_theta
        prev_theta = theta_unwrapped
        model_phase_accum += dtheta_real

        if model_phase_accum >= dtheta:
            xc_now = float(info.get('cutter_x', plant.x0_cutter))
            yc_now = float(plant.y_cutter)
            sched.set_now(theta_unwrapped, xc_now, yc_now)

            model_u = np.array([omega_cmd, ap_cmd]).reshape(-1, 1)
            sim.make_step(model_u)
            z_pred = np.asarray(sim.x0.cat, dtype=np.float64).reshape(-1)

            w_pred = pp.Phi @ z_pred[:K]
            model_w.append(w_pred)
            model_eta.append(z_pred[:K])
            model_times.append(t)

            # Model force (reconstruction)
            try:
                wd = z_pred[K + K + M - 1]
                qf = pp.qf_model(theta_unwrapped, xc_now, yc_now,
                                 z_pred[:K], wd, ap_cmd, args.sharp,
                                 args.chip_eps, dtheta, args.n_quad)
                phi_cur = pp.phi_at(xc_now, yc_now)
                Mm = pp.Mm
                Fz_model = 0.0
                cnt = 0
                for j in range(K):
                    if abs(phi_cur[j]) > 1e-12:
                        Fz_model += qf[j] * Mm[j] / phi_cur[j]
                        cnt += 1
                if cnt > 0:
                    Fz_model /= cnt
                else:
                    Fz_model = np.nan
            except Exception:
                Fz_model = np.nan
            model_Fz.append(Fz_model)

            model_phase_accum -= dtheta

    env.close()

    # Convert to arrays (handle empty gracefully)
    if len(times) == 0:
        print("No data collected. Aborting.")
        return

    times = np.array(times)
    plant_w = np.vstack(plant_w) if isinstance(plant_w[0], np.ndarray) else np.array(plant_w).reshape(-1, 1)
    plant_eta = np.vstack(plant_eta) if isinstance(plant_eta[0], np.ndarray) else np.array(plant_eta).reshape(-1, 1)
    plant_Fz = np.array(plant_Fz)
    model_w = np.vstack(model_w) if len(model_w)>0 and isinstance(model_w[0], np.ndarray) else np.array(model_w)
    model_eta = np.vstack(model_eta) if len(model_eta)>0 and isinstance(model_eta[0], np.ndarray) else np.array(model_eta)
    model_Fz = np.array(model_Fz) if len(model_Fz)>0 else np.empty(0)
    model_times = np.array(model_times) if len(model_times)>0 else np.empty(0)

    # ---- PLOTTING ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(times, plant_w[:, 0]*1e6, label='Plant (sensor 0)')
    if len(model_w) > 0:
        ax.plot(model_times, model_w[:, 0]*1e6, 'ro', markersize=4, label='Model')
    ax.set_ylabel('Displacement [µm]')
    ax.set_title('Sensor 0 displacement')
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot(times, plant_eta[:, 0]*1e6, label='Plant')
    if len(model_eta) > 0:
        ax.plot(model_times, model_eta[:, 0]*1e6, 'ro', markersize=4, label='Model')
    ax.set_ylabel('Modal disp [µm]')
    ax.set_title('Mode 1 displacement')
    ax.legend(); ax.grid(True)

    ax = axes[2]
    ax.plot(times, plant_Fz, label='Plant Fz')
    if len(model_Fz) > 0:
        ax.plot(model_times, model_Fz, 'ro', markersize=4, label='Model Fz')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Axial force [N]')
    ax.set_title('Cutting force')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'model_validation.png', dpi=150)
    print(f"Saved plot to {out_dir / 'model_validation.png'}")

    # ---- METRICS ----
    if len(model_w) > 0 and len(model_times) > 1:
        f_w = interp1d(model_times, model_w[:, 0], kind='linear', fill_value='extrapolate')
        model_w_interp = f_w(times)
        rmse_w = np.sqrt(np.mean((plant_w[:, 0] - model_w_interp)**2))
        range_w = np.max(plant_w[:, 0]) - np.min(plant_w[:, 0])
        nrms_w = rmse_w / (range_w + 1e-12) * 100
        corr_w = np.corrcoef(plant_w[:, 0], model_w_interp)[0, 1]
        max_err_w = np.max(np.abs(plant_w[:, 0] - model_w_interp))
        print(f"Sensor 0: RMSE = {rmse_w*1e6:.2e} µm, NRMSE = {nrms_w:.2f}%, corr = {corr_w:.4f}, max error = {max_err_w*1e6:.2e} µm")

        if len(model_Fz) > 0:
            f_Fz = interp1d(model_times, model_Fz, kind='linear', fill_value='extrapolate')
            model_Fz_interp = f_Fz(times)
            rmse_Fz = np.sqrt(np.mean((plant_Fz - model_Fz_interp)**2))
            range_Fz = np.max(plant_Fz) - np.min(plant_Fz)
            nrms_Fz = rmse_Fz / (range_Fz + 1e-12) * 100
            corr_Fz = np.corrcoef(plant_Fz, model_Fz_interp)[0, 1]
            max_err_Fz = np.max(np.abs(plant_Fz - model_Fz_interp))
            print(f"Force: RMSE = {rmse_Fz:.2f} N, NRMSE = {nrms_Fz:.2f}%, corr = {corr_Fz:.4f}, max error = {max_err_Fz:.2f} N")

        f_eta = interp1d(model_times, model_eta[:, 0], kind='linear', fill_value='extrapolate')
        eta_interp = f_eta(times)
        rmse_eta = np.sqrt(np.mean((plant_eta[:, 0] - eta_interp)**2))
        range_eta = np.max(plant_eta[:, 0]) - np.min(plant_eta[:, 0])
        nrms_eta = rmse_eta / (range_eta + 1e-12) * 100
        print(f"Mode 1: RMSE = {rmse_eta*1e6:.2e} µm, NRMSE = {nrms_eta:.2f}%")

        if len(times) > 10:
            corr = correlate(plant_w[:, 0] - np.mean(plant_w[:, 0]),
                             model_w_interp - np.mean(model_w_interp),
                             mode='full')
            lag = np.argmax(corr) - (len(times) - 1)
            time_lag = lag * env_dt if lag != 0 else 0.0
            print(f"Time lag (sensor 0): {time_lag*1e3:.2f} ms")
    else:
        print("No model predictions available for metrics.")

    print(f"\nValidation complete. Results saved in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced model validation.')
    parser.add_argument('--env-id', default='CustomODEPlate-v0')
    parser.add_argument('--reward', default='dense')
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--n-substeps', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=100000)
    parser.add_argument('--duration', type=float, default=0.2)
    parser.add_argument('--omega-avg', type=float, default=100.0, help='Safe default: 100 rad/s')
    parser.add_argument('--ap-avg', type=float, default=0.05, help='Safe default: 0.05 mm')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-modes', type=int, default=3)
    parser.add_argument('--n-delay', type=int, default=None)
    parser.add_argument('--dtheta-frac', type=float, default=0.1)
    parser.add_argument('--chip-eps', type=float, default=2e-3)
    parser.add_argument('--n-quad', type=int, default=2)
    parser.add_argument('--sharp', type=float, default=0.0)
    parser.add_argument('--command-type', choices=['step', 'sine'], default='step')
    parser.add_argument('--sine-freq', type=float, default=5.0)
    parser.add_argument('--sine-amp', type=float, default=0.2, help='Fraction of average')
    parser.add_argument('--sine-amp-ap', type=float, default=0.2, help='Fraction of average')
    parser.add_argument('--output-dir', default='validation_plots')
    parser.add_argument('--unsafe', action='store_true', help='Disable safety caps on commands')
    args = parser.parse_args()
    run_validation(args)


if __name__ == '__main__':
    main()