#!/usr/bin/env python3
"""
Rigorous model fidelity validation: compare the Gymnasium plant against the
reduced phase-domain internal model used by NMPC-MHE.

This script implements a phase‑cell‑synchronous comparison, correctly aligns
plant and model in phase, computes cell‑averaged forces, and uses the same
delay‑order selection and discretisation as the controller.

All approximations are isolated, and the validation is performed over a
specified range of operating points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

import gymnasium as gym

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from mpc_mhe_dompc import PhasePlant, build_model, cheb_lobatto_D, PhaseSchedule, auto_delay_order

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


def _cell_averaged_plant_force(plant_states, cell_start_idx, cell_end_idx):
    """Compute average axial force over a phase cell from plant samples."""
    if cell_end_idx <= cell_start_idx:
        return np.nan
    forces = []
    for k in range(cell_start_idx, cell_end_idx):
        f = plant_states['Fz'][k]
        if np.isfinite(f):
            forces.append(f)
    return float(np.mean(forces)) if forces else np.nan


def _ensure_1d(arr, expected_len):
    """Convert to 1D array of given length; raise error if mismatch."""
    arr = np.asarray(arr).flatten()
    if arr.shape[0] != expected_len:
        raise ValueError(f"Expected length {expected_len}, got {arr.shape[0]}")
    return arr


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

    # Use same delay-order selection as controller
    M = args.n_delay if args.n_delay is not None else auto_delay_order(pp, tol=1e-6)
    print(f"Delay order M = {M}")

    dtheta = args.dtheta_frac * p

    # Build do-mpc model
    model, xi = build_model(pp, M, args.chip_eps, args.n_quad)
    sim = Simulator(model)
    sim.settings.t_step = dtheta
    sim.settings.integration_tool = 'idas'
    sim.settings.t_rel = 1e-6
    sim.settings.t_abs = 1e-8
    sim.settings.supress_exit_warning = True

    # Phase schedule
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
    max_env_steps = int(total_time / env_dt)

    print("===== Validation Configuration =====")
    print(f"  env dt = {env_dt*1e3:.3f} ms, control dtheta = {dtheta:.4f} rad")
    print(f"  max steps: {max_env_steps} (≈ {max_env_steps*env_dt:.3f} s)")
    print(f"  retained modes: {K}, delay order: {M}")

    # ---- WARM-UP ----
    warmup_omega = pp.omega_min * 1.1
    warmup_ap = 0.05
    warmup_ap = np.clip(warmup_ap, pp.ap_min, pp.ap_max)
    warmup_cmd = np.array([warmup_omega, warmup_ap], dtype=float)
    low = np.asarray(plant._ctrl_low, dtype=float)
    high = np.asarray(plant._ctrl_high, dtype=float)
    act_warm = 2.0 * (warmup_cmd[:len(low)] - low) / (high - low) - 1.0
    act_warm = np.clip(act_warm, -1.0, 1.0)

    min_warmup_steps = int(np.ceil(5 * p / (warmup_omega * env_dt))) + 20
    warmup_steps = max(300, min_warmup_steps)
    print(f"Warming up for {warmup_steps} env steps ...")

    obs, info = env.reset(seed=args.seed)
    warmup_theta = []
    warmup_wc = []
    for _ in range(warmup_steps):
        obs, _, term, trunc, info = env.step(act_warm)
        if term or trunc:
            print(f"Warm-up terminated: {info.get('termination_reason')}")
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
        print("Warm-up failed. Aborting.")
        env.close()
        return

    warmup_theta = np.array(warmup_theta)
    warmup_wc = np.array(warmup_wc)
    warmup_theta_unwrapped = np.unwrap(warmup_theta)

    # Current state at end of warm-up
    xm_current = info.get('x_modal', np.zeros(2*pp.K_full))
    z_block_current = _deinterleave_modal(xm_current, pp.K_full)
    eta_cur = z_block_current[:pp.K_full][pp.sel]
    v_cur = z_block_current[pp.K_full:][pp.sel]
    theta_now = warmup_theta_unwrapped[-1]
    xc_now = float(info.get('cutter_x', plant.x0_cutter))
    yc_now = float(plant.y_cutter)

    # Seed delay state
    xi_full, _ = cheb_lobatto_D(M)
    xi_nodes = xi_full[1:]  # length M
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

    # ---- MAIN COMMAND SEQUENCE (piecewise constant per phase cell) ----
    if args.unsafe:
        omega_avg = args.omega_avg
        ap_avg = args.ap_avg
    else:
        omega_avg = args.omega_avg if args.omega_avg < 200 else 100.0
        ap_avg = args.ap_avg if args.ap_avg < 0.2 else 0.05

    estimated_cells = int(np.ceil(total_time * omega_avg / dtheta)) + 10
    cell_omega_seq = []
    cell_ap_seq = []
    for k in range(estimated_cells):
        t_cell = k * dtheta / omega_avg  # approximate time
        if args.command_type == 'sine':
            om = omega_avg + 0.2 * omega_avg * np.sin(2 * np.pi * args.sine_freq * t_cell)
            om = np.clip(om, pp.omega_min, pp.omega_max)
            ap = ap_avg + 0.2 * ap_avg * np.sin(2 * np.pi * args.sine_freq * t_cell)
            ap = np.clip(ap, pp.ap_min, pp.ap_max)
        else:
            om = omega_avg
            ap = ap_avg
        cell_omega_seq.append(om)
        cell_ap_seq.append(ap)

    # Data structures
    plant_states = {'times': [], 'theta': [], 'w': [], 'eta': [], 'Fz': []}
    cell_data = []

    # Initial plant state (t=0)
    plant_states['times'].append(0.0)
    plant_states['theta'].append(theta_now)
    plant_states['w'].append(info.get('w_sensor', np.zeros(pp.nS)))
    xm0 = info.get('x_modal', np.zeros(2*pp.K_full))
    plant_states['eta'].append(xm0[0::2][pp.sel])
    plant_states['Fz'].append(_extract_plant_force(info))

    # Loop over cells
    cell_idx = 0
    cell_start_theta = theta_now
    cell_start_time = 0.0
    cell_start_state = z0_model.copy()
    cell_plant_start_idx = 0
    plant_step_idx = 0

    while plant_step_idx < max_env_steps:
        if cell_idx >= len(cell_omega_seq):
            break
        omega_cmd = cell_omega_seq[cell_idx]
        ap_cmd = cell_ap_seq[cell_idx]

        u_phys = np.array([omega_cmd, ap_cmd], dtype=float)
        act = 2.0 * (u_phys[:len(low)] - low) / (high - low) - 1.0
        act = np.clip(act, -1.0, 1.0)

        target_theta = cell_start_theta + dtheta
        while plant_step_idx < max_env_steps:
            obs, _, term, trunc, info = env.step(act)
            plant_step_idx += 1
            t = plant_step_idx * env_dt

            theta_raw = float(info.get('spindle_phase_rad', 0.0))
            if plant_step_idx == 1:
                theta_unwrapped = theta_raw
            else:
                delta = theta_raw - plant_states['theta'][-1]
                if delta < -np.pi:
                    delta += 2*np.pi
                elif delta > np.pi:
                    delta -= 2*np.pi
                theta_unwrapped = plant_states['theta'][-1] + delta

            plant_states['times'].append(t)
            plant_states['theta'].append(theta_unwrapped)
            plant_states['w'].append(info.get('w_sensor', np.zeros(pp.nS)))
            xm = info.get('x_modal', np.zeros(2*pp.K_full))
            plant_states['eta'].append(xm[0::2][pp.sel])
            plant_states['Fz'].append(_extract_plant_force(info))

            if term or trunc:
                print(f"Episode terminated at step {plant_step_idx}: {info.get('termination_reason')}")
                break

            if theta_unwrapped >= target_theta:
                break

        # Step the model for the completed cell
        sched.set_now(cell_start_theta, xc_now, yc_now)
        model_u = np.array([omega_cmd, ap_cmd]).reshape(-1, 1)
        sim.x0 = cell_start_state.copy()
        sim.set_initial_guess()
        sim.make_step(model_u)
        z_pred = np.asarray(sim.x0.cat, dtype=np.float64).reshape(-1)

        # Compute model's cell-averaged force (using state at cell start)
        phi_cell, gvec_cell, svec_cell = sched.at(0)
        wc_start = float((phi_cell @ cell_start_state[:K]).item())
        wd = cell_start_state[K + K + M - 1]
        h = pp.ft * svec_cell + pp.kr * (wc_start - wd)
        eps = args.chip_eps
        root = np.sqrt(h**2 + eps**2)
        h_eff = 0.5 * (h + root)
        gate = 0.5 * (1.0 + h / root)
        Fz_model_cell = ap_cmd * np.sum(gvec_cell * (pp.Ka * h_eff + pp.Kae * gate))

        # Cell-average plant force
        cell_plant_end_idx = len(plant_states['theta'])
        Fz_plant_cell = _cell_averaged_plant_force(plant_states, cell_plant_start_idx, cell_plant_end_idx)

        # Store cell data (ensuring 1D arrays)
        w_start_1d = _ensure_1d(pp.Phi @ cell_start_state[:K], pp.nS)
        w_end_1d   = _ensure_1d(pp.Phi @ z_pred[:K], pp.nS)
        eta_start_1d = _ensure_1d(cell_start_state[:K], K)
        eta_end_1d   = _ensure_1d(z_pred[:K], K)

        cell_data.append({
            'cell_idx': cell_idx,
            'theta_start': cell_start_theta,
            'theta_end': target_theta,
            'time_start': cell_start_time,
            'time_end': plant_states['times'][-1],
            'omega_cmd': omega_cmd,
            'ap_cmd': ap_cmd,
            'w_start': w_start_1d,
            'w_end': w_end_1d,
            'eta_start': eta_start_1d,
            'eta_end': eta_end_1d,
            'Fz_plant_cell': Fz_plant_cell,
            'Fz_model_cell': Fz_model_cell,
            'plant_samples': list(range(cell_plant_start_idx, cell_plant_end_idx)),
        })

        if term or trunc:
            break

        # Prepare next cell
        cell_idx += 1
        cell_start_theta = target_theta
        cell_start_time = plant_states['times'][-1]
        cell_start_state = z_pred.copy()
        cell_plant_start_idx = cell_plant_end_idx
        xc_now = float(info.get('cutter_x', plant.x0_cutter))
        yc_now = float(plant.y_cutter)

    env.close()

    # --- Plotting ---
    if not cell_data:
        print("No cells were completed. Aborting.")
        return

    # Build model time series for interpolation (all 1D now)
    model_times = [d['time_start'] for d in cell_data]
    model_w_vals = [d['w_start'] for d in cell_data]
    model_eta_vals = [d['eta_start'] for d in cell_data]
    # Add final state
    model_times.append(cell_data[-1]['time_end'])
    model_w_vals.append(cell_data[-1]['w_end'])
    model_eta_vals.append(cell_data[-1]['eta_end'])

    # Stack into 2D arrays (rows = time, columns = sensor/mode)
    model_w_vals = np.vstack(model_w_vals)      # (N, nS)
    model_eta_vals = np.vstack(model_eta_vals)  # (N, K)

    plant_times = np.array(plant_states['times'])
    plant_w = np.vstack(plant_states['w']) if plant_states['w'] else np.empty((0, pp.nS))
    plant_eta = np.vstack(plant_states['eta']) if plant_states['eta'] else np.empty((0, K))
    plant_Fz = np.array(plant_states['Fz'])

    # Interpolate model to plant time grid
    if len(model_times) > 1:
        f_w_model = interp1d(model_times, model_w_vals, kind='linear', fill_value='extrapolate', axis=0)
        model_w_interp = f_w_model(plant_times)
        f_eta_model = interp1d(model_times, model_eta_vals, kind='linear', fill_value='extrapolate', axis=0)
        model_eta_interp = f_eta_model(plant_times)
    else:
        model_w_interp = np.full_like(plant_w, np.nan)
        model_eta_interp = np.full_like(plant_eta, np.nan)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(plant_times, plant_w[:, 0]*1e6, label='Plant (sensor 0)')
    ax.plot(plant_times, model_w_interp[:, 0]*1e6, 'r--', label='Model (interp)')
    ax.set_ylabel('Displacement [µm]')
    ax.set_title('Sensor 0 displacement')
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot(plant_times, plant_eta[:, 0]*1e6, label='Plant')
    ax.plot(plant_times, model_eta_interp[:, 0]*1e6, 'r--', label='Model')
    ax.set_ylabel('Modal disp [µm]')
    ax.set_title('Mode 1 displacement')
    ax.legend(); ax.grid(True)

    ax = axes[2]
    ax.plot(plant_times, plant_Fz, label='Plant instantaneous')
    # Cell-averaged forces
    cell_centers = [(d['time_start'] + d['time_end'])/2 for d in cell_data]
    plant_Fz_cell = [d['Fz_plant_cell'] for d in cell_data]
    model_Fz_cell = [d['Fz_model_cell'] for d in cell_data]
    ax.step(cell_centers, plant_Fz_cell, where='mid', label='Plant cell-averaged')
    ax.plot(cell_centers, model_Fz_cell, 'ro', label='Model cell force')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Axial force [N]')
    ax.set_title('Cutting force')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'model_validation_corrected.png', dpi=150)
    print(f"Saved plot to {out_dir / 'model_validation_corrected.png'}")

    # ---- METRICS ----
    valid_cells = [d for d in cell_data if np.isfinite(d['Fz_plant_cell']) and np.isfinite(d['Fz_model_cell'])]
    if valid_cells:
        Fz_plant_vals = np.array([d['Fz_plant_cell'] for d in valid_cells])
        Fz_model_vals = np.array([d['Fz_model_cell'] for d in valid_cells])
        rmse_Fz = np.sqrt(np.mean((Fz_plant_vals - Fz_model_vals)**2))
        range_Fz = np.max(Fz_plant_vals) - np.min(Fz_plant_vals)
        nrms_Fz = rmse_Fz / (range_Fz + 1e-12) * 100
        corr_Fz = np.corrcoef(Fz_plant_vals, Fz_model_vals)[0, 1]
        print(f"Force (cell-averaged): RMSE = {rmse_Fz:.2f} N, NRMSE = {nrms_Fz:.2f}%, corr = {corr_Fz:.4f}")

        # Displacement at cell ends
        w_plant_end = np.array([pp.Phi @ plant_states['eta'][d['plant_samples'][-1]] if d['plant_samples'] else np.nan for d in valid_cells])
        w_model_end = np.array([d['w_end'] for d in valid_cells])
        valid_w = ~np.isnan(w_plant_end) & ~np.isnan(w_model_end)
        if np.any(valid_w):
            w_plant_end = w_plant_end[valid_w]
            w_model_end = w_model_end[valid_w]
            rmse_w = np.sqrt(np.mean((w_plant_end - w_model_end)**2))
            range_w = np.max(w_plant_end) - np.min(w_plant_end)
            nrms_w = rmse_w / (range_w + 1e-12) * 100
            corr_w = np.corrcoef(w_plant_end, w_model_end)[0, 1]
            print(f"Displacement (cell end): RMSE = {rmse_w*1e6:.2e} µm, NRMSE = {nrms_w:.2f}%, corr = {corr_w:.4f}")

    print(f"\nValidation complete. Results saved in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Rigorous model validation.')
    parser.add_argument('--env-id', default='CustomODEPlate-v0')
    parser.add_argument('--reward', default='dense')
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--n-substeps', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=100000)   # <-- ADDED
    parser.add_argument('--duration', type=float, default=0.2)
    parser.add_argument('--omega-avg', type=float, default=100.0)
    parser.add_argument('--ap-avg', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-modes', type=int, default=3)
    parser.add_argument('--n-delay', type=int, default=None)
    parser.add_argument('--dtheta-frac', type=float, default=0.1)
    parser.add_argument('--chip-eps', type=float, default=2e-3)
    parser.add_argument('--n-quad', type=int, default=2)
    parser.add_argument('--sharp', type=float, default=0.0)
    parser.add_argument('--command-type', choices=['step', 'sine'], default='step')
    parser.add_argument('--sine-freq', type=float, default=5.0)
    parser.add_argument('--sine-amp', type=float, default=0.2)
    parser.add_argument('--sine-amp-ap', type=float, default=0.2)
    parser.add_argument('--output-dir', default='validation_plots')
    parser.add_argument('--unsafe', action='store_true', help='Disable safety caps on commands')
    args = parser.parse_args()
    run_validation(args)


if __name__ == '__main__':
    main()