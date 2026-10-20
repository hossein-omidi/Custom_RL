"""Research-oriented RL / MPC face-milling comparison.

This script never equates raw sums of per-environment-step rewards to economic
performance. Both records must correspond to comparable physical scenarios.
Run with --allow-mismatch for exploratory (explicitly non-comparable) plots.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PASS_REASONS = {'pass_completed', 'pass_completed_90percent'}
META_KEYS = (
    'env_id', 'dt', 'n_substeps', 'step_dt', 'w_limit_m',
    'sensor_points_m', 'L1_m', 'L2_m', 'h_m', 'E_Pa', 'nu',
    'rho_areal_kg_m2', 'modal_damping_ratio', 'n_teeth',
    'feed_per_tooth_mm', 'ae_default_mm', 'D_mm',
    'gamma_L_deg', 'gamma_r_deg', 'gamma_a_deg',
    'Kt_N_mm2', 'Kr_N_mm2', 'Ka_N_mm2',
    'Kte_N_mm', 'Kre_N_mm', 'Kae_N_mm',
    'mode_clamped_axis', 'milling_mode', 'use_process_damping',
    'force_projection_mode', 'dynamics_uncertainty_std',
    'physical_action_low', 'physical_action_high',
    'initial_eta_std', 'initial_etad_std', 'process_noise_std', 'obs_noise_std',
)


def _numeric(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _same(a, b) -> bool:
    try:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if x.shape == y.shape and x.size and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            return bool(np.allclose(x, y, rtol=1e-9, atol=1e-12))
    except (TypeError, ValueError):
        pass
    return a == b


def _primary(ep):
    runs = ep.get('runs')
    return runs[0] if isinstance(runs, list) and runs else ep


def _load(directory: Path) -> dict[int, list[dict]]:
    files = {}
    for p in sorted(directory.glob('trajectories_seed*.json')):
        try:
            seed = int(p.stem.removeprefix('trajectories_seed'))
        except ValueError:
            continue
        data = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f'{p}: expected list of episode objects')
        files[seed] = data
    return files


def _series(a, key, n=None):
    arr = np.asarray(a, dtype=float)
    if arr.ndim > 1:
        raise ValueError(f'{key}: expected 1-dimensional signal')
    if n is None or len(arr) == n:
        return arr
    out = np.full(n, np.nan)
    out[:min(n, len(arr))] = arr[:min(n, len(arr))]
    return out


def _get_times(ep):
    t = _series(ep.get('times', []), 'times')
    if len(t) and (not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0) or t[0] <= 0):
        raise ValueError('times must be finite, strictly increasing, and positive')
    return t


def _get_physical_actions(ep, t, key):
    n = len(t)
    raw = np.asarray(ep.get('physical_actions', []), dtype=float)
    if raw.ndim == 2 and raw.shape[0] == n and raw.shape[1] >= 2:
        return raw
    # Legacy diagnostics use samples at the environment rather than controller rate.
    process = ep.get('process', {}) or {}
    omega = _series(process.get('omega_rad_s', []), 'omega_rad_s', n)
    if not np.any(np.isfinite(omega)):
        omega = _series(process.get('omega_rpm', []), 'omega_rpm', n)*2*np.pi/60
    ap = _series(process.get('ap_mm', []), 'ap_mm', n)
    if len(omega) != n or len(ap) != n or not (np.all(np.isfinite(omega)) and np.all(np.isfinite(ap))):
        raise ValueError(f'{key}: no complete per-step physical actions')
    return np.column_stack((omega, ap))


def _feed(ep, metadata, n):
    pr = ep.get('process', {}) or {}
    out = _series(pr.get('feed_progress', []), 'feed_progress', n)
    L1 = _numeric(metadata.get('L1_m', 1.0))
    # Position fallback works for terminal steps for which plant.termination returns early.
    cutter_x = _series(pr.get('cutter_x', []), 'cutter_x', n)
    obs = np.asarray(ep.get('observations', []), dtype=float)
    xnorm = (obs[:, -2] if obs.ndim == 2 and obs.shape[0] == n and obs.shape[1] >= 2 else np.full(n, np.nan))
    if np.isfinite(L1) and L1 > 0:
        inferred = np.where(np.isfinite(cutter_x), 1-cutter_x/L1, 1-xnorm)
        out = np.where(np.isfinite(out), out, inferred)
    return out


def _metrics(ep, metadata):
    ep = _primary(ep)
    t = _get_times(ep)
    n = len(t)
    if not n:
        raise ValueError('no samples')
    dt = np.diff(np.concatenate(([0.0], t)))
    w = np.asarray(ep.get('w_sensor', []), dtype=float)
    if w.ndim != 2 or w.shape[0] != n or not np.all(np.isfinite(w)):
        raise ValueError('no complete w_sensor data')
    wpeak = np.max(np.abs(w), axis=1)
    actions = _get_physical_actions(ep, t, 'actions')
    omega, ap = actions[:, 0], actions[:, 1]
    if not np.all(np.isfinite(actions)):
        raise ValueError('non-finite physical actions')
    feed = _feed(ep, metadata, n)
    ft = _numeric(metadata.get('feed_per_tooth_mm'))
    N = _numeric(metadata.get('n_teeth'))
    ae = _numeric(metadata.get('ae_default_mm'))
    if (ep.get('process') or {}).get('ae_mm'):
        ae_series = _series(ep['process']['ae_mm'], 'ae_mm', n)
        ae_series = np.where(np.isfinite(ae_series), ae_series, ae)
    else:
        ae_series = np.full(n, ae)
    # The physical nominal removed-volume proxy is based on table feed x axial depth x radial width.
    if all(math.isfinite(v) and v > 0 for v in (ft, N)) and np.all(np.isfinite(ae_series)):
        nominal_mrr = (ft * N / (2*np.pi)) * omega * np.maximum(ap, 0) * np.maximum(ae_series, 0) # mm^3 / s
        increment = nominal_mrr * dt
        nominal_volume = float(np.sum(increment))
        cumulative_volume = np.cumsum(increment)
    else:
        nominal_volume = math.nan
        cumulative_volume = np.full(n, np.nan)
    fp_valid = feed[np.isfinite(feed)]
    reason = ep.get('termination_reason', (ep.get('summary') or {}).get('termination_reason'))
    passed = bool(ep.get('pass_completed', reason in PASS_REASONS))
    limit = _numeric(metadata.get('w_limit_m', metadata.get('w_limit')))
    raw_rewards = np.asarray(ep.get('rewards', []), dtype=float)
    return {
        'pass_completed': passed, 'termination_reason': reason,
        'simulated_time_s': float(t[-1]), 'n_environment_steps': n,
        'nominal_removed_volume_mm3': nominal_volume,
        'mean_nominal_mrr_mm3_s': (nominal_volume / t[-1] if t[-1] > 0 else math.nan),
        'peak_abs_sensor_displacement_m': float(np.max(wpeak)),
        'time_weighted_sensor_rms_m': float(np.sqrt(np.sum(dt*np.mean(w*w, axis=1)) / np.sum(dt))),
        'sensor_limit_m': limit,
        'sensor_limit_exceeded': bool(np.max(wpeak) > limit) if math.isfinite(limit) else None,
        'feed_progress_final': float(fp_valid[-1]) if fp_valid.size else math.nan,
        'total_variation_omega_rad_s': float(np.sum(np.abs(np.diff(omega)))),
        'total_variation_ap_mm': float(np.sum(np.abs(np.diff(ap)))),
        'omega_mean_rpm_time_weighted': float(np.sum(dt*omega)/np.sum(dt)*60/(2*np.pi)),
        'ap_mean_mm_time_weighted': float(np.sum(dt*ap)/np.sum(dt)),
        # Provided for inspection only; never interpret raw reward as comparable across discretizations.
        'raw_reward_sum_NOT_COMPARABLE': float(np.sum(raw_rewards)) if raw_rewards.size else math.nan,
    }, {'t': t, 'w': wpeak, 'omega': omega*60/(2*np.pi), 'ap': ap,
        'feed': feed, 'force': _series((ep.get('process') or {}).get('Fz_N', []), 'Fz_N', n),
        'cumvol': cumulative_volume}


def _scenario(ep, meta):
    ep = _primary(ep)
    return {
        'milling_line_y_m': ep.get('y_line_m', (ep.get('reset_info') or {}).get('y_cutter')),
        'reset_seed': ep.get('seed', ep.get('scenario_seed')),
        'reward_id': meta.get('reward_id'),
    }


def compare_scenarios(rl_ep, mpc_ep):
    er, em = _primary(rl_ep), _primary(mpc_ep)
    r = er.get('metadata') or rl_ep.get('metadata') or {}
    m = em.get('metadata') or mpc_ep.get('metadata') or {}
    sr, sm = _scenario(rl_ep, r), _scenario(mpc_ep, m)
    problems, unknown = [], []
    for key in META_KEYS:
        rv, mv = r.get(key), m.get(key)
        if rv is None or mv is None:
            unknown.append(f'metadata.{key}')
        elif not _same(rv, mv):
            problems.append({'field': key, 'rl': rv, 'mpc': mv})
    for key in ('milling_line_y_m', 'reset_seed'):
        rv, mv = sr.get(key), sm.get(key)
        if rv is None or mv is None or (isinstance(rv, float) and not math.isfinite(rv)) or (isinstance(mv, float) and not math.isfinite(mv)):
            unknown.append(key)
        elif not _same(rv, mv):
            problems.append({'field': key, 'rl': rv, 'mpc': mv})
    rr, mr = sr.get('reward_id'), sm.get('reward_id')
    if rr is None or mr is None:
        unknown.append('reward_id')
    elif rr != mr and set((rr, mr)) != {'dense', 'productive'}:
        problems.append({'field': 'reward_id', 'rl': rr, 'mpc': mr})
    # Validate controller labels and physical time grids, not equality of their whole trajectories.
    # Different physical durations are expected if cutting rates differ.
    matched = not problems and not unknown
    return {'comparable': matched, 'mismatches': problems, 'unknown_metadata': unknown,
            'rl_scenario': sr, 'mpc_scenario': sm,
            'note': 'Identical time step / plant parameters and scenario are necessary, but not sufficient, for independent stochastic-path pairing.'}


def _plot(rl, mpc, out: Path, comparable: bool, label: str, limit: float):
    titles = ('Maximum absolute sensor displacement (across sensors) [m]', 'Spindle speed [rpm]',
              'Axial depth of cut [mm]', 'Axial cutting force [N]',
              'Cumulative nominal removed volume [mm³]', 'Feed progress')
    keys = ('w', 'omega', 'ap', 'force', 'cumvol', 'feed')
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for ax, key, title in zip(axes.flatten(), keys, titles):
        for name, sig, color, ls in (('RL', rl, '#1f77b4', '-'), ('MPC', mpc, '#d62728', '--')):
            x, y = sig['t'], sig[key]
            ok = np.isfinite(x) & np.isfinite(y)
            if np.any(ok):
                ax.plot(x[ok], y[ok], color=color, ls=ls, linewidth=1.3, label=name)
        if key == 'w' and math.isfinite(limit):
            ax.axhline(limit, color='black', ls=':', linewidth=1, label=f'Limit {limit*1000:g} mm')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Physical machining time [s]')
        ax.grid(alpha=0.25)
        if key == 'w':
            ax.legend()
    flag = 'MATCHED METADATA' if comparable else 'EXPLORATORY ONLY — SCENARIO MISMATCH / UNKNOWN'
    fig.suptitle(f'{label} | {flag}', fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _clean_json(v):
    if isinstance(v, dict):
        return {str(k): _clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_json(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def main(argv=None):
    p = argparse.ArgumentParser(description='Validated physical RL vs MPC performance comparison')
    p.add_argument('--rl-dir', default='eval_trajectories')
    p.add_argument('--mpc-dir', default='mpc_phase_trajectories')
    p.add_argument('--out-dir', default='plots/mpc_vs_rl')
    p.add_argument('--seeds', type=int, nargs='*', default=None)
    p.add_argument('--strict', dest='strict', action='store_true', default=True, help='Refuse misleading plots if scenarios cannot be verified')
    p.add_argument('--allow-mismatch', dest='strict', action='store_false', help='Produce explicitly exploratory plots for unpaired data')
    a = p.parse_args(argv)
    rl, mpc = _load(Path(a.rl_dir)), _load(Path(a.mpc_dir))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {'method': 'physical-time-integrated nominal volume and sensor safety; raw reward sums never ranked', 'episodes': []}
    seeds = sorted(set(rl) | set(mpc)) if not a.seeds else a.seeds
    if not seeds:
        print('No saved trajectories found'); return 2
    flagged = False
    for seed in seeds:
        for i in range(max(len(rl.get(seed, [])), len(mpc.get(seed, [])))):
            item = {'seed': seed, 'episode': i}
            re = rl.get(seed, [])[i] if i < len(rl.get(seed, [])) else None
            me = mpc.get(seed, [])[i] if i < len(mpc.get(seed, [])) else None
            if re is None or me is None:
                item['status'] = 'MISSING_RL_OR_MPC'
                flagged = True
            else:
                try:
                    validity = compare_scenarios(re, me)
                    item.update(validity)
                    item['rl_metrics'], rs = _metrics(_primary(re), re.get('metadata') or _primary(re).get('metadata') or {})
                    item['mpc_metrics'], ms = _metrics(_primary(me), me.get('metadata') or _primary(me).get('metadata') or {})
                    item['status'] = 'MATCHED_METADATA' if validity['comparable'] else 'UNMATCHED'
                    if not validity['comparable']:
                        flagged = True
                    if validity['comparable'] or not a.strict:
                        _plot(rs, ms, out/f'compare_seed{seed}_ep{i}.png', validity['comparable'],
                              f'Seed {seed} episode {i}', item['rl_metrics']['sensor_limit_m'])
                except (ValueError, KeyError, IndexError) as exc:
                    item['status'] = 'INVALID_DATA'; item['error'] = str(exc)
                    flagged = True
            report['episodes'].append(item)
            print(f"seed={seed} ep={i} status={item['status']}")
            if item.get('mismatches'):
                for m in item['mismatches']:
                    print(f"  MISMATCH {m['field']}: RL={m['rl']!r}; MPC={m['mpc']!r}")
            if item.get('unknown_metadata'):
                print('  UNKNOWN metadata: '+', '.join(item['unknown_metadata']))
            for key in ('rl_metrics', 'mpc_metrics'):
                if key in item:
                    v = item[key]
                    print(f"  {key}: pass={v['pass_completed']}, feed={v['feed_progress_final']:.3f}, "
                          f"peak={v['peak_abs_sensor_displacement_m']*1000:.3f} mm, "
                          f"volume_proxy={v['nominal_removed_volume_mm3']:.1f} mm3, "
                          f"duration={v['simulated_time_s']:.3f} s")
    (out/'paired_comparison_report.json').write_text(json.dumps(_clean_json(report), indent=2, allow_nan=False), encoding='utf-8')
    print(f"Report: {out/'paired_comparison_report.json'}")
    if flagged and a.strict:
        print('FAIL: comparison not certified. Fix run setup or use --allow-mismatch for visual inspection only.')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
