"""Install minimally invasive paired RL/MPC evaluation changes.

Only affects scripts/eval_policy.py, scripts/MPC/mpc_face_milling_chatgpt_V3.py,
and scripts/compare_mpc_rl.py. No training/plant/controller objective edits.
Always backups before writing and never silently applies an unknown source revision.
"""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

PATCH_TAG = 'FAIR_EVAL_2026_10'


def replace_once(src: str, old: str, new: str, name: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f'{name}: expected exactly one known source pattern; found {count}. No modifications applied.')
    return src.replace(old, new, 1)


def modify_rl(s: str) -> str:
    s = replace_once(s,
        '    parser.add_argument("--n-substeps", type=int, default=EVAL_DEFAULT_N_SUBSTEPS)',
        '    parser.add_argument("--n-substeps", type=int, default=EVAL_DEFAULT_N_SUBSTEPS)\n'
        '    parser.add_argument("--policy-hold-env-steps", type=int, default=1,\n'
        '                        help="Hold one RL policy action for this many environment steps; physical safety is checked every environment step.")\n'
        '    parser.add_argument("--reset-seed-offset", type=int, default=1000,\n'
        '                        help="Offset used for actual environment reset seed (default preserves historical RL evaluation).")\n'
        '    parser.add_argument("--plant-override", default=None,\n'
        '                        help="JSON object limited to initial_eta_std, initial_etad_std, process_noise_std, obs_noise_std.")',
        'RL CLI insertion')
    s = replace_once(s, '    if args.n_substeps <= 0:\n',
        '    if args.policy_hold_env_steps <= 0:\n'
        '        raise SystemExit("--policy-hold-env-steps must be positive.")\n'
        '    if args.n_substeps <= 0:\n', 'RL validate hold')
    old = '''        env_kwargs = plate_env_kwargs(
            reward_id=args.reward,
            dt=args.dt,
            n_substeps=args.n_substeps,
            max_episode_steps=args.max_episode_steps,
            dynamics_uncertainty_std=args.dynamics_uncertainty_std,
            randomize_y0=args.randomize_y0,
        )'''
    new = old + '''
        if args.plant_override:
            import json as _fair_json
            overrides = _fair_json.loads(args.plant_override)
            allowed = {"initial_eta_std", "initial_etad_std", "process_noise_std", "obs_noise_std"}
            if not isinstance(overrides, dict) or not set(overrides).issubset(allowed):
                raise ValueError(f"Fair evaluation --plant-override accepts only {sorted(allowed)}")
            env_kwargs.update(overrides)'''
    s = replace_once(s, old, new, 'RL plant overrides')
    s = replace_once(s,
        '        metadata["eval_control_step_s"] = float(control_dt)',
        '        metadata["eval_control_step_s"] = float(control_dt * args.policy_hold_env_steps)\n'
        '        metadata["eval_environment_step_s"] = float(control_dt)\n'
        '        metadata["eval_policy_hold_env_steps"] = int(args.policy_hold_env_steps)\n'
        '        metadata["eval_reset_seed_offset"] = int(args.reset_seed_offset)',
        'RL timing metadata')
    s = replace_once(s, '''        plant = env.unwrapped.plant
        tau_min = 2.0 * np.pi''', '''        plant = env.unwrapped.plant
        # Record the complete stochastic/initialization protocol for paired audit.
        metadata["initial_eta_std"] = float(plant.initial_eta_std)
        metadata["initial_etad_std"] = float(plant.initial_etad_std)
        metadata["process_noise_std"] = float(env.unwrapped.process_noise_std)
        metadata["obs_noise_std"] = float(env.unwrapped.obs_noise_std)
        tau_min = 2.0 * np.pi''', 'RL full uncertainty metadata')
    s = replace_once(s,
        '                run_seed = seed + 1000 + ep * args.n_mc + mc',
        '                run_seed = seed + args.reset_seed_offset + ep * args.n_mc + mc',
        'RL reset seed')
    old = '''                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)'''
    new = '''                # Policy decisions retain the training-time cadence; the plant is
                # stepped and safety-tested at every (possibly finer) env interval.
                fair_eval_step = 0
                while True:
                    if fair_eval_step % args.policy_hold_env_steps == 0:
                        action, _ = model.predict(obs, deterministic=True)
                    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
                    fair_eval_step += 1'''
    s = replace_once(s, old, new, 'RL action zero-order hold')
    ast.parse(s)
    return s


def modify_mpc(s: str) -> str:
    old = '''    P.add_argument("--randomize-y0", action=argparse.BooleanOptionalAction, default=True)'''
    new = old + '''
    P.add_argument("--reset-seed-offset", type=int, default=None,
                   help="Optional paired-evaluation seed offset. Without this flag retain historical MPC episode seeds.")'''
    s = replace_once(s, old, new, 'MPC reset seed CLI')
    old = '''            r = run_episode(env, pp, model, M, a, seed=int(seed * 1000 + ep),'''
    new = '''            actual_seed = int(seed * 1000 + ep) if a.reset_seed_offset is None else int(seed + a.reset_seed_offset + ep)
                r = run_episode(env, pp, model, M, a, seed=actual_seed,'''
    s = replace_once(s, old, new, 'MPC reset seed calculation')
    old = '''            print(f"      [{n_ctrl:5d} ctrl / {step:6d} env steps] {el:6.1f}s "
                  f"({1e3*el/max(n_ctrl,1):.0f} ms/ctrl) omega={u[0]:7.1f} ap={u[1]:6.3f} "
                  f"max|w|={wmax:.2e} mpc_ok={100*np.mean(d['feasible']):.0f}% "'''
    new = '''            simulated_seconds = float(logs["times"][-1]) if logs["times"] else float(step * pp.env_dt)
            print(f"      [{n_ctrl:5d} ctrl / {step:6d} env steps] wall={el:6.1f}s sim={simulated_seconds:.3f}s "
                  f"({1e3*el/max(n_ctrl,1):.0f} wall-ms/ctrl) omega={u[0]:7.1f} ap={u[1]:6.3f} "
                  f"peak|w|_so_far={wmax:.2e} mpc_ok={100*np.mean(d['feasible']):.0f}% "'''
    s = replace_once(s, old, new, 'MPC explicit physical/wall progress')
    s = replace_once(s, '    meta = plant_plot_metadata(env)',
        '    meta = plant_plot_metadata(env)\n'
        '    meta["initial_eta_std"] = float(pp.plant.initial_eta_std)\n'
        '    meta["initial_etad_std"] = float(pp.plant.initial_etad_std)\n'
        '    meta["process_noise_std"] = float(env.unwrapped.process_noise_std)\n'
        '    meta["obs_noise_std"] = float(env.unwrapped.obs_noise_std)',
        'MPC full uncertainty metadata')
    ast.parse(s)
    return s


def install(repo: Path, package: Path, dry: bool):
    paths = [repo/'scripts/eval_policy.py', repo/'scripts/MPC/mpc_face_milling_chatgpt_V3.py', repo/'scripts/compare_mpc_rl.py']
    if not all(x.exists() for x in paths):
        raise FileNotFoundError('Run from a Custom_RL repository root containing scripts/eval_policy.py and scripts/MPC/mpc_face_milling_chatgpt_V3.py')
    (rl, mpc, comp) = paths
    src_rl, src_mpc = rl.read_text(encoding='utf-8'), mpc.read_text(encoding='utf-8')
    if PATCH_TAG in src_rl or PATCH_TAG in src_mpc:
        raise RuntimeError('Already patched; not applying changes twice.')
    revised = [modify_rl(src_rl), modify_mpc(src_mpc)]
    revised = [s + '\n# ' + PATCH_TAG + '\n' for s in revised]
    comparison_src = (package/'scripts/compare_mpc_rl.py').read_text(encoding='utf-8')
    ast.parse(comparison_src)
    if dry:
        print('Dry run passed: all expected source patterns present; Python parses cleanly.')
        return
    # No writes until all patterns have been checked. Backups protect local edits.
    for p in paths:
        backup = p.with_suffix(p.suffix+'.before_fair_eval.bak')
        if backup.exists():
            raise FileExistsError(f'{backup} already exists; remove/move it after reviewing your previous backup')
    for p in paths:
        shutil.copy2(p, p.with_suffix(p.suffix+'.before_fair_eval.bak'))
    for p, content in zip(paths, (*revised, comparison_src)):
        p.write_text(content, encoding='utf-8')
    print('Installed: '+', '.join(str(p.relative_to(repo)) for p in paths))
    print('Backups: *.before_fair_eval.bak')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.', help='Local Custom_RL repository root')
    parser.add_argument('--dry-run', action='store_true')
    a = parser.parse_args()
    install(Path(a.repo).resolve(), Path(__file__).resolve().parents[1], a.dry_run)


if __name__ == '__main__':
    main()
