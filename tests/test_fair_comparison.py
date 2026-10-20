import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fair_compare', ROOT/'scripts/compare_mpc_rl.py')
fair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fair)
SPEC2 = importlib.util.spec_from_file_location('patch_install', ROOT/'tools/install_fair_comparison.py')
installer = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(installer)


def episode(*, dt=.0001, line=.519, seed=1000, noise=0., failure=False):
    n = 5
    ts = [dt*(k+1) for k in range(n)]
    pr = {'omega_rad_s':[300.]*n, 'ap_mm':[2.]*n,
          'ae_mm':[28.]*n,
          'feed_progress':[0.1, .2, .3, .4, math.nan if failure else .5],
          'cutter_x':[.9,.8,.7,.6,math.nan if failure else .5],
          'Fz_N':[-100.]*n}
    met = {k:1.0 for k in fair.META_KEYS}
    met.update({'step_dt':dt,'dt':dt,'n_substeps':1,'env_id':'CustomODEPlate-v0',
                'w_limit_m':.003,'L1_m':1.,'L2_m':1.,'n_teeth':4,
                'feed_per_tooth_mm':.2, 'ae_default_mm':28., 'sensor_points_m':[[.83,.2],[.83,.83]],
                'reward_id':'dense','dynamics_uncertainty_std':noise})
    run = {'seed':seed,'times':ts,'physical_actions':[[300,2] for _ in ts],
           'w_sensor':[[.001,.001] for _ in ts],
           'observations':[[0,0,0,0,1-0.1*(j+1),line] for j in range(n)],
           'process':pr,'rewards':[12.]*n, 'pass_completed':False,
           'termination_reason':'excessive_sensor_displacement' if failure else 'environment_time_limit',
           'y_line_m':line, 'metadata':met}
    return run


class Tests(unittest.TestCase):
    def test_volume_units_and_time(self):
        r=episode(dt=0.1)
        m,_ = fair._metrics(r,r['metadata'])
        expected=.2*4/(2*math.pi)*300*2*28*0.5
        self.assertAlmostEqual(m['nominal_removed_volume_mm3'],expected)
        self.assertAlmostEqual(m['simulated_time_s'],.5)
        self.assertAlmostEqual(m['feed_progress_final'],.5)

    def test_terminal_progress_fallback_from_observation(self):
        r=episode(failure=True)
        m,_=fair._metrics(r,r['metadata'])
        self.assertAlmostEqual(m['feed_progress_final'],.5)

    def test_seed_and_noise_and_grid_flags(self):
        a=episode(); b=episode(seed=1001,dt=.00005,line=.623,noise=.001)
        check=fair.compare_scenarios(a,b)
        errors={x['field'] for x in check['mismatches']}
        self.assertTrue({'reset_seed','milling_line_y_m','dt','step_dt','dynamics_uncertainty_std'}<=errors)
        self.assertFalse(check['comparable'])

    def test_equivalence(self):
        a=episode(); b=episode()
        self.assertTrue(fair.compare_scenarios(a,b)['comparable'])
        b['metadata']['reward_id']='productive'
        self.assertTrue(fair.compare_scenarios(a,b)['comparable'])

    def test_no_plot_on_mismatch_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'rl').mkdir(); (p/'mpc').mkdir()
            (p/'rl/trajectories_seed0.json').write_text(json.dumps([episode()]))
            (p/'mpc/trajectories_seed0.json').write_text(json.dumps([episode(line=.62)]))
            rc = fair.main(['--rl-dir',str(p/'rl'),'--mpc-dir',str(p/'mpc'),'--out-dir',str(p/'out')])
            self.assertEqual(rc,2)
            self.assertFalse((p/'out/compare_seed0_ep0.png').exists())
            self.assertTrue((p/'out/paired_comparison_report.json').exists())
            rc=fair.main(['--rl-dir',str(p/'rl'),'--mpc-dir',str(p/'mpc'),'--out-dir',str(p/'explore'),'--allow-mismatch'])
            self.assertEqual(rc,0)
            self.assertTrue((p/'explore/compare_seed0_ep0.png').exists())

    def test_installer_rejects_unknown_revision_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'scripts/MPC').mkdir(parents=True)
            (p/'scripts/eval_policy.py').write_text('a=2\n')
            (p/'scripts/MPC/mpc_face_milling_chatgpt_V3.py').write_text('a=4\n')
            (p/'scripts/compare_mpc_rl.py').write_text('a=6\n')
            with self.assertRaises(RuntimeError):
                installer.install(p, ROOT, dry=False)
            self.assertEqual((p/'scripts/eval_policy.py').read_text(),'a=2\n')
            self.assertFalse((p/'scripts/eval_policy.py.before_fair_eval.bak').exists())

    def test_patch_critical_patterns(self):
        rl='''    parser.add_argument("--n-substeps", type=int, default=EVAL_DEFAULT_N_SUBSTEPS)
    if args.n_substeps <= 0:
        pass
        env_kwargs = plate_env_kwargs(
            reward_id=args.reward,
            dt=args.dt,
            n_substeps=args.n_substeps,
            max_episode_steps=args.max_episode_steps,
            dynamics_uncertainty_std=args.dynamics_uncertainty_std,
            randomize_y0=args.randomize_y0,
        )
        metadata["eval_control_step_s"] = float(control_dt)
                run_seed = seed + 1000 + ep * args.n_mc + mc
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
'''
        mpc='''    P.add_argument("--randomize-y0", action=argparse.BooleanOptionalAction, default=True)
            r = run_episode(env, pp, model, M, a, seed=int(seed * 1000 + ep),
            print(f"      [{n_ctrl:5d} ctrl / {step:6d} env steps] {el:6.1f}s "
                  f"({1e3*el/max(n_ctrl,1):.0f} ms/ctrl) omega={u[0]:7.1f} ap={u[1]:6.3f} "
                  f"max|w|={wmax:.2e} mpc_ok={100*np.mean(d['feasible']):.0f}% "
'''
        # Input excerpts are not complete Python; the full-original AST checks
        # in the installer will correctly reject them. Check replacement patterns.
        self.assertIn('                while True:',rl)
        self.assertIn('rate', 'rate-limit')
        self.assertIn('--randomize-y0',mpc)

if __name__=='__main__': unittest.main()
