"""Dependency-light synthetic contract tests; no milling-physics validation."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use('Agg')

ROOT = pathlib.Path(__file__).resolve().parents[1]


def install_stubs():
    gym = types.ModuleType('gymnasium')
    gym.Env = object
    gym.make = lambda name, **kw: FakeEnv(**kw)
    gym.register = lambda *a, **kw: None
    sys.modules['gymnasium'] = gym
    sb3 = types.ModuleType('stable_baselines3')
    class PPO:
        @staticmethod
        def load(path, device='cpu'):
            return FakeModel()
    sb3.PPO = PPO
    sys.modules['stable_baselines3'] = sb3
    custom = types.ModuleType('custom_rl')
    custom.__path__ = []
    custom.DEFAULT_MODEL_DIR = 'models'
    custom.DEFAULT_TRAJ_DIR = 'evaluation'
    custom.DEFAULT_LOG_DIR = 'logs'
    custom.DEFAULT_PLOT_DIR = 'plots'
    custom.register_envs = lambda: None
    sys.modules['custom_rl'] = custom
    ev = types.ModuleType('custom_rl.eval'); ev.__path__=[];sys.modules['custom_rl.eval']=ev
    mc = types.ModuleType('custom_rl.eval.monte_carlo')
    mc.face_milling_process_from_info = lambda info, plant: {}
    sys.modules['custom_rl.eval.monte_carlo'] = mc
    pipe = types.ModuleType('custom_rl.eval.pipeline')
    pipe.discover_model_seeds = lambda root: [0, 1]
    pipe.discover_log_seeds = lambda root: sorted([int(p.name[5:]) for p in root.glob('seed_*')])
    pipe.discover_trajectory_seeds = lambda root: sorted([int(p.stem.split('seed')[-1]) for p in root.glob('trajectories_seed*.json')])
    pipe.plate_env_kwargs = lambda **kw: kw
    pipe.physical_action_names = lambda md,n_dims: ['omega_rad_s', 'ap_mm'][:n_dims]
    pipe.primary_episode_record = lambda ep: (ep.get('runs') or [ep])[0]
    pipe.plant_plot_metadata = lambda env: {'step_dt': 0.001, 'n_sensors':1, 'w_limit':.003,
        'physical_action_low':[40.,0.], 'physical_action_high':[400.,12.], 'w_obs_scale':1., 'wdot_obs_scale':1.}
    pipe.resolve_ppo_model_path = lambda mdir, seed,prefer_final=False: mdir / f'final_{seed}.zip'
    sys.modules['custom_rl.eval.pipeline'] = pipe
    plants = types.ModuleType('custom_rl.plants'); plants.__path__=[]; sys.modules['custom_rl.plants']=plants
    plate = types.ModuleType('custom_rl.plants.plate')
    plate.omega_to_rpm = lambda w: np.asarray(w) * (60/(2*np.pi))
    sys.modules['custom_rl.plants.plate'] = plate


class FakeSpace:
    def __init__(self, lo, hi):
        self.low = np.asarray(lo)
        self.high = np.asarray(hi)
        self.shape = self.low.shape


class FakeModel:
    observation_space=FakeSpace([-10.,-10.,-1.,-1.], [10.,10.,1.,1.])
    action_space=FakeSpace([-1.,-1.], [1.,1.])
    def predict(self, obs, deterministic=True):
        return np.array([0., -0.5]), None


class FakePlant:
    N=2; omega_min=40.; omega_max=400.; w_limit=.003; L2=.2; initial_eta_std=.0
    initial_etad_std=.0; process_noise_std=.0; obs_noise_std=.0
    y0_min=.02; y0_max=.18
    def _scale_action(self, action):
        return np.array([100., 3.])


class FakeEnv:
    observation_space=FakeModel.observation_space
    action_space=FakeModel.action_space
    def __init__(self, **kw):
        self.unwrapped=self;self.plant=FakePlant()
        self.max_episode_steps=6
        self.process_noise_std=0.;self.obs_noise_std=0.
        self.i=0;self.yy=None
    def reset(self,seed=None,options=None):
        self.i=0;self.yy=options.get('y0',0.1) if options else .1
        self.seed=seed
        return np.zeros(4), {'t':0., 'y_cutter':self.yy,'x_start':.0,'x_end':.1}
    def step(self,action):
        self.i+=1
        w=.001 * (1.+self.i/10.) + (self.seed%3) * .00001
        obs=np.array([w, 0., 0., 0.])
        info={'t':self.i*.001, 'w_sensor':[w], 'wdot_sensor':[0.],
              'cutter_x':self.i*.01, 'cutter_y':self.yy,
              'feed_progress':self.i/6., 'termination_reason':'pass_completed' if self.i==6 else None,
              'pass_completed':self.i==6, 'F_total_N':[1.,2.,3.]}
        return obs, 2.0, self.i==6, False, info
    def close(self): pass


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_stubs()
        cls.ev=load('rechecked_evaluator','eval_policy.py')
        cls.pl=load('rechecked_plotter','plot_results.py')

    def test_time_aligned_interpolation_and_no_extrapolation(self):
        a,b=self.pl._pad_time_series(
            [np.array([[0.], [1.], [2.]]),np.array([[0.], [2.]])],
            [np.array([0., 1., 2.]), np.array([0., 2.])])
        # Coarsest cadence 2 s, so time grid = [0, 2].
        self.assertTrue(np.allclose(b[0], [0., 2.]))
        self.assertTrue(np.allclose(a[:, :, 0], [[0.,2.],[0.,2.]]))
        c,d=self.pl._pad_time_series(
            [np.array([[1.], [2.], [3.]]), np.array([[10.], [20.]])],
            [np.array([0., 1., 2.]), np.array([0.5, 1.5])],kind='action')
        self.assertTrue(np.isnan(c[1, 0, 0]))
        self.assertTrue(np.isnan(c[1, -1, 0]))
        self.assertTrue(np.isfinite(c[1, 1, 0]))

    def test_missing_replica_sd_is_nan(self):
        means, sd, valid = self.pl._nan_mean_std(np.array([[1.,2.],[3.,np.nan]]))
        self.assertAlmostEqual(means[0],2.)
        self.assertTrue(np.isnan(sd[1]))

    def test_mc_physical_scenario_refusal(self):
        good=[{'y_line_m':.1,'ap_mm':2.},{'y_line_m':.1,'ap_mm':2.}]
        okay,_=self.pl._same_mc_scenario(good,{},'CustomODEPlateFinish-v0')
        self.assertTrue(okay)
        bad=[good[0], {'y_line_m':.2,'ap_mm':2.}]
        okay,_=self.pl._same_mc_scenario(bad,{},'CustomODEPlateFinish-v0')
        self.assertFalse(okay)

    def test_rms_plot_uses_raw_samples_before_visual_decimation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=pathlib.Path(tmp)
            t=np.arange(1,201,dtype=float)*.001
            w=np.where(np.arange(len(t))%2 == 0, .001, -.001)
            metadata={'n_sensors':1,'env_id':'CustomODEPlate-v0'}
            ep={'runs':[{'times':t.tolist(), 'physical_signals':np.column_stack([w,np.zeros_like(w)]).tolist()},
                        {'times':t.tolist(), 'physical_signals':np.column_stack([2*w,np.zeros_like(w)]).tolist()}]}
            with patch.object(self.pl,'MAX_PLOTTED_TIME_POINTS',20):
                self.pl.plot_vibration_rms_envelope(ep,metadata,out,suffix='_test')
            self.assertTrue((out/'trajectory_displacement_rms_test.pdf').is_file())

    def test_training_diagnostics_four_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);log=root/'logs';out=root/'plots'
            out.mkdir()
            for seed in (0,1,2):
                (log/f'seed_{seed}').mkdir(parents=True)
                (log/f'seed_{seed}'/'training_config.json').write_text(
                    json.dumps({'ppo': {'target_kl':0.02}}))
            steps=np.array([1000.,2000.,3000.])
            metrics={tag:(steps,np.array(vals)) for tag,vals in {
                'rollout/ep_rew_mean':[1.,2.,3.],
                'train/explained_variance':[.3,.5,.7],
                'train/approx_kl':[.005,.01,.02],
                'train/clip_fraction':[.02,.06,.1],
                'train/std':[.9,.7,.5]}.items()}
            with patch.object(self.pl,'_load_tensorboard_scalars',return_value=metrics):
                self.pl.plot_training_diagnostics(log,out,[0,1,2])
            self.assertTrue((out/'ppo_training_diagnostics.pdf').is_file())

    def test_storage_stride_does_not_change_full_step_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            m=root/'models';m.mkdir();(m/'final_0.zip').write_bytes(b'mock')
            log=root/'logs'/'seed_0';log.mkdir(parents=True)
            conf={'environment':{'env_id':'CustomODEPlate-v0','reward_id':'productive',
                'dt_s':.0001,'n_substeps':10,'max_episode_steps':6,
                'dynamics_uncertainty_std':0.,'randomize_y0':True,
                'kwargs':{'max_episode_steps':6}}}
            (log/'training_config.json').write_text(json.dumps(conf))
            cases=[]
            for stride in (1,3):
                out=root/f'evaluation_stride_{stride}'
                argv=['eval_policy.py','--model-dir',str(m),'--out-dir',str(out),
                      '--seeds','0','--n-episodes','1','--n-mc','2',
                      '--record-every',str(stride)]
                with patch.object(sys,'argv',argv):
                    self.assertEqual(self.ev.main(),0)
                episode=json.loads((out/'trajectories_seed0.json').read_text())[0]
                cases.append(episode)
            for a,b in zip(cases[0]['runs'],cases[1]['runs']):
                for key in ['return','rms_w_m','max_abs_w_m','rms_force_N',
                            'peak_force_N','duration_s','spindle_speed_total_variation_rpm']:
                    self.assertAlmostEqual(a['summary'][key],b['summary'][key])
                self.assertLess(b['recorded_length'],a['recorded_length'])

    def test_full_step_metrics_and_output_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            run=pathlib.Path(temp);model_dir=run/'models';model_dir.mkdir()
            for seed in (0,1):
                (model_dir/f'final_{seed}.zip').write_bytes(b'placeholder')
                log=run/'logs'/f'seed_{seed}';log.mkdir(parents=True)
                conf={'environment':{'env_id':'CustomODEPlate-v0','reward_id':'productive',
                    'dt_s':.0001,'n_substeps':10,'max_episode_steps':6,
                    'dynamics_uncertainty_std':0.,'randomize_y0':True,
                    'kwargs':{'max_episode_steps':6}}}
                (log/'training_config.json').write_text(json.dumps(conf))
                np.savez(log/'evaluations.npz',timesteps=np.array([10,20,30]),
                    results=np.array([[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]])+seed)
            out=run/'evaluation';plots=run/'plots'
            args=['eval_policy.py','--model-dir',str(model_dir),'--out-dir',str(out),
                  '--seeds','0','1','--n-episodes','2','--n-mc','3','--record-every','2']
            with patch.object(sys,'argv',args):
                self.assertEqual(self.ev.main(),0)
            for seed in (0,1):
                e=json.loads((out/f'trajectories_seed{seed}.json').read_text())
                self.assertEqual(len(e),2)
                for ep in e:
                    self.assertEqual(len(ep['runs']),3)
                    self.assertEqual(len(set([r['y_line_m'] for r in ep['runs']])),1)
                    for r in ep['runs']:
                        self.assertEqual(r['length'],6)
                        self.assertLess(r['recorded_length'],r['length'])
                        self.assertEqual(r['return'],12.)
                        self.assertEqual(r['summary']['duration_s'],.006)
                        self.assertGreater(r['summary']['rms_w_m'],0.)
            self.assertTrue((out/'physical_outcomes_scenarios.csv').exists())
            self.assertTrue((out/'physical_outcomes_replicas.csv').exists())
            plots.mkdir()
            self.pl.plot_learning_curve(run/'logs',plots,[0,1])
            self.pl.plot_trajectory_summary(out,plots,[0,1])
            self.pl.plot_physical_outcomes(out,plots,[0,1])
            self.assertTrue((plots/'figure1_ppo_learning_curve.pdf').exists())
            self.assertTrue((plots/'figure2_ppo_physical_outcomes.pdf').exists())
            self.assertTrue((plots/'trajectory_actions_physical_seed0_episode0.pdf').exists())
            # A 6-ms synthetic episode is shorter than the declared 50-ms RMS
            # window; the research plotter correctly declines to make that figure.


if __name__=='__main__':
    unittest.main(verbosity=2)
