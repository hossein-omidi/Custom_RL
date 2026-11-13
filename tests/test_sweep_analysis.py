"""Offline tests for standalone sweep-level plotting and policy screening."""
from __future__ import annotations
import csv, json, sys, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_ppo_sweep import check_training_runs, load_evaluation, plot, select, screen


def write_json(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data), encoding='utf-8')


def make_run(folder, name, seed_range=(0,1,2), completion=0.7):
    folder.mkdir(parents=True,exist_ok=True)
    args=dict(seeds=list(seed_range),total_timesteps=12000)
    write_json(folder/'run_manifest.json',dict(status='completed',arguments=args))
    for seed in seed_range:
        d=folder/'logs'/f'seed_{seed}'
        write_json(d/'training_config.json', dict(seed=seed, environment={
            'env_id':'CustomODEPlate-v0','reward_id':'productive','kwargs':{'dt':.0001,'productivity_weight':240.0},
            'controller_sample_time_s':.001,'max_episode_steps':270000},
            evaluation_callback={'episodes_per_evaluation':2,'evaluation_env_seed':10000}))
        write_json(d/'training_summary.json', dict(status='completed',seed=seed,timesteps_reported_by_model=12000,
            best_evaluation_metrics=dict(pass_completion_rate=completion,safety_termination_rate=0,
                median_completed_productivity=completion*10,median_progress=completion,mean_return=completion*100,
                mean_max_displacement_m=0.001)))
        steps=np.array([3000,6000,9000,12000])
        np.savez_compressed(d/'evaluations.npz',timesteps=steps,
            results=np.tile(np.array([1.,2.])+seed*.1,(len(steps),1))+completion,
            ep_lengths=np.ones((len(steps),2))*100,
            pass_completion_rate=np.ones(len(steps))*completion,
            safety_termination_rate=np.zeros(len(steps)),
            truncation_rate=np.zeros(len(steps)),
            median_completed_return=np.ones(len(steps))*1.,
            median_completed_productivity=np.ones(len(steps))*1.,
            median_progress=np.ones(len(steps))*completion,
            mean_max_displacement_m=np.ones(len(steps))*.001)
        (folder/'models'/f'best_{seed}').mkdir(parents=True,exist_ok=True)
        (folder/'models'/f'best_{seed}'/'best_model.zip').write_bytes(b'FAKE CHECKPOINT FOR PATH TEST ONLY')
    models=[dict(training_seed=seed,checkpoint_path=str((folder/'models'/f'best_{seed}'/'best_model.zip').resolve())) for seed in seed_range]
    write_json(folder/'evaluation'/'evaluation_manifest.json',dict(env_id='CustomODEPlate-v0',evaluation_seed_base=1000,n_episodes=3,n_mc=2,
        controller_hold_env_steps=1,controller_sample_time_s=.001, policy_deterministic=True,models=models))
    fn=folder/'evaluation'/'physical_outcomes_replicas.csv'
    with fn.open('w',newline='',encoding='utf-8') as f:
        cols=['training_seed','scenario_index','mc_index','reset_seed','y_line_m','ap_fixed_mm','pass_completed',
              'termination_reason','feed_progress','return','rms_w_m']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for seed in seed_range:
            for epi in range(3):
                for mc in range(2):
                    k=epi*2+mc
                    complete=int(k<round(completion*6))
                    w.writerow(dict(training_seed=seed,scenario_index=epi,mc_index=mc,reset_seed=1000+k,
                        y_line_m=.20+epi*.01,ap_fixed_mm='',pass_completed=bool(complete),
                        termination_reason='pass_completed_90percent' if complete else 'time_limit_truncated',
                        feed_progress=1 if complete else .7,**{'return':completion*100+seed*.01},rms_w_m=.001))

class TestSweepAnalysis(unittest.TestCase):
    def test_train_figures_and_best_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runs={name:root/name for name in ('fast','balanced','precise')}
            for name,frac in [('fast',.30),('balanced',.65),('precise',.95)]:make_run(runs[name],name,completion=frac)
            seeds,histories=check_training_runs(runs)
            self.assertEqual(seeds,[0,1,2]);self.assertEqual(len(histories),3)
            plot(runs,root/'plots')
            self.assertTrue((root/'plots'/'ppo_sweep_learning_curve.png').is_file())
            self.assertTrue((root/'plots'/'ppo_sweep_learning_curve_stats.csv').is_file())
            early=screen(runs,root/'screen')
            self.assertEqual(early['candidate_configuration'],'precise')
            self.assertTrue((root/'screen'/'ppo_training_checkpoint_screening.csv').is_file())
            selection=select(runs,root/'selection')
            self.assertEqual(selection['suggested_configuration'],'precise')
            self.assertEqual(selection['status'],'screening_recommendation_not_final_test')
            self.assertTrue((root/'selection'/'ppo_selection_recommendation.json').is_file())
    def test_held_action_incomplete_reward_components_are_not_plotted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runs={name:root/name for name in ('fast','balanced','precise')}
            for name in runs:
                make_run(runs[name],name)
                for seed in (0,1,2):
                    path=runs[name]/'logs'/f'seed_{seed}'/'training_config.json'
                    cfg=json.loads(path.read_text(encoding='utf-8'))
                    cfg['environment']['controller_hold_env_steps']=5
                    cfg['environment']['controller_sample_time_s']=0.005
                    write_json(path,cfg)
            plot(runs,root/'plots')
            with (root/'plots'/'ppo_sweep_reliability_metrics_stats.csv').open(newline='') as fd:
                metrics={row['metric'] for row in csv.DictReader(fd)}
            self.assertIn('pass_completion_rate',metrics)
            self.assertNotIn('median_completed_productivity',metrics)
            self.assertNotIn('mean_max_displacement_m',metrics)
            provenance=json.loads((root/'plots'/'ppo_sweep_plot_provenance.json').read_text())
            self.assertEqual(provenance['controller_hold_env_steps'],5)
            self.assertEqual(set(provenance['excluded_training_callback_metrics']),
                             {'median_completed_productivity','mean_max_displacement_m'})
    def test_reject_physical_condition_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs={n:root/n for n in ('fast','balanced','precise')}
            for name in runs:make_run(runs[name],name)
            p=runs['fast']/'logs'/'seed_0'/'training_config.json'; d=json.loads(p.read_text());d['environment']['kwargs']['dt']=.002;write_json(p,d)
            with self.assertRaisesRegex(ValueError,'Effective physical/reward'):check_training_runs(runs)
    def test_reject_unpaired_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs={n:root/n for n in ('fast','balanced','precise')}
            for name in runs:make_run(runs[name],name)
            p=runs['fast']/'evaluation'/'physical_outcomes_replicas.csv'
            text=p.read_text();p.write_text(text.replace('1000,0.2','9000,0.2',1))
            with self.assertRaises(ValueError):load_evaluation(runs)


# The second class checks auto-discovery. Group selection is directory-based,
# never by choosing the best-looking return or a global LATEST pointer.
class TestLatestDiscovery(unittest.TestCase):
    def _group(self, root, mode, group_name, *, complete=True, manifest=False):
        group = root / mode / group_name
        for name in ('fast','balanced','precise'):
            run = group / name / f'{mode}_{name}'
            make_run(run, name)
            if not complete and name == 'precise':
                m = json.loads((run/'run_manifest.json').read_text())
                m['status'] = 'running'
                write_json(run/'run_manifest.json', m)
        if manifest:
            write_json(group/'sweep_manifest.json', {
                'schema_version': 3, 'status': 'completed' if complete else 'running', 'mode': mode,
                'configurations': [
                    {'name':n,'status':'completed' if complete else 'running',
                     'run_dir':f'{n}/{mode}_{n}'} for n in ('fast','balanced','precise')
                ],
            })
        return group

    def test_latest_legacy_and_ignores_incomplete(self):
        from analyze_ppo_sweep import _latest_run_group
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs' / 'ppo_sweep'
            older = self._group(root, 'roughing', '20261111_194426')
            self._group(root, 'roughing', '20261112_194426', complete=False)
            runs, selected = _latest_run_group(root, 'roughing')
            self.assertEqual(selected, older)
            self.assertEqual(set(runs), {'fast', 'balanced', 'precise'})
            self.assertTrue(runs['balanced'].name.startswith('roughing'))

    def test_latest_manifest_and_cli_resolve_only(self):
        from analyze_ppo_sweep import _latest_run_group, main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs' / 'ppo_sweep'
            group = self._group(root, 'finishing', '20261112_194426', manifest=True)
            runs, selected = _latest_run_group(root, 'finishing')
            self.assertEqual(selected, group)
            self.assertEqual(len(runs), 3)
            self.assertEqual(main(['plot','--latest','--mode','finishing',
                                   '--sweep-root',str(root),'--resolve-only']),0)

    def test_both_modes_must_be_disambiguated(self):
        from analyze_ppo_sweep import _latest_run_group
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'runs'/'ppo_sweep'
            self._group(root,'roughing','20261111_194426')
            self._group(root,'finishing','20261112_194426')
            with self.assertRaisesRegex(ValueError,'BOTH roughing and finishing'):
                _latest_run_group(root,None)

    def test_incomplete_manifest_not_bypassed(self):
        from analyze_ppo_sweep import _latest_run_group
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'runs'/'ppo_sweep'
            self._group(root,'roughing','20261112_194426',complete=False,manifest=True)
            with self.assertRaises(FileNotFoundError):
                _latest_run_group(root,'roughing')


class TestTensorBoardProcessing(unittest.TestCase):
    def test_diagnostic_plot_seed_alignment_and_stats(self):
        from unittest.mock import patch
        from plot_ppo_sweep import _aggregate_diagnostic_series, _plot_training_diagnostics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs={n:root/n for n in ('fast','balanced','precise')}
            tags = ('rollout/ep_rew_mean', 'train/explained_variance',
                    'train/approx_kl', 'train/clip_fraction', 'train/std')
            def read_fake(seed_folder):
                seed=int(seed_folder.name.split('_')[1])
                # Different profiles have different update schedules; seeds
                # within a profile share the observed PPO update timestamps.
                profile=seed_folder.parents[1].name
                spacing={'fast': 1000.,'balanced': 2000.,'precise': 4000.}[profile]
                xs=np.array([spacing,2*spacing,3*spacing],dtype=float)
                return {tag:(xs,np.array([0.1,0.2,0.3])+seed*.01) for tag in tags}
            with patch('plot_ppo_sweep._load_tensorboard_scalars',side_effect=read_fake):
                rows=_plot_training_diagnostics(runs, {n:n.title() for n in runs}, [0,1,2],root/'out')
            self.assertTrue((root/'out'/'ppo_sweep_training_diagnostics.pdf').is_file())
            self.assertEqual({r[0] for r in rows},set(tags))
            self.assertEqual({r[3] for r in rows},{3})
            self.assertTrue(all(np.isfinite(float(r[4])) for r in rows))
            x=np.array([0.,1.,2.,3.]); y=np.array([1.,2.,3.,4.]);
            grid,mean,sd=_aggregate_diagnostic_series([
                (x,y),
                (np.array([0.,2.,3.]), np.array([2.,4.,5.])),
                (np.array([0.,1.,3.]), np.array([3.,4.,6.])),
            ])
            np.testing.assert_array_equal(grid,np.array([0.,3.]))
            np.testing.assert_allclose(mean,np.array([2.,5.]))
            np.testing.assert_allclose(sd,np.array([1.,1.]))

    def test_reject_disjoint_optimization_update_steps(self):
        from plot_ppo_sweep import _aggregate_diagnostic_series
        x=np.array([100.,200.,300.]); y=np.array([1.,2.,3.]);
        self.assertIsNone(_aggregate_diagnostic_series([
            (x,y),(x+50,y+1),
        ]))

    def test_seed_level_return_aggregation_not_episode_pooling(self):
        from plot_ppo_sweep import aggregate_seed_curves
        checkpoints=np.array([100.,200.])
        # These values stand for per-seed MEANS over their evaluation episodes.
        mu,sd=aggregate_seed_curves([
            (checkpoints,np.array([10.,20.])),
            (checkpoints,np.array([20.,30.])),
            (checkpoints,np.array([30.,40.])),
        ])
        np.testing.assert_array_equal(mu,np.array([20.,30.]))
        np.testing.assert_array_equal(sd,np.array([10.,10.]))

    def test_reject_missing_required_diagnostics(self):
        from analyze_ppo_sweep import plot
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs={n:root/n for n in ('fast','balanced','precise')}
            for name in runs: make_run(runs[name],name)
            with self.assertRaisesRegex(RuntimeError, 'Required PPO diagnostics'):
                plot(runs,root/'plots',require_diagnostics=True)


class TestChronologicalLatest(unittest.TestCase):
    def test_directory_timestamp_beats_modified_time(self):
        from analyze_ppo_sweep import _latest_run_group
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'runs'/'ppo_sweep'
            old=TestLatestDiscovery()._group(root,'roughing','20261111_194426')
            new=TestLatestDiscovery()._group(root,'roughing','20261112_194426')
            # Simulate copying/touching the older run AFTER the newer run.
            for p in old.rglob('run_manifest.json'):
                os.utime(p, None)
            _,selected=_latest_run_group(root,'roughing')
            self.assertEqual(selected,new)

if __name__ == '__main__':
    unittest.main()
