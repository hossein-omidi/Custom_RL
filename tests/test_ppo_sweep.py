from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# The production repository already contains scripts/run_paths.py. This small
# fallback lets the isolated patch's unit tests import experiment.py without
# copying or replacing that existing support module.
if not (SCRIPTS / "run_paths.py").is_file():
    stub = types.ModuleType("run_paths")
    stub.DEFAULT_OUTPUT_ROOT = Path("runs/ppo")
    for _name in (
        "comparison_dir", "evaluation_dir", "list_runs", "load_run_manifest",
        "load_training_config", "logs_dir", "models_dir", "mpc_dir",
        "plots_dir", "resolve_run_dir", "write_latest_pointer",
    ):
        setattr(stub, _name, lambda *args, **kwargs: None)
    sys.modules["run_paths"] = stub

from experiment import _flatten_train_cfg, _load_experiment_config
from sweep_ppo import CONFIG_DIR, calculate_common_budget, load_resolved_configs
from plot_ppo_sweep import aggregate_seed_curves, build_common_grid


class PPOSweepTests(unittest.TestCase):
    def test_three_configs_have_valid_rollout_geometry(self):
        configs = load_resolved_configs(["fast", "balanced", "precise"])
        self.assertEqual(set(configs), {"fast", "balanced", "precise"})
        for cfg in configs.values():
            rollout = int(cfg["n_steps"]) * int(cfg["n_envs"])
            self.assertEqual(rollout % int(cfg["batch_size"]), 0)
            self.assertEqual(float(cfg["gamma"]), 0.9999)
            self.assertEqual(float(cfg["gae_lambda"]), 0.98)

    def test_common_budget_is_exact_for_all_rollout_sizes(self):
        configs = load_resolved_configs(["fast", "balanced", "precise"])
        requested, effective, multiple = calculate_common_budget(configs)
        self.assertEqual(requested, 1_007_616)
        self.assertEqual(effective, 1_007_616)
        self.assertEqual(multiple, 12_288)
        for cfg in configs.values():
            self.assertEqual(effective % (int(cfg["n_steps"]) * int(cfg["n_envs"])), 0)

    def test_config_prevents_roughing_ap_max_leak(self):
        for name in ("fast", "balanced", "precise"):
            path = CONFIG_DIR / f"{name}.yaml"
            cfg = _flatten_train_cfg(_load_experiment_config(path))
            self.assertNotIn("ap_max", cfg)

    def test_common_grid_and_seed_aggregation(self):
        curves = [
            (np.array([100., 200., 300.]), np.array([1., 2., 3.])),
            (np.array([100., 200., 300.]), np.array([2., 3., 4.])),
            (np.array([100., 250., 300.]), np.array([3., 4., 5.])),
        ]
        grid, interpolated = build_common_grid(curves)
        self.assertFalse(interpolated)
        self.assertTrue(np.array_equal(grid, np.array([100., 300.])))
        mean, sd = aggregate_seed_curves(curves, grid)
        self.assertTrue(np.allclose(mean, [2., 4.]))
        self.assertTrue(np.allclose(sd, [1., 1.]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
