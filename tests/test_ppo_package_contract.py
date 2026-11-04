from __future__ import annotations
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ppo_reliability import reliability_checkpoint_rank, summarize_reliability_episodes


class ReliabilityRankingTests(unittest.TestCase):
    def test_completion_rate_is_primary(self):
        a = {"pass_completion_rate": 1.0, "safety_termination_rate": 0.0, "truncation_rate": 0.0,
             "median_completed_productivity": 100.0, "median_progress": 1.0, "mean_return": 100.0}
        b = {"pass_completion_rate": 0.8, "safety_termination_rate": 0.0, "truncation_rate": 0.0,
             "median_completed_productivity": 10000.0, "median_progress": 1.0, "mean_return": 10000.0}
        self.assertGreater(reliability_checkpoint_rank(a), reliability_checkpoint_rank(b))

    def test_safety_rate_breaks_completion_tie(self):
        a = {"pass_completion_rate": 0.8, "safety_termination_rate": 0.0, "truncation_rate": 0.0,
             "median_completed_productivity": 10.0, "median_progress": 0.8, "mean_return": 10.0}
        b = {"pass_completion_rate": 0.8, "safety_termination_rate": 0.2, "truncation_rate": 0.0,
             "median_completed_productivity": 1000.0, "median_progress": 0.8, "mean_return": 1000.0}
        self.assertGreater(reliability_checkpoint_rank(a), reliability_checkpoint_rank(b))

    def test_completed_return_only_uses_completed_episodes(self):
        rows = [
            {"return": 100.0, "pass_completed": True, "safety_termination": False, "truncated": False, "final_progress": 1.0, "max_displacement_m": 0.001, "productivity_reward_sum": 80.0},
            {"return": 9000.0, "pass_completed": False, "safety_termination": True, "truncated": False, "final_progress": 0.2, "max_displacement_m": 0.0031, "productivity_reward_sum": 7000.0},
            {"return": 200.0, "pass_completed": True, "safety_termination": False, "truncated": False, "final_progress": 1.0, "max_displacement_m": 0.0015, "productivity_reward_sum": 100.0},
        ]
        m = summarize_reliability_episodes(rows)
        self.assertAlmostEqual(m["pass_completion_rate"], 2/3)
        self.assertAlmostEqual(m["safety_termination_rate"], 1/3)
        self.assertAlmostEqual(m["median_completed_return"], 150.0)
        self.assertAlmostEqual(m["median_completed_productivity"], 90.0)


class StaticConfigurationTests(unittest.TestCase):
    def test_1ms_and_ppo_defaults(self):
        source = (ROOT / "scripts/train_ppo_unified.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try: values[target.id] = ast.literal_eval(node.value)
                        except Exception: pass
        self.assertEqual(values["DEFAULT_CONTROLLER_HOLD_ENV_STEPS"], 1)
        self.assertEqual(values["PPO_N_STEPS"], 2048)
        self.assertEqual(values["PPO_BATCH_SIZE"], 256)
        self.assertEqual(values["DEFAULT_TOTAL_TIMESTEPS"], 1_000_000)
        self.assertEqual(values["DEFAULT_N_EVAL_EPISODES"], 10)
        self.assertEqual(values["PPO_GAMMA_PER_ENV_STEP"], 0.999995)
        self.assertEqual(values["PPO_GAE_LAMBDA_PER_ENV_STEP"], 0.9995)
        self.assertEqual(values["PPO_LOG_STD_INIT"], -2.5)
        self.assertIn("self.eval_env.seed(episode_seed)", source)
        self.assertIn("else 10_000", source)

    def test_evaluator_uses_1ms_fallback_and_saved_reward_scale(self):
        source = (ROOT / "scripts/eval_policy.py").read_text(encoding="utf-8")
        self.assertIn("EVAL_DEFAULT_CONTROLLER_HOLD_ENV_STEPS = 1", source)
        self.assertIn('"running_reward_scale": float(settings["training_env_kwargs"].get("running_reward_scale", 1.0))', source)
        self.assertIn('("running_reward_scale", training_kwargs.get("running_reward_scale"), env_kwargs.get("running_reward_scale"))', source)

    def test_full_axial_depth_range_not_overridden(self):
        source = (ROOT / "scripts/train_ppo_unified.py").read_text(encoding="utf-8")
        build = source.split('def _build_env_kwargs',1)[1].split('def _compute_safe_action_bias',1)[0]
        self.assertIn('if args.env_id == ENV_ID_FINISHING:', build)
        self.assertNotIn('ap_max=', source.split('REWARD_KWARGS = {',1)[1].split('}',1)[0])
        self.assertIn('"productivity_weight": 240.0', source)
        self.assertIn('"omega_cost_weight": 3.0', source)
        self.assertIn('"pass_completion_bonus": 1000.0', source)
        self.assertIn('"termination_penalty": 500.0', source)
        self.assertIn('"negative_ap_weight": 2.0', source)
        self.assertIn('"running_reward_scale": DEFAULT_RUNNING_REWARD_SCALE', source)

    def test_reward_scaling_keeps_terminal_adjustment_outside_scale(self):
        patcher = (ROOT / "apply_reward_scaling.py").read_text(encoding="utf-8")
        self.assertIn("reward = self.running_reward_scale * running_reward_unscaled", patcher)
        self.assertIn("Terminal completion/failure adjustments", patcher)
        self.assertIn("old_reward", patcher)
        self.assertIn("new_reward", patcher)
        self.assertIn('"pass_completion_bonus": 1000.0', (ROOT / "scripts/train_ppo_unified.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
