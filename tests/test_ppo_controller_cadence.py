"""Unit tests for the 5 ms PPO controller / 1 ms plant-step contract.

These tests stub Stable-Baselines3 and the project package so the action-repeat
and saved-config contract can be checked without starting training or importing
the full milling model.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class _Wrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.unwrapped = env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def _install_import_stubs():
    gym = types.ModuleType("gymnasium")
    gym.Wrapper = _Wrapper
    gym.Env = object
    gym.spaces = types.SimpleNamespace(Box=object)
    gym.envs = types.SimpleNamespace(registry={})
    gym.register = lambda *args, **kwargs: None
    gym.make = lambda *args, **kwargs: None

    sb3 = types.ModuleType("stable_baselines3")
    sb3.PPO = type("PPO", (), {})
    common = types.ModuleType("stable_baselines3.common")
    callbacks = types.ModuleType("stable_baselines3.common.callbacks")
    callbacks.CheckpointCallback = type("CheckpointCallback", (), {})
    callbacks.EvalCallback = type("EvalCallback", (), {})
    env_util = types.ModuleType("stable_baselines3.common.env_util")
    env_util.make_vec_env = lambda *args, **kwargs: None
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")
    vec_env.DummyVecEnv = type("DummyVecEnv", (), {})
    vec_env.SubprocVecEnv = type("SubprocVecEnv", (), {})

    custom = types.ModuleType("custom_rl")
    custom.__path__ = []
    custom.DEFAULT_MODEL_DIR = "models"
    custom.DEFAULT_TRAJ_DIR = "evaluation"
    custom.DEFAULT_LOG_DIR = "logs"
    custom.DEFAULT_PLOT_DIR = "plots"
    custom.register_envs = lambda: None

    custom_eval = types.ModuleType("custom_rl.eval")
    custom_eval.__path__ = []
    pipeline = types.ModuleType("custom_rl.eval.pipeline")
    pipeline.plate_env_kwargs = lambda **kwargs: dict(kwargs)
    pipeline.discover_model_seeds = lambda root: []
    pipeline.discover_log_seeds = lambda root: []
    pipeline.discover_trajectory_seeds = lambda root: []
    pipeline.plant_plot_metadata = lambda env: {}
    pipeline.resolve_ppo_model_path = lambda *args, **kwargs: None
    pipeline.physical_action_names = lambda metadata, n_dims: []
    pipeline.primary_episode_record = lambda episode: episode
    monte_carlo = types.ModuleType("custom_rl.eval.monte_carlo")
    monte_carlo.face_milling_process_from_info = lambda info, plant: {}

    plants = types.ModuleType("custom_rl.plants")
    plants.__path__ = []
    plate = types.ModuleType("custom_rl.plants.plate")
    plate.omega_to_rpm = lambda value: np.asarray(value) * 60.0 / (2.0 * np.pi)

    return {
        "gymnasium": gym,
        "stable_baselines3": sb3,
        "stable_baselines3.common": common,
        "stable_baselines3.common.callbacks": callbacks,
        "stable_baselines3.common.env_util": env_util,
        "stable_baselines3.common.vec_env": vec_env,
        "custom_rl": custom,
        "custom_rl.eval": custom_eval,
        "custom_rl.eval.pipeline": pipeline,
        "custom_rl.eval.monte_carlo": monte_carlo,
        "custom_rl.plants": plants,
        "custom_rl.plants.plate": plate,
    }


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _ToyEnv:
    def __init__(self, terminate_at=None):
        self.action_space = object()
        self.observation_space = object()
        self.unwrapped = self
        self.dt = 1e-4
        self.n_substeps = 10
        self._step_dt = 1e-3
        self.count = 0
        self.terminate_at = terminate_at
        self.actions = []

    def step(self, action):
        self.count += 1
        self.actions.append(np.asarray(action).copy())
        terminated = self.terminate_at is not None and self.count >= self.terminate_at
        info = {"t": self.count * self._step_dt}
        return np.array([self.count], dtype=float), float(self.count), terminated, False, info


class ControllerCadenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._patcher = patch.dict(sys.modules, _install_import_stubs())
        cls._patcher.start()
        cls.trainer = _load_module(
            "ppo_controller_trainer_under_test", ROOT / "scripts" / "train_ppo_unified.py"
        )
        cls.evaluator = _load_module(
            "ppo_controller_evaluator_under_test", ROOT / "scripts" / "eval_policy.py"
        )
        cls.plotter = _load_module(
            "ppo_controller_plotter_under_test", ROOT / "scripts" / "plot_results.py"
        )

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def test_five_ms_action_repeat_keeps_one_ms_physics_steps(self):
        env = _ToyEnv()
        wrapper = self.trainer.ControllerSampleTimeWrapper(
            env, hold_env_steps=5, gamma_per_env_step=0.9
        )
        action = np.array([0.25, -0.5])
        obs, reward, terminated, truncated, info = wrapper.step(action)

        expected = sum((0.9 ** i) * float(i + 1) for i in range(5))
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(env.count, 5)
        self.assertEqual(len(env.actions), 5)
        self.assertTrue(all(np.array_equal(a, action) for a in env.actions))
        self.assertAlmostEqual(info["controller_sample_time_s"], 0.005)
        self.assertEqual(info["controller_hold_env_steps"], 5)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs.tolist(), [5.0])

    def test_training_factory_installs_the_same_action_repeat_wrapper(self):
        env = _ToyEnv()
        gym_stub = sys.modules["gymnasium"]
        with patch.object(gym_stub, "make", return_value=env):
            wrapped = self.trainer.make_registered_plate_env(
                _controller_hold_env_steps=5,
                _gamma_per_env_step=0.9999,
            )
        self.assertIsInstance(wrapped, self.trainer.ControllerSampleTimeWrapper)
        self.assertEqual(wrapped.hold_env_steps, 5)
        self.assertTrue(math.isclose(wrapped.gamma_per_env_step, 0.9999, rel_tol=1e-12))

    def test_action_repeat_stops_immediately_on_termination(self):
        env = _ToyEnv(terminate_at=3)
        wrapper = self.trainer.ControllerSampleTimeWrapper(
            env, hold_env_steps=5, gamma_per_env_step=1.0
        )
        _, reward, terminated, truncated, info = wrapper.step(np.array([0.0]))
        self.assertEqual(env.count, 3)
        self.assertEqual(reward, 1.0 + 2.0 + 3.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["controller_hold_env_steps"], 3)
        self.assertAlmostEqual(info["controller_sample_time_s"], 0.003)

    def test_trainer_defaults_are_five_ms_and_preserve_training_budget(self):
        with patch.object(sys, "argv", ["train_ppo_unified.py"]):
            args = self.trainer._parse_args()
        self.trainer._resolve_mode_defaults(args)
        self.trainer._validate_args(args)

        self.assertEqual(args.dt, 1e-4)
        self.assertEqual(args.n_substeps, 10)
        self.assertEqual(args.controller_hold_env_steps, 5)
        self.assertAlmostEqual(args.dt * args.n_substeps * args.controller_hold_env_steps, 0.005)
        self.assertEqual(args.total_timesteps, 200_000)
        self.assertEqual(args.n_steps, 400)
        self.assertEqual(args.batch_size, 50)
        self.assertEqual(args.eval_freq, 10_000)
        self.assertEqual(args.checkpoint_freq, 40_000)
        self.assertTrue(math.isclose(args.gamma, 0.9999 ** 5, rel_tol=1e-12))
        self.assertTrue(math.isclose(args.gae_lambda, 0.98 ** 5, rel_tol=1e-12))

        env_kwargs = self.trainer._build_env_kwargs(args)
        self.assertEqual(env_kwargs["omega_cost_weight"], 3.0)
        self.assertEqual(env_kwargs["pass_completion_bonus"], 1000.0)

    def test_evaluator_reads_controller_hold_and_discount_from_training_config(self):
        args = SimpleNamespace(
            env_id=None,
            reward=None,
            dt=None,
            n_substeps=None,
            max_episode_steps=None,
            dynamics_uncertainty_std=None,
            randomize_y0=None,
            policy_hold_env_steps=None,
            ap_max=None,
        )
        gamma_controller = 0.9999 ** 5
        config = {
            "environment": {
                "env_id": "CustomODEPlate-v0",
                "reward_id": "productive",
                "dt_s": 1e-4,
                "n_substeps": 10,
                "max_episode_steps": 250_000,
                "dynamics_uncertainty_std": 0.001,
                "randomize_y0": True,
                "controller_hold_env_steps": 5,
                "kwargs": {
                    "omega_cost_weight": 3.0,
                    "pass_completion_bonus": 1000.0,
                },
            },
            "ppo": {"gamma": gamma_controller},
        }
        settings = self.evaluator._resolve_eval_settings(args, config)
        self.assertEqual(settings["controller_hold_env_steps"], 5)
        self.assertAlmostEqual(
            settings["dt"] * settings["n_substeps"] * settings["controller_hold_env_steps"],
            0.005,
        )
        self.assertTrue(
            math.isclose(settings["gamma_per_env_step"], 0.9999, rel_tol=1e-12)
        )

    def test_plotter_labels_learning_axis_as_controller_transitions(self):
        source = (ROOT / "scripts" / "plot_results.py").read_text(encoding="utf-8")
        self.assertIn('ax.set_xlabel("PPO controller transitions")', source)
        self.assertIn('ax_reward.set_xlabel("PPO controller transitions")', source)

    def test_legacy_checkpoint_fallback_remains_one_ms(self):
        args = SimpleNamespace(
            env_id=None,
            reward=None,
            dt=None,
            n_substeps=None,
            max_episode_steps=None,
            dynamics_uncertainty_std=None,
            randomize_y0=None,
            policy_hold_env_steps=None,
            ap_max=None,
        )
        settings = self.evaluator._resolve_eval_settings(args, None)
        self.assertEqual(settings["controller_hold_env_steps"], 1)
        self.assertAlmostEqual(settings["dt"] * settings["n_substeps"], 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
