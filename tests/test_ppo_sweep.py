from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sweep_ppo import (  # noqa: E402
    CONFIG_DIR,
    _command_for,
    calculate_common_budget,
    load_resolved_configs,
    validate_completed_run,
    validate_shared_settings,
)
from plot_ppo_sweep import (  # noqa: E402
    aggregate_seed_curves,
    build_common_grid,
    load_seed_metrics,
    main as plot_main,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_fake_run(
    run_dir: Path, cfg: dict, mode: str, seeds: list[int], total: int, ap_max_mm: float = 2.0
) -> dict:
    env_id = "CustomODEPlate-v0" if mode == "roughing" else "CustomODEPlateFinish-v0"
    reward_id = "productive" if mode == "roughing" else "dense"
    n_envs = int(cfg["n_envs"])
    eval_calls = max(1, min(int(cfg["eval_freq"]), total) // n_envs)
    eval_interval = eval_calls * n_envs
    steps = np.arange(eval_interval, total + 1, eval_interval, dtype=np.int64)
    env = {
        "env_id": env_id,
        "reward_id": reward_id,
        "dt_s": float(cfg["dt"]),
        "n_substeps": int(cfg["n_substeps"]),
        "controller_hold_env_steps": int(cfg["controller_hold_env_steps"]),
        "controller_sample_time_s": float(cfg["dt"] * cfg["n_substeps"] * cfg["controller_hold_env_steps"]),
        "max_episode_steps": int(cfg["max_episode_steps"]),
        "dynamics_uncertainty_std": float(cfg["dynamics_uncertainty_std"]),
        "randomize_y0": bool(cfg["randomize_y0"]),
        "kwargs": {
            "dynamics_uncertainty_std": float(cfg["dynamics_uncertainty_std"]),
            "mode": mode,
            "ap_max": 12.0 if mode == "roughing" else float(ap_max_mm),
        },
        "probe": {"ap_min": 0.0, "ap_max": 12.0 if mode == "roughing" else float(ap_max_mm)},
    }
    signature = {
        "dt_s": env["dt_s"],
        "n_substeps": env["n_substeps"],
        "controller_hold_env_steps": env["controller_hold_env_steps"],
        "max_episode_steps": env["max_episode_steps"],
        "dynamics_uncertainty_std": env["dynamics_uncertainty_std"],
        "randomize_y0": env["randomize_y0"],
        "env_id": env["env_id"],
        "reward_id": env["reward_id"],
        "controller_sample_time_s": env["controller_sample_time_s"],
        "env_kwargs_sha256": __import__("hashlib").sha256(
            json.dumps(env["kwargs"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "plant_probe_sha256": __import__("hashlib").sha256(
            json.dumps(env["probe"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_args = {
        key: cfg[key] for key in (
            "n_envs", "vec_env", "start_method", "device", "progress_bar", "dt",
            "n_substeps", "controller_hold_env_steps", "max_episode_steps",
            "dynamics_uncertainty_std", "randomize_y0", "gamma", "gae_lambda",
            "n_eval_episodes", "eval_freq", "checkpoint_freq", "callback_eval_seed",
        )
    }
    run_arguments = {
        **shared_args,
        "total_timesteps": total,
        "seeds": seeds,
        "env_id": env_id,
        "reward": reward_id,
        "ap_max": None if mode == "roughing" else float(ap_max_mm),
    }
    _write_json(run_dir / "run_manifest.json", {
        "status": "completed",
        "arguments": run_arguments,
    })
    software = {
        "git_commit": "test-commit",
        "script_sha256": "test-script-hash",
        "python": "3.x",
        "packages": {"stable-baselines3": "test"},
    }

    for seed in seeds:
        seed_dir = run_dir / "logs" / f"seed_{seed}"
        ppo = {
            "learning_rate_initial": float(cfg["learning_rate"]),
            "learning_rate_final": float(cfg["final_learning_rate"]),
            "n_steps": int(cfg["n_steps"]),
            "n_envs": int(cfg["n_envs"]),
            "batch_size": int(cfg["batch_size"]),
            "n_epochs": int(cfg["n_epochs"]),
            "gamma": float(cfg["gamma"]),
            "gae_lambda": float(cfg["gae_lambda"]),
            "clip_range_initial": float(cfg["clip_range"]),
            "clip_range_final": float(cfg["final_clip_range"]),
            "target_kl": 0.02,
            "ent_coef": 0.002,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "log_std_init": -2.5,
            "adam_epsilon": 1e-5,
            "normalize_observations": False,
            "normalize_rewards": False,
            "safe_action_bias": [0.0, -0.9],
        }
        _write_json(seed_dir / "training_config.json", {
            "seed": seed,
            "arguments": run_arguments,
            "ppo": ppo,
            "environment": env,
            "software": software,
            "evaluation_callback": {
                "episodes_per_evaluation": int(cfg["n_eval_episodes"]),
                "evaluation_env_seed": int(cfg["callback_eval_seed"]),
                "frequency_controller_transitions_actual": eval_interval,
            },
        })
        _write_json(seed_dir / "training_summary.json", {
            "seed": seed,
            "status": "completed",
            "timesteps_reported_by_model": total,
        })
        n_eval = int(cfg["n_eval_episodes"])
        results = np.tile(np.linspace(1.0, 2.0, n_eval), (steps.size, 1)) + seed * 0.1
        ep_lengths = np.full_like(results, 100.0)
        np.savez_compressed(
            seed_dir / "evaluations.npz",
            timesteps=steps,
            results=results,
            ep_lengths=ep_lengths,
            pass_completion_rate=np.linspace(0.1, 0.9, steps.size),
            safety_termination_rate=np.linspace(0.5, 0.1, steps.size),
            truncation_rate=np.linspace(0.2, 0.05, steps.size),
            median_completed_return=np.linspace(1.0, 2.0, steps.size),
            median_completed_productivity=np.linspace(0.0, 3.0, steps.size),
            median_progress=np.linspace(0.2, 0.95, steps.size),
            mean_max_displacement_m=np.linspace(0.001, 0.0015, steps.size),
        )
    fixed_ppo = {
        "ent_coef": 0.002,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "log_std_init": -2.5,
        "adam_epsilon": 1e-5,
        "normalize_observations": False,
        "normalize_rewards": False,
        "safe_action_bias": [0.0, -0.9],
    }
    return {"environment": signature, "fixed_ppo": fixed_ppo, "software": software}


class PPOSweepTests(unittest.TestCase):
    def test_three_configs_have_valid_rollout_geometry(self):
        configs = load_resolved_configs(["fast", "balanced", "precise"])
        self.assertEqual(set(configs), {"fast", "balanced", "precise"})
        expected = {"fast": (1024, 512, 4), "balanced": (2048, 256, 8), "precise": (4096, 256, 10)}
        for name, cfg in configs.items():
            self.assertEqual((cfg["n_steps"], cfg["batch_size"], cfg["n_epochs"]), expected[name])
            rollout = int(cfg["n_steps"]) * int(cfg["n_envs"])
            self.assertEqual(rollout % int(cfg["batch_size"]), 0)
            self.assertEqual(float(cfg["gamma"]), 0.999995)
            self.assertEqual(float(cfg["gae_lambda"]), 0.9995)

    def test_common_budget_is_exact_for_all_rollout_sizes(self):
        configs = load_resolved_configs(["fast", "balanced", "precise"])
        requested, effective, multiple = calculate_common_budget(configs)
        self.assertEqual((requested, effective, multiple), (1_007_616, 1_007_616, 12_288))
        for cfg in configs.values():
            self.assertEqual(effective % (int(cfg["n_steps"]) * int(cfg["n_envs"])), 0)

    def test_shared_conditions_are_identical_and_checked(self):
        configs = load_resolved_configs(["fast", "balanced", "precise"])
        validate_shared_settings(configs)
        changed = {name: dict(cfg) for name, cfg in configs.items()}
        changed["precise"] = dict(changed["precise"])
        changed["precise"]["dynamics_uncertainty_std"] = 0.002
        with self.assertRaisesRegex(ValueError, "dynamics_uncertainty_std"):
            validate_shared_settings(changed)

    def test_mode_command_only_passes_ap_max_for_finishing(self):
        config = CONFIG_DIR / "fast.yaml"
        roughing = _command_for(config, Path("/tmp/rough"), "roughing", [0, 1], 1000,
                                ap_max=2.0, device=None)
        finishing = _command_for(config, Path("/tmp/finish"), "finishing", [0, 1], 1000,
                                 ap_max=2.0, device=None)
        self.assertNotIn("--ap-max", roughing)
        self.assertIn("--ap-max", finishing)

    def test_common_grid_rejects_missing_or_different_checkpoints(self):
        curves = [
            (np.array([100.0, 200.0, 300.0]), np.array([1.0, 2.0, 3.0])),
            (np.array([100.0, 200.0, 300.0]), np.array([2.0, 3.0, 4.0])),
        ]
        grid, interpolated = build_common_grid(curves)
        self.assertFalse(interpolated)
        self.assertTrue(np.array_equal(grid, [100.0, 200.0, 300.0]))
        mismatch = [curves[0], (np.array([100.0, 250.0, 300.0]), curves[1][1])]
        with self.assertRaisesRegex(ValueError, "checkpoint schedule mismatch"):
            build_common_grid(mismatch)

    def test_seed_mean_and_sample_sd(self):
        curves = [
            (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
            (np.array([1.0, 2.0]), np.array([2.0, 3.0])),
            (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        ]
        mean, sd = aggregate_seed_curves(curves)
        self.assertTrue(np.allclose(mean, [2.0, 3.0]))
        self.assertTrue(np.allclose(sd, [1.0, 1.0]))

    def test_completed_run_validation_checks_actual_settings_and_budget(self):
        configs = load_resolved_configs(["fast"])
        cfg = configs["fast"]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            signature = _make_fake_run(run_dir, cfg, "roughing", [0, 1], 1_007_616)
            actual = validate_completed_run(run_dir, cfg, "roughing", [0, 1], 1_007_616)
            self.assertEqual(signature, actual)

    def test_finishing_ap_max_is_checked_against_the_actual_environment(self):
        configs = load_resolved_configs(["fast"])
        cfg = configs["fast"]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_fake_run(run_dir, cfg, "finishing", [0, 1], 1_007_616, ap_max_mm=2.0)
            validate_completed_run(
                run_dir, cfg, "finishing", [0, 1], 1_007_616, ap_max_mm=2.0
            )
            path = run_dir / "logs" / "seed_0" / "training_config.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["environment"]["kwargs"]["ap_max"] = 1.5
            _write_json(path, record)
            with self.assertRaisesRegex(RuntimeError, "environment kwargs ap_max"):
                validate_completed_run(
                    run_dir, cfg, "finishing", [0, 1], 1_007_616, ap_max_mm=2.0
                )

    def test_plotter_generates_evaluation_figures_from_synthetic_runs(self):
        configs = load_resolved_configs(["fast", "balanced"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sweep"
            root.mkdir()
            entries = []
            signatures = []
            for name, cfg in configs.items():
                run_dir = root / name / "run"
                signature = _make_fake_run(run_dir, cfg, "roughing", [0, 1], 1_007_616)
                signatures.append(signature)
                entries.append({
                    "name": name,
                    "label": name.title(),
                    "status": "completed",
                    "run_dir": str(Path(name) / "run"),
                    "hyperparameters": {
                        "learning_rate": float(cfg["learning_rate"]),
                        "final_learning_rate": float(cfg["final_learning_rate"]),
                        "n_steps": int(cfg["n_steps"]),
                        "n_envs": int(cfg["n_envs"]),
                        "batch_size": int(cfg["batch_size"]),
                        "n_epochs": int(cfg["n_epochs"]),
                        "gamma": float(cfg["gamma"]),
                        "gae_lambda": float(cfg["gae_lambda"]),
                        "clip_range": float(cfg["clip_range"]),
                        "final_clip_range": float(cfg["final_clip_range"]),
                    },
                    "effective_run_signature": signature,
                })
            manifest = {
                "schema_version": 3,
                "status": "completed",
                "mode": "roughing",
                "environment": "CustomODEPlate-v0",
                "reward": "productive",
                "seeds": [0, 1],
                "effective_total_timesteps_per_seed": 1_007_616,
                "actual_evaluation_interval_transitions": 49_998,
                "common_training_settings": {
                    key: configs["fast"][key] for key in (
                        "n_envs", "vec_env", "start_method", "device", "progress_bar", "dt",
                        "n_substeps", "controller_hold_env_steps", "max_episode_steps",
                        "dynamics_uncertainty_std", "randomize_y0", "gamma", "gae_lambda",
                        "n_eval_episodes", "eval_freq", "checkpoint_freq", "callback_eval_seed",
                    )
                },
                "configurations": entries,
            }
            manifest_path = root / "sweep_manifest.json"
            _write_json(manifest_path, manifest)
            output = root / "plots"
            self.assertEqual(plot_main(["--manifest", str(manifest_path), "--out-dir", str(output)]), 0)
            self.assertTrue((output / "ppo_sweep_learning_curve.pdf").is_file())
            self.assertTrue((output / "ppo_sweep_reliability_metrics.pdf").is_file())
            self.assertTrue((output / "ppo_sweep_learning_curve_stats.csv").is_file())


    def test_plotter_validates_finishing_ap_max(self):
        configs = load_resolved_configs(["fast", "balanced"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sweep"
            root.mkdir()
            entries = []
            for name, cfg in configs.items():
                run_dir = root / name / "run"
                signature = _make_fake_run(
                    run_dir, cfg, "finishing", [0, 1], 1_007_616, ap_max_mm=2.0
                )
                entries.append({
                    "name": name,
                    "label": name.title(),
                    "status": "completed",
                    "run_dir": str(Path(name) / "run"),
                    "hyperparameters": {
                        "learning_rate": float(cfg["learning_rate"]),
                        "final_learning_rate": float(cfg["final_learning_rate"]),
                        "n_steps": int(cfg["n_steps"]),
                        "n_envs": int(cfg["n_envs"]),
                        "batch_size": int(cfg["batch_size"]),
                        "n_epochs": int(cfg["n_epochs"]),
                        "gamma": float(cfg["gamma"]),
                        "gae_lambda": float(cfg["gae_lambda"]),
                        "clip_range": float(cfg["clip_range"]),
                        "final_clip_range": float(cfg["final_clip_range"]),
                    },
                    "effective_run_signature": signature,
                })
            manifest = {
                "schema_version": 3,
                "status": "completed",
                "mode": "finishing",
                "environment": "CustomODEPlateFinish-v0",
                "reward": "dense",
                "ap_max_mm": 2.0,
                "seeds": [0, 1],
                "effective_total_timesteps_per_seed": 1_007_616,
                "actual_evaluation_interval_transitions": 49_998,
                "common_training_settings": {
                    key: configs["fast"][key] for key in (
                        "n_envs", "vec_env", "start_method", "device", "progress_bar", "dt",
                        "n_substeps", "controller_hold_env_steps", "max_episode_steps",
                        "dynamics_uncertainty_std", "randomize_y0", "gamma", "gae_lambda",
                        "n_eval_episodes", "eval_freq", "checkpoint_freq", "callback_eval_seed",
                    )
                },
                "configurations": entries,
            }
            manifest_path = root / "sweep_manifest.json"
            _write_json(manifest_path, manifest)
            output = root / "plots"
            self.assertEqual(plot_main(["--manifest", str(manifest_path), "--out-dir", str(output)]), 0)
            self.assertTrue((output / "ppo_sweep_learning_curve.pdf").is_file())



if __name__ == "__main__":
    unittest.main(verbosity=2)
