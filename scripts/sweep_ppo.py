#!/usr/bin/env python
"""Run matched Fast / Balanced / Precise PPO training configurations.

This is an additive orchestration layer. It reuses the repository's existing
experiment.py and train_ppo_unified.py and does not replace either file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment import _flatten_train_cfg, _load_experiment_config  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "ppo_sweep"
EXPERIMENT_SCRIPT = SCRIPT_DIR / "experiment.py"
CONFIG_NAMES = ("fast", "balanced", "precise")
LABELS = {"fast": "Fast", "balanced": "Balanced", "precise": "Precise"}
MODE_SETTINGS = {
    "roughing": {"env_id": "CustomODEPlate-v0", "reward": "productive"},
    "finishing": {"env_id": "CustomODEPlateFinish-v0", "reward": "dense"},
}

# These are effective settings passed through the existing experiment/trainer
# CLI. Nested plant/reward blocks inherited from base.yaml are configuration
# metadata in the current pipeline; they are not forwarded as complete kwargs.
SHARED_CONFIG_KEYS = (
    "n_envs", "vec_env", "start_method", "device", "progress_bar",
    "dt", "n_substeps", "controller_hold_env_steps", "max_episode_steps",
    "dynamics_uncertainty_std", "randomize_y0", "gamma", "gae_lambda",
    "n_eval_episodes", "eval_freq", "checkpoint_freq", "callback_eval_seed",
)
REQUIRED_EVALUATION_ARRAYS = {
    "timesteps", "results", "ep_lengths", "pass_completion_rate",
    "safety_termination_rate", "truncation_rate", "median_completed_return",
    "median_completed_productivity", "median_progress", "mean_max_displacement_m",
}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any, name: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric in {path}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite in {path}.")
    return number


def load_resolved_configs(names: list[str]) -> dict[str, dict[str, Any]]:
    """Resolve the repository's YAML inheritance and validate PPO geometry."""
    resolved: dict[str, dict[str, Any]] = {}
    for name in names:
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Sweep config not found: {path}")
        cfg = _flatten_train_cfg(_load_experiment_config(path))
        required = (
            "n_steps", "n_envs", "batch_size", "n_epochs", "total_timesteps",
            "learning_rate", "final_learning_rate", "gamma", "gae_lambda",
            "clip_range", "final_clip_range", "eval_freq", "n_eval_episodes",
            "checkpoint_freq", "seeds", "dt", "n_substeps",
            "controller_hold_env_steps", "max_episode_steps", "dynamics_uncertainty_std",
            "randomize_y0", "callback_eval_seed",
        )
        missing = [key for key in required if cfg.get(key) is None]
        if missing:
            raise ValueError(f"{path} does not resolve required setting(s): {missing}.")

        n_steps = int(cfg["n_steps"])
        n_envs = int(cfg["n_envs"])
        batch_size = int(cfg["batch_size"])
        n_epochs = int(cfg["n_epochs"])
        if n_steps <= 1 or n_envs <= 0 or batch_size <= 1 or n_epochs <= 0:
            raise ValueError(f"Invalid PPO rollout/minibatch dimensions in {path}.")
        rollout_size = n_steps * n_envs
        if batch_size > rollout_size or rollout_size % batch_size:
            raise ValueError(
                f"Invalid PPO batch geometry in {path}: rollout_size={rollout_size}, "
                f"batch_size={batch_size}; batch_size must divide rollout_size exactly."
            )

        if int(cfg["total_timesteps"]) <= 0:
            raise ValueError(f"total_timesteps must be positive in {path}.")
        lr = _finite_number(cfg["learning_rate"], "learning_rate", path)
        final_lr = _finite_number(cfg["final_learning_rate"], "final_learning_rate", path)
        gamma = _finite_number(cfg["gamma"], "gamma", path)
        gae = _finite_number(cfg["gae_lambda"], "gae_lambda", path)
        clip = _finite_number(cfg["clip_range"], "clip_range", path)
        final_clip = _finite_number(cfg["final_clip_range"], "final_clip_range", path)
        if lr <= 0 or final_lr < 0 or final_lr > lr:
            raise ValueError(f"Require 0 <= final_learning_rate <= learning_rate in {path}.")
        if not (0 < gamma <= 1 and 0 < gae <= 1):
            raise ValueError(f"gamma and gae_lambda must be in (0, 1] in {path}.")
        if not (0 < clip <= 1 and 0 < final_clip <= clip):
            raise ValueError(f"Require 0 < final_clip_range <= clip_range <= 1 in {path}.")

        for key in ("eval_freq", "n_eval_episodes", "checkpoint_freq", "dt",
                    "n_substeps", "controller_hold_env_steps", "max_episode_steps"):
            if _finite_number(cfg[key], key, path) <= 0:
                raise ValueError(f"{key} must be positive in {path}.")
        if _finite_number(cfg["dynamics_uncertainty_std"], "dynamics_uncertainty_std", path) < 0:
            raise ValueError(f"dynamics_uncertainty_std cannot be negative in {path}.")
        if int(cfg["max_episode_steps"]) % int(cfg["controller_hold_env_steps"]):
            raise ValueError(
                f"max_episode_steps must be divisible by controller_hold_env_steps in {path}."
            )
        if cfg["vec_env"] == "dummy" and n_envs != 1:
            raise ValueError(f"DummyVecEnv requires n_envs=1 in {path}.")
        seeds = cfg["seeds"] if isinstance(cfg["seeds"], list) else [cfg["seeds"]]
        seeds = [int(seed) for seed in seeds]
        if len(seeds) < 2 or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"Each sweep config must define at least two unique non-negative seeds: {path}.")
        if int(cfg["callback_eval_seed"]) < 0:
            raise ValueError(f"callback_eval_seed must be non-negative in {path}.")
        resolved[name] = cfg

    if not resolved:
        raise ValueError("No sweep configurations were selected.")
    return resolved


def validate_shared_settings(configs: dict[str, dict[str, Any]]) -> None:
    """Reject differences in environment timing and evaluation protocol."""
    names = list(configs)
    reference_name = names[0]
    reference = configs[reference_name]
    reference_seeds = [int(seed) for seed in reference["seeds"]]
    for name, cfg in list(configs.items())[1:]:
        for key in SHARED_CONFIG_KEYS:
            if cfg.get(key) != reference.get(key):
                raise ValueError(
                    f"Unmatched shared setting {key!r}: {reference_name}={reference.get(key)!r}, "
                    f"{name}={cfg.get(key)!r}. Keep this setting identical across configurations."
                )
        seeds = [int(seed) for seed in cfg["seeds"]]
        if seeds != reference_seeds:
            raise ValueError(
                f"Training seeds differ between {reference_name} and {name}; "
                "use the same ordered seed list for every configuration."
            )


def calculate_common_budget(
    configs: dict[str, dict[str, Any]], requested_total: int | None = None
) -> tuple[int, int, int]:
    """Round the requested aggregate transition budget to a common rollout multiple."""
    n_envs_values = {int(cfg["n_envs"]) for cfg in configs.values()}
    if len(n_envs_values) != 1:
        raise ValueError("All sweep configurations must use the same n_envs.")
    configured_budgets = {int(cfg["total_timesteps"]) for cfg in configs.values()}
    if requested_total is None:
        if len(configured_budgets) != 1:
            raise ValueError("All configurations must have the same configured total_timesteps.")
        requested_total = configured_budgets.pop()
    if requested_total <= 0:
        raise ValueError("total_timesteps must be positive.")
    rollout_sizes = [int(cfg["n_steps"]) * int(cfg["n_envs"]) for cfg in configs.values()]
    common_multiple = math.lcm(*rollout_sizes)
    effective_total = ((int(requested_total) + common_multiple - 1) // common_multiple) * common_multiple
    return int(requested_total), int(effective_total), int(common_multiple)


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _command_for(
    config_path: Path,
    output_root: Path,
    mode: str,
    seeds: list[int],
    total_timesteps: int,
    *,
    ap_max: float,
    device: str | None,
    preflight: bool = False,
) -> list[str]:
    mode_cfg = MODE_SETTINGS[mode]
    command = [
        sys.executable, str(EXPERIMENT_SCRIPT), "train",
        "--config", str(config_path.resolve()),
        "--output-root", str(output_root.resolve()),
        "--env-id", mode_cfg["env_id"],
        "--reward", mode_cfg["reward"],
        "--seeds", *[str(seed) for seed in seeds],
        "--total-timesteps", str(total_timesteps),
    ]
    if mode == "finishing":
        command.extend(["--ap-max", str(float(ap_max))])
    if device is not None:
        command.extend(["--device", device])
    if preflight:
        command.append("--preflight-only")
    return command


def _latest_run(output_root: Path) -> Path:
    pointer = output_root / "LATEST"
    if not pointer.is_file():
        raise FileNotFoundError(f"Trainer did not write its LATEST pointer: {pointer}")
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"Trainer wrote an empty LATEST pointer: {pointer}")
    run_dir = Path(raw).expanduser()
    if not run_dir.is_absolute():
        run_dir = output_root / run_dir
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(output_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"LATEST points outside the configuration output root: {run_dir}") from exc
    if not (run_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(f"Run manifest missing from expected run directory: {run_dir}")
    return run_dir


def _expected_eval_steps(cfg: dict[str, Any], total_timesteps: int) -> np.ndarray:
    n_envs = int(cfg["n_envs"])
    eval_calls = max(1, min(int(cfg["eval_freq"]), int(total_timesteps)) // n_envs)
    actual_interval = eval_calls * n_envs
    return np.arange(actual_interval, int(total_timesteps) + 1, actual_interval, dtype=np.int64)


def validate_completed_run(
    run_dir: Path,
    cfg: dict[str, Any],
    mode: str,
    seeds: list[int],
    total_timesteps: int,
    *,
    device_override: str | None = None,
    ap_max_mm: float | None = None,
) -> dict[str, Any]:
    """Verify that the trainer actually used the requested config and completed every seed."""
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("status") != "completed":
        raise RuntimeError(f"Trainer run is not completed: {run_manifest.get('status')!r} in {run_dir}")
    args = run_manifest.get("arguments", {})
    if int(args.get("total_timesteps", -1)) != int(total_timesteps):
        raise RuntimeError("Run manifest total_timesteps does not match the sweep budget.")
    if [int(x) for x in args.get("seeds", [])] != seeds:
        raise RuntimeError("Run manifest training seeds do not match the sweep seed list.")
    expected_mode = MODE_SETTINGS[mode]
    if args.get("env_id") != expected_mode["env_id"] or args.get("reward") != expected_mode["reward"]:
        raise RuntimeError("Run manifest environment/reward selector does not match the requested mode.")
    if mode == "finishing":
        if ap_max_mm is None or not math.isfinite(float(ap_max_mm)) or float(ap_max_mm) <= 0.0:
            raise RuntimeError("A finite, positive ap_max_mm is required to validate finishing runs.")
        actual_ap_max = args.get("ap_max")
        if actual_ap_max is None or not math.isclose(float(actual_ap_max), float(ap_max_mm), rel_tol=1e-10, abs_tol=1e-9):
            raise RuntimeError(
                f"Run manifest ap_max={actual_ap_max!r} mm does not match requested finishing depth {ap_max_mm:g} mm."
            )
    elif args.get("ap_max") is not None:
        raise RuntimeError("Roughing run unexpectedly received the finishing-only --ap-max argument.")
    expected_device = device_override or str(cfg["device"])
    expected_arguments = {
        key: (expected_device if key == "device" else cfg[key])
        for key in SHARED_CONFIG_KEYS
    }
    for key, expected in expected_arguments.items():
        actual = args.get(key)
        if actual != expected:
            raise RuntimeError(f"Effective trainer argument {key}={actual!r}, expected {expected!r}.")

    expected_ppo = {
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
    }
    expected_steps = _expected_eval_steps(cfg, total_timesteps)
    environment_signature: dict[str, Any] | None = None
    fixed_ppo_signature: dict[str, Any] | None = None
    software_signature: dict[str, Any] | None = None
    evaluation_steps_reference: np.ndarray | None = None

    for seed in seeds:
        seed_dir = run_dir / "logs" / f"seed_{seed}"
        config_path = seed_dir / "training_config.json"
        summary_path = seed_dir / "training_summary.json"
        eval_path = seed_dir / "evaluations.npz"
        for required_path in (config_path, summary_path, eval_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Required completed-seed artifact is missing: {required_path}")

        seed_config = _read_json(config_path)
        summary = _read_json(summary_path)
        seed_arguments = seed_config.get("arguments", {})
        if mode == "finishing":
            seed_ap_max = seed_arguments.get("ap_max")
            if seed_ap_max is None or not math.isclose(float(seed_ap_max), float(ap_max_mm), rel_tol=1e-10, abs_tol=1e-9):
                raise RuntimeError(f"Seed {seed} recorded ap_max={seed_ap_max!r} mm, expected {ap_max_mm:g} mm.")
        elif seed_arguments.get("ap_max") is not None:
            raise RuntimeError(f"Roughing seed {seed} unexpectedly recorded a finishing-only ap_max argument.")
        if int(seed_config.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed metadata mismatch in {config_path}.")
        if summary.get("status") != "completed" or int(summary.get("seed", -1)) != seed:
            raise RuntimeError(f"Training summary is not completed for seed {seed}: {summary_path}")
        if int(summary.get("timesteps_reported_by_model", -1)) != int(total_timesteps):
            raise RuntimeError(
                f"Seed {seed} trained for {summary.get('timesteps_reported_by_model')} transitions, "
                f"expected {total_timesteps}."
            )

        actual_ppo = seed_config.get("ppo", {})
        for key, expected in expected_ppo.items():
            actual = actual_ppo.get(key)
            if isinstance(expected, float):
                if actual is None or not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(f"Seed {seed} PPO setting {key}={actual!r}, expected {expected!r}.")
            elif actual != expected:
                raise RuntimeError(f"Seed {seed} PPO setting {key}={actual!r}, expected {expected!r}.")

        env = seed_config.get("environment", {})
        if env.get("env_id") != expected_mode["env_id"] or env.get("reward_id") != expected_mode["reward"]:
            raise RuntimeError(f"Seed {seed} environment/reward does not match mode {mode}.")
        if mode == "finishing":
            actual_kwargs_ap = env.get("kwargs", {}).get("ap_max")
            actual_probe_ap = env.get("probe", {}).get("ap_max")
            for source, actual in (("environment kwargs", actual_kwargs_ap), ("plant probe", actual_probe_ap)):
                if actual is None or not math.isclose(float(actual), float(ap_max_mm), rel_tol=1e-10, abs_tol=1e-9):
                    raise RuntimeError(
                        f"Seed {seed} {source} ap_max={actual!r} mm does not match requested {ap_max_mm:g} mm."
                    )
        expected_env_values = {
            "dt_s": float(cfg["dt"]),
            "n_substeps": int(cfg["n_substeps"]),
            "controller_hold_env_steps": int(cfg["controller_hold_env_steps"]),
            "max_episode_steps": int(cfg["max_episode_steps"]),
            "dynamics_uncertainty_std": float(cfg["dynamics_uncertainty_std"]),
            "randomize_y0": bool(cfg["randomize_y0"]),
        }
        for key, expected in expected_env_values.items():
            actual = env.get(key)
            if isinstance(expected, float):
                if actual is None or not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(f"Seed {seed} environment setting {key}={actual!r}, expected {expected!r}.")
            elif actual != expected:
                raise RuntimeError(f"Seed {seed} environment setting {key}={actual!r}, expected {expected!r}.")

        callback = seed_config.get("evaluation_callback", {})
        if int(callback.get("episodes_per_evaluation", -1)) != int(cfg["n_eval_episodes"]):
            raise RuntimeError(f"Seed {seed} evaluation episode count differs from the sweep configuration.")
        if int(callback.get("evaluation_env_seed", -1)) != int(cfg["callback_eval_seed"]):
            raise RuntimeError(f"Seed {seed} evaluation seed differs from the sweep configuration.")
        if int(callback.get("frequency_controller_transitions_actual", -1)) != int(expected_steps[0]):
            raise RuntimeError(f"Seed {seed} actual evaluation frequency differs from the expected schedule.")

        signature = {
            **{key: env.get(key) for key in expected_env_values},
            "env_id": env.get("env_id"),
            "reward_id": env.get("reward_id"),
            "controller_sample_time_s": env.get("controller_sample_time_s"),
            "env_kwargs_sha256": _canonical_hash(env.get("kwargs", {})),
            "plant_probe_sha256": _canonical_hash(env.get("probe", {})),
        }
        fixed_keys = (
            "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "net_arch",
            "log_std_init", "adam_epsilon", "normalize_observations",
            "normalize_rewards", "safe_action_bias",
        )
        fixed_values = {key: actual_ppo.get(key) for key in fixed_keys}
        if any(value is None for value in fixed_values.values()):
            raise RuntimeError(f"A fixed PPO setting is missing from {config_path}.")
        software = seed_config.get("software", {})
        software_values = {
            "git_commit": software.get("git_commit"),
            "script_sha256": software.get("script_sha256"),
            "python": software.get("python"),
            "packages": software.get("packages"),
        }
        if fixed_ppo_signature is None:
            fixed_ppo_signature = fixed_values
        elif fixed_values != fixed_ppo_signature:
            raise RuntimeError(f"Fixed PPO settings differ between training seeds in {run_dir}.")
        if software_signature is None:
            software_signature = software_values
        elif software_values != software_signature:
            raise RuntimeError(f"Software versions or trainer source differ between training seeds in {run_dir}.")

        if environment_signature is None:
            environment_signature = signature
        elif signature != environment_signature:
            raise RuntimeError(f"Effective environment settings differ between training seeds in {run_dir}.")

        with np.load(eval_path, allow_pickle=False) as data:
            missing = sorted(REQUIRED_EVALUATION_ARRAYS.difference(data.files))
            if missing:
                raise RuntimeError(f"{eval_path} is missing evaluation arrays: {missing}")
            steps = np.asarray(data["timesteps"], dtype=np.int64).reshape(-1)
            results = np.asarray(data["results"], dtype=np.float64)
            if results.ndim == 1:
                results = results.reshape(-1, 1)
            if results.ndim != 2 or results.shape != (steps.size, int(cfg["n_eval_episodes"])):
                raise RuntimeError(f"Unexpected evaluation results shape in {eval_path}: {results.shape}.")
            if not np.all(np.isfinite(results)):
                raise RuntimeError(f"Non-finite evaluation episode returns found in {eval_path}.")
            if not np.array_equal(steps, expected_steps):
                raise RuntimeError(
                    f"Evaluation checkpoints in {eval_path} do not match the expected schedule. "
                    f"Expected {expected_steps[:3].tolist()} ... {expected_steps[-1]}, "
                    f"received {steps[:3].tolist()} ... {steps[-1] if steps.size else 'none'}."
                )
            for metric in REQUIRED_EVALUATION_ARRAYS - {"timesteps", "results", "ep_lengths"}:
                values = np.asarray(data[metric], dtype=np.float64).reshape(-1)
                if values.size != steps.size or not np.all(np.isfinite(values)):
                    raise RuntimeError(f"Invalid or non-finite {metric} values in {eval_path}.")
            lengths = np.asarray(data["ep_lengths"], dtype=np.float64)
            if lengths.shape != results.shape or not np.all(np.isfinite(lengths)):
                raise RuntimeError(f"Invalid evaluation episode lengths in {eval_path}.")

        if evaluation_steps_reference is None:
            evaluation_steps_reference = steps
        elif not np.array_equal(steps, evaluation_steps_reference):
            raise RuntimeError("Evaluation checkpoints differ between seeds in the same configuration.")

    if environment_signature is None or fixed_ppo_signature is None or software_signature is None:
        raise RuntimeError(f"No complete seed records were validated in {run_dir}.")
    return {
        "environment": environment_signature,
        "fixed_ppo": fixed_ppo_signature,
        "software": software_signature,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched Fast/Balanced/Precise PPO sweep using the existing experiment pipeline."
    )
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS), default="roughing")
    parser.add_argument("--configs", nargs="+", choices=CONFIG_NAMES, default=list(CONFIG_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Aggregate PPO transitions per training seed; rounded up to a common rollout multiple.")
    parser.add_argument("--output-root", default="runs/ppo_sweep")
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--ap-max", type=float, default=2.0,
                        help="Finishing-only maximum axial depth in mm (default: 2.0).")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--parallel-configs", type=int, default=1,
                        help="Concurrent configurations; sequential execution is recommended for runtime comparisons.")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(set(args.configs)) != len(args.configs):
        raise SystemExit("Duplicate configuration names are not allowed.")
    if args.seeds is not None and (
        len(args.seeds) < 2 or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds)
    ):
        raise SystemExit("Use at least two unique, non-negative training seeds.")
    if args.mode == "finishing" and (not math.isfinite(args.ap_max) or args.ap_max <= 0):
        raise SystemExit("--ap-max must be finite and positive for finishing mode.")
    if args.parallel_configs < 1 or args.parallel_configs > len(args.configs):
        raise SystemExit("--parallel-configs must be between 1 and the number of selected configurations.")

    configs = load_resolved_configs(args.configs)
    validate_shared_settings(configs)
    config_seeds = [int(seed) for seed in next(iter(configs.values()))["seeds"]]
    seeds = [int(seed) for seed in (args.seeds if args.seeds is not None else config_seeds)]
    if len(seeds) < 2 or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise SystemExit("Use at least two unique, non-negative training seeds for mean ± sample-SD comparison.")

    requested, effective, common_multiple = calculate_common_budget(configs, args.total_timesteps)
    base_output = _resolve_path(args.output_root)
    sweep_name = args.sweep_name or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", sweep_name):
        raise SystemExit("--sweep-name may contain only letters, numbers, dot, underscore, and hyphen.")
    sweep_root = (base_output / args.mode / sweep_name).resolve()
    if not args.dry_run and sweep_root.exists() and any(sweep_root.iterdir()):
        raise SystemExit(f"Sweep directory already exists and is not empty: {sweep_root}")

    print("\n========== PPO SWEEP PLAN ==========")
    print(f"Mode: {args.mode} | environment: {MODE_SETTINGS[args.mode]['env_id']} | reward: {MODE_SETTINGS[args.mode]['reward']}")
    print(f"Training seeds: {seeds} | aggregate transitions per seed: {effective:,}")
    if effective != requested:
        print(f"Requested budget {requested:,} rounded up to {effective:,} (common rollout multiple={common_multiple:,}).")
    print(f"Sweep output: {sweep_root}")

    commands: dict[str, dict[str, list[str]]] = {}
    output_roots: dict[str, Path] = {}
    for name, cfg in configs.items():
        config_path = CONFIG_DIR / f"{name}.yaml"
        output_root = sweep_root / name
        output_roots[name] = output_root
        commands[name] = {
            "preflight": _command_for(config_path, output_root, args.mode, seeds, effective,
                                      ap_max=args.ap_max, device=args.device, preflight=True),
            "train": _command_for(config_path, output_root, args.mode, seeds, effective,
                                  ap_max=args.ap_max, device=args.device, preflight=False),
        }
        rollout = int(cfg["n_steps"]) * int(cfg["n_envs"])
        minibatches = (rollout // int(cfg["batch_size"])) * int(cfg["n_epochs"])
        updates = effective // rollout
        print(
            f"  {LABELS[name]:8s}: rollout={rollout:5d}, batch={int(cfg['batch_size']):4d}, "
            f"epochs={int(cfg['n_epochs']):2d}, minibatches/update={minibatches:3d}, "
            f"nominal minibatch updates/seed={updates * minibatches:,} (upper bound; target-KL may stop epochs early)"
        )

    if args.dry_run:
        for name in configs:
            if not args.skip_preflight:
                print("\n[preflight] " + LABELS[name])
                print(" ".join(commands[name]["preflight"]))
            print("\n[training] " + LABELS[name])
            print(" ".join(commands[name]["train"]))
        return 0

    if not args.skip_preflight:
        print("\n========== PREFLIGHT: ALL CONFIGURATIONS ==========")
        for name in configs:
            print(f"\n--- Preflight: {LABELS[name]} ---")
            result = subprocess.run(commands[name]["preflight"], cwd=str(REPO_ROOT), check=False)
            if result.returncode != 0:
                print(f"Preflight failed for {name}; no sweep training was started.")
                return int(result.returncode or 2)

    reference_cfg = next(iter(configs.values()))
    n_envs = int(reference_cfg["n_envs"])
    eval_calls = max(1, min(int(reference_cfg["eval_freq"]), effective) // n_envs)
    actual_eval_interval = eval_calls * n_envs
    checkpoint_calls = max(1, int(reference_cfg["checkpoint_freq"]) // n_envs)
    actual_checkpoint_interval = checkpoint_calls * n_envs

    common_training_settings = {key: reference_cfg.get(key) for key in SHARED_CONFIG_KEYS}
    if args.device is not None:
        common_training_settings["device"] = args.device

    manifest: dict[str, Any] = {
        "schema_version": 3,
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "environment": MODE_SETTINGS[args.mode]["env_id"],
        "reward": MODE_SETTINGS[args.mode]["reward"],
        "ap_max_mm": float(args.ap_max) if args.mode == "finishing" else None,
        "seeds": seeds,
        "requested_total_timesteps_per_seed": requested,
        "effective_total_timesteps_per_seed": effective,
        "common_rollout_multiple": common_multiple,
        "parallel_configs": int(args.parallel_configs),
        "common_training_settings": common_training_settings,
        "actual_evaluation_interval_transitions": actual_eval_interval,
        "actual_checkpoint_interval_transitions": actual_checkpoint_interval,
        "statistical_note": "Bands are mean ± 1 sample SD across independent training-seed metrics; they are not confidence intervals.",
        "configuration_comparison_note": "Each profile changes a bundle of PPO hyperparameters; this sweep does not isolate the causal effect of individual hyperparameters.",
        "optimizer_update_count_note": "Nominal minibatch-update counts are upper bounds because Stable-Baselines3 PPO may stop an epoch early when target_kl is exceeded.",
        "configurations": [],
    }
    for name, cfg in configs.items():
        rollout = int(cfg["n_steps"]) * int(cfg["n_envs"])
        minibatches = (rollout // int(cfg["batch_size"])) * int(cfg["n_epochs"])
        manifest["configurations"].append({
            "name": name,
            "label": LABELS[name],
            "config_file": (CONFIG_DIR / f"{name}.yaml").relative_to(REPO_ROOT).as_posix(),
            "output_root": output_roots[name].relative_to(sweep_root).as_posix(),
            "run_dir": None,
            "status": "pending",
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
            "rollout_size": rollout,
            "rollout_duration_s": float(cfg["n_steps"] * cfg["dt"] * cfg["n_substeps"] * cfg["controller_hold_env_steps"]),
            "minibatches_per_update": minibatches,
            "nominal_optimizer_minibatch_updates_per_seed": (effective // rollout) * minibatches,
        })

    manifest_path = sweep_root / "sweep_manifest.json"
    _write_manifest(manifest_path, manifest)

    def train_one(name: str) -> tuple[str, int, str | None, dict[str, Any] | None, str | None]:
        print(f"\n========== TRAINING {LABELS[name].upper()} ==========")
        result = subprocess.run(commands[name]["train"], cwd=str(REPO_ROOT), check=False)
        try:
            run_dir = _latest_run(output_roots[name])
        except Exception as exc:
            return name, int(result.returncode or 2), None, None, str(exc)
        if result.returncode != 0:
            return name, int(result.returncode), str(run_dir), None, "Training subprocess returned a non-zero exit code."
        try:
            signature = validate_completed_run(
                run_dir,
                configs[name],
                args.mode,
                seeds,
                effective,
                device_override=args.device,
                ap_max_mm=float(args.ap_max) if args.mode == "finishing" else None,
            )
        except Exception as exc:
            return name, 2, str(run_dir), None, str(exc)
        return name, 0, str(run_dir), signature, None

    results: dict[str, tuple[int, str | None, dict[str, Any] | None, str | None]] = {}
    if args.parallel_configs == 1:
        for name in configs:
            name_out, code, run_dir, signature, error = train_one(name)
            results[name_out] = (code, run_dir, signature, error)
            entry = next(item for item in manifest["configurations"] if item["name"] == name_out)
            entry["status"] = "completed" if code == 0 else "failed"
            entry["run_dir"] = Path(run_dir).relative_to(sweep_root).as_posix() if run_dir else None
            entry["effective_run_signature"] = signature
            entry["return_code"] = code
            if error:
                entry["error"] = error
            _write_manifest(manifest_path, manifest)
            if code != 0:
                break
    else:
        print("WARNING: parallel configuration runs can cause CPU/GPU contention and distort wall-clock comparisons.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_configs) as pool:
            futures = [pool.submit(train_one, name) for name in configs]
            for future in concurrent.futures.as_completed(futures):
                name_out, code, run_dir, signature, error = future.result()
                results[name_out] = (code, run_dir, signature, error)
                entry = next(item for item in manifest["configurations"] if item["name"] == name_out)
                entry["status"] = "completed" if code == 0 else "failed"
                entry["run_dir"] = Path(run_dir).relative_to(sweep_root).as_posix() if run_dir else None
                entry["effective_run_signature"] = signature
                entry["return_code"] = code
                if error:
                    entry["error"] = error
                _write_manifest(manifest_path, manifest)

    all_ok = len(results) == len(configs) and all(code == 0 for code, _, _, _ in results.values())
    if all_ok:
        signatures = [signature for _, _, signature, _ in results.values()]
        reference_signature = signatures[0]
        if any(signature != reference_signature for signature in signatures[1:]):
            all_ok = False
            manifest["environment_consistency_error"] = (
                "The effective environment settings recorded by the trainer differ across configurations."
            )
            for entry in manifest["configurations"]:
                if entry["status"] == "completed":
                    entry["status"] = "invalidated_environment_mismatch"
    manifest["status"] = "completed" if all_ok else "failed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    _write_manifest(manifest_path, manifest)
    print(f"\nSweep status: {manifest['status']}")
    print(f"Sweep manifest: {manifest_path}")
    if all_ok:
        print("Generate learning, diagnostic and reliability plots with:")
        print(f'  python scripts/plot_ppo_sweep.py --manifest "{manifest_path}"')
        return 0
    print("The sweep is incomplete or failed validation. Do not use it for cross-configuration comparison.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
