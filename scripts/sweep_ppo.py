#!/usr/bin/env python
"""Run matched Fast / Balanced / Precise PPO training configurations.

The existing experiment.py and train_ppo_unified.py remain the execution engine.
This wrapper only validates the sweep, runs preflight checks, isolates outputs,
and records a reproducible sweep manifest.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
    "roughing": {"env_id": "CustomODEPlate-v0", "reward": "productive", "ap_max": None},
    "finishing": {"env_id": "CustomODEPlateFinish-v0", "reward": "dense", "ap_max": 2.0},
}


def load_resolved_configs(names: list[str]) -> dict[str, dict[str, Any]]:
    """Load the actual inherited config values using experiment.py's own resolver."""
    resolved: dict[str, dict[str, Any]] = {}
    for name in names:
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Sweep config not found: {path}")
        cfg = _flatten_train_cfg(_load_experiment_config(path))
        for key in ("n_steps", "n_envs", "batch_size", "n_epochs", "total_timesteps"):
            if cfg.get(key) is None:
                raise ValueError(f"{path} does not resolve required setting {key!r}.")
        n_steps = int(cfg["n_steps"])
        n_envs = int(cfg["n_envs"])
        batch_size = int(cfg["batch_size"])
        if n_steps <= 1 or n_envs <= 0 or batch_size <= 1:
            raise ValueError(f"Invalid rollout/minibatch dimensions in {path}.")
        rollout_size = n_steps * n_envs
        if batch_size > rollout_size or rollout_size % batch_size:
            raise ValueError(
                f"Invalid PPO batch geometry in {path}: rollout_size={rollout_size}, "
                f"batch_size={batch_size}; batch_size must divide rollout_size exactly."
            )
        resolved[name] = cfg
    if not resolved:
        raise ValueError("No sweep configurations were selected.")
    return resolved


def calculate_common_budget(
    configs: dict[str, dict[str, Any]], requested_total: int | None = None
) -> tuple[int, int, int]:
    """Round the budget up to a common multiple of all rollout sizes.

    PPO completes whole vectorized rollouts. Choosing a common multiple ensures
    every selected configuration receives exactly the same number of transitions.
    """
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
    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT), "train",
        "--config", str(config_path.resolve()),
        "--output-root", str(output_root.resolve()),
        "--env-id", mode_cfg["env_id"],
        "--reward", mode_cfg["reward"],
        "--seeds", *[str(seed) for seed in seeds],
        "--total-timesteps", str(total_timesteps),
    ]
    if mode == "finishing":
        cmd.extend(["--ap-max", str(float(ap_max))])
    if device is not None:
        cmd.extend(["--device", device])
    if preflight:
        cmd.append("--preflight-only")
    return cmd


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
    if not (run_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(f"Run manifest missing from expected run directory: {run_dir}")
    return run_dir


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched Fast/Balanced/Precise PPO sweep using the existing experiment pipeline."
    )
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS), default="roughing")
    parser.add_argument("--configs", nargs="+", choices=CONFIG_NAMES, default=list(CONFIG_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Requested transitions per training seed; rounded up to a common rollout multiple.")
    parser.add_argument("--output-root", default="runs/ppo_sweep")
    parser.add_argument("--sweep-name", default=None,
                        help="Optional unique folder name; defaults to a timestamp.")
    parser.add_argument("--ap-max", type=float, default=2.0,
                        help="Finishing-only maximum axial depth in mm (default: 2.0).")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--parallel-configs", type=int, default=1,
                        help="Concurrent configurations (default 1; sequential is recommended for fair runtime comparisons).")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the three configuration preflight checks (not recommended).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without launching preflight or training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parallel_configs < 1 or args.parallel_configs > len(args.configs):
        raise SystemExit("--parallel-configs must be between 1 and the number of selected configurations.")
    if args.ap_max <= 0:
        raise SystemExit("--ap-max must be positive.")
    if args.seeds is not None:
        if not args.seeds or any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
            raise SystemExit("Seeds must be unique, non-negative integers.")

    configs = load_resolved_configs(args.configs)
    config_seeds = [int(s) for s in next(iter(configs.values())).get("seeds", [0])]
    for name, cfg in configs.items():
        if args.seeds is None and [int(s) for s in cfg.get("seeds", [0])] != config_seeds:
            raise SystemExit(f"Training seeds differ in {name}.yaml; use identical seeds across configurations.")
    seeds = [int(s) for s in (args.seeds if args.seeds is not None else config_seeds)]
    if not seeds or any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise SystemExit("Seeds must be unique, non-negative integers.")

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
    print(f"Training seeds: {seeds} | transitions per seed: {effective:,}")
    if effective != requested:
        print(f"Requested budget {requested:,} rounded up to {effective:,} to align all rollout sizes (LCM={common_multiple:,}).")
    print(f"Sweep output: {sweep_root}")
    for name, cfg in configs.items():
        n_steps, n_envs = int(cfg["n_steps"]), int(cfg["n_envs"])
        rollout = n_steps * n_envs
        minibatches = (rollout // int(cfg["batch_size"])) * int(cfg["n_epochs"])
        updates = effective // rollout
        print(
            f"  {LABELS[name]:8s}: rollout={rollout:5d}, batch={int(cfg['batch_size']):4d}, "
            f"epochs={int(cfg['n_epochs']):2d}, minibatches/update={minibatches:3d}, "
            f"optimizer minibatch updates/seed={updates * minibatches:,}"
        )

    commands: dict[str, dict[str, list[str]]] = {}
    output_roots: dict[str, Path] = {}
    for name, cfg in configs.items():
        config_path = CONFIG_DIR / f"{name}.yaml"
        out_root = sweep_root / name
        output_roots[name] = out_root
        commands[name] = {
            "preflight": _command_for(config_path, out_root, args.mode, seeds, effective,
                                      ap_max=args.ap_max, device=args.device, preflight=True),
            "train": _command_for(config_path, out_root, args.mode, seeds, effective,
                                  ap_max=args.ap_max, device=args.device, preflight=False),
        }

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

    manifest: dict[str, Any] = {
        "schema_version": 1,
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
        "configurations": [],
    }
    for name, cfg in configs.items():
        n_steps, n_envs = int(cfg["n_steps"]), int(cfg["n_envs"])
        rollout = n_steps * n_envs
        minibatches = (rollout // int(cfg["batch_size"])) * int(cfg["n_epochs"])
        manifest["configurations"].append({
            "name": name,
            "label": LABELS[name],
            "config_file": str((CONFIG_DIR / f"{name}.yaml").relative_to(REPO_ROOT)),
            "output_root": str(output_roots[name]),
            "run_dir": None,
            "status": "pending",
            "hyperparameters": {
                key: cfg.get(key) for key in (
                    "learning_rate", "final_learning_rate", "n_steps", "n_envs", "batch_size",
                    "n_epochs", "gamma", "gae_lambda", "clip_range", "final_clip_range",
                )
            },
            "rollout_size": rollout,
            "minibatches_per_update": minibatches,
            "optimizer_minibatch_updates_per_seed": (effective // rollout) * minibatches,
        })
    manifest_path = sweep_root / "sweep_manifest.json"
    _write_manifest(manifest_path, manifest)

    def train_one(name: str) -> tuple[str, int, str | None]:
        print(f"\n========== TRAINING {LABELS[name].upper()} ==========")
        result = subprocess.run(commands[name]["train"], cwd=str(REPO_ROOT), check=False)
        try:
            run_dir = _latest_run(output_roots[name])
            resolved_run = str(run_dir)
        except Exception as exc:
            print(f"Could not resolve run directory for {name}: {exc}")
            resolved_run = None
        if result.returncode != 0:
            return name, int(result.returncode), resolved_run
        if resolved_run is None:
            return name, 2, None
        return name, 0, resolved_run

    results: dict[str, tuple[int, str | None]] = {}
    if args.parallel_configs == 1:
        for name in configs:
            name_out, code, run_dir = train_one(name)
            results[name_out] = (code, run_dir)
            entry = next(x for x in manifest["configurations"] if x["name"] == name_out)
            entry["status"] = "completed" if code == 0 else "failed"
            entry["run_dir"] = run_dir
            entry["return_code"] = code
            _write_manifest(manifest_path, manifest)
            if code != 0:
                break
    else:
        print("WARNING: parallel configuration runs can cause CPU/GPU contention and distort wall-clock comparisons.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_configs) as pool:
            futures = [pool.submit(train_one, name) for name in configs]
            for future in concurrent.futures.as_completed(futures):
                name_out, code, run_dir = future.result()
                results[name_out] = (code, run_dir)
                entry = next(x for x in manifest["configurations"] if x["name"] == name_out)
                entry["status"] = "completed" if code == 0 else "failed"
                entry["run_dir"] = run_dir
                entry["return_code"] = code
                _write_manifest(manifest_path, manifest)

    all_ok = len(results) == len(configs) and all(code == 0 for code, _ in results.values())
    manifest["status"] = "completed" if all_ok else "failed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    _write_manifest(manifest_path, manifest)
    print(f"\nSweep status: {manifest['status']}")
    print(f"Sweep manifest: {manifest_path}")
    if all_ok:
        print("Create the cross-configuration learning curve with:")
        print(f'  python scripts/plot_ppo_sweep.py --manifest "{manifest_path}"')
        print("For full physical trajectory evaluation, evaluate each run separately with experiment.py eval.")
        return 0
    print("At least one configuration failed. Completed runs and failure codes are recorded in the manifest.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
