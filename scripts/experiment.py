#!/usr/bin/env python
"""Config-driven experiment workflow: adjust settings → train → eval → plot.

Everyday usage (from repository root)::

    # Edit configs/roughing.yaml, then run the full cycle:
    python scripts/experiment.py run --config configs/roughing.yaml

    # Or stage by stage against the latest auto-created run directory:
    python scripts/experiment.py train --config configs/roughing.yaml
    python scripts/experiment.py eval  --run latest
    python scripts/experiment.py plot  --run latest
    python scripts/experiment.py list

All artifacts land under ``runs/ppo/<mode>_<timestamp>/`` (models, TensorBoard
logs, evaluation trajectories, plots). You do not pass output directories by hand.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_paths import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    evaluation_dir,
    list_runs,
    load_run_manifest,
    logs_dir,
    models_dir,
    plots_dir,
    resolve_run_dir,
    write_latest_pointer,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyYAML is required for experiment configs. "
            "Install with: pip install pyyaml"
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyYAML is required for experiment configs. "
            "Install with: pip install pyyaml"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _python() -> str:
    return sys.executable


def _resolve_output_root(value: str | Path | None) -> Path:
    root = Path(value or DEFAULT_OUTPUT_ROOT).expanduser()
    if not root.is_absolute():
        root = (_REPO_ROOT / root).resolve()
    else:
        root = root.resolve()
    return root


def _run_subprocess(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("\n>> " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd or _REPO_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _bool_flag(name: str, value: bool) -> list[str]:
    return [name if value else name.replace("--", "--no-", 1)]


def _config_to_train_argv(cfg: dict[str, Any], overrides: argparse.Namespace) -> list[str]:
    """Translate YAML + CLI overrides into train_ppo_unified.py argv."""
    merged = dict(cfg)
    evaluation = merged.pop("evaluation", None)
    if evaluation is not None and not isinstance(evaluation, dict):
        raise SystemExit("'evaluation' in the config must be a mapping.")

    # Explicit CLI overrides win over YAML.
    for key in (
        "seeds",
        "env_id",
        "reward",
        "ap_max",
        "output_root",
        "run_name",
        "total_timesteps",
        "n_envs",
        "vec_env",
        "start_method",
        "dt",
        "n_substeps",
        "controller_hold_env_steps",
        "max_episode_steps",
        "dynamics_uncertainty_std",
        "learning_rate",
        "final_learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "final_clip_range",
        "n_eval_episodes",
        "callback_eval_seed",
        "eval_freq",
        "checkpoint_freq",
        "device",
        "preflight_only",
    ):
        value = getattr(overrides, key, None)
        if value is not None:
            merged[key] = value

    if getattr(overrides, "randomize_y0", None) is not None:
        merged["randomize_y0"] = bool(overrides.randomize_y0)
    if getattr(overrides, "progress_bar", None) is not None:
        merged["progress_bar"] = bool(overrides.progress_bar)

    argv: list[str] = [_python(), str(_SCRIPTS_DIR / "train_ppo_unified.py")]

    seeds = merged.get("seeds", [0])
    if isinstance(seeds, int):
        seeds = [seeds]
    argv.extend(["--seeds", *[str(int(s)) for s in seeds]])

    def add_opt(flag: str, key: str, *, as_bool: bool = False) -> None:
        if key not in merged or merged[key] is None:
            return
        value = merged[key]
        if as_bool:
            argv.extend(_bool_flag(flag, bool(value)))
        else:
            argv.extend([flag, str(value)])

    add_opt("--env-id", "env_id")
    add_opt("--reward", "reward")
    add_opt("--ap-max", "ap_max")
    add_opt("--output-root", "output_root")
    add_opt("--run-name", "run_name")
    add_opt("--total-timesteps", "total_timesteps")
    add_opt("--n-envs", "n_envs")
    add_opt("--vec-env", "vec_env")
    add_opt("--start-method", "start_method")
    add_opt("--dt", "dt")
    add_opt("--n-substeps", "n_substeps")
    add_opt("--controller-hold-env-steps", "controller_hold_env_steps")
    add_opt("--max-episode-steps", "max_episode_steps")
    add_opt("--dynamics-uncertainty-std", "dynamics_uncertainty_std")
    add_opt("--randomize-y0", "randomize_y0", as_bool=True)
    add_opt("--learning-rate", "learning_rate")
    add_opt("--final-learning-rate", "final_learning_rate")
    add_opt("--n-steps", "n_steps")
    add_opt("--batch-size", "batch_size")
    add_opt("--n-epochs", "n_epochs")
    add_opt("--gamma", "gamma")
    add_opt("--gae-lambda", "gae_lambda")
    add_opt("--clip-range", "clip_range")
    add_opt("--final-clip-range", "final_clip_range")
    add_opt("--n-eval-episodes", "n_eval_episodes")
    add_opt("--callback-eval-seed", "callback_eval_seed")
    add_opt("--eval-freq", "eval_freq")
    add_opt("--checkpoint-freq", "checkpoint_freq")
    add_opt("--device", "device")
    add_opt("--progress-bar", "progress_bar", as_bool=True)

    if merged.get("preflight_only"):
        argv.append("--preflight-only")

    return argv


def _find_new_run_dir(output_root: Path, before: set[Path]) -> Path:
    after = {p.resolve() for p in list_runs(output_root)}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if created:
        return created[0]
    # Fallback: LATEST pointer written by the trainer
    from run_paths import read_latest_pointer

    pointed = read_latest_pointer(output_root)
    if pointed is not None:
        return pointed
    raise FileNotFoundError(
        f"Training finished but no new run directory was found under {output_root}."
    )


def cmd_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (_REPO_ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    cfg = _load_yaml(config_path)
    output_root = _resolve_output_root(args.output_root or cfg.get("output_root"))

    before = {p.resolve() for p in list_runs(output_root)}
    # Force absolute output-root in overrides so the trainer LATEST pointer is stable.
    args.output_root = str(output_root)
    train_argv = _config_to_train_argv(cfg, args)
    _run_subprocess(train_argv)

    if args.preflight_only:
        return 0

    run_dir = _find_new_run_dir(output_root, before)
    write_latest_pointer(run_dir, output_root)
    frozen = dict(cfg)
    frozen["_source_config"] = str(config_path)
    frozen["_run_dir"] = str(run_dir)
    _dump_yaml(run_dir / "config.yaml", frozen)
    print(f"\nFrozen config written to: {run_dir / 'config.yaml'}")
    print(f"LATEST -> {run_dir}")
    return 0


def _eval_settings_from_run(run_dir: Path, cfg_eval: dict[str, Any]) -> list[str]:
    """Build eval_policy argv that only needs --run-dir (settings from training)."""
    argv = [
        _python(),
        str(_SCRIPTS_DIR / "eval_policy.py"),
        "--run-dir",
        str(run_dir),
        "--n-episodes",
        str(int(cfg_eval.get("n_episodes", 5))),
        "--n-mc",
        str(int(cfg_eval.get("n_mc", 5))),
    ]
    if cfg_eval.get("overwrite", True):
        argv.append("--overwrite")
    if cfg_eval.get("prefer_final", False):
        argv.append("--prefer-final")

    # Seeds from manifest / training config when present
    try:
        manifest = load_run_manifest(run_dir)
        seeds = manifest.get("arguments", {}).get("seeds")
        if seeds:
            argv.extend(["--seeds", *[str(int(s)) for s in seeds]])
    except FileNotFoundError:
        pass
    return argv


def cmd_eval(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(getattr(args, "output_root", None))
    run_dir = resolve_run_dir(
        run=None if args.run_dir else args.run,
        run_dir=args.run_dir,
        output_root=output_root,
    )
    write_latest_pointer(run_dir, run_dir.parent)

    cfg_eval: dict[str, Any] = {}
    frozen = run_dir / "config.yaml"
    if frozen.is_file():
        cfg_eval = dict(_load_yaml(frozen).get("evaluation") or {})
    if args.config:
        cfg_eval.update(_load_yaml(Path(args.config).expanduser().resolve()).get("evaluation") or {})

    # CLI overrides for eval intensity
    if args.n_episodes is not None:
        cfg_eval["n_episodes"] = args.n_episodes
    if args.n_mc is not None:
        cfg_eval["n_mc"] = args.n_mc
    if args.overwrite:
        cfg_eval["overwrite"] = True
    if args.prefer_final:
        cfg_eval["prefer_final"] = True

    argv = _eval_settings_from_run(run_dir, cfg_eval)
    if args.ap is not None:
        argv.extend(["--ap", str(args.ap)])
    _run_subprocess(argv)
    print(f"\nEvaluation artifacts: {evaluation_dir(run_dir)}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(getattr(args, "output_root", None))
    run_dir = resolve_run_dir(
        run=None if args.run_dir else args.run,
        run_dir=args.run_dir,
        output_root=output_root,
    )
    argv = [
        _python(),
        str(_SCRIPTS_DIR / "plot_results.py"),
        "--run-dir",
        str(run_dir),
    ]
    if args.seeds:
        argv.extend(["--seeds", *[str(int(s)) for s in args.seeds]])
    _run_subprocess(argv)
    print(f"\nPlots written under: {plots_dir(run_dir)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Full cycle: train → eval → plot into one auto-created run directory."""
    rc = cmd_train(args)
    if rc != 0 or args.preflight_only:
        return rc

    output_root = _resolve_output_root(args.output_root)
    # Prefer LATEST written by train
    run_dir = resolve_run_dir(run="latest", output_root=output_root)

    # Re-bind eval/plot to this run
    eval_args = argparse.Namespace(
        run="latest",
        run_dir=str(run_dir),
        output_root=str(run_dir.parent),
        config=args.config,
        n_episodes=args.n_episodes,
        n_mc=args.n_mc,
        overwrite=True,
        prefer_final=bool(getattr(args, "prefer_final", False)),
        ap=getattr(args, "ap", None),
    )
    cmd_eval(eval_args)

    plot_args = argparse.Namespace(
        run=None,
        run_dir=str(run_dir),
        output_root=str(run_dir.parent),
        seeds=getattr(args, "seeds", None),
    )
    cmd_plot(plot_args)

    print("\n========== EXPERIMENT COMPLETE ==========")
    print(f"Run directory : {run_dir}")
    print(f"  models      : {models_dir(run_dir)}")
    print(f"  logs / TB   : {logs_dir(run_dir)}")
    print(f"  evaluation  : {evaluation_dir(run_dir)}")
    print(f"  plots       : {plots_dir(run_dir)}")
    print(f"TensorBoard   : tensorboard --logdir \"{logs_dir(run_dir)}\"")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = _resolve_output_root(args.output_root)
    runs = list_runs(root)
    if not runs:
        print(f"No runs under {root}")
        return 0
    print(f"Runs under {root} (newest first):\n")
    for path in runs:
        status = "?"
        created = ""
        try:
            manifest = load_run_manifest(path)
            status = str(manifest.get("status", "?"))
            created = str(manifest.get("created_at", ""))
        except Exception:
            pass
        marker = ""
        from run_paths import read_latest_pointer

        latest = read_latest_pointer(root)
        if latest is not None and latest.resolve() == path.resolve():
            marker = "  [LATEST]"
        print(f"  {path.name:40s}  status={status:10s}  {created}{marker}")
        print(f"    {path}")
    return 0


def _add_train_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--env-id", default=None,
                        choices=["CustomODEPlate-v0", "CustomODEPlateFinish-v0"])
    parser.add_argument("--reward", default=None,
                        choices=["dense", "productive", "quadratic", "sparse"])
    parser.add_argument("--ap-max", type=float, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default=None)
    parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork"], default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--n-substeps", type=int, default=None)
    parser.add_argument("--controller-hold-env-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--dynamics-uncertainty-std", type=float, default=None)
    parser.add_argument("--randomize-y0", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--final-learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--final-clip-range", type=float, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=None)
    parser.add_argument("--callback-eval-seed", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    # Eval intensity when used with `run`
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Post-training eval outer episodes (run/eval).")
    parser.add_argument("--n-mc", type=int, default=None,
                        help="Post-training eval MC replicas (run/eval).")
    parser.add_argument("--prefer-final", action="store_true",
                        help="Eval final_<seed>.zip instead of best checkpoint.")
    parser.add_argument("--ap", type=float, default=None,
                        help="Finishing-only: fixed ap [mm] for evaluation.")


def _add_run_select_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", default="latest",
                        help="Run name under output-root, absolute path, or 'latest'.")
    parser.add_argument("--run-dir", default=None,
                        help="Explicit run directory (overrides --run).")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Organized PPO experiment workflow (config → train → eval → plot).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Train, then evaluate and plot into the new run dir.")
    _add_train_override_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_train = sub.add_parser("train", help="Train only; creates an auto-named run directory.")
    _add_train_override_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate policies for a run (default: latest).")
    _add_run_select_args(p_eval)
    p_eval.add_argument("--config", default=None,
                        help="Optional YAML whose 'evaluation' block sets n_episodes/n_mc.")
    p_eval.add_argument("--n-episodes", type=int, default=None)
    p_eval.add_argument("--n-mc", type=int, default=None)
    p_eval.add_argument("--overwrite", action="store_true")
    p_eval.add_argument("--prefer-final", action="store_true")
    p_eval.add_argument("--ap", type=float, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_plot = sub.add_parser("plot", help="Plot training + evaluation for a run (default: latest).")
    _add_run_select_args(p_plot)
    p_plot.add_argument("--seeds", nargs="*", type=int, default=None)
    p_plot.set_defaults(func=cmd_plot)

    p_list = sub.add_parser("list", help="List organized runs under runs/ppo/.")
    p_list.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p_list.set_defaults(func=cmd_list)

    return parser


def main() -> int:
    # Ensure we resolve relative paths from the repo root
    if Path.cwd().resolve() != _REPO_ROOT.resolve():
        # Still allow running from elsewhere; subprocess cwd is repo root.
        pass
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
