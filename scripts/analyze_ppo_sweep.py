#!/usr/bin/env python3
"""Scientific post-processing of an existing Fast/Balanced/Precise PPO sweep.

Additive; does not change training, evaluation, checkpoint selection, or `LATEST`.

`plot`: mean +/- sample SD over independent training seeds for each profile.
`select`: a *screening* recommendation from an explicit held-out evaluation set.
           Never uses a time-ordered/latest-run pointer to pick a policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Reuse tested v3 plot functions; do not duplicate statistical/figure code.
from plot_ppo_sweep import (
    _plot_reliability_metrics,
    _plot_return_curve,
    _plot_training_diagnostics,
    _write_csv,
    build_common_grid,
    load_seed_metrics,
    _validate_run_records,
)

PROFILES = ("fast", "balanced", "precise")
LABELS = {p: p.title() for p in PROFILES}
SAFETY_REASONS = {"excessive_sensor_displacement", "invalid_state"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def _latest_run_group(sweep_root: Path, mode: str | None) -> tuple[dict[str, Path], Path]:
    """Resolve the newest *complete* sweep group; never merge unrelated sweeps.

    v3 manifests are preferred within a group. Older groups with Fast/Balanced/
    Precise subdirectories are supported when they lack a sweep manifest.
    The optional mode avoids accidentally mixing roughing and finishing.
    """
    root = sweep_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Sweep output root does not exist: {root}")
    modes = [mode] if mode else ["roughing", "finishing"]
    candidates: list[tuple[float, str, dict[str, Path], Path]] = []

    def group_timestamp(group: Path, manifest_path: Path | None, fallback: float) -> float:
        # Sweep directory timestamps are stable even when the experiment folder
        # is copied or plot files are regenerated later.
        match = re.search(r"\d{8}_\d{6}", group.name)
        if match:
            try:
                return datetime.strptime(match.group(), "%Y%m%d_%H%M%S").timestamp()
            except ValueError:
                pass
        if manifest_path and manifest_path.is_file():
            completed_at = read_json(manifest_path).get("completed_at")
            if completed_at:
                try:
                    return datetime.fromisoformat(str(completed_at)).timestamp()
                except ValueError:
                    pass
        return fallback
    for mode_name in modes:
        mode_root = root / mode_name
        if not mode_root.is_dir():
            continue
        for group in sorted((p for p in mode_root.iterdir() if p.is_dir())):
            manifest_file = group / "sweep_manifest.json"
            found: dict[str, Path] = {}
            if manifest_file.is_file():
                try:
                    manifest = read_json(manifest_file)
                    if manifest.get("status") != "completed" or manifest.get("mode", mode_name) != mode_name:
                        continue
                    for entry in manifest.get("configurations", []):
                        name = str(entry.get("name", ""))
                        if name not in PROFILES or entry.get("status") != "completed" or not entry.get("run_dir"):
                            continue
                        raw = Path(str(entry["run_dir"])).expanduser()
                        found[name] = (raw if raw.is_absolute() else group / raw).resolve()
                except (ValueError, TypeError, KeyError):
                    continue
                # Do not bypass a manifest that reports an incomplete sweep.
            else:
                # Older sweeps: <mode>/<sweep-id>/<profile>/<run>/run_manifest.json.
                for name in PROFILES:
                    profile_dir = group / name
                    if not profile_dir.is_dir():
                        continue
                    completed = []
                    for candidate_file in profile_dir.glob("*/run_manifest.json"):
                        try:
                            if read_json(candidate_file).get("status") == "completed":
                                completed.append(candidate_file)
                        except (OSError, ValueError):
                            continue
                    if completed:
                        chosen = max(completed, key=lambda path: (path.stat().st_mtime, str(path)))
                        found[name] = chosen.parent.resolve()
            if set(found) != set(PROFILES):
                continue
            if len(set(found.values())) != len(PROFILES):
                continue
            if all((r / "run_manifest.json").is_file() for r in found.values()):
                newest = (manifest_file.stat().st_mtime if manifest_file.is_file()
                          else max((r / "run_manifest.json").stat().st_mtime for r in found.values()))
                order_time = group_timestamp(group, manifest_file if manifest_file.is_file() else None, newest)
                candidates.append((order_time, mode_name, {name: found[name] for name in PROFILES}, group))

    if not candidates:
        suffix = f" for mode {mode!r}" if mode else ""
        raise FileNotFoundError(f"No complete three-profile sweep group found under {root}{suffix}.")
    available_modes = {item[1] for item in candidates}
    if mode is None and len(available_modes) > 1:
        raise ValueError(
            "Completed sweeps exist for BOTH roughing and finishing. "
            "Specify --mode roughing or --mode finishing to avoid selecting the wrong experiment."
        )
    newest, selected_mode, runs, group = max(candidates, key=lambda item: (item[0], str(item[3])))
    print(f"Automatically selected latest complete {selected_mode} sweep: {group}")
    for name, folder in runs.items():
        print(f"  {name:8s}: {folder}")
    return runs, group


def resolve_runs(args: argparse.Namespace) -> dict[str, Path]:
    """Use --latest, an explicit manifest, or the original three path flags."""
    supplied = [getattr(args, p, None) for p in PROFILES]
    latest = bool(getattr(args, "latest", False))
    mode = getattr(args, "mode", None)
    if latest:
        if getattr(args, "manifest", None) or any(supplied):
            raise ValueError("--latest cannot be combined with --manifest or --fast/--balanced/--precise.")
        sweep_root = getattr(args, "sweep_root", None)
        if sweep_root is None:
            sweep_root = Path(__file__).resolve().parents[1] / "runs" / "ppo_sweep"
        out, group = _latest_run_group(Path(sweep_root), mode)
        args._selected_group = group
    elif args.manifest:
        if mode:
            raise ValueError("--mode is only used together with --latest.")
        if any(supplied):
            raise ValueError("Specify either --manifest or individual --fast/--balanced/--precise run paths.")
        manifest_file = Path(args.manifest).expanduser().resolve()
        manifest = read_json(manifest_file)
        if manifest.get("status") != "completed":
            raise ValueError("Sweep manifest is not completed; incomplete sweeps must not be compared.")
        out: dict[str, Path] = {}
        for item in manifest.get("configurations", []):
            name = str(item.get("name", ""))
            if name not in PROFILES:
                continue
            if name in out:
                raise ValueError(f"Duplicate sweep manifest configuration: {name}")
            if item.get("status") != "completed" or not item.get("run_dir"):
                raise ValueError(f"Incomplete configuration in sweep manifest: {name}")
            raw = Path(str(item["run_dir"])).expanduser()
            out[name] = (raw if raw.is_absolute() else manifest_file.parent / raw).resolve()
        if len(out) < 2:
            raise ValueError("At least two completed PPO profiles are required.")
    else:
        if mode:
            raise ValueError("--mode is only used together with --latest.")
        if not all(supplied):
            raise ValueError("Specify --latest, --manifest, or all three --fast/--balanced/--precise paths.")
        out = {name: Path(getattr(args, name)).expanduser().resolve() for name in PROFILES}

    out = {name: out[name] for name in PROFILES if name in out}
    if len(set(out.values())) != len(out):
        raise ValueError("Each configuration must point to a distinct run directory.")
    for name, run_dir in out.items():
        if not (run_dir / "run_manifest.json").is_file():
            raise FileNotFoundError(f"Missing run_manifest.json for {name}: {run_dir}")
    return out


def validate_v3_sweep_manifest(args: argparse.Namespace, runs: dict[str, Path]) -> None:
    """Reuse v3's strict training-configuration/provenance validation when available."""
    path: Path | None = None
    if getattr(args, "manifest", None):
        path = Path(args.manifest).expanduser().resolve()
    elif getattr(args, "latest", False) and getattr(args, "_selected_group", None):
        candidate = args._selected_group / "sweep_manifest.json"
        if candidate.is_file():
            path = candidate
    if path is None:
        return  # Older sweep groups and explicit run directories remain supported.
    manifest = read_json(path)
    if manifest.get("schema_version") != 3:
        return
    seeds = [int(s) for s in manifest.get("seeds", [])]
    entries = {str(item.get("name")): item for item in manifest.get("configurations", [])}
    if set(entries) != set(runs):
        raise ValueError("Version-3 manifest profiles do not match the selected run directories.")
    if len(seeds) < 2:
        raise ValueError("Version-3 manifest must contain at least two training seeds.")
    for name, run_dir in runs.items():
        _validate_run_records(run_dir, entries[name], manifest, seeds)


def check_training_runs(runs: dict[str, Path]) -> tuple[list[int], dict[str, dict[int, dict]]]:
    """Validate paired seed sets, data completeness and effective plant/reward settings."""
    all_histories: dict[str, dict[int, dict]] = {}
    common_seeds: list[int] | None = None
    reference_conditions: str | None = None
    reference_callback: tuple[int, int] | None = None
    reference_budget: int | None = None
    for name, run_dir in runs.items():
        manifest = read_json(run_dir / "run_manifest.json")
        if manifest.get("status") != "completed":
            raise ValueError(f"Training run was not marked completed: {name}, {run_dir}")
        seeds = [int(x) for x in manifest.get("arguments", {}).get("seeds", [])]
        if len(seeds) < 2 or len(seeds) != len(set(seeds)):
            raise ValueError(f"At least two independent training seeds are required in {name}")
        if common_seeds is None:
            common_seeds = seeds
        elif seeds != common_seeds:
            raise ValueError(f"Seed IDs differ across profiles: {name} has {seeds}; expected {common_seeds}")
        budget = int(manifest.get("arguments", {}).get("total_timesteps", -1))
        if budget <= 0 or (reference_budget is not None and reference_budget != budget):
            raise ValueError(f"Different or invalid training budgets for configuration {name}")
        reference_budget = budget
        all_histories[name] = {}
        for seed in seeds:
            d = run_dir / "logs" / f"seed_{seed}"
            config = read_json(d / "training_config.json")
            summary = read_json(d / "training_summary.json")
            if int(config.get("seed", -1)) != seed or int(summary.get("seed", -1)) != seed:
                raise ValueError(f"Training seed metadata mismatch: {d}")
            if summary.get("status") != "completed" or int(summary.get("timesteps_reported_by_model", -1)) != budget:
                raise ValueError(f"Incomplete PPO training budget: {d}")
            env = config.get("environment", {})
            # The effective environment kwargs contain the actual reward coefficients,
            # unlike inherited YAML sections which are not necessarily forwarded.
            conditions = {
                "env_id": env.get("env_id"),
                "reward_id": env.get("reward_id"),
                "kwargs": env.get("kwargs"),
                "sample_dt": env.get("controller_sample_time_s"),
                "max_episode_steps": env.get("max_episode_steps"),
            }
            signature = json.dumps(conditions, sort_keys=True, allow_nan=False)
            if reference_conditions is None:
                reference_conditions = signature
            elif reference_conditions != signature:
                raise ValueError(f"Effective physical/reward conditions differ: {name}, training seed {seed}")
            cb = config.get("evaluation_callback", {})
            episodes = int(cb.get("episodes_per_evaluation", -1))
            callback_seed = int(cb.get("evaluation_env_seed", -1))
            if episodes < 1 or callback_seed < 0:
                raise ValueError(f"Missing callback evaluation metadata: {d}")
            if reference_callback is None:
                reference_callback = (episodes, callback_seed)
            elif reference_callback != (episodes, callback_seed):
                raise ValueError(f"Different evaluation episode counts/seeds in {name} seed {seed}")
            all_histories[name][seed] = load_seed_metrics(run_dir, seed, expected_eval_episodes=episodes)
    assert common_seeds is not None
    curves = [all_histories[name][seed]["evaluation_return"] for name in runs for seed in common_seeds]
    build_common_grid(curves)  # Explicitly rejects silently interpolated evaluation checkpoints.
    return common_seeds, all_histories


def plot(runs: dict[str, Path], out_dir: Path, *, require_diagnostics: bool = False) -> None:
    seeds, histories = check_training_runs(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_seed_config = read_json(
        next(iter(runs.values())) / "logs" / f"seed_{seeds[0]}" / "training_config.json"
    )
    hold_steps = int(primary_seed_config.get("environment", {}).get("controller_hold_env_steps", 1))
    excluded_metrics = (
        frozenset({"median_completed_productivity", "mean_max_displacement_m"})
        if hold_steps > 1 else frozenset()
    )
    if excluded_metrics:
        print("WARNING: controller_hold_env_steps>1. Existing training callback records "
              "only the last fine-step reward components and sensor peak per held action. "
              "Suppressing affected reliability panels, but evaluation-return curves remain valid.")
    provenance = {
        "excluded_training_callback_metrics": sorted(excluded_metrics),
        "controller_hold_env_steps": hold_steps,
        "environment_id": read_json(
            next(iter(runs.values())) / "logs" / f"seed_{seeds[0]}" / "training_config.json"
        ).get("environment", {}).get("env_id"),
        "run_directories": {name: str(folder) for name, folder in runs.items()},
        "independent_training_seeds": seeds,
        "evaluation_episodes": int(read_json(
            next(iter(runs.values())) / "logs" / f"seed_{seeds[0]}" / "training_config.json"
        )["evaluation_callback"]["episodes_per_evaluation"]),
        "learning_and_reliability": "Mean of seed-level metrics ± 1 sample SD across independent training seeds, ddof=1; no interpolation between evaluation checkpoints.",
        "training_diagnostics": "Seed-level TensorBoard metrics are aggregated only at exactly shared, actually recorded update steps within each configuration, by mean ± 1 sample SD. Configurations retain their own step grids; no interpolation, extrapolation or smoothing.",
        "selection_note": "Plots do not select a best policy; checkpoint screening and held-out configuration selection are separate procedures.",
    }
    (out_dir / "ppo_sweep_plot_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    labels = {name: LABELS[name] for name in runs}
    return_stats, summary_rows = _plot_return_curve(histories, labels, seeds, out_dir)
    reliability_stats = _plot_reliability_metrics(
        histories, labels, seeds, out_dir, excluded_metrics=excluded_metrics
    )
    diagnostic_stats = _plot_training_diagnostics(runs, labels, seeds, out_dir)
    provenance["available_training_diagnostic_tags"] = sorted({row[0] for row in diagnostic_stats})
    provenance["tensorboard_diagnostics_generated"] = bool(diagnostic_stats)
    (out_dir / "ppo_sweep_plot_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if require_diagnostics:
        diagnostic_tags = {row[0] for row in diagnostic_stats}
        core = {"train/explained_variance", "train/approx_kl", "train/clip_fraction"}
        if not core.issubset(diagnostic_tags) or not diagnostic_tags.intersection({"train/std", "train/entropy_loss"}):
            raise RuntimeError(
                "Required PPO diagnostics are incomplete. Need explained variance, "
                "approximate KL divergence, clipping fraction, and either policy "
                "standard deviation or entropy loss for every seed/configuration."
            )
    _write_csv(out_dir / "ppo_sweep_learning_curve_stats.csv", [
        "configuration", "training_transitions", "n_training_seeds", "mean_eval_return",
        "sample_sd_across_seeds", *[f"seed_{s}_mean_return" for s in seeds],
    ], return_stats)
    _write_csv(out_dir / "ppo_sweep_reliability_metrics_stats.csv", [
        "metric", "configuration", "training_transitions", "n_training_seeds",
        "mean_across_seeds", "sample_sd_across_seeds", *[f"seed_{s}_value" for s in seeds],
    ], reliability_stats)
    _write_csv(out_dir / "ppo_sweep_final_summary.csv", [
        "configuration", "label", "n_training_seeds", "final_common_timestep",
        "final_mean_eval_return", "final_sample_sd", "peak_mean_return_timestep", "peak_mean_return",
    ], summary_rows)
    if diagnostic_stats:
        _write_csv(out_dir / "ppo_sweep_training_diagnostics_stats.csv", [
            "tensorboard_tag", "configuration", "training_transitions", "n_training_seeds",
            "mean_across_seeds", "sample_sd_across_seeds", *[f"seed_{s}_value" for s in seeds],
        ], diagnostic_stats)
    raw_rows: list[list[Any]] = []
    for name, run_dir in runs.items():
        for seed in seeds:
            with np.load(run_dir / "logs" / f"seed_{seed}" / "evaluations.npz", allow_pickle=False) as record:
                steps = np.asarray(record["timesteps"], dtype=np.int64).reshape(-1)
                outcomes = np.asarray(record["results"], dtype=np.float64)
            if outcomes.ndim == 1:
                outcomes = outcomes[:, None]
            for step, episode_returns in zip(steps, outcomes):
                for episode_number, episode_return in enumerate(episode_returns, start=1):
                    raw_rows.append([name, seed, int(step), episode_number, float(episode_return)])
    _write_csv(out_dir / "ppo_sweep_raw_evaluation_returns.csv", [
        "configuration", "training_seed", "training_transitions", "evaluation_episode", "episode_return",
    ], raw_rows)
    print(f"Cross-profile figures and aggregate CSV tables saved in: {out_dir}")
    print("Each line is a configuration-level mean; each band is +/-1 sample SD across independent training seeds.")
    if not diagnostic_stats:
        print("NOTE: No common TensorBoard diagnostic series were found; learning and reliability plots remain valid.")


def _float(row: dict[str, str], key: str) -> float:
    try:
        x = float(row[key])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Required finite numeric evaluation column is missing/invalid: {key}") from exc
    if not math.isfinite(x):
        raise ValueError(f"Non-finite evaluation value in {key}")
    return x


def _bool(row: dict[str, str], key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"Required evaluation boolean {key} must be true/false, received {value!r}")
    return value in {"true", "1"}


def load_evaluation(runs: dict[str, Path]) -> tuple[dict[str, dict[int, dict]], dict[str, Any]]:
    """Load held-out evaluation, validating paired physical scenarios and checkpoints."""
    training_seeds, _ = check_training_runs(runs)
    records: dict[str, dict[int, dict]] = {}
    common_protocol: tuple[Any, ...] | None = None
    reference_scenarios: dict[int, list[tuple]] | None = None
    report_meta: dict[str, Any] = {}
    for name, run_dir in runs.items():
        d = run_dir / "evaluation"
        manifest = read_json(d / "evaluation_manifest.json")
        protocol = (
            manifest.get("env_id"),
            int(manifest.get("evaluation_seed_base", -1)),
            int(manifest.get("n_episodes", -1)),
            int(manifest.get("n_mc", -1)),
            manifest.get("controller_hold_env_steps"),
            manifest.get("controller_sample_time_s"),
            bool(manifest.get("policy_deterministic")),
        )
        if protocol[1] < 0 or protocol[2] < 1 or protocol[3] < 1 or not protocol[-1]:
            raise ValueError(f"Invalid or non-deterministic held-out evaluation for {name}")
        # Ensure optional evaluation overrides match as well (the selected policy
        # may be compared only under the same process/noise settings).
        option_names = ("plant_override", "y0", "y_position", "ap", "ap_max",
                        "dynamics_uncertainty_std", "randomize_y0", "dt", "n_substeps",
                        "policy_hold_env_steps", "reward", "max_episode_steps", "prefer_final")
        supplied_args = manifest.get("arguments", {})
        option_signature = {key: supplied_args.get(key) for key in option_names}
        if common_protocol is None:
            common_protocol = protocol
            report_meta = {
                "env_id": protocol[0], "evaluation_seed_base": protocol[1],
                "n_episodes": protocol[2], "n_mc": protocol[3],
                "matched_evaluation_overrides": option_signature,
            }
        elif common_protocol != protocol:
            raise ValueError(f"Different held-out evaluation protocols for profile {name}")
        elif option_signature != report_meta["matched_evaluation_overrides"]:
            raise ValueError(f"Different evaluation overrides/physical scenarios for profile {name}")
        # Check the selection set is distinct from training callback episodes.
        callback_config = read_json(run_dir / "logs" / f"seed_{training_seeds[0]}" / "training_config.json")
        cb = callback_config.get("evaluation_callback", {})
        train_seed_base = int(cb.get("evaluation_env_seed", -1))
        train_count = int(cb.get("episodes_per_evaluation", -1))
        if train_seed_base < 0 or train_count < 1:
            raise ValueError(f"Cannot verify disjoint screening and training callback scenarios for {name}")
        training_ids = set(range(train_seed_base, train_seed_base + train_count))
        heldout_ids = set(range(protocol[1], protocol[1] + protocol[2] * protocol[3]))
        if training_ids & heldout_ids:
            raise ValueError(f"Selection evaluation seed range overlaps the training callback evaluation scenarios: {name}")
        models = manifest.get("models", [])
        if sorted(int(x["training_seed"]) for x in models) != sorted(training_seeds):
            raise ValueError(f"Held-out evaluation does not contain all expected training seeds: {name}")
        for model in models:
            cp = Path(str(model.get("checkpoint_path", "")))
            if cp.name != "best_model.zip" or not cp.is_file():
                raise ValueError(
                    f"Held-out evaluation for {name} must use a checkpoint-selected best_model.zip, "
                    "not an arbitrary final/latest checkpoint."
                )
        replicas_path = d / "physical_outcomes_replicas.csv"
        if not replicas_path.is_file():
            raise FileNotFoundError(f"Independent evaluation outcomes missing: {replicas_path}")
        with replicas_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            seed = int(row["training_seed"])
            if seed not in training_seeds:
                raise ValueError(f"Unexpected seed {seed} in evaluation of {name}")
            by_seed[seed].append(row)
        if sorted(by_seed) != sorted(training_seeds):
            raise ValueError(f"Incomplete held-out evaluation rows for {name}")
        scenario_by_seed: dict[int, list[tuple]] = {}
        for seed in training_seeds:
            rs = sorted(by_seed[seed], key=lambda r: (int(r["scenario_index"]), int(r["mc_index"])))
            expected = protocol[2] * protocol[3]
            if len(rs) != expected:
                raise ValueError(f"Expected {expected} independent scenario/replica outcomes for {name} seed {seed}; got {len(rs)}")
            scenario_list = [(int(r["scenario_index"]), int(r["mc_index"]), int(r["reset_seed"]),
                              _float(r, "y_line_m"),
                              str(r.get("ap_fixed_mm", ""))) for r in rs]
            if len(set((s, mc) for s, mc, *_ in scenario_list)) != expected:
                raise ValueError(f"Duplicate held-out scenario/replica index for {name} seed {seed}")
            scenario_by_seed[seed] = scenario_list
            data = {
                "completion": float(np.mean([_bool(r, "pass_completed") for r in rs])),
                "safety": float(np.mean([r["termination_reason"] in SAFETY_REASONS for r in rs])),
                "progress": float(np.mean([_float(r, "feed_progress") for r in rs])),
                "return": float(np.mean([_float(r, "return") for r in rs])),
                "rms_displacement_m": float(np.mean([_float(r, "rms_w_m") for r in rs])),
            }
            records.setdefault(name, {})[seed] = data
        if reference_scenarios is None:
            reference_scenarios = scenario_by_seed
        else:
            for seed in training_seeds:
                orig = reference_scenarios[seed]
                newer = scenario_by_seed[seed]
                if len(orig) != len(newer):
                    raise ValueError("Held-out scenario counts differ between configurations.")
                for a, b in zip(orig, newer):
                    if a[:3] != b[:3] or not math.isclose(a[3], b[3], rel_tol=0, abs_tol=1e-10) or a[4] != b[4]:
                        raise ValueError(f"Held-out scenario inputs differ for configuration {name}, seed {seed}")
    assert common_protocol is not None
    return records, report_meta


def screen(runs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    """Cheap *preliminary* configuration screening from saved callback metrics.

    Every seed already has a reliability-ranked best_model.zip checkpoint. These
    checkpoint-selection metrics are *not* independent test outcomes.
    """
    seeds, _ = check_training_runs(runs)
    mode = None
    per_config: dict[str, dict[str, float]] = {}
    all_metrics: dict[str, dict[int, dict[str, float]]] = {}
    wanted = ("pass_completion_rate", "safety_termination_rate",
              "median_completed_productivity", "median_progress", "mean_return",
              "mean_max_displacement_m")
    for name, run_dir in runs.items():
        by_seed = {}
        for seed in seeds:
            config = read_json(run_dir / "logs" / f"seed_{seed}" / "training_config.json")
            mode = config.get("environment", {}).get("env_id")
            summary = read_json(run_dir / "logs" / f"seed_{seed}" / "training_summary.json")
            metrics = summary.get("best_evaluation_metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"No reliability-selected checkpoint metrics in {name} seed {seed}")
            parsed: dict[str, float] = {}
            for key in wanted:
                try:
                    val = float(metrics[key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Missing {key} in {name} seed {seed} best metrics") from exc
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite metric {key} in {name} seed {seed}")
                parsed[key] = val
            checkpoint = run_dir / "models" / f"best_{seed}" / "best_model.zip"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Best checkpoint missing: {checkpoint}")
            by_seed[seed] = parsed
        all_metrics[name] = by_seed
        per_config[name] = {key: float(np.mean([v[key] for v in by_seed.values()])) for key in wanted}

    def key(name: str):
        v = per_config[name]
        if mode == "CustomODEPlateFinish-v0":
            return (-v["pass_completion_rate"], v["safety_termination_rate"],
                    v["mean_max_displacement_m"], -v["median_progress"], -v["mean_return"], name)
        return (-v["pass_completion_rate"], v["safety_termination_rate"],
                -v["median_completed_productivity"], -v["median_progress"], -v["mean_return"], name)
    ranked = sorted(per_config, key=key)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ppo_training_checkpoint_screening.csv", ["configuration", "n_seeds", *wanted], [
        [name, len(seeds), *[per_config[name][k] for k in wanted]] for name in ranked
    ])
    result = {
        "status": "preliminary_screening_uses_checkpoint_selection_data",
        "rank": ranked,
        "candidate_configuration": ranked[0],
        "selected_checkpoints_per_seed": {
            name: {str(s): str((runs[name] / "models" / f"best_{s}" / "best_model.zip").resolve()) for s in seeds}
            for name in runs
        },
        "warning": "Checkpoint metrics reuse the training callback's selection scenarios and are optimistically biased. Do not use as final, independent performance evidence.",
        "next_step": "Independently evaluate all profiles to select a configuration; retest that choice against NMPC on different held-out scenarios.",
    }
    (out_dir / "ppo_training_checkpoint_screening.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Preliminary (not independently tested) candidate: {ranked[0]}")
    print("Saved checkpoint screening table; no evaluation or training was run.")
    return result


def _rank_key(values: dict[str, float], mode: str) -> tuple[float, ...]:
    """Prespecified screening order; finishing favours low RMS after reliability."""
    if mode == "CustomODEPlateFinish-v0":
        return (-values["completion"], values["safety"], values["rms_displacement_m"],
                -values["progress"], -values["return"])
    return (-values["completion"], values["safety"], -values["progress"],
            -values["return"], values["rms_displacement_m"])


def select(runs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    records, metadata = load_evaluation(runs)
    mode = metadata["env_id"]
    mean_by_name: dict[str, dict[str, float]] = {}
    sd_by_name: dict[str, dict[str, float]] = {}
    for name, by_seed in records.items():
        mean_by_name[name] = {
            metric: float(np.mean([row[metric] for row in by_seed.values()]))
            for metric in ("completion", "safety", "progress", "return", "rms_displacement_m")
        }
        sd_by_name[name] = {
            metric: float(np.std([row[metric] for row in by_seed.values()], ddof=1))
            for metric in mean_by_name[name]
        }
    ranked = sorted(mean_by_name, key=lambda n: (*_rank_key(mean_by_name[n], mode), n))
    winner = ranked[0]
    seed_ranked = sorted(records[winner], key=lambda s: (*_rank_key(records[winner][s], mode), s))
    # Median-ranked representative avoids choosing the most favourable single seed.
    representative_seed = seed_ranked[len(seed_ranked) // 2]
    checkpoint = (runs[winner] / "models" / f"best_{representative_seed}" / "best_model.zip").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Representative checkpoint missing: {checkpoint}")
    rows = []
    for name in ranked:
        stats = mean_by_name[name]
        sd = sd_by_name[name]
        rows.append([name, len(records[name]), *[stats[k] for k in stats],
                     *[sd[k] for k in stats]])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(next(iter(mean_by_name.values())))
    _write_csv(out_dir / "ppo_heldout_configuration_selection.csv", [
        "configuration", "n_training_seeds", *["mean_" + k for k in metrics],
        *["sample_sd_" + k for k in metrics],
    ], rows)
    result = {
        "status": "screening_recommendation_not_final_test",
        "selection_evaluation_protocol": metadata,
        "ranking_rule": "completion desc, safety failures asc, then roughing: progress desc/return desc/RMS asc; finishing: RMS asc/progress desc/return desc",
        "configuration_rank": ranked,
        "suggested_configuration": winner,
        "representative_training_seed": representative_seed,
        "representative_checkpoint_path": str(checkpoint),
        "recommended_paper_reporting": (
            "Use all independent training seeds of the chosen configuration in the final PPO-vs-NMPC comparison. "
            "The median-ranked seed is for an illustrative trajectory only, not the full statistical result."
        ),
        "independent_final_test_required": True,
        "warning": (
            "These held-out scenarios were used for model selection. Do not re-use them as independent "
            "unbiased final test results; evaluate the selected configuration on new scenario seeds."
        ),
    }
    (out_dir / "ppo_selection_recommendation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Suggested PPO configuration: {winner}; illustrative median-ranked seed: {representative_seed}")
    print(f"Representative checkpoint: {checkpoint}")
    print("FINAL TEST: evaluate selected configuration + NMPC on a new, matched scenario set.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plot", "screen", "select"))
    parser.add_argument("--manifest", help="Existing completed sweep_manifest.json (v3 or compatible).")
    parser.add_argument("--latest", action="store_true", help="Discover the newest complete three-configuration sweep automatically.")
    parser.add_argument("--mode", choices=("roughing", "finishing"), help="Mode to select when using --latest (required if both modes exist).")
    parser.add_argument("--sweep-root", default=None, help="Sweep search root for --latest (default: runs/ppo_sweep under the repository).")
    parser.add_argument("--resolve-only", action="store_true", help="Display selected run paths without reading data or generating outputs.")
    parser.add_argument("--require-diagnostics", action="store_true", help="For plot, fail if the common TensorBoard diagnostics are unavailable.")
    for name in PROFILES:
        parser.add_argument("--" + name, help=f"Individual {name} run directory (no sweep manifest required).")
    parser.add_argument("--out-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = resolve_runs(args)
    if args.resolve_only:
        for name, folder in runs.items():
            print(f"{name}: {folder}")
        return 0
    validate_v3_sweep_manifest(args, runs)
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif args.latest:
        out_dir = args._selected_group / "plots"
    elif args.manifest:
        out_dir = Path(args.manifest).expanduser().resolve().parent / "plots"
    else:
        out_dir = next(iter(runs.values())).parent.parent / "ppo_comparison_plots"
    if args.command != "plot" and args.require_diagnostics:
        raise ValueError("--require-diagnostics applies only to the plot command.")
    if args.command == "plot":
        plot(runs, out_dir, require_diagnostics=args.require_diagnostics)
    elif args.command == "screen":
        screen(runs, out_dir)
    else:
        select(runs, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
