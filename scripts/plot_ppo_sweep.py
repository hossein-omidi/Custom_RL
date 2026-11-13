#!/usr/bin/env python
"""Plot cross-seed PPO learning, optimization and reliability comparisons.

Evaluation metrics are first averaged within each training seed. Mean and
sample SD are then calculated across independent training seeds. Training
TensorBoard diagnostics are aggregated in the same seed-first order.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RELIABILITY_METRICS = {
    "pass_completion_rate": {
        "title": "Pass completion rate",
        "ylabel": "Rate [%]",
        "scale": 100.0,
        "ylim": (0.0, 100.0),
    },
    "safety_termination_rate": {
        "title": "Safety-termination rate",
        "ylabel": "Rate [%]",
        "scale": 100.0,
        "ylim": (0.0, 100.0),
    },
    "truncation_rate": {
        "title": "Time-limit truncation rate",
        "ylabel": "Rate [%]",
        "scale": 100.0,
        "ylim": (0.0, 100.0),
    },
    "median_completed_productivity": {
        "title": "Completed-pass productivity reward",
        "ylabel": "Cumulative reward",
        "scale": 1.0,
        "ylim": None,
    },
    "median_progress": {
        "title": "Final pass progress",
        "ylabel": "Pass progress [%]",
        "scale": 100.0,
        "ylim": (0.0, 100.0),
    },
    "mean_max_displacement_m": {
        "title": "Episode-wise peak sensor displacement",
        "ylabel": "Mean peak displacement [mm]",
        "scale": 1000.0,
        "ylim": None,
    },
}

DIAGNOSTIC_SPECS = {
    "rollout/ep_rew_mean": ("Training episode return", "Mean rolling episode return"),
    "train/explained_variance": ("Value-function fit", "Explained variance"),
    "train/approx_kl": ("Policy-update size", "Approximate KL divergence"),
    "train/clip_fraction": ("Policy clipping", "Clip fraction"),
    "train/std": ("Policy exploration", "Policy distribution SD (pre-action clipping)"),
    "train/entropy_loss": ("Policy exploration", "Entropy loss"),
}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    import hashlib
    return hashlib.sha256(encoded).hexdigest()


def _clean_curve(steps: np.ndarray, values: np.ndarray, *, source: Path) -> tuple[np.ndarray, np.ndarray]:
    steps = np.asarray(steps, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if steps.size != values.size or steps.size < 2:
        raise ValueError(f"Invalid curve lengths in {source}: {steps.size} steps, {values.size} values.")
    if not np.all(np.isfinite(steps)) or not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {source}; refusing to silently drop observations.")
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"Timesteps must be strictly increasing with no duplicates in {source}.")
    return steps, values


def load_seed_metrics(
    run_dir: Path,
    seed: int,
    *,
    expected_eval_episodes: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load one seed's callback history, preserving episode-first aggregation."""
    path = run_dir / "logs" / f"seed_{seed}" / "evaluations.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation history for training seed {seed}: {path}")

    required = {
        "timesteps", "results", "ep_lengths", *RELIABILITY_METRICS.keys(),
        "median_completed_return",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing required callback arrays: {missing}")
        steps = np.asarray(data["timesteps"], dtype=np.float64).reshape(-1)
        results = np.asarray(data["results"], dtype=np.float64)
        lengths = np.asarray(data["ep_lengths"], dtype=np.float64)
        metric_arrays = {
            key: np.asarray(data[key], dtype=np.float64).reshape(-1)
            for key in RELIABILITY_METRICS
        }
        completed_return = np.asarray(data["median_completed_return"], dtype=np.float64).reshape(-1)

    if results.ndim == 1:
        results = results.reshape(-1, 1)
    expected_shape = (steps.size, int(expected_eval_episodes))
    if results.shape != expected_shape:
        raise ValueError(f"Expected results shape {expected_shape} in {path}; received {results.shape}.")
    if lengths.shape != results.shape:
        raise ValueError(f"Episode-length array shape {lengths.shape} does not match results {results.shape} in {path}.")
    if not np.all(np.isfinite(results)) or not np.all(np.isfinite(lengths)):
        raise ValueError(f"Non-finite episode returns or lengths found in {path}.")

    steps, _ = _clean_curve(steps, np.arange(steps.size, dtype=float), source=path)
    return_mean = np.mean(results, axis=1)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "evaluation_return": (steps, return_mean),
    }
    for key, values in metric_arrays.items():
        if values.size != steps.size:
            raise ValueError(f"Invalid {key!r} length in {path}: expected {steps.size}, received {values.size}.")
        curves[key] = _clean_curve(steps, values, source=path)
    if completed_return.size != steps.size or not np.all(np.isfinite(completed_return)):
        raise ValueError(f"Invalid median_completed_return values in {path}.")
    return curves


def build_common_grid(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, bool]:
    """Require identical evaluation checkpoints; do not interpolate missing evaluations."""
    if not curves:
        raise ValueError("No valid learning curves were supplied.")
    reference = curves[0][0]
    for index, (steps, _) in enumerate(curves[1:], start=2):
        if not np.array_equal(steps, reference):
            raise ValueError(
                f"Evaluation checkpoint schedule mismatch in curve {index}. "
                "All configurations and training seeds must use the same evaluation frequency and budget."
            )
    if reference.size < 2:
        raise ValueError("At least two common evaluation checkpoints are required.")
    return reference.copy(), False


def aggregate_seed_curves(
    curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and sample SD across seed-level metrics at matched checkpoints."""
    if len(curves) < 2:
        raise ValueError("At least two independent training seeds are required for sample SD.")
    reference = curves[0][0]
    for steps, _ in curves[1:]:
        if not np.array_equal(steps, reference):
            raise ValueError("Seed curves must have identical checkpoints; interpolation is not used for evaluation metrics.")
    if grid is not None and not np.array_equal(np.asarray(grid), reference):
        raise ValueError("Requested aggregation grid differs from the recorded evaluation checkpoints.")
    matrix = np.vstack([values for _, values in curves])
    return np.mean(matrix, axis=0), np.std(matrix, axis=0, ddof=1)


def _validate_run_records(
    run_dir: Path,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    seeds: list[int],
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Missing run_manifest.json: {run_manifest_path}")
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise ValueError(f"Run is not marked completed: {run_dir}")

    total = int(manifest["effective_total_timesteps_per_seed"])
    mode = str(manifest.get("mode"))
    if mode not in {"roughing", "finishing"}:
        raise ValueError(f"Unsupported sweep mode in manifest: {mode!r}.")
    expected_env = "CustomODEPlate-v0" if mode == "roughing" else "CustomODEPlateFinish-v0"
    expected_reward = "productive" if mode == "roughing" else "dense"
    if manifest.get("environment") != expected_env or manifest.get("reward") != expected_reward:
        raise ValueError("Sweep manifest environment/reward identifiers are inconsistent with its mode.")
    expected_ap_max = manifest.get("ap_max_mm")
    if mode == "finishing":
        if expected_ap_max is None or not math.isfinite(float(expected_ap_max)) or float(expected_ap_max) <= 0.0:
            raise ValueError("Finishing sweep manifest must contain a finite, positive ap_max_mm.")
    elif expected_ap_max is not None:
        raise ValueError("Roughing sweep manifest must not specify a finishing-only ap_max_mm.")
    expected_hyper = entry["hyperparameters"]
    expected_episodes = int(manifest["common_training_settings"]["n_eval_episodes"])
    expected_seeds = [int(seed) for seed in seeds]
    histories: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    run_signature = entry.get("effective_run_signature")
    if not isinstance(run_signature, dict):
        raise ValueError(f"Missing effective run signature for configuration {entry.get('name')}.")
    environment_signature = run_signature.get("environment")
    fixed_ppo_signature = run_signature.get("fixed_ppo")
    software_signature = run_signature.get("software")

    for seed in expected_seeds:
        seed_dir = run_dir / "logs" / f"seed_{seed}"
        config_path = seed_dir / "training_config.json"
        summary_path = seed_dir / "training_summary.json"
        if not config_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete training records for seed {seed} in {run_dir}.")
        config = _read_json(config_path)
        summary = _read_json(summary_path)
        if int(config.get("seed", -1)) != seed:
            raise ValueError(f"Training config seed mismatch in {config_path}.")
        if summary.get("status") != "completed" or int(summary.get("seed", -1)) != seed:
            raise ValueError(f"Training summary is incomplete for seed {seed}: {summary_path}")
        if int(summary.get("timesteps_reported_by_model", -1)) != total:
            raise ValueError(f"Training seed {seed} did not complete the prescribed transition budget.")

        actual_args = config.get("arguments", {})
        if mode == "finishing":
            actual_arg_ap = actual_args.get("ap_max")
            if actual_arg_ap is None or not math.isclose(float(actual_arg_ap), float(expected_ap_max), rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f"Finishing ap_max argument mismatch for seed {seed} in {run_dir}.")
        elif actual_args.get("ap_max") is not None:
            raise ValueError(f"Roughing run unexpectedly contains a finishing-only ap_max argument for seed {seed}.")
        for key, expected in manifest["common_training_settings"].items():
            if actual_args.get(key) != expected:
                raise ValueError(f"Actual training argument {key} differs from the sweep manifest for seed {seed}.")

        ppo = config.get("ppo", {})
        ppo_map = {
            "learning_rate": "learning_rate_initial",
            "final_learning_rate": "learning_rate_final",
            "n_steps": "n_steps",
            "n_envs": "n_envs",
            "batch_size": "batch_size",
            "n_epochs": "n_epochs",
            "gamma": "gamma",
            "gae_lambda": "gae_lambda",
            "clip_range": "clip_range_initial",
            "final_clip_range": "clip_range_final",
        }
        for manifest_key, config_key in ppo_map.items():
            expected = expected_hyper[manifest_key]
            actual = ppo.get(config_key)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                if actual is None or not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(
                        f"Actual PPO parameter {config_key}={actual!r} for seed {seed} "
                        f"does not match sweep manifest value {expected!r}."
                    )
            elif actual != expected:
                raise ValueError(f"Actual PPO parameter {config_key} does not match sweep manifest.")

        env = config.get("environment", {})
        if env.get("env_id") != expected_env or env.get("reward_id") != expected_reward:
            raise ValueError(f"Environment/reward mismatch for seed {seed} in {run_dir}.")
        if mode == "finishing":
            actual_kwargs_ap = env.get("kwargs", {}).get("ap_max")
            actual_probe_ap = env.get("probe", {}).get("ap_max")
            for source, actual in (("environment kwargs", actual_kwargs_ap), ("plant probe", actual_probe_ap)):
                if actual is None or not math.isclose(float(actual), float(expected_ap_max), rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError(
                        f"Finishing ap_max mismatch in {source} for seed {seed}: "
                        f"recorded {actual!r}, expected {expected_ap_max:g} mm."
                    )
        actual_environment_signature = {
            "dt_s": env.get("dt_s"),
            "n_substeps": env.get("n_substeps"),
            "controller_hold_env_steps": env.get("controller_hold_env_steps"),
            "max_episode_steps": env.get("max_episode_steps"),
            "dynamics_uncertainty_std": env.get("dynamics_uncertainty_std"),
            "randomize_y0": env.get("randomize_y0"),
            "env_id": env.get("env_id"),
            "reward_id": env.get("reward_id"),
            "controller_sample_time_s": env.get("controller_sample_time_s"),
            "env_kwargs_sha256": _canonical_hash(env.get("kwargs", {})),
            "plant_probe_sha256": _canonical_hash(env.get("probe", {})),
        }
        if actual_environment_signature != environment_signature:
            raise ValueError(f"Effective environment signature mismatch for seed {seed} in {run_dir}.")

        fixed_keys = (
            "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "net_arch",
            "log_std_init", "adam_epsilon", "normalize_observations",
            "normalize_rewards", "safe_action_bias",
        )
        actual_fixed_ppo = {key: ppo.get(key) for key in fixed_keys}
        if actual_fixed_ppo != fixed_ppo_signature:
            raise ValueError(f"Fixed PPO settings differ from the sweep manifest for seed {seed}.")
        software = config.get("software", {})
        actual_software = {
            "git_commit": software.get("git_commit"),
            "script_sha256": software.get("script_sha256"),
            "python": software.get("python"),
            "packages": software.get("packages"),
        }
        if actual_software != software_signature:
            raise ValueError(f"Software/trainer version mismatch for seed {seed} in {run_dir}.")

        callback = config.get("evaluation_callback", {})
        if int(callback.get("episodes_per_evaluation", -1)) != expected_episodes:
            raise ValueError(f"Evaluation episode count mismatch for seed {seed}.")
        expected_eval_seed = int(manifest["common_training_settings"]["callback_eval_seed"])
        if int(callback.get("evaluation_env_seed", -1)) != expected_eval_seed:
            raise ValueError(f"Evaluation seed mismatch for seed {seed}.")
        if int(callback.get("frequency_controller_transitions_actual", -1)) != int(manifest["actual_evaluation_interval_transitions"]):
            raise ValueError(f"Evaluation frequency mismatch for seed {seed}.")
        histories[seed] = load_seed_metrics(run_dir, seed, expected_eval_episodes=expected_episodes)

    all_curves = [histories[seed]["evaluation_return"] for seed in expected_seeds]
    build_common_grid(all_curves)
    return histories


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_return_curve(
    histories: dict[str, dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]],
    labels: dict[str, str],
    seeds: list[int],
    out_dir: Path,
) -> tuple[list[list[Any]], list[list[Any]]]:
    all_curves = [histories[name][seed]["evaluation_return"] for name in histories for seed in seeds]
    grid, _ = build_common_grid(all_curves)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    stats_rows: list[list[Any]] = []
    summary_rows: list[list[Any]] = []

    for index, name in enumerate(histories):
        seed_curves = [histories[name][seed]["evaluation_return"] for seed in seeds]
        mean, sd = aggregate_seed_curves(seed_curves, grid)
        color = colors[index % len(colors)]
        ax.plot(grid, mean, color=color, linewidth=2.0, marker="o", markersize=3.2,
                label=f"{labels[name]} (n={len(seeds)})")
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.17, linewidth=0)
        for j, step in enumerate(grid):
            per_seed = [float(histories[name][seed]["evaluation_return"][1][j]) for seed in seeds]
            stats_rows.append([name, int(step), len(seeds), float(mean[j]), float(sd[j]), *per_seed])
        peak = int(np.argmax(mean))
        summary_rows.append([
            name, labels[name], len(seeds), int(grid[-1]), float(mean[-1]), float(sd[-1]),
            int(grid[peak]), float(mean[peak]),
        ])

    ax.set_xlabel("Training transitions (across all environments)")
    ax.set_ylabel("Evaluation episode return")
    ax.grid(axis="y", alpha=0.22, linewidth=0.65)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.015),
              ncol=len(histories), borderaxespad=0.0)
    ax.margins(x=0.01)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_dir / f"ppo_sweep_learning_curve{suffix}", dpi=300 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return stats_rows, summary_rows


def _plot_reliability_metrics(
    histories: dict[str, dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]],
    labels: dict[str, str],
    seeds: list[int],
    out_dir: Path,
    *,
    excluded_metrics: frozenset[str] = frozenset(),
) -> list[list[Any]]:
    """Main 2x2 reliability panel and supplementary 1x2 panel.

    Statistical data are *not* clipped. Only the displayed SD envelope is
    bounded for quantities that are physically confined to [0, 100] percent.
    """
    metric_keys = list(RELIABILITY_METRICS)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    stats_rows: list[list[Any]] = []
    main_keys = (
        "pass_completion_rate", "safety_termination_rate",
        "median_completed_productivity", "median_progress",
    )
    supplemental_keys = ("truncation_rate", "mean_max_displacement_m")

    def render(keys: tuple[str, ...], filename: str) -> None:
        fig, axes = plt.subplots(
            2 if len(keys) == 4 else 1, 2,
            figsize=(10.2, 6.9 if len(keys) == 4 else 3.65),
            squeeze=False, sharex=True, constrained_layout=True,
        )
        axs = list(axes.flat)
        for i, metric in enumerate(keys):
            spec = RELIABILITY_METRICS[metric]
            ax = axs[i]
            if metric in excluded_metrics:
                ax.set_title(f"({chr(ord('a') + i)}) {spec['title']}", loc="left")
                ax.text(0.5, 0.5, "Not reported\nHeld-action callback omits\nintermediate-step data",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            for config_index, name in enumerate(histories):
                curves = [histories[name][seed][metric] for seed in seeds]
                grid, _ = build_common_grid(curves)
                mean, sd = aggregate_seed_curves(curves, grid)
                color = colors[config_index % len(colors)]
                scale = float(spec["scale"])
                lower, upper = (mean - sd) * scale, (mean + sd) * scale
                if spec["ylim"] is not None:
                    lower = np.clip(lower, *spec["ylim"])
                    upper = np.clip(upper, *spec["ylim"])
                ax.plot(grid, mean * scale, color=color, linewidth=1.8,
                        marker="o", markersize=2.8, label=labels[name])
                ax.fill_between(grid, lower, upper, color=color, alpha=0.16, linewidth=0)
                if metric not in metric_set_recorded:
                    for j, step in enumerate(grid):
                        values = [float(histories[name][seed][metric][1][j]) for seed in seeds]
                        stats_rows.append([
                            metric, name, int(step), len(seeds), float(mean[j]), float(sd[j]), *values
                        ])
            metric_set_recorded.add(metric)
            ax.set_title(f"({chr(ord('a') + i)}) {spec['title']}", loc="left")
            ax.set_ylabel(spec["ylabel"])
            ax.grid(axis="y", alpha=0.24, linewidth=0.6)
            if spec["ylim"] is not None:
                ax.set_ylim(*spec["ylim"])
            ax.margins(x=0.01)
            if len(keys) == 2 or i >= 2:
                ax.set_xlabel("Training transitions (across all environments)")
        handles, captions = axs[0].get_legend_handles_labels()
        fig.legend(handles, captions, loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=len(histories), frameon=False)
        fig.savefig(out_dir / (filename + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / (filename + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    metric_set_recorded: set[str] = set()
    render(main_keys, "ppo_sweep_reliability_metrics")
    render(supplemental_keys, "ppo_sweep_reliability_supplement")
    return stats_rows


def _load_tensorboard_scalars(seed_log_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    event_files = list(seed_log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard event files exist but TensorBoard is not installed. "
            "Install it in the project environment with: pip install tensorboard"
        ) from exc

    by_tag: dict[str, dict[int, tuple[float, float]]] = {}
    for event_dir in sorted({path.parent for path in event_files}):
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0, "tensors": 0})
        accumulator.Reload()
        tags = accumulator.Tags()
        for tag in tags.get("scalars", []):
            for event in accumulator.Scalars(tag):
                step, value = int(event.step), float(event.value)
                if math.isfinite(value):
                    previous = by_tag.setdefault(tag, {}).get(step)
                    if previous is None or float(event.wall_time) >= previous[0]:
                        by_tag[tag][step] = (float(event.wall_time), value)
        tensor_tags = tags.get("tensors", [])
        if tensor_tags:
            try:
                from tensorboard.util import tensor_util
            except ImportError:
                tensor_util = None
            if tensor_util is not None:
                for tag in tensor_tags:
                    for event in accumulator.Tensors(tag):
                        try:
                            array = np.asarray(tensor_util.make_ndarray(event.tensor_proto)).reshape(-1)
                            value = float(array[0]) if array.size else float("nan")
                        except Exception:
                            continue
                        if math.isfinite(value):
                            step = int(event.step)
                            previous = by_tag.setdefault(tag, {}).get(step)
                            if previous is None or float(event.wall_time) >= previous[0]:
                                by_tag[tag][step] = (float(event.wall_time), value)

    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, values in by_tag.items():
        ordered = np.asarray(sorted(values), dtype=np.float64)
        output[tag] = (ordered, np.asarray([values[int(step)][1] for step in ordered], dtype=np.float64))
    return output


def _aggregate_diagnostic_series(
    series: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate at shared, actually recorded TensorBoard steps; do not interpolate.

    Different PPO configurations may log after different rollout sizes. Within
    one configuration, independent seeds should share update-step coordinates.
    If an event is missing, use the intersection of observed steps instead of
    fabricating a value at an unobserved optimizer update.
    """
    if len(series) < 2:
        return None
    for x, y in series:
        if x.size != y.size or x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return None
        if np.any(np.diff(x) <= 0):
            return None
    grid = np.asarray(series[0][0], dtype=np.float64)
    for x, _ in series[1:]:
        grid = np.intersect1d(grid, x, assume_unique=True)
    if grid.size < 2:
        return None
    rows = []
    for x, y in series:
        indices = np.searchsorted(x, grid)
        if np.any(indices >= x.size) or not np.array_equal(x[indices], grid):
            raise ValueError("Internal error: common TensorBoard steps could not be mapped to raw samples.")
        rows.append(y[indices])
    matrix = np.vstack(rows)
    return grid, np.mean(matrix, axis=0), np.std(matrix, axis=0, ddof=1)


def _plot_training_diagnostics(
    run_dirs: dict[str, Path],
    labels: dict[str, str],
    seeds: list[int],
    out_dir: Path,
) -> list[list[Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scalar_data: dict[str, dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    try:
        for name, run_dir in run_dirs.items():
            scalar_data[name] = {}
            for seed in seeds:
                scalar_data[name][seed] = _load_tensorboard_scalars(run_dir / "logs" / f"seed_{seed}")
    except RuntimeError as exc:
        print(f"WARNING: PPO TensorBoard diagnostics were skipped: {exc}")
        return []

    if not any(series for config in scalar_data.values() for series in config.values()):
        print("WARNING: no TensorBoard event files were found; optimization-diagnostic figure was not generated.")
        return []

    # Prefer train/std only when it is available for every seed/configuration;
    # otherwise use entropy_loss consistently rather than mixing quantities.
    std_tag = None
    for candidate in ("train/std", "train/entropy_loss"):
        if all(candidate in scalar_data[name][seed] for name in scalar_data for seed in seeds):
            std_tag = candidate
            break

    tag_specs = [
        ("rollout/ep_rew_mean", "Training episode return", "Mean rolling episode return"),
        ("train/explained_variance", "Value-function fit", "Explained variance"),
        ("train/approx_kl", "Policy-update size", "Approximate KL divergence (epoch average)"),
        ("train/clip_fraction", "Policy clipping", "Clip fraction"),
    ]
    if std_tag is not None:
        if std_tag == "train/std":
            tag_specs.append((std_tag, "Policy exploration", "Policy Gaussian SD (pre-clipping)"))
        else:
            tag_specs.append((std_tag, "Policy exploration", "Entropy loss"))

    # Only retain tags recorded for every seed of every configuration.
    available_specs = [
        spec for spec in tag_specs
        if all(spec[0] in scalar_data[name][seed] for name in scalar_data for seed in seeds)
    ]
    if not available_specs:
        print("WARNING: TensorBoard files contain no common PPO diagnostic tags across all runs.")
        return []

    ncols = 2
    nrows = int(math.ceil(len(available_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.7 * nrows), squeeze=False, constrained_layout=True)
    axes_flat = list(axes.flat)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    stats_rows: list[list[Any]] = []
    for panel_index, (tag, title, ylabel) in enumerate(available_specs):
        ax = axes_flat[panel_index]
        for config_index, name in enumerate(run_dirs):
            series = [scalar_data[name][seed][tag] for seed in seeds]
            aggregated = _aggregate_diagnostic_series(series)
            if aggregated is None:
                raise ValueError(f"Could not align TensorBoard metric {tag!r} across seeds for {name}.")
            grid, mean, sd = aggregated
            color = colors[config_index % len(colors)]
            ax.plot(grid, mean, color=color, linewidth=1.7, label=labels[name])
            ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
            for j, step in enumerate(grid):
                per_seed = []
                for seed in seeds:
                    raw_x, raw_y = scalar_data[name][seed][tag]
                    raw_idx = np.searchsorted(raw_x, step)
                    if raw_idx >= raw_x.size or raw_x[raw_idx] != step:
                        raise ValueError(f"Unobserved TensorBoard step {step:g}: {name}/seed_{seed}/{tag}")
                    per_seed.append(float(raw_y[raw_idx]))
                stats_rows.append([tag, name, int(step), len(seeds), float(mean[j]), float(sd[j]), *per_seed])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training transitions (across all environments)")
        ax.grid(axis="y", alpha=0.24, linewidth=0.6)

    for ax in axes_flat[len(available_specs):]:
        ax.set_visible(False)
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(run_dirs), frameon=False)
    fig.suptitle("PPO optimization diagnostics across training configurations")
    # PPO configurations have different optimizer-update schedules, so each
    # configuration retains its own observed x-coordinates. Within a profile,
    # only TensorBoard steps recorded for EVERY training seed are aggregated.
    # No interpolation, extrapolation or visual smoothing is performed.
    for suffix in (".png", ".pdf"):
        fig.savefig(out_dir / f"ppo_sweep_training_diagnostics{suffix}", dpi=300 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return stats_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot cross-seed PPO learning, diagnostics and reliability comparisons.")
    parser.add_argument("--manifest", required=True, help="sweep_manifest.json created by sweep_ppo.py")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Sweep manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 3 or manifest.get("status") != "completed":
        raise SystemExit("Only a fully completed schema-version-3 sweep can be plotted.")

    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    entries = manifest.get("configurations", [])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise SystemExit("At least two unique training seeds are required for mean ± sample-SD plots.")
    if len(entries) < 2:
        raise SystemExit("At least two completed configurations are required for comparison plots.")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else manifest_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    histories: dict[str, dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    labels: dict[str, str] = {}
    run_dirs: dict[str, Path] = {}
    raw_rows: list[list[Any]] = []
    seen_names: set[str] = set()

    for entry in entries:
        name = str(entry.get("name"))
        if name in seen_names:
            raise SystemExit(f"Duplicate configuration entry in manifest: {name}")
        seen_names.add(name)
        if entry.get("status") != "completed" or not entry.get("run_dir"):
            raise SystemExit(f"Configuration {name!r} is incomplete in the sweep manifest.")
        run_value = Path(str(entry["run_dir"])).expanduser()
        run_dir = run_value.resolve() if run_value.is_absolute() else (manifest_path.parent / run_value).resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory for {name!r} does not exist: {run_dir}")
        seed_histories = _validate_run_records(run_dir, entry, manifest, seeds)
        histories[name] = seed_histories
        labels[name] = str(entry.get("label") or name.title())
        run_dirs[name] = run_dir

        for seed in seeds:
            npz_path = run_dir / "logs" / f"seed_{seed}" / "evaluations.npz"
            with np.load(npz_path, allow_pickle=False) as data:
                steps = np.asarray(data["timesteps"], dtype=np.int64).reshape(-1)
                results = np.asarray(data["results"], dtype=np.float64)
            if results.ndim == 1:
                results = results.reshape(-1, 1)
            for step, row in zip(steps, results):
                for episode_index, value in enumerate(row, start=1):
                    raw_rows.append([name, seed, int(step), episode_index, float(value)])

    # Require one exact evaluation schedule across all configurations and seeds.
    all_return_curves = [histories[name][seed]["evaluation_return"] for name in histories for seed in seeds]
    common_steps, _ = build_common_grid(all_return_curves)

    return_stats, summary_rows = _plot_return_curve(histories, labels, seeds, out_dir)
    held_steps = int(next(iter(manifest["configurations"]))["effective_run_signature"]["environment"]["controller_hold_env_steps"])
    excluded_metrics = (
        frozenset({"median_completed_productivity", "mean_max_displacement_m"})
        if held_steps > 1 else frozenset()
    )
    if excluded_metrics:
        print("WARNING: held-action callback did not aggregate intermediate-step reward components/peaks; "
              "the affected reliability panels have been suppressed.")
    reliability_stats = _plot_reliability_metrics(
        histories, labels, seeds, out_dir, excluded_metrics=excluded_metrics
    )
    diagnostic_stats = _plot_training_diagnostics(run_dirs, labels, seeds, out_dir)

    _write_csv(out_dir / "ppo_sweep_raw_evaluation_returns.csv", [
        "configuration", "training_seed", "training_transitions", "evaluation_episode", "episode_return",
    ], raw_rows)
    _write_csv(out_dir / "ppo_sweep_learning_curve_stats.csv", [
        "configuration", "training_transitions", "n_training_seeds", "mean_eval_return",
        "sample_sd_across_seeds", *[f"seed_{seed}_mean_return" for seed in seeds],
    ], return_stats)
    _write_csv(out_dir / "ppo_sweep_reliability_metrics_stats.csv", [
        "metric", "configuration", "training_transitions", "n_training_seeds", "mean_across_seeds",
        "sample_sd_across_seeds", *[f"seed_{seed}_value" for seed in seeds],
    ], reliability_stats)
    if diagnostic_stats:
        _write_csv(out_dir / "ppo_sweep_training_diagnostics_stats.csv", [
            "tensorboard_tag", "configuration", "training_transitions", "n_training_seeds", "mean_across_seeds",
            "sample_sd_across_seeds", *[f"seed_{seed}_value" for seed in seeds],
        ], diagnostic_stats)

    _write_csv(out_dir / "ppo_sweep_final_summary.csv", [
        "configuration", "label", "n_training_seeds", "final_common_timestep", "final_mean_eval_return",
        "final_sample_sd", "peak_mean_return_timestep", "peak_mean_return",
    ], summary_rows)

    print(f"Evaluation checkpoints: {common_steps[0]:,} to {common_steps[-1]:,} ({len(common_steps)} points).")
    for filename in (
        "ppo_sweep_learning_curve.png", "ppo_sweep_learning_curve.pdf",
        "ppo_sweep_reliability_metrics.png", "ppo_sweep_reliability_metrics.pdf",
    ):
        print(f"Saved: {out_dir / filename}")
    if diagnostic_stats:
        print(f"Saved: {out_dir / 'ppo_sweep_training_diagnostics.png'}")
        print(f"Saved: {out_dir / 'ppo_sweep_training_diagnostics.pdf'}")
    else:
        print("Training-diagnostic figure was not generated because common TensorBoard tags were unavailable.")
    print(f"Saved: {out_dir / 'ppo_sweep_raw_evaluation_returns.csv'}")
    print(f"Saved: {out_dir / 'ppo_sweep_learning_curve_stats.csv'}")
    print(f"Saved: {out_dir / 'ppo_sweep_reliability_metrics_stats.csv'}")
    print(f"Saved: {out_dir / 'ppo_sweep_final_summary.csv'}")
    print("Bands: mean ± 1 sample SD across independent training seeds; not confidence intervals.")
    print("Evaluation episode returns are averaged within each seed before cross-seed statistics are computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
