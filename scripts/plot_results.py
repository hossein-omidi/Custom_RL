"""Plot PPO evaluation learning curves, training diagnostics, and scenario-scoped trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import numpy as np

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_PLOT_DIR, DEFAULT_TRAJ_DIR
from custom_rl.eval.pipeline import (
    discover_log_seeds,
    discover_trajectory_seeds,
    physical_action_names,
    primary_episode_record,
)
from custom_rl.plants.plate import omega_to_rpm


# A descriptive one-standard-deviation spread, not a confidence interval.
PLOT_STD_MULT = 1.0
MAX_PLOTTED_TIME_POINTS = 30000  # display-only; full saved trajectories remain unmodified


def _load_evaluation_history(log_dir: Path, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load EvalCallback checkpoint returns for one independently trained seed.

    Stable-Baselines3 stores:
        timesteps: (n_checkpoints,)
        results:   (n_checkpoints, n_eval_episodes)

    The returned results matrix preserves every raw evaluation episode return.
    """
    path = log_dir / f"seed_{seed}" / "evaluations.npz"
    if not path.is_file():
        print(f"Note: missing evaluation history for seed {seed}: {path}")
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            if "timesteps" not in data.files or "results" not in data.files:
                raise ValueError("evaluations.npz must contain 'timesteps' and 'results'.")
            timesteps = np.asarray(data["timesteps"], dtype=np.float64).reshape(-1)
            results = np.asarray(data["results"], dtype=np.float64)
    except Exception as exc:
        print(f"Warning: could not read evaluation history for seed {seed}: {exc}")
        return None

    if results.ndim == 1:
        results = results.reshape(-1, 1)
    if results.ndim != 2 or results.shape[0] != timesteps.size:
        print(
            f"Warning: invalid evaluations.npz shapes for seed {seed}: "
            f"timesteps={timesteps.shape}, results={results.shape}."
        )
        return None

    # Treat both NaN and +/-inf episode returns as missing observations.
    results = np.where(np.isfinite(results), results, np.nan)
    valid_rows = np.isfinite(timesteps) & np.any(np.isfinite(results), axis=1)
    timesteps = timesteps[valid_rows]
    results = results[valid_rows]
    if timesteps.size == 0:
        print(f"Warning: no finite evaluation returns for seed {seed}.")
        return None

    order = np.argsort(timesteps, kind="stable")
    timesteps = timesteps[order]
    results = results[order]

    # If a resumed/merged run contains duplicate checkpoint steps, keep the
    # last recorded row for each step rather than counting it twice.
    last_index_by_step: dict[float, int] = {}
    for index, step in enumerate(timesteps):
        last_index_by_step[float(step)] = index
    keep = np.asarray(sorted(last_index_by_step.values()), dtype=int)
    return timesteps[keep], results[keep]


def _align_seed_learning_curves(
    histories: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Align per-seed mean returns without extrapolating beyond shared training."""
    if len(histories) < 2:
        return None

    means = {
        seed: np.nanmean(results, axis=1)
        for seed, (_, results) in histories.items()
    }
    step_arrays = [histories[seed][0] for seed in sorted(histories)]
    common_steps = step_arrays[0]
    for steps in step_arrays[1:]:
        common_steps = np.intersect1d(common_steps, steps)

    if common_steps.size:
        rows = []
        for seed in sorted(histories):
            steps = histories[seed][0]
            indices = np.searchsorted(steps, common_steps)
            rows.append(means[seed][indices])
        return common_steps, np.asarray(rows, dtype=np.float64), False

    # Different evaluation schedules are unusual for a single unified run.
    # If they occur, interpolate only over the common observed interval.
    lower = max(float(steps[0]) for steps in step_arrays)
    upper = min(float(steps[-1]) for steps in step_arrays)
    if upper < lower:
        return None
    grid = np.unique(
        np.concatenate(
            [steps[(steps >= lower) & (steps <= upper)] for steps in step_arrays]
        )
    )
    if grid.size == 0:
        return None
    matrix = np.vstack(
        [
            np.interp(grid, histories[seed][0], means[seed])
            for seed in sorted(histories)
        ]
    )
    return grid, matrix, True


def _write_figure1_data(
    out_dir: Path,
    histories: dict[int, tuple[np.ndarray, np.ndarray]],
    aligned: tuple[np.ndarray, np.ndarray, bool] | None,
) -> None:
    """Write raw episode returns and across-seed statistics for auditability."""
    raw_path = out_dir / "figure1_ppo_learning_curve_data.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["training_seed", "training_timesteps", "evaluation_episode", "episode_return", "seed_mean_return"]
        )
        for seed in sorted(histories):
            steps, results = histories[seed]
            seed_means = np.nanmean(results, axis=1)
            for checkpoint_index, (step, episode_returns) in enumerate(zip(steps, results)):
                for episode_index, value in enumerate(episode_returns, start=1):
                    if np.isfinite(value):
                        writer.writerow(
                            [seed, int(step), episode_index, float(value), float(seed_means[checkpoint_index])]
                        )

    print(f"Saved {raw_path}")
    if aligned is None:
        return
    steps, seed_mean_matrix, interpolated = aligned
    across_seed_mean = np.mean(seed_mean_matrix, axis=0)
    across_seed_std = np.std(seed_mean_matrix, axis=0, ddof=1)
    summary_path = out_dir / "figure1_ppo_learning_curve_across_seed_stats.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["training_timesteps", *[f"seed_{s}_mean_return" for s in sorted(histories)],
             "across_seed_mean", "across_seed_sample_sd", "n_training_seeds", "interpolated_alignment"]
        )
        for index, step in enumerate(steps):
            writer.writerow(
                [int(step), *seed_mean_matrix[:, index].tolist(),
                 float(across_seed_mean[index]), float(across_seed_std[index]),
                 len(histories), bool(interpolated)]
            )
    print(f"Saved {summary_path}")


def plot_learning_curve(
    log_dir: Path,
    out_dir: Path,
    seeds: list[int],
    smooth: int = 10,
) -> None:
    """Create Figure 1 from raw EvalCallback returns, not Monitor training returns.

    Each seed line is the mean of that policy's evaluation episodes at a
    checkpoint. Individual episode returns are shown as markers. The shaded
    band is ±1 *sample* standard deviation across the seed-level means.

    ``smooth`` is retained for CLI/API compatibility; this scientific figure
    intentionally does not smooth or otherwise alter the checkpoint values.
    """
    del smooth
    histories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        history = _load_evaluation_history(log_dir, seed)
        if history is not None:
            histories[seed] = history

    if not histories:
        print(f"No EvalCallback evaluations.npz files found under {log_dir} for seeds {seeds}.")
        return

    missing = [seed for seed in seeds if seed not in histories]
    if missing:
        print(f"Figure 1 warning: missing evaluation history for seed(s) {missing}.")
    if len(histories) != 3:
        print(
            f"Figure 1 note: found {len(histories)} training seed(s); the target figure "
            "uses three independent seeds. The shaded sample-SD band requires at least two."
        )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    color_by_seed = {seed: colors[i % len(colors)] for i, seed in enumerate(sorted(histories))}

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    aligned = _align_seed_learning_curves(histories)
    legend_handles: list = []

    if aligned is not None:
        band_steps, seed_mean_matrix, interpolated = aligned
        across_seed_mean = np.mean(seed_mean_matrix, axis=0)
        across_seed_std = np.std(seed_mean_matrix, axis=0, ddof=1)
        ax.fill_between(
            band_steps,
            across_seed_mean - across_seed_std,
            across_seed_mean + across_seed_std,
            color="0.55",
            alpha=0.20,
            linewidth=0,
            zorder=1,
        )
        legend_handles.append(
            Patch(
                facecolor="0.55",
                alpha=0.20,
                label=f"Across-seed mean ± 1 sample SD (n={len(histories)})",
            )
        )
        if interpolated:
            print(
                "Figure 1 note: seed checkpoint grids did not match exactly; "
                "the between-seed band was interpolated only within their shared range."
            )
    else:
        print("Figure 1 note: fewer than two comparable seed histories; no across-seed SD band was drawn.")

    for seed in sorted(histories):
        steps, results = histories[seed]
        seed_means = np.nanmean(results, axis=1)
        color = color_by_seed[seed]
        n_eval = results.shape[1]

        ax.plot(
            steps,
            seed_means,
            color=color,
            linewidth=2.0,
            marker=None,
            zorder=4,
            label=f"Seed {seed}: mean of {n_eval} eval episodes",
        )

        raw_x = np.repeat(steps, results.shape[1])
        raw_y = results.reshape(-1)
        finite = np.isfinite(raw_x) & np.isfinite(raw_y)
        ax.scatter(
            raw_x[finite],
            raw_y[finite],
            s=24,
            marker="o",
            facecolors=color,
            edgecolors=color,
            linewidths=0.5,
            alpha=0.42,
            zorder=3,
            label="_nolegend_",
        )
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2.0,
                   label=f"Seed {seed}: mean of {n_eval} eval episodes")
        )

    legend_handles.append(
        Line2D([0], [0], linestyle="none", marker="o", markersize=5,
               markerfacecolor="0.45", markeredgecolor="0.45", alpha=0.55,
               label="Individual evaluation episode returns")
    )

    ax.set_xlabel("PPO controller transitions")
    ax.set_ylabel("Evaluation episode return")
    ax.set_title(f"PPO learning curve: {len(histories)} independent training policies")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=8)
    fig.text(
        0.5, 0.006,
        "Band: ±1 sample SD across seed-level means (not a 95% CI); "
        "spread also reflects stochastic EvalCallback evaluation conditions.",
        ha="center", va="bottom", fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out_png = out_dir / "figure1_ppo_learning_curve.png"
    out_pdf = out_dir / "figure1_ppo_learning_curve.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    _write_figure1_data(out_dir, histories, aligned)


def _load_tensorboard_scalars(seed_log_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read SB3 TensorBoard scalar histories for one training seed.

    TensorBoard is imported lazily so plotting trajectory JSONs still works in
    environments where TensorBoard is not installed.
    """
    event_files = list(seed_log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard event files exist, but the 'tensorboard' package is not installed. "
            "Install it in the project environment with: pip install tensorboard"
        ) from exc

    events_by_tag: dict[str, dict[int, tuple[float, float]]] = {}
    event_dirs = sorted({path.parent for path in event_files})
    for event_dir in event_dirs:
        accumulator = EventAccumulator(
            str(event_dir),
            size_guidance={"scalars": 0, "tensors": 0},
        )
        accumulator.Reload()
        tags = accumulator.Tags()

        for tag in tags.get("scalars", []):
            for event in accumulator.Scalars(tag):
                step = int(event.step)
                value = float(event.value)
                if np.isfinite(value):
                    current = events_by_tag.setdefault(tag, {}).get(step)
                    candidate = (float(event.wall_time), value)
                    if current is None or candidate[0] >= current[0]:
                        events_by_tag[tag][step] = candidate

        # Support TensorBoard versions that expose scalar summaries through the
        # tensor plugin rather than the legacy scalar plugin.
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
                        if np.isfinite(value):
                            step = int(event.step)
                            current = events_by_tag.setdefault(tag, {}).get(step)
                            candidate = (float(event.wall_time), value)
                            if current is None or candidate[0] >= current[0]:
                                events_by_tag[tag][step] = candidate

    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, step_values in events_by_tag.items():
        ordered_steps = np.asarray(sorted(step_values), dtype=np.float64)
        values = np.asarray([step_values[int(step)][1] for step in ordered_steps], dtype=np.float64)
        output[tag] = (ordered_steps, values)
    return output


def _plot_seed_metric(
    ax,
    scalar_data: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    seeds: list[int],
    tag: str,
    color_by_seed: dict[int, str],
    *,
    linestyle: str = "-",
    label_suffix: str = "",
    show_missing: bool = True,
) -> int:
    """Plot one TensorBoard scalar for each seed; return number of plotted runs."""
    plotted = 0
    for seed in seeds:
        series = scalar_data.get(seed, {}).get(tag)
        if series is None:
            continue
        steps, values = series
        ax.plot(
            steps,
            values,
            color=color_by_seed[seed],
            linewidth=1.5,
            linestyle=linestyle,
            label=f"Seed {seed}{label_suffix}",
        )
        plotted += 1
    if plotted == 0 and show_missing:
        ax.text(0.5, 0.5, f"Metric not found: {tag}", ha="center", va="center", transform=ax.transAxes)
    return plotted


def _read_target_kl(log_dir: Path, seeds: list[int]) -> float | None:
    """Read the configured PPO target_kl from the first available seed config."""
    for seed in seeds:
        config_path = log_dir / f"seed_{seed}" / "training_config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            value = config.get("ppo", {}).get("target_kl")
            if value is not None and np.isfinite(float(value)):
                return float(value)
        except Exception:
            continue
    return None


def plot_training_diagnostics(log_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Plot four high-value PPO diagnostics as a 2x2 multi-seed figure.

    Panels: rolling training episode return, critic explained variance, policy
    update size (approximate KL + clip fraction), and continuous-policy action
    standard deviation. Every curve remains separated by training seed.
    """
    scalar_data: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    try:
        for seed in seeds:
            seed_dir = log_dir / f"seed_{seed}"
            if seed_dir.is_dir():
                scalar_data[seed] = _load_tensorboard_scalars(seed_dir)
    except RuntimeError as exc:
        print(f"Training diagnostics skipped: {exc}")
        return

    if not any(scalar_data.values()):
        print(
            f"No TensorBoard scalar histories found under {log_dir}. "
            "The PPO diagnostics figure was not created."
        )
        return

    available_seeds = [seed for seed in seeds if scalar_data.get(seed)]
    if not available_seeds:
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    color_by_seed = {seed: colors[i % len(colors)] for i, seed in enumerate(sorted(seeds))}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax_reward, ax_ev, ax_kl, ax_std = axes.reshape(-1)

    reward_plotted = _plot_seed_metric(ax_reward, scalar_data, seeds, "rollout/ep_rew_mean", color_by_seed)
    ax_reward.set_title("Training performance")
    ax_reward.set_ylabel("Mean training episode return (rolling window)")
    ax_reward.set_xlabel("PPO controller transitions")
    if reward_plotted:
        ax_reward.legend(frameon=False, fontsize=8)

    _plot_seed_metric(ax_ev, scalar_data, seeds, "train/explained_variance", color_by_seed)
    ax_ev.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, label="0: no predictive skill")
    ax_ev.set_title("Value-function fit")
    ax_ev.set_ylabel("Explained variance")
    ax_ev.set_xlabel("PPO controller transitions")
    ax_ev.legend(frameon=False, fontsize=8)

    kl_plotted = _plot_seed_metric(
        ax_kl, scalar_data, seeds, "train/approx_kl", color_by_seed,
        label_suffix=" KL", show_missing=False,
    )
    ax_kl.set_title("PPO policy-update stability")
    ax_kl.set_ylabel("Approximate KL divergence")
    ax_kl.set_xlabel("PPO controller transitions")
    ax_clip = ax_kl.twinx()
    clip_plotted = _plot_seed_metric(
        ax_clip, scalar_data, seeds, "train/clip_fraction", color_by_seed,
        linestyle="--", label_suffix=" clip", show_missing=False,
    )
    ax_clip.set_ylabel("Clip fraction")
    target_kl = _read_target_kl(log_dir, seeds)
    if target_kl is not None and kl_plotted:
        stop_threshold = 1.5 * target_kl
        ax_kl.axhline(
            stop_threshold, color="0.25", linestyle=":", linewidth=1.2,
            label=f"SB3 early-stop threshold ≈ 1.5 × target_kl ({stop_threshold:g})",
        )
    left_handles, left_labels = ax_kl.get_legend_handles_labels()
    right_handles, right_labels = ax_clip.get_legend_handles_labels()
    if left_handles or right_handles:
        ax_kl.legend(left_handles + right_handles, left_labels + right_labels,
                     frameon=False, fontsize=7, loc="best")
    if kl_plotted == 0 and clip_plotted == 0:
        ax_kl.text(
            0.5, 0.5, "PPO update metrics not found",
            ha="center", va="center", transform=ax_kl.transAxes,
        )

    std_plotted = _plot_seed_metric(
        ax_std, scalar_data, seeds, "train/std", color_by_seed, show_missing=False
    )
    if std_plotted:
        ax_std.set_title("Policy exploration")
        ax_std.set_ylabel("Mean action std (normalized action units)")
    else:
        # Fallback for older SB3 logs or policies that do not expose train/std.
        entropy_plotted = _plot_seed_metric(
            ax_std, scalar_data, seeds, "train/entropy_loss", color_by_seed,
            show_missing=False,
        )
        if entropy_plotted == 0:
            ax_std.text(
                0.5, 0.5, "Neither train/std nor train/entropy_loss was found",
                ha="center", va="center", transform=ax_std.transAxes,
            )
        ax_std.set_title("Policy exploration (entropy fallback)")
        ax_std.set_ylabel("Entropy loss (negative entropy)")
        if entropy_plotted:
            ax_std.legend(frameon=False, fontsize=8)
    ax_std.set_xlabel("PPO controller transitions")
    if std_plotted:
        ax_std.legend(frameon=False, fontsize=8)

    for ax in (ax_reward, ax_ev, ax_kl, ax_std):
        ax.grid(True, alpha=0.25)

    fig.suptitle("PPO training diagnostics across independent training seeds", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = out_dir / "ppo_training_diagnostics.png"
    out_pdf = out_dir / "ppo_training_diagnostics.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")



def _physical_signal_labels(n_sensors: int) -> list[str]:
    """Create labels for physical sensor displacement and velocity."""
    labels = []

    for i in range(n_sensors):
        labels.append(f"w_sensor_{i + 1} (m)")

    for i in range(n_sensors):
        labels.append(f"wdot_sensor_{i + 1} (m/s)")

    return labels


def _nan_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pointwise mean and sample SD, tolerating unequal rollout lengths."""
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    valid = counts > 0
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)
    for idx in np.flatnonzero(valid):
        column = values[finite[:, idx], idx]
        mean[idx] = float(np.mean(column))
        # One survivor after earlier run terminations cannot estimate variability.
        std[idx] = float(np.std(column, ddof=1)) if column.size > 1 else np.nan
    return mean, std, valid


def _physical_signals_from_episode(ep: dict) -> np.ndarray:
    """
    Return unscaled physical signals from one saved episode.

    Preferred new format:
        ep["physical_signals"] = [w_sensor, wdot_sensor]

    Fallback:
        ep["observations"] begins with scaled [w_sensor, wdot_sensor], so unscale
        using metadata["w_obs_scale"] and metadata["wdot_obs_scale"].  Any
        additional observation terms, such as normalized cutter path coordinates,
        are ignored for physical sensor plots.

    Old trajectory files with only ep["states"] are intentionally skipped because
    they may contain modal coordinates and should not be relabeled as physical
    sensor response.
    """
    record = primary_episode_record(ep)
    metadata = ep.get("metadata", record.get("metadata", {}))

    if "physical_signals" in record and len(record["physical_signals"]) > 0:
        signals = np.asarray(record["physical_signals"], dtype=np.float64)
    elif "physical_signals" in ep and len(ep["physical_signals"]) > 0:
        signals = np.asarray(ep["physical_signals"], dtype=np.float64)

    elif "observations" in record and len(record["observations"]) > 0:
        obs = np.asarray(record["observations"], dtype=np.float64)

        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        n_sensors = int(metadata.get("n_sensors", 0) or 0)
        if n_sensors <= 0:
            # Current observation appends normalized cutter_x and cutter_y after
            # the displacement/velocity channels; they are not sensor channels.
            n_sensors = max((obs.shape[1] - 2) // 2, 0)
        w_scale = float(metadata.get("w_obs_scale", metadata.get("w_obs_scale_m", 1.0)))
        wdot_scale = float(metadata.get("wdot_obs_scale", metadata.get("wdot_obs_scale_m_s", 1.0)))

        if not np.isfinite(w_scale) or abs(w_scale) < 1e-12:
            w_scale = 1.0

        if not np.isfinite(wdot_scale) or abs(wdot_scale) < 1e-12:
            wdot_scale = 1.0

        if n_sensors > 0 and obs.shape[1] >= 2 * n_sensors:
            signals = obs[:, : 2 * n_sensors].copy()
            signals[:, :n_sensors] *= w_scale
            signals[:, n_sensors : 2 * n_sensors] *= wdot_scale
        else:
            return np.asarray([], dtype=np.float64)

    else:
        return np.asarray([], dtype=np.float64)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    return signals


def _non_empty_sequence(value) -> bool:
    """Return True for non-empty saved list/array-like values."""
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def _saved_actions_from_episode(ep: dict, metadata: dict) -> tuple[np.ndarray, bool]:
    """Return actions for plotting and whether they are already physical.

    New eval files save both normalized ``actions`` and ``physical_actions``.
    Prefer the physical values.  For older files, fall back to normalized
    actions instead of incorrectly assuming that ``actions`` are physical just
    because the metadata describes the plant's physical action convention.
    """
    record = primary_episode_record(ep)

    physical = record.get("physical_actions", ep.get("physical_actions", []))
    if _non_empty_sequence(physical):
        return np.asarray(physical, dtype=np.float64), True

    actions = record.get("actions", ep.get("actions", []))
    if _non_empty_sequence(actions):
        return np.asarray(actions, dtype=np.float64), False

    return np.asarray([], dtype=np.float64), False


def _time_grid(T: np.ndarray | None, max_t: int) -> tuple[np.ndarray, str]:
    """Return representative time grid."""
    if T is None:
        return np.arange(max_t), "Step"

    valid = np.any(~np.isnan(T), axis=0)

    if not np.any(valid):
        return np.arange(max_t), "Step"

    grid = np.arange(max_t, dtype=np.float64)
    grid[valid] = np.nanmean(T[:, valid], axis=0)

    return grid, "Time (s)"


def plot_state_trajectories(
    S: np.ndarray,
    T: np.ndarray | None,
    metadata: dict,
    out_dir: Path,
    *,
    suffix: str = "",
) -> None:
    """Plot physical sensor signals with mean±std and displacement safety bounds."""
    state_dim = S.shape[2]
    n_sensors = int(metadata.get("n_sensors", state_dim // 2))

    if n_sensors <= 0:
        n_sensors = state_dim // 2 if state_dim >= 2 else state_dim

    labels = _physical_signal_labels(n_sensors)

    if len(labels) < state_dim:
        labels.extend(f"signal{idx + 1}" for idx in range(len(labels), state_dim))

    t_grid, x_label = _time_grid(T, S.shape[1])

    w_limit = metadata.get("w_limit", None)

    n_cols = 2
    n_rows = int(np.ceil(state_dim / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 3.0 * n_rows),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for dim in range(state_dim):
        ax = axes[dim]

        mean_s, std_s, valid = _nan_mean_std(S[:, :, dim])

        if S.shape[0] > 1:
            ax.fill_between(
                t_grid[valid],
                mean_s[valid] - PLOT_STD_MULT * std_s[valid],
                mean_s[valid] + PLOT_STD_MULT * std_s[valid],
                alpha=0.25,
            )
        ax.plot(t_grid[valid], mean_s[valid], lw=1.5)

        # Show physical displacement limit only for displacement channels.
        if w_limit is not None and np.isfinite(w_limit) and dim < n_sensors:
            ax.axhline(+float(w_limit), linestyle="--", linewidth=1)
            ax.axhline(-float(w_limit), linestyle="--", linewidth=1)

        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.3)

    for ax in axes[state_dim:]:
        ax.axis("off")

    for ax in axes[:state_dim]:
        if ax.get_visible() and ax.axison:
            ax.set_xlabel(x_label)

    band_text = f"mean ± {PLOT_STD_MULT:g} SD across {S.shape[0]} MC runs" if S.shape[0] > 1 else "single evaluation rollout"
    fig.suptitle(f"Physical sensor response — {band_text}")
    fig.tight_layout()

    out_path = out_dir / f"trajectory_physical_sensor_response{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")


def _physical_action_plot_label(name: str) -> str:
    """Human-readable axis label for physical action channels."""
    if name == "omega_rad_s":
        return "Spindle speed (rpm)"
    if name == "ap_mm":
        return "Axial depth of cut ap (mm)"
    if name == "ae_mm":
        return "Radial immersion ae (mm)"
    return name


def _normalized_action_plot_label(name: str, dim: int) -> str:
    """Human-readable axis label for normalized action channels."""
    if name == "omega_rad_s":
        return "u_omega (normalized)"
    if name == "ap_mm":
        return "u_ap (normalized)"
    if name == "ae_mm":
        return "u_ae (normalized)"
    return f"u_{dim} (normalized)"


def _maybe_convert_action_channel_for_plot(
    values: np.ndarray,
    *,
    action_name: str,
    actions_are_physical: bool,
) -> np.ndarray:
    """Convert omega from rad/s to rpm for plotting; leave other channels unchanged."""
    if actions_are_physical and action_name == "omega_rad_s":
        return omega_to_rpm(values)
    return values


def plot_action_trajectories(
    A: np.ndarray,
    T: np.ndarray | None,
    metadata: dict,
    out_dir: Path,
    *,
    suffix: str = "",
) -> None:
    """Plot policy actions with correct face-milling labels and bounds."""
    action_dim = A.shape[2]
    t_grid, x_label = _time_grid(T, A.shape[1])
    actions_are_physical_for_plot = bool(metadata.get("_plot_actions_are_physical", False))
    names = physical_action_names(metadata, n_dims=action_dim)

    finishing = metadata.get("env_id") == "CustomODEPlateFinish-v0"
    if actions_are_physical_for_plot:
        action_labels = [_physical_action_plot_label(name) for name in names]
        if finishing:
            action_labels = [
                "Fixed axial depth ap (mm; not a PPO action)" if name == "ap_mm" else label
                for name, label in zip(names, action_labels)
            ]
    else:
        action_labels = [_normalized_action_plot_label(name, idx) for idx, name in enumerate(names)]

    physical_low = metadata.get("physical_action_low", None)
    physical_high = metadata.get("physical_action_high", None)

    if physical_low is not None:
        physical_low = np.asarray(physical_low, dtype=np.float64)

    if physical_high is not None:
        physical_high = np.asarray(physical_high, dtype=np.float64)

    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(8, 3.0 * action_dim),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for dim in range(action_dim):
        ax = axes[dim]
        action_name = names[dim] if dim < len(names) else f"action_{dim}"

        series = _maybe_convert_action_channel_for_plot(
            A[:, :, dim].copy(),
            action_name=action_name,
            actions_are_physical=actions_are_physical_for_plot,
        )

        mean_a, std_a, valid = _nan_mean_std(series)
        band = PLOT_STD_MULT * std_a

        if A.shape[0] > 1:
            ax.fill_between(
                t_grid[valid],
                mean_a[valid] - band[valid],
                mean_a[valid] + band[valid],
                alpha=0.25,
                step="post",
            )
        ax.step(t_grid[valid], mean_a[valid], where="post", lw=1.5)

        if actions_are_physical_for_plot:
            if (
                physical_low is not None
                and physical_high is not None
                and dim < len(physical_low)
                and dim < len(physical_high)
            ):
                low = float(physical_low[dim])
                high = float(physical_high[dim])
                if action_name == "omega_rad_s":
                    low = float(omega_to_rpm(low))
                    high = float(omega_to_rpm(high))
                ax.axhline(low, linestyle="--", linewidth=1)
                ax.axhline(high, linestyle="--", linewidth=1)
        else:
            ax.axhline(-1.0, linestyle="--", linewidth=1)
            ax.axhline(+1.0, linestyle="--", linewidth=1)

        ax.set_ylabel(action_labels[dim])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)

    title_mode = (
        "spindle command and fixed process settings" if finishing and actions_are_physical_for_plot
        else "physical control actions" if actions_are_physical_for_plot
        else "normalized PPO control actions"
    )
    band_text = f"mean ± {PLOT_STD_MULT:g} SD across {A.shape[0]} MC runs" if A.shape[0] > 1 else "single evaluation rollout"
    fig.suptitle(f"{title_mode}: {band_text}")
    fig.tight_layout()

    out_path = out_dir / f"trajectory_actions_{'physical' if actions_are_physical_for_plot else 'normalized'}{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")


PROCESS_PLOT_SPECS = [
    ("cutter_x", "Cutter x-position (m)"),
    ("cutter_y", "Milling line y-position (m)"),
    ("feed_progress", "Feed progress"),
    ("omega_rpm", "Spindle speed (rpm)"),
    ("ap_mm", "Axial depth of cut ap (mm)"),
    ("ae_mm", "Radial immersion ae (mm)"),
    ("mean_chip_mm", "Mean chip thickness (mm)"),
    ("max_chip_mm", "Max chip thickness (mm)"),
]


def plot_process_arrays(
    P: np.ndarray,
    T: np.ndarray,
    keys: list[str],
    out_dir: Path,
    *,
    suffix: str = "",
) -> None:
    """Plot process signals for one episode's MC replicas, never mixing test scenarios."""
    if P.size == 0 or not keys:
        return
    label_lookup = dict(PROCESS_PLOT_SPECS)
    t_grid, x_label = _time_grid(T, P.shape[1])
    n_channels = P.shape[2]
    n_cols = 2
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.0 * n_rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for dim, key in enumerate(keys):
        ax = axes[dim]
        mean_p, std_p, valid = _nan_mean_std(P[:, :, dim])
        if P.shape[0] > 1:
            band = PLOT_STD_MULT * std_p
            ax.fill_between(t_grid[valid], mean_p[valid] - band[valid], mean_p[valid] + band[valid], alpha=0.25)
        ax.plot(t_grid[valid], mean_p[valid], lw=1.5)
        ax.set_ylabel(label_lookup.get(key, key))
        ax.grid(True, alpha=0.3)

    for ax in axes[n_channels:]:
        ax.axis("off")
    for ax in axes[:n_channels]:
        if ax.get_visible() and ax.axison:
            ax.set_xlabel(x_label)
    band_text = f"mean ± {PLOT_STD_MULT:g} SD across {P.shape[0]} MC runs" if P.shape[0] > 1 else "single evaluation rollout"
    fig.suptitle(f"Face-milling process signals — {band_text}")
    fig.tight_layout()
    out_path = out_dir / f"trajectory_process_signals{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _scenario_value(run: dict, episode: dict, key: str) -> float | None:
    value = run.get(key, episode.get(key))
    if value is None:
        reset_options = run.get("reset_options") or {}
        reset_info = run.get("reset_info") or {}
        if key == "y_line_m":
            value = reset_options.get("y0", reset_info.get("y_cutter"))
        elif key == "ap_mm":
            value = reset_options.get("ap", reset_info.get("ap_mm", reset_info.get("ap")))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _episode_env_id(episode: dict) -> str:
    metadata = episode.get("metadata", {})
    env_id = metadata.get("env_id") if isinstance(metadata, dict) else None
    if env_id:
        return str(env_id)
    runs = episode.get("runs") or [episode]
    for run in runs:
        actions = run.get("physical_actions") or run.get("actions") or []
        try:
            array = np.asarray(actions)
            if array.ndim >= 2 and array.shape[-1] == 1:
                return "CustomODEPlateFinish-v0"
            if array.ndim >= 2 and array.shape[-1] >= 2:
                return "CustomODEPlate-v0"
        except Exception:
            continue
    return "CustomODEPlate-v0"


def _same_mc_scenario(runs: list[dict], episode: dict, env_id: str) -> tuple[bool, str]:
    """Check that MC replicas share the same physical scenario before averaging."""
    if len(runs) <= 1:
        return True, ""
    for key in ("y_line_m", "ap_mm"):
        values = [_scenario_value(run, episode, key) for run in runs]
        if key == "y_line_m" and any(value is None for value in values):
            return False, "milling-line position is not recorded for every MC replica"
        if env_id == "CustomODEPlateFinish-v0" and key == "ap_mm" and any(value is None for value in values):
            return False, "finishing ap is not recorded for every MC replica"
        known = [value for value in values if value is not None]
        if known and (max(known) - min(known) > 1e-9):
            return False, f"MC replicas use different {key} values"
    return True, ""


def _pad_time_series(
    series: list[np.ndarray],
    times: list[np.ndarray],
    *,
    expected_dim: int | None = None,
    kind: str = "continuous",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Align MC replicas by *physical time*, not by row number.

    Continuous sensor/process values are linearly interpolated. Zero-order-hold
    actions use the last action applied at/before the grid time. Neither kind
    extrapolates past a replica's final timestamp. A coarsened grid is strictly
    for visualization and does not modify stored evaluation samples/statistics.
    """
    if kind not in {"continuous", "action"}:
        raise ValueError(f"Unknown time-series interpolation kind: {kind}")
    valid_runs = []
    for values, t in zip(series, times):
        values = np.asarray(values, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        n = min(values.shape[0], t.size)
        if n <= 0:
            continue
        values, t = values[:n], t[:n]
        if expected_dim is None:
            expected_dim = values.shape[1]
        if values.shape[1] != expected_dim:
            raise ValueError("MC time-series dimensions differ; cannot average.")
        if not np.all(np.isfinite(t)) or (n > 1 and not np.all(np.diff(t) > 0)):
            raise ValueError("MC timestamps must be finite and strictly increasing.")
        valid_runs.append((values, t))
    if not valid_runs:
        return None, None
    if len(valid_runs) == 1:
        values, times0 = valid_runs[0]
        if len(times0) > MAX_PLOTTED_TIME_POINTS:
            indices = np.unique(np.r_[np.arange(0, len(times0),
                int(np.ceil(len(times0) / MAX_PLOTTED_TIME_POINTS))), len(times0)-1]).astype(int)
            print("Plot note: single-trajectory display decimated; use full-rate saved data for high-frequency inference.")
            values, times0 = values[indices], times0[indices]
        return values[np.newaxis, :, :], times0[np.newaxis, :]

    # Use the slowest observed cadence to avoid artificially increasing
    # time resolution; cap display points without modifying raw records.
    increments = [float(np.median(np.diff(t))) for _, t in valid_runs if len(t) > 1]
    if not increments:
        return None, None
    dt = max(increments)
    earliest = min(float(t[0]) for _, t in valid_runs)
    latest = max(float(t[-1]) for _, t in valid_runs)
    if latest <= earliest:
        return None, None
    point_count = int(np.ceil((latest - earliest) / dt)) + 1
    if point_count > MAX_PLOTTED_TIME_POINTS:
        dt = (latest - earliest) / (MAX_PLOTTED_TIME_POINTS - 1)
        print("Plot note: time grid coarsened for display; do not infer vibration frequency content from this figure.")
    grid = np.arange(earliest, latest + dt * 0.25, dt, dtype=np.float64)
    if grid[-1] < latest - 0.1 * dt:
        grid = np.append(grid, latest)
    out = np.full((len(valid_runs), len(grid), int(expected_dim)), np.nan, dtype=np.float64)
    for i, (values, t) in enumerate(valid_runs):
        # No extrapolation, particularly beyond an earlier safety termination.
        supported = (grid >= t[0]) & (grid <= t[-1])
        if not np.any(supported):
            continue
        q = grid[supported]
        for j in range(int(expected_dim)):
            signal = values[:, j]
            finite = np.isfinite(signal)
            if not np.any(finite):
                continue
            if kind == "action":
                indices = np.searchsorted(t, q, side="right") - 1
                out[i, supported, j] = signal[np.clip(indices, 0, len(t) - 1)]
            else:
                if np.sum(finite) >= 2:
                    out[i, supported, j] = np.interp(q, t[finite], signal[finite], left=np.nan, right=np.nan)
                elif np.sum(finite) == 1:
                    close = np.isclose(q, t[finite][0], rtol=1e-8, atol=1e-10)
                    positions = np.where(supported)[0][close]
                    out[i, positions, j] = signal[finite][0]
    return out, np.broadcast_to(grid, (len(valid_runs), len(grid))).copy()


def _collect_episode_mc_arrays(episode: dict) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict, list[str]]:
    metadata = dict(episode.get("metadata", {}))
    runs = episode.get("runs") or [episode]
    state_runs: list[np.ndarray] = []
    state_times: list[np.ndarray] = []
    action_runs: list[np.ndarray] = []
    action_times: list[np.ndarray] = []
    process_runs: list[dict] = []
    process_times: list[np.ndarray] = []
    action_physical_flags: list[bool] = []

    for run in runs:
        wrapped = dict(run)
        wrapped["metadata"] = metadata
        states = _physical_signals_from_episode(wrapped)
        actions, is_physical = _saved_actions_from_episode(wrapped, metadata)
        times = np.asarray(run.get("times", []), dtype=np.float64)
        a_times = np.asarray(run.get("action_times", run.get("times", [])), dtype=np.float64)
        if states.size:
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            state_runs.append(states)
            state_times.append(times)
        if actions.size:
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            # The last action is held until the final post-step physical time.
            # Include that endpoint for ZOH resampling without inventing a new
            # policy decision or an extra step in the trajectory record.
            n = min(len(actions), len(a_times))
            actions, a_times = actions[:n], a_times[:n]
            if n and times.size and np.isfinite(times[-1]) and times[-1] > a_times[-1]:
                actions = np.vstack([actions, actions[-1:]])
                a_times = np.r_[a_times, times[-1]]
            action_runs.append(actions)
            action_times.append(a_times)
            action_physical_flags.append(bool(is_physical))
        process = run.get("process", {})
        if isinstance(process, dict) and times.size:
            process_runs.append(process)
            process_times.append(times)

    S, T = _pad_time_series(state_runs, state_times)
    if len(set(action_physical_flags)) > 1:
        A, TA = None, None
    else:
        A, TA = _pad_time_series(action_runs, action_times, kind="action")
        metadata["_plot_actions_are_physical"] = bool(action_physical_flags and action_physical_flags[0])

    available_keys = {
        key for key, _ in PROCESS_PLOT_SPECS
        if any(_non_empty_sequence(process.get(key)) for process in process_runs)
    }
    process_keys = [key for key, _ in PROCESS_PLOT_SPECS if key in available_keys]
    P = PT = None
    if process_keys and process_runs:
        process_arrays = []
        for process, t in zip(process_runs, process_times):
            n = len(t)
            values = np.full((n, len(process_keys)), np.nan, dtype=np.float64)
            for key_idx, key in enumerate(process_keys):
                channel = np.asarray(process.get(key, []), dtype=np.float64).reshape(-1)
                m = min(n, len(channel))
                values[:m, key_idx] = channel[:m]
            process_arrays.append(values)
        P, PT = _pad_time_series(process_arrays, process_times, kind="continuous")

    return S, T, A, TA, P, PT, metadata, process_keys

def plot_trajectory_summary(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Plot MC variability separately for each training seed and fixed test episode."""
    available = [seed for seed in seeds if (traj_dir / f"trajectories_seed{seed}.json").is_file()]
    missing = [seed for seed in seeds if seed not in available]
    if missing:
        print(f"Note: no trajectory file for seed(s) {missing} in {traj_dir}")
    if not available:
        print(f"No trajectory JSON files in {traj_dir}")
        return

    for seed in available:
        path = traj_dir / f"trajectories_seed{seed}.json"
        with path.open("r", encoding="utf-8") as stream:
            episodes = json.load(stream)
        if not isinstance(episodes, list):
            print(f"Skipping malformed trajectory file: {path}")
            continue

        for ep_index, episode in enumerate(episodes):
            metadata = episode.get("metadata", {})
            env_id = _episode_env_id(episode)
            runs = episode.get("runs") or [episode]
            same_scenario, reason = _same_mc_scenario(runs, episode, env_id)
            if not same_scenario:
                print(
                    f"Skipping MC averaging for seed {seed}, episode {ep_index}: {reason}. "
                    "Use one fixed y0/ap per episode before interpreting a band as stochastic variability."
                )
                continue

            S, T, A, TA, P, PT, plot_metadata, process_keys = _collect_episode_mc_arrays(episode)
            suffix = f"_seed{seed}_episode{ep_index}"
            if S is not None:
                plot_state_trajectories(S, T, plot_metadata, out_dir, suffix=suffix)
                plot_vibration_rms_envelope(episode, plot_metadata, out_dir, suffix=suffix)
            if A is not None:
                plot_action_trajectories(A, TA, plot_metadata, out_dir, suffix=suffix)
            if P is not None and PT is not None:
                plot_process_arrays(P, PT, process_keys, out_dir, suffix=suffix)

def _windowed_rms(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Trailing RMS at each sample using only finite, within-episode observations."""
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    powers = np.where(finite, arr ** 2, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(powers, dtype=np.float64)))
    cc = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    indices = np.arange(arr.size)
    starts = np.maximum(indices + 1 - window_samples, 0)
    ends = indices + 1
    sums = cs[ends] - cs[starts]
    counts = cc[ends] - cc[starts]
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    supported = (counts > 0) & finite
    out[supported] = np.sqrt(np.maximum(sums[supported] / counts[supported], 0.0))
    return out


def plot_vibration_rms_envelope(
    episode: dict, metadata: dict, out_dir: Path, *, suffix: str = "",
    window_s: float = 0.05,
) -> None:
    """Compute RMS on ORIGINAL saved physical samples, then align for display.

    Taking RMS after plotting decimation can severely alias milling vibrations,
    so the order of operations is deliberate: rolling RMS for each independent
    physical rollout, then physical-time alignment, then mean and sample SD.
    """
    runs = episode.get("runs") or [episode]
    n_sensors = int(metadata.get("n_sensors", 0))
    if n_sensors <= 0:
        return
    rms_runs = []
    rms_times = []
    for run in runs:
        wrapped = dict(run)
        wrapped["metadata"] = metadata
        signal = _physical_signals_from_episode(wrapped)
        t = np.asarray(run.get("times", []), dtype=float).reshape(-1)
        n = min(len(signal), len(t))
        if n < 5 or signal.ndim != 2 or signal.shape[1] < n_sensors:
            continue
        signal, t = signal[:n], t[:n]
        intervals = np.diff(t)
        if not np.all(np.isfinite(intervals)) or not np.all(intervals > 0):
            print(f"RMS envelope skipped {suffix}: invalid physical-time samples.")
            return
        dt = float(np.median(intervals))
        if dt > window_s / 5.0:
            print(f"RMS envelope skipped {suffix}: recorded dt={dt:g} s is too coarse for {window_s:g} s window.")
            return
        n_window = max(5, int(round(window_s / dt)))
        envelopes = np.column_stack([_windowed_rms(signal[:, j], n_window) for j in range(n_sensors)])
        rms_runs.append(envelopes)
        rms_times.append(t)
    if len(rms_runs) != len(runs):
        print(f"RMS envelope skipped {suffix}: physical signal/time missing for one or more MC replicas.")
        return
    R, T = _pad_time_series(rms_runs, rms_times, kind="continuous")
    if R is None or T is None:
        return
    grid, label = _time_grid(T, R.shape[1])
    fig, axes = plt.subplots(n_sensors, 1, figsize=(9, max(3.2, 2.4*n_sensors)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for j, ax in enumerate(axes):
        envelopes = R[:, :, j] * 1e6
        means, sds, supported = _nan_mean_std(envelopes)
        valid = supported & np.isfinite(means)
        ax.plot(grid[valid], means[valid], linewidth=1.4, color="C0")
        band = valid & np.isfinite(sds)
        if R.shape[0] > 1 and np.any(band):
            ax.fill_between(grid[band], means[band]-sds[band], means[band]+sds[band], alpha=0.25, color="C0")
        ax.set_ylabel(f"Sensor {j+1} RMS (µm)")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(label)
    fig.suptitle(f"Sliding displacement RMS ({window_s*1000:g}-ms window; across-MC mean ± 1 SD)")
    fig.tight_layout()
    out = out_dir / f"trajectory_displacement_rms{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_physical_outcomes(traj_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    """Paper-quality physical outcomes, one observation per *outer scenario*.

    Each point is the mean of MC replicas of a fixed physical scenario. Do
    not treat replicas as additional independent test scenarios.
    """
    path = traj_dir / "physical_outcomes_scenarios.csv"
    if not path.is_file():
        print(f"Physical outcomes figure skipped: missing {path}")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows_by_seed = {}
    for seed in seeds:
        rows = [r for r in all_rows if r.get("training_seed") == str(seed)]
        rows.sort(key=lambda r: int(r["scenario_index"]))
        if rows:
            rows_by_seed[seed] = rows
    if not rows_by_seed:
        return
    # Do not silently call different y0/ap conditions a matched comparison.
    replica_path = traj_dir / "physical_outcomes_replicas.csv"
    conditions_matched = False
    if replica_path.is_file() and len(rows_by_seed) > 1:
        with replica_path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
        by_pair = {}
        for row in raw:
            try:
                key = (int(row["training_seed"]), int(row["scenario_index"]), int(row["mc_index"]))
                by_pair[key] = (int(row["reset_seed"]), float(row["y_line_m"]),
                                float(row["ap_fixed_mm"]) if row.get("ap_fixed_mm") not in ("", None) else None)
            except (TypeError, ValueError, KeyError):
                continue
        pairs = [int(seed) for seed in rows_by_seed]
        shared = set((sc, mc) for seed, sc, mc in by_pair if seed == pairs[0])
        for seed in pairs[1:]:
            shared &= set((sc, mc) for s, sc, mc in by_pair if s == seed)
        if shared and all((seed, sc, mc) in by_pair for seed in pairs for sc, mc in shared):
            def same_conditions(a: tuple, b: tuple) -> bool:
                return (a[0] == b[0]
                    and abs(a[1] - b[1]) < 1e-9
                    and ((a[2] is None and b[2] is None)
                         or (a[2] is not None and b[2] is not None
                             and abs(a[2] - b[2]) < 1e-9)))
            conditions_matched = all(
                same_conditions(by_pair[(s, sc, mc)], by_pair[(pairs[0], sc, mc)])
                for sc, mc in shared
                for s in pairs[1:]
            )
            # Paired status requires the SAME complete set of MC cases for each seed.
            conditions_matched = conditions_matched and all(
                set((sc, mc) for ss, sc, mc in by_pair if ss == s) == shared
                for s in pairs
            )
    if len(rows_by_seed) > 1 and not conditions_matched:
        print("Physical outcomes warning: scenarios could not be verified as paired across seeds; figure labels them as unpaired.")
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=False)
    spec = [
        ("rms_w_m_mean", "rms_w_m_sample_sd", 1e6, "MC-mean sensor displacement RMS (µm)"),
        ("max_abs_w_m_mean", "max_abs_w_m_sample_sd", 1e6, "MC-mean peak |displacement| (µm)"),
        ("final_feed_progress_mean", "final_feed_progress_sample_sd", 100.0, "Final feed progress (%)"),
        ("pass_completion_fraction", None, 100.0, "Pass completion (%)"),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    sorted_seeds = sorted(rows_by_seed)
    for idx, (field, sd_field, scale, ylabel) in enumerate(spec):
        ax = axs.ravel()[idx]
        for rank, seed in enumerate(sorted_seeds):
            rows = rows_by_seed[seed]
            offset = (rank - (len(sorted_seeds) - 1)/2) * 0.15
            x = np.asarray([int(r["scenario_index"]) for r in rows], dtype=float) + offset
            y = np.asarray([float(r[field]) if r.get(field) not in (None, "") else np.nan for r in rows], dtype=float) * scale
            color = colors[rank % len(colors)]
            if sd_field is not None:
                err = np.asarray([float(r[sd_field]) if r.get(sd_field) not in (None, "") else np.nan for r in rows], dtype=float) * scale
                if np.any(np.isfinite(err)):
                    ax.errorbar(x, y, yerr=np.where(np.isfinite(err), err, 0.0),
                                fmt="o", markersize=4.5, elinewidth=1, capsize=2,
                                color=color, label=f"Seed {seed}")
                else:
                    ax.scatter(x, y, s=26, color=color, label=f"Seed {seed}", zorder=3)
            else:
                ax.scatter(x, y, s=32, color=color, label=f"Seed {seed}", zorder=3)
        if not any(
            np.isfinite(float(row[field]))
            for rows in rows_by_seed.values()
            for row in rows
            if row.get(field) not in (None, "")
        ):
            ax.text(0.5, 0.5, "Metric not recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Held-out scenario" + (" index" if conditions_matched else " index (unpaired)"))
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if idx == 3:
            ax.set_ylim(-5, 105)
        ax.set_xticks(sorted({int(row["scenario_index"]) for rows in rows_by_seed.values() for row in rows}))
    axs.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("PPO physical outcomes across " + ("matched" if conditions_matched else "seed-specific") + " held-out scenarios")
    fig.text(.5, .007,
             "Each marker: scenario-level MC mean; error bars: ±1 MC sample SD where available (not 95% CI).",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    out = out_dir / "figure2_ppo_physical_outcomes.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _resolve_plot_seeds(
    seeds: list[int] | None,
    log_dir: Path,
    traj_dir: Path,
) -> tuple[list[int], list[int]]:
    """Return (log_seeds, traj_seeds), auto-discovering when seeds is None."""
    if seeds:
        return list(seeds), list(seeds)

    log_seeds = discover_log_seeds(log_dir)
    # EvalCallback history can exist before a worker has completed its first
    # monitored episode, so also discover seed folders directly from evaluations.npz.
    for seed_dir in log_dir.glob("seed_*") if log_dir.is_dir() else []:
        try:
            seed_value = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        if (seed_dir / "evaluations.npz").is_file() and seed_value not in log_seeds:
            log_seeds.append(seed_value)
    log_seeds = sorted(set(log_seeds))
    traj_seeds = discover_trajectory_seeds(traj_dir)

    if log_seeds or traj_seeds:
        combined = sorted(set(log_seeds) | set(traj_seeds))
        return combined, combined

    return [], []


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and trajectory results")

    parser.add_argument(
        "--log-dir",
        default=None,
        help=f"EvalCallback logs; inferred as a sibling of --traj-dir when possible (fallback: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--traj-dir",
        default=None,
        help=f"eval_policy.py output; inferred as a sibling of --log-dir when possible (fallback: {DEFAULT_TRAJ_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Plot output directory; inferred as <run-dir>/plots when possible (fallback: {DEFAULT_PLOT_DIR})",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds to plot (default: auto-discover from log-dir and traj-dir)",
    )
    parser.add_argument(
        "--smooth", type=int, default=1,
        help="Retained for compatibility; the scientific Figure 1 is intentionally unsmoothed.",
    )

    args = parser.parse_args()

    if args.log_dir is not None:
        log_dir = Path(args.log_dir).resolve()
    elif args.traj_dir is not None and (Path(args.traj_dir).resolve().parent / "logs").is_dir():
        log_dir = (Path(args.traj_dir).resolve().parent / "logs").resolve()
    else:
        log_dir = Path(DEFAULT_LOG_DIR).resolve()

    if args.traj_dir is not None:
        traj_dir = Path(args.traj_dir).resolve()
    elif args.log_dir is not None and (Path(args.log_dir).resolve().parent / "evaluation").is_dir():
        traj_dir = (Path(args.log_dir).resolve().parent / "evaluation").resolve()
    else:
        traj_dir = Path(DEFAULT_TRAJ_DIR).resolve()

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    elif log_dir.parent == traj_dir.parent and (log_dir.parent / "logs").is_dir():
        out_dir = (log_dir.parent / "plots").resolve()
    elif (traj_dir.parent / "logs").is_dir():
        out_dir = (traj_dir.parent / "plots").resolve()
    elif (log_dir.parent / "evaluation").is_dir():
        out_dir = (log_dir.parent / "plots").resolve()
    else:
        out_dir = Path(DEFAULT_PLOT_DIR).resolve()

    log_seeds, traj_seeds = _resolve_plot_seeds(args.seeds, log_dir, traj_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Log directory  : {log_dir}")
    print(f"Trajectory dir : {traj_dir}")
    print(f"Plot output    : {out_dir}")
    print(f"Learning curve seeds: {log_seeds}")
    print(f"Trajectory seeds    : {traj_seeds}")

    plot_learning_curve(log_dir, out_dir, log_seeds, smooth=args.smooth)
    plot_training_diagnostics(log_dir, out_dir, log_seeds)
    plot_trajectory_summary(traj_dir, out_dir, traj_seeds)
    plot_physical_outcomes(traj_dir, out_dir, traj_seeds)


if __name__ == "__main__":
    main()