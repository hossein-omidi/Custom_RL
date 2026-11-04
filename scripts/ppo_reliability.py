"""Pure reliability metrics used to rank PPO checkpoints."""
from __future__ import annotations
from statistics import median, mean
from typing import Any, Iterable


def summarize_reliability_episodes(episodes: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(episodes)
    if not rows:
        raise ValueError("At least one evaluation episode is required.")
    n = len(rows)
    completed = [r for r in rows if bool(r.get("pass_completed", False))]
    completed_returns = [float(r["return"]) for r in completed]
    completed_productivity = [float(r.get("productivity_reward_sum", 0.0)) for r in completed]
    return {
        "n_episodes": float(n),
        "completed_episodes": float(len(completed)),
        "pass_completion_rate": float(len(completed) / n),
        "safety_termination_rate": float(sum(bool(r.get("safety_termination", False)) for r in rows) / n),
        "truncation_rate": float(sum(bool(r.get("truncated", False)) for r in rows) / n),
        "median_completed_return": float(median(completed_returns)) if completed_returns else 0.0,
        "median_completed_productivity": float(median(completed_productivity)) if completed_productivity else 0.0,
        "median_progress": float(median(float(r.get("final_progress", 0.0)) for r in rows)),
        "mean_return": float(mean(float(r["return"]) for r in rows)),
        "mean_max_displacement_m": float(mean(float(r.get("max_displacement_m", 0.0)) for r in rows)),
    }


def reliability_checkpoint_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    """Higher tuple is better: completion first, then safety/truncation, completed-pass productivity, progress."""
    return (
        float(metrics["pass_completion_rate"]),
        -float(metrics["safety_termination_rate"]),
        -float(metrics["truncation_rate"]),
        float(metrics["median_completed_productivity"]),
        float(metrics["median_progress"]),
        float(metrics["mean_return"]),
    )


def is_better_checkpoint(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    return incumbent is None or reliability_checkpoint_rank(candidate) > reliability_checkpoint_rank(incumbent)
