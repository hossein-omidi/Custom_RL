"""Checkpoint resolution and verified selection for config-based runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

ENV_ID = "CustomODEPlate-v0"


@dataclass
class CheckpointSelection:
    """Result of checkpoint resolution for one training seed."""

    seed: int
    path: str
    source: str
    policy: str
    mean_return: float | None = None
    std_return: float | None = None
    n_verify_episodes: int = 0
    eval_callback_peak_mean: float | None = None
    eval_callback_best_timestep: int | None = None
    candidates: list[dict[str, Any]] | None = None


def _paths(model_dir: Path, seed: int) -> tuple[Path, Path]:
    return (
        model_dir / f"best_{seed}" / "best_model.zip",
        model_dir / f"final_{seed}.zip",
    )


def load_eval_callback_history(log_dir: Path, seed: int) -> dict[str, Any] | None:
    """
  Read SB3 EvalCallback evaluations.npz written during training.

  Returns peak mean eval return and the timestep index where it occurred.
  """
    npz_path = log_dir / f"seed_{seed}" / "evaluations.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    results = np.asarray(data["results"], dtype=np.float64)
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)

    if results.size == 0:
        return None

    mean_per_eval = np.mean(results, axis=1)
    best_idx = int(np.argmax(mean_per_eval))

    return {
        "peak_mean_return": float(mean_per_eval[best_idx]),
        "peak_timestep": int(timesteps[best_idx]) if timesteps.size > best_idx else None,
        "n_eval_points": int(mean_per_eval.size),
        "last_mean_return": float(mean_per_eval[-1]),
    }


def rollout_mean_return(
    model_path: Path,
    env_kwargs: dict[str, Any],
    *,
    n_episodes: int,
    seed_base: int,
    deterministic: bool = True,
) -> tuple[float, float]:
    """Empirical mean±std episode return over n_episodes (hold-out verification)."""
    from custom_rl import register_envs

    register_envs()
    model = PPO.load(str(model_path), device="cpu")
    env = gym.make(ENV_ID, **env_kwargs)
    max_steps = int(env.unwrapped.max_episode_steps)

    returns: list[float] = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + 10_000 + i)
        total = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)

    env.close()
    arr = np.asarray(returns, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def select_checkpoint(
    model_dir: Path,
    log_dir: Path,
    seed: int,
    *,
    policy: str = "verified",
    env_kwargs: dict[str, Any] | None = None,
    n_verify_episodes: int = 5,
    deterministic: bool = True,
) -> CheckpointSelection | None:
    """
    Resolve the policy checkpoint for evaluation.

    Policies
    --------
    best:
        EvalCallback ``best_model.zip`` (training-time peak eval return).
        Falls back to ``final_<seed>.zip`` if best is missing.

    final:
        Last snapshot ``final_<seed>.zip`` only.

    auto:
        ``best`` if present, else ``final``.

    verified (recommended):
        1. Read EvalCallback ``evaluations.npz`` peak (training metric).
        2. Run hold-out rollouts on ``best`` and ``final`` (if both exist).
        3. Pick the checkpoint with highest verified mean return.
        4. Tie-break toward ``best`` (matches EvalCallback intent).
    """
    model_dir = Path(model_dir)
    log_dir = Path(log_dir)
    best_path, final_path = _paths(model_dir, seed)
    eval_hist = load_eval_callback_history(log_dir, seed)
    callback_peak = eval_hist["peak_mean_return"] if eval_hist else None
    callback_step = eval_hist["peak_timestep"] if eval_hist else None

    if policy in {"best", "auto"}:
        if policy == "best" or best_path.exists():
            if best_path.exists():
                return CheckpointSelection(
                    seed=seed,
                    path=str(best_path),
                    source="best",
                    policy=policy,
                    eval_callback_peak_mean=callback_peak,
                    eval_callback_best_timestep=callback_step,
                )
        if final_path.exists():
            return CheckpointSelection(
                seed=seed,
                path=str(final_path),
                source="final",
                policy=policy,
                eval_callback_peak_mean=callback_peak,
                eval_callback_best_timestep=callback_step,
            )
        return None

    if policy == "final":
        if final_path.exists():
            return CheckpointSelection(
                seed=seed,
                path=str(final_path),
                source="final",
                policy=policy,
                eval_callback_peak_mean=callback_peak,
                eval_callback_best_timestep=callback_step,
            )
        return None

    if policy == "verified":
        if env_kwargs is None:
            raise ValueError("env_kwargs required for verified checkpoint policy")

        candidates: list[dict[str, Any]] = []
        for label, path in (("best", best_path), ("final", final_path)):
            if not path.exists():
                continue
            mean_r, std_r = rollout_mean_return(
                path,
                env_kwargs,
                n_episodes=n_verify_episodes,
                seed_base=seed,
                deterministic=deterministic,
            )
            candidates.append(
                {
                    "label": label,
                    "path": str(path),
                    "mean_return": mean_r,
                    "std_return": std_r,
                }
            )

        if not candidates:
            return None

        candidates.sort(key=lambda c: (-c["mean_return"], 0 if c["label"] == "best" else 1))
        winner = candidates[0]
        return CheckpointSelection(
            seed=seed,
            path=winner["path"],
            source=f"verified_{winner['label']}",
            policy=policy,
            mean_return=winner["mean_return"],
            std_return=winner["std_return"],
            n_verify_episodes=n_verify_episodes,
            eval_callback_peak_mean=callback_peak,
            eval_callback_best_timestep=callback_step,
            candidates=candidates,
        )

    raise ValueError(f"Unknown checkpoint policy: {policy}")


def resolve_model_path(
    model_dir: Path,
    seed: int,
    checkpoint: str = "best",
    **kwargs: Any,
) -> Path | None:
    """Backward-compatible path resolver (no verification rollouts)."""
    sel = select_checkpoint(
        model_dir,
        model_dir.parent / "logs",
        seed,
        policy="best" if checkpoint == "best" else ("final" if checkpoint == "final" else "auto"),
        env_kwargs=kwargs.get("env_kwargs"),
        n_verify_episodes=0,
    )
    return Path(sel.path) if sel else None


def save_checkpoint_report(selections: list[CheckpointSelection], report_path: Path) -> None:
    """Write JSON report of checkpoint decisions for reproducibility."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selections": [asdict(s) for s in selections],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
