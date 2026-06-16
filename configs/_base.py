"""Shared defaults for experiment configurations."""

from __future__ import annotations

from typing import Any


def make_config(
    *,
    name: str,
    description: str,
    total_timesteps: int,
    seeds: list[int],
    ppo: dict[str, Any],
    env: dict[str, Any],
    reward: dict[str, Any],
    train: dict[str, Any] | None = None,
    eval_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a full CONFIG dict with standard directory layout under runs/<name>/.

    Checkpoint policy:
        eval.checkpoint = "best"  → EvalCallback best_model.zip (preferred)
        eval.checkpoint = "final" → final_<seed>.zip
    """
    train = train or {}
    eval_cfg = eval_cfg or {}

    default_train = {
        "n_envs": 1,
        "vec_env": "dummy",
        "n_eval_episodes": 3,
        "eval_freq": 5000,
        "device": "cpu",
    }
    default_train.update(train)

    default_eval = {
        "checkpoint": "verified",
        "n_verify_episodes": 8,
        "seeds": list(seeds),
        "policy_seeds": list(seeds),
        "n_episodes": 10,
        "mc_rollouts": 40,
        "mc_disable_uncertainty": False,
        "deterministic_policy": True,
        "plot_smooth": 10,
        "run_mc": True,
        "run_trajectories": True,
        "run_plots": True,
    }
    default_eval.update(eval_cfg)

    return {
        "config_name": name,
        "description": description,
        "run_dir": f"runs/{name}",
        "seeds": list(seeds),
        "total_timesteps": int(total_timesteps),
        "ppo": dict(ppo),
        "env": dict(env),
        "reward": dict(reward),
        "train": default_train,
        "eval": default_eval,
    }
