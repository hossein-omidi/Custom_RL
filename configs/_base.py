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
    stability_lobe: dict[str, Any] | None = None,
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

    default_stability = {
        "omega_grid": [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0],
        "ac_grid": [0.5, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0],
        "n_rollouts": 5,
        "seeds": list(seeds),
        "horizon_steps": 120,
        "transient_fraction": 0.25,
        "unstable_threshold": 0.5,
        "rms_unstable_factor": 3.0,
        "controller_modes": ["uncontrolled"],
    }
    if stability_lobe:
        default_stability.update(stability_lobe)

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
        "stability_lobe": default_stability,
    }
