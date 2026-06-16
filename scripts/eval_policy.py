"""Evaluate trained policy and save trajectories for plotting.

.. note::
   **Preferred workflow:** use ``python result.py --config conf1`` which loads
   checkpoints from ``runs/<config>/models/`` with verified selection.

   This script is a **legacy** convenience wrapper around ``DEFAULT_MODEL_DIR``
   (``models/ppo_plate``). It only works with models trained on the **current**
   6-dimensional sensor observation interface.

   Error ``observation shape (6,) ... please use (8,)`` means the checkpoint is
   from an **old modal-observation** policy. Retrain with ``python training.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import load_config
from custom_rl import DEFAULT_MODEL_DIR, DEFAULT_TRAJ_DIR, register_envs

ENV_ID = "CustomODEPlate-v0"


def _validate_model_env(model: PPO, env: gym.Env, model_path: Path) -> None:
    """Fail fast when checkpoint observation space does not match the env."""
    model_shape = tuple(model.observation_space.shape)
    env_shape = tuple(env.observation_space.shape)
    if model_shape == env_shape:
        return

    raise ValueError(
        f"\nObservation space mismatch for {model_path}:\n"
        f"  model expects {model_shape}\n"
        f"  env provides  {env_shape}\n\n"
        "Cause: checkpoints trained on the old 8-dim *modal* observation are "
        "incompatible with the current 6-dim *sensor* interface "
        "(4 sensor signals + 2 previous actions).\n\n"
        "Fix:\n"
        "  1. Retrain:  python training.py --config conf1\n"
        "  2. Evaluate: python result.py --config conf1\n"
        "  Or point --model-dir to runs/<config>/models after retraining.\n"
    )


def _build_env_kwargs_from_config(cfg: dict) -> dict:
    kw = dict(cfg["env"])
    kw.update(cfg["reward"])
    return kw


def _get_metadata(env: gym.Env, max_episode_steps: int, extra: dict | None = None) -> dict:
    base_env = env.unwrapped
    plant = base_env.plant
    metadata = {
        "env_id": ENV_ID,
        "dt": float(base_env.dt),
        "n_substeps": int(base_env.n_substeps),
        "step_dt": float(base_env._step_dt),
        "max_episode_steps": int(max_episode_steps),
        "observation_shape": list(env.observation_space.shape),
        "sensor_coords": np.asarray(plant.sensor_coords, dtype=np.float64).tolist(),
        "disp_norm_scale": float(plant.disp_norm_scale),
        "vel_norm_scale": float(plant.vel_norm_scale),
        "obs_clip": plant.obs_clip,
        "displacement_failure_limit": float(plant.displacement_failure_limit),
        "velocity_failure_limit": float(plant.velocity_failure_limit),
        "n_pass_lines": int(plant.n_pass_lines),
        "feed_speed": float(plant.feed_speed),
        "pass_duration": float(plant.pass_duration),
    }
    if hasattr(plant, "get_interface_metadata"):
        metadata["interface"] = plant.get_interface_metadata()
    if hasattr(plant, "u_phys_low"):
        metadata["physical_action_low"] = plant.u_phys_low.tolist()
        metadata["physical_action_high"] = plant.u_phys_high.tolist()
    if extra:
        metadata.update(extra)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy policy evaluation (prefer: python result.py --config confN)"
    )
    parser.add_argument("--config", default=None, help="Use runs/<config>/ models & traj dirs")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--reward",
        default=None,
        choices=["dense", "productive", "quadratic", "sparse"],
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
    args = parser.parse_args()

    register_envs()

    env_kwargs: dict = {}
    if args.config:
        cfg = load_config(args.config)
        env_kwargs = _build_env_kwargs_from_config(cfg)
        model_dir = Path(args.model_dir or cfg["model_dir"])
        out_dir = Path(args.out_dir or cfg["traj_dir"])
        if args.reward is None:
            args.reward = cfg["env"].get("reward_id", "productive")
    else:
        model_dir = Path(args.model_dir or DEFAULT_MODEL_DIR)
        out_dir = Path(args.out_dir or DEFAULT_TRAJ_DIR)
        if args.reward is None:
            args.reward = "productive"

    make_kwargs = dict(env_kwargs)
    make_kwargs.setdefault("reward_id", args.reward)
    if args.max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = args.max_episode_steps

    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for rel in (f"best_{seed}/best_model.zip", f"final_{seed}.zip"):
            model_path = model_dir / rel
            if model_path.exists():
                break
        else:
            print(f"Skip seed {seed}: no model in {model_dir}")
            continue

        model = PPO.load(str(model_path), device="cpu")
        env = gym.make(ENV_ID, **make_kwargs)
        _validate_model_env(model, env, model_path)

        max_steps = int(env.unwrapped.max_episode_steps)
        metadata = _get_metadata(
            env,
            max_steps,
            extra={"model_path": str(model_path), "config": args.config},
        )
        trajectories = []

        for ep in range(args.n_episodes):
            obs, reset_info = env.reset(seed=seed + 1000 + ep)
            episode_context = dict(reset_info)

            states, actions, physical_actions = [], [], []
            sensor_w, sensor_w_dot, reward_components, rewards, times = [], [], [], [], []
            current_time = 0.0
            termination_reason = "unknown"

            while True:
                action, _ = model.predict(obs, deterministic=True)
                plant = env.unwrapped.plant

                states.append(obs.tolist())
                actions.append(np.asarray(action, dtype=np.float64).tolist())
                physical_actions.append(plant.normalized_to_physical_action(action).tolist())
                times.append(float(current_time))

                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                if "sensor_w" in info:
                    sensor_w.append(np.asarray(info["sensor_w"]).tolist())
                if "sensor_w_dot" in info:
                    sensor_w_dot.append(np.asarray(info["sensor_w_dot"]).tolist())
                if "reward_components" in info:
                    reward_components.append(info["reward_components"])
                current_time = float(info.get("t", current_time + metadata["step_dt"]))

                if terminated or truncated:
                    termination_reason = str(info.get("termination_reason", "unknown"))
                    break

            trajectories.append(
                {
                    "metadata": metadata,
                    "episode_context": episode_context,
                    "termination_reason": termination_reason,
                    "times": times,
                    "states": states,
                    "actions": actions,
                    "physical_actions": physical_actions,
                    "sensor_w": sensor_w,
                    "sensor_w_dot": sensor_w_dot,
                    "reward_components": reward_components,
                    "rewards": rewards,
                    "return": float(sum(rewards)),
                    "length": len(rewards),
                }
            )

        out_path = out_dir / f"trajectories_seed{seed}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2)
        print(f"Saved {args.n_episodes} episodes → {out_path}")
        env.close()


if __name__ == "__main__":
    main()
