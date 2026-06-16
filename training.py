#!/usr/bin/env python3
"""
Config-based PPO training for plate milling chatter control.

Usage:
    python training.py --config conf1
    python training.py --config conf2
    python training.py --config conf3

Each config writes to an isolated run directory:
    runs/<config>/logs/
    runs/<config>/models/       (best_<seed>/best_model.zip + final_<seed>.zip)
    runs/<config>/reports/      (created at eval time)
    runs/<config>/config_snapshot.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from configs import list_configs, load_config, save_config_snapshot
from custom_rl import register_envs

ENV_ID = "CustomODEPlate-v0"


def _build_env_kwargs(cfg: dict) -> dict:
    env_kwargs = dict(cfg["env"])
    env_kwargs.update(cfg["reward"])
    return env_kwargs


def train_from_config(cfg: dict) -> None:
    register_envs()

    log_dir = Path(cfg["log_dir"])
    model_dir = Path(cfg["model_dir"])
    run_dir = Path(cfg["run_dir"])
    report_dir = Path(cfg["report_dir"])

    for d in (log_dir, model_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(cfg, run_dir / "config_snapshot.json")

    vec_env_cls = SubprocVecEnv if cfg["train"]["vec_env"] == "subproc" else DummyVecEnv
    env_kwargs = _build_env_kwargs(cfg)
    ppo_kwargs = dict(cfg["ppo"])
    train_cfg = cfg["train"]

    print("=" * 60)
    print(f"Training  : {cfg['config_name']}")
    print(f"Description: {cfg['description']}")
    print(f"Run dir   : {run_dir}")
    print(f"Timesteps : {cfg['total_timesteps']}")
    print(f"Seeds     : {cfg['seeds']}")
    print("=" * 60)

    for seed in cfg["seeds"]:
        seed_dir = log_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        best_path = model_dir / f"best_{seed}"

        env = make_vec_env(
            env_id=ENV_ID,
            n_envs=train_cfg["n_envs"],
            seed=seed,
            vec_env_cls=vec_env_cls,
            monitor_dir=str(seed_dir),
            env_kwargs=env_kwargs,
        )

        eval_env = make_vec_env(
            env_id=ENV_ID,
            n_envs=1,
            seed=seed + 10000,
            vec_env_cls=DummyVecEnv,
            env_kwargs=env_kwargs,
        )

        eval_freq = max(int(train_cfg["eval_freq"]) // int(train_cfg["n_envs"]), 1)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_path),
            log_path=str(seed_dir),
            eval_freq=eval_freq,
            n_eval_episodes=int(train_cfg["n_eval_episodes"]),
            deterministic=True,
        )

        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            device=train_cfg.get("device", "cpu"),
            **ppo_kwargs,
        )

        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=eval_callback,
        )

        final_path = model_dir / f"final_{seed}"
        model.save(str(final_path))
        print(f"Seed {seed}: best → {best_path}/best_model.zip")
        print(f"Seed {seed}: final → {final_path}.zip")

        env.close()
        eval_env.close()

    print()
    print("Training complete.")
    print(f"  Evaluate with:  python result.py --config {cfg['config_name']}")
    print(f"  Logs   → {log_dir}")
    print(f"  Models → {model_dir}")


def main() -> None:
    available = ", ".join(list_configs()) or "conf1"
    parser = argparse.ArgumentParser(description="Train PPO from configs/confN.py")
    parser.add_argument(
        "--config",
        default="conf1",
        help=f"Config name ({available})",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
