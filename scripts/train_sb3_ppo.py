"""Multi-seed PPO training with Stable-Baselines3 and parallel environments."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_MODEL_DIR, register_envs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CustomODECartPole")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--reward", default="dense", choices=["dense", "sparse"])
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments per training run (vectorized)",
    )
    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default="dummy",
        help="Vectorized env type: 'dummy' (sequential) or 'subproc' (multiprocess)",
    )
    args = parser.parse_args()

    register_envs()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv

    for seed in args.seeds:
        seed_dir = Path(args.log_dir) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(Path(args.save_dir) / f"best_{seed}")

        # Official SB3 make_vec_env: creates n_envs parallel envs with proper seeding
        # Each env gets seed, seed+1, seed+2, ... automatically
        # monitor_dir enables Monitor wrapper with per-env log files
        env = make_vec_env(
            env_id="CustomODECartPole-v0",
            n_envs=args.n_envs,
            seed=seed,
            vec_env_cls=vec_env_cls,
            monitor_dir=str(seed_dir),
            env_kwargs={"reward_id": args.reward},
        )

        # Eval env: also vectorized for consistent interface
        eval_env = make_vec_env(
            env_id="CustomODECartPole-v0",
            n_envs=1,
            seed=seed + 10000,
            vec_env_cls=DummyVecEnv,
            env_kwargs={"reward_id": args.reward},
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=str(seed_dir),
            eval_freq=max(5000 // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        )

        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
        )
        model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

        model.save(str(Path(args.save_dir) / f"final_{seed}"))
        env.close()
        eval_env.close()

    print(f"Training done. Logs: {args.log_dir}, Models: {args.save_dir}")


if __name__ == "__main__":
    main()
