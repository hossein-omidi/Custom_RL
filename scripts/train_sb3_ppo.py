"""Multi-seed PPO training with Stable-Baselines3 and parallel environments."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_MODEL_DIR, register_envs


ENV_ID = "CustomODEPlate-v0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CustomODEPlate")

    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    parser.add_argument(
        "--reward",
        default="productive",
        choices=["dense", "productive", "quadratic", "sparse"],
        help="Reward function for the plate environment",
    )

    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments per training run",
    )

    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default="dummy",
        help="Vectorized env type: 'dummy' or 'subproc'",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: auto from pass duration)",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes",
    )

    args = parser.parse_args()

    register_envs()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv

    env_kwargs = {
        "reward_id": args.reward,
    }
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    for seed in args.seeds:
        seed_dir = Path(args.log_dir) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        save_path = str(Path(args.save_dir) / f"best_{seed}")

        env = make_vec_env(
            env_id=ENV_ID,
            n_envs=args.n_envs,
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

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=str(seed_dir),
            eval_freq=max(5000 // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
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
            device="cpu",
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
        )

        model.save(str(Path(args.save_dir) / f"final_{seed}"))

        env.close()
        eval_env.close()

    print(f"Training done. Logs: {args.log_dir}, Models: {args.save_dir}")


if __name__ == "__main__":
    main()