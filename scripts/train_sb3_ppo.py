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
from custom_rl.eval.pipeline import plate_env_kwargs
from custom_rl.plants.plate import RPM_MAX, RPM_MIN


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

    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument(
        "--n-envs",
        type=int,
        default=2,
        help="Parallel training environments per seed",
    )

    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default="dummy",
        help="Vectorized env type: 'dummy' or 'subproc'",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="ODE integration step size [s]",
    )

    parser.add_argument(
        "--n-substeps",
        type=int,
        default=1,
        help="RK4 substeps per env step",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: auto from pass duration)",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Evaluation episodes per callback (random y0 explores pass lines)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20_000,
        help="Evaluate every N total env steps (higher = less eval overhead)",
    )

    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=0.0,
        help="Modal acceleration disturbance std [0=off, e.g. 0.01 for stochastic plant]",
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample milling start y0 each episode (uniform over y0 range)",
    )

    args = parser.parse_args()

    register_envs()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Spindle speed range: {RPM_MIN:.0f} - {RPM_MAX:.0f} rpm "
        f"(physics uses rad/s internally)"
    )
    print(
        f"Stochastic plant: uncertainty_std={args.dynamics_uncertainty_std}, "
        f"randomize_y0={args.randomize_y0}"
    )
    print(
        "Directories:\n"
        f"  logs     -> {Path(args.log_dir).resolve()}/seed_<N>/\n"
        f"  models   -> {Path(args.save_dir).resolve()}/best_<N>/best_model.zip\n"
        f"            {Path(args.save_dir).resolve()}/final_<N>.zip (after training)\n"
        f"  eval     -> python scripts/eval_policy.py --save-dir {args.save_dir}\n"
        f"  plots    -> python scripts/plot_results.py --log-dir {args.log_dir}"
    )

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv

    env_kwargs = plate_env_kwargs(
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=args.n_substeps,
        max_episode_steps=args.max_episode_steps,
        dynamics_uncertainty_std=args.dynamics_uncertainty_std,
        randomize_y0=args.randomize_y0,
    )

    # Show practical episode cap (pass completion usually ends sooner).
    _probe = gym.make(ENV_ID, **env_kwargs)
    print(f"Max episode steps (cap): {_probe.unwrapped.max_episode_steps}")
    print(
        "Note: first PPO rollout (~8192 env steps with defaults) can take several "
        "minutes on CPU before progress logs appear."
    )
    _probe.close()

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
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )

        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
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