"""Train the second-mode (finishing) PPO model over a FEASIBLE fixed-depth range.

The registered ``CustomODEPlateFinish-v0`` samples the fixed depth ap uniformly
over [0, 18] mm, but only ap <~ 1 mm keeps the plate under the 1 mm displacement
limit, so most episodes there are doomed regardless of control.  This trainer
builds the finishing env with a small ``--ap-max`` so the per-episode fixed depth
is drawn from a feasible finishing range and the policy actually learns to
modulate spindle speed.  The action space (omega only) and observation are
identical to the registered env, so the trained model evaluates directly on
``CustomODEPlateFinish-v0`` with a pinned ``--ap``.

Same PPO hyper-parameters as ``train_sb3_ppo.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs

# PPO defaults identical to scripts/train_sb3_ppo.py
PPO_KW = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=0.999,
    gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
    max_grad_norm=0.5, target_kl=0.03,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
)


def make_finish_env_factory(ap_max, dt, n_substeps, max_episode_steps,
                            dynamics_uncertainty_std, randomize_y0):
    def _make():
        register_envs()
        kw = plate_env_kwargs(
            reward_id="dense", dt=dt, n_substeps=n_substeps,
            max_episode_steps=max_episode_steps,
            dynamics_uncertainty_std=dynamics_uncertainty_std,
            randomize_y0=randomize_y0,
        )
        kw["ap_max"] = float(ap_max)  # feasible finishing-depth range
        return gym.make("CustomODEPlateFinish-v0", **kw)
    return _make


def main() -> None:
    p = argparse.ArgumentParser(description="Train second-mode (finishing) PPO over a feasible depth range.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--total-timesteps", type=int, default=300000)
    p.add_argument("--ap-max", type=float, default=1.0,
                   help="Feasible finishing-depth upper bound [mm] for training randomization.")
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--n-substeps", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--n-eval-episodes", type=int, default=1)
    p.add_argument("--eval-freq", type=int, default=15000)
    p.add_argument("--dynamics-uncertainty-std", type=float, default=0.001)
    p.add_argument("--randomize-y0", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-dir", default="models/ppo_plate_finish")
    p.add_argument("--log-dir", default="logs/ppo_plate_finish")
    args = p.parse_args()

    register_envs()
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Second-mode (finishing) training: ap_max={args.ap_max} mm (feasible range), "
          f"uncertainty_std={args.dynamics_uncertainty_std}, randomize_y0={args.randomize_y0}")
    print(f"  models -> {save_dir}/best_<seed>/best_model.zip , {save_dir}/final_<seed>.zip")

    factory = make_finish_env_factory(
        args.ap_max, args.dt, args.n_substeps, args.max_episode_steps,
        args.dynamics_uncertainty_std, args.randomize_y0,
    )

    for seed in args.seeds:
        env = make_vec_env(factory, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv,
                           monitor_dir=str(log_dir / f"seed_{seed}"))
        eval_env = make_vec_env(factory, n_envs=1, seed=seed + 10000, vec_env_cls=DummyVecEnv)
        cb = EvalCallback(eval_env, best_model_save_path=str(save_dir / f"best_{seed}"),
                          log_path=str(log_dir / f"seed_{seed}"),
                          eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
                          deterministic=True)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **PPO_KW)
        model.learn(total_timesteps=args.total_timesteps, callback=cb)
        model.save(str(save_dir / f"final_{seed}"))
        env.close(); eval_env.close()
        print(f"  seed {seed}: done -> {save_dir}/final_{seed}.zip")

    print("FINISH_TRAIN_DONE")


if __name__ == "__main__":
    main()
