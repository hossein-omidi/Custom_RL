"""Train the second-mode (finishing) PPO model over a FEASIBLE fixed-depth range.

The registered ``CustomODEPlateFinish-v0`` samples the fixed depth ap uniformly
over [0, 18] mm, but only ap <~ 1 mm keeps the plate under the 1 mm displacement
limit, so most episodes there are doomed regardless of control.  This trainer
builds the finishing env with a small ``--ap-max`` so the per-episode fixed depth
is drawn from a feasible finishing range and the policy actually learns to
modulate spindle speed.  The action space (omega only) and observation are
identical to the registered env, so the trained model evaluates directly on
``CustomODEPlateFinish-v0`` with a pinned ``--ap``.

Training hyper-parameters mirror ``scripts/train_sb3_ppo.py`` exactly: obs+return
VecNormalize (fixes the dead critic / policy collapse), a linear learning-rate
decay for KL-spike stability, and the saved ``vecnormalize_<seed>.pkl`` stats so
``eval_policy.py`` reproduces the exact normalization the policy trained with.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs

# PPO defaults identical to scripts/train_sb3_ppo.py
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.999
PPO_KW = dict(
    n_steps=2048, batch_size=256, n_epochs=10, gamma=PPO_GAMMA,
    gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
    max_grad_norm=0.5, target_kl=0.03,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
)

# VecNormalize clipping, identical to scripts/train_sb3_ppo.py.
VECNORM_CLIP_OBS = 10.0
VECNORM_CLIP_REWARD = 10.0


def _linear_schedule(initial_value: float):
    """Linear decay from initial_value (progress_remaining=1) to 0 (=0).

    Identical to scripts/train_sb3_ppo.py: late-training LR decay is the standard
    cure for residual KL spikes, giving a smooth, stable final policy.
    """
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * float(initial_value)

    return schedule


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
    p.add_argument("--n-substeps", type=int, default=10,
                   help="RK4 substeps per control step. Matches train_sb3_ppo.py "
                        "and eval_policy.py so control_dt=dt*n_substeps is consistent.")
    p.add_argument("--max-episode-steps", type=int, default=None,
               help="Default (None) auto-computes from the plant's own "
                    "L1/feed-rate/omega_min so a full pass always fits.")
    p.add_argument("--n-eval-episodes", type=int, default=5)
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
        # Normalize obs + returns for training (fixes the dead critic).
        env = VecNormalize(
            env, norm_obs=True, norm_reward=True,
            clip_obs=VECNORM_CLIP_OBS, clip_reward=VECNORM_CLIP_REWARD, gamma=PPO_GAMMA,
        )

        eval_env = make_vec_env(factory, n_envs=1, seed=seed + 10000, vec_env_cls=DummyVecEnv)
        # Eval env: normalize obs identically, but do NOT normalize reward and do
        # NOT update stats (training=False) so eval returns stay in real units.
        # EvalCallback auto-syncs the obs running-mean from the training env.
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, training=False,
            clip_obs=VECNORM_CLIP_OBS, gamma=PPO_GAMMA,
        )

        cb = EvalCallback(eval_env, best_model_save_path=str(save_dir / f"best_{seed}"),
                          log_path=str(log_dir / f"seed_{seed}"),
                          eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
                          deterministic=True)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                    learning_rate=_linear_schedule(PPO_LEARNING_RATE), **PPO_KW)
        model.learn(total_timesteps=args.total_timesteps, callback=cb)
        model.save(str(save_dir / f"final_{seed}"))
        # Save VecNormalize stats so eval_policy.py reproduces the exact obs
        # normalization the policy trained with.
        env.save(str(save_dir / f"vecnormalize_{seed}.pkl"))
        env.close(); eval_env.close()
        print(f"  seed {seed}: done -> {save_dir}/final_{seed}.zip")

    print("FINISH_TRAIN_DONE")


if __name__ == "__main__":
    main()
