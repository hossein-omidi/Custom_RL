"""Train a second-mode (finishing) PPO model over a FEASIBLE fixed-depth range.

The registered CustomODEPlateFinish-v0 randomizes ap over [0, 18] mm, but only
ap <~ 1 mm keeps the plate under the 1 mm displacement limit, so most episodes
would be doomed regardless of control.  Here we build the finishing env with
ap_max=1.0 so the per-episode fixed depth is sampled from a feasible range and
the policy actually learns to modulate spindle speed.  The action space (omega
only) and observation are identical to the registered env, so the trained model
evaluates directly on CustomODEPlateFinish-v0 with a pinned --ap.
"""
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs

SAVE = Path("models/ppo_plate_finish")
SAVE.mkdir(parents=True, exist_ok=True)
LOG = Path("logs/ppo_plate_finish"); LOG.mkdir(parents=True, exist_ok=True)


def make_finish_env():
    register_envs()
    kw = plate_env_kwargs(reward_id="dense", dt=1e-4, n_substeps=10,
                          max_episode_steps=2000, randomize_y0=True)
    kw["ap_max"] = 1.0  # feasible finishing-depth range for training
    return gym.make("CustomODEPlateFinish-v0", **kw)


def main():
    register_envs()
    env = make_vec_env(make_finish_env, n_envs=1, seed=0, vec_env_cls=DummyVecEnv,
                       monitor_dir=str(LOG / "seed_0"))
    eval_env = make_vec_env(make_finish_env, n_envs=1, seed=10000, vec_env_cls=DummyVecEnv)
    cb = EvalCallback(eval_env, best_model_save_path=str(SAVE / "best_0"),
                      log_path=str(LOG / "seed_0"), eval_freq=15000,
                      n_eval_episodes=1, deterministic=True)
    model = PPO("MlpPolicy", env, seed=0, learning_rate=3e-4, n_steps=2048,
                batch_size=256, n_epochs=10, gamma=0.999, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
                target_kl=0.03,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                   log_std_init=-1.0),
                verbose=0, device="cpu")
    model.learn(total_timesteps=130000, callback=cb)
    model.save(str(SAVE / "final_0"))
    env.close(); eval_env.close()
    print("FINISH_TRAIN_DONE")


if __name__ == "__main__":
    main()
