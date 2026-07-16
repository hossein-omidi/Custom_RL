"""Multi-seed PPO training with Stable-Baselines3 and parallel environments."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_MODEL_DIR, register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from custom_rl.plants.plate import omega_to_rpm


ENV_ID = "CustomODEPlate-v0"

# PPO defaults selected for the current face-milling plate environment.
# control_dt = dt * n_substeps = 0.001 s; gamma=0.999 gives ~1s effective
# discount horizon. ent_coef>0 encourages exploration of the narrow stable band.
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.999
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.005
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_TARGET_KL = 0.03
PPO_NET_ARCH = [256, 256]
PPO_LOG_STD_INIT = -1.0

# Fraction of the physical ap range used as the safe-start target.
# 0.05 means 5% of (ap_max - ap_min) above ap_min.
SAFE_AP_FRACTION = 0.05


def _compute_safe_action_bias(plant) -> list[float]:
    """Derive a normalized-action bias from the plant's actual bounds so the
    untrained policy defaults to a safe operating point (low ap, mid omega)."""
    ap_range = float(plant.ap_max - plant.ap_min)
    if ap_range > 0:
        safe_ap = float(plant.ap_min) + SAFE_AP_FRACTION * ap_range
        u_ap = 2.0 * (safe_ap - float(plant.ap_min)) / ap_range - 1.0
    else:
        u_ap = 0.0

    bias = [0.0, float(np.clip(u_ap, -1.0, 1.0))]

    if int(getattr(plant, "action_dim", 2)) > 2:
        bias.append(0.0)

    return bias


def _init_action_bias(model: PPO, bias: list[float]) -> None:
    """Set the action-net output bias so the untrained policy defaults to safe ap."""
    import torch

    action_net = model.policy.action_net
    if hasattr(action_net, "bias") and action_net.bias is not None:
        n = min(len(bias), action_net.bias.numel())
        with torch.no_grad():
            action_net.bias[:n] = torch.tensor(bias[:n], dtype=torch.float32)


def make_registered_plate_env(**env_kwargs):
    """Create the registered plate environment inside the active process.

    This makes SubprocVecEnv robust on spawn-based systems such as Windows:
    each worker process registers the custom env before calling gym.make().
    """
    register_envs()
    return gym.make(ENV_ID, **env_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CustomODEPlate")

    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument(
        "--reward",
        default="productive",
        choices=["dense", "productive", "quadratic", "sparse"],
        help="Reward function for the plate environment",
    )

    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument(
        "--n-envs",
        type=int,
        default=2,
        help=(
            "Parallel training environments per seed. Use DummyVecEnv only with n_envs=1; "
            "for n_envs>1 use SubprocVecEnv because the face-milling force "
            "module stores regenerative history in module-level globals."
        ),
    )

    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default="subproc",
        help="Vectorized env type. DummyVecEnv is safe only with n_envs=1 for this plant.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0e-4,
        help="RK4 integration substep [s]",
    )

    parser.add_argument(
        "--n-substeps",
        type=int,
        default=5,
        help="RK4 substeps per environment/control step",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=25000,
        help="Max environment/control steps per episode",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=2,
        help="Evaluation episodes per callback. Keep small because one stable pass is long.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluate every N total env steps (higher = less eval overhead)",
    )

    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=0.001,
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

    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.vec_env == "dummy" and args.n_envs != 1:
        raise ValueError(
            "DummyVecEnv with n_envs > 1 is unsafe for this face-milling plant because "
            "the regenerative force module uses module-level history/global parameters. "
            "Use --vec-env subproc for parallel training, or keep --n-envs 1."
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
    probe_plant = _probe.unwrapped.plant
    control_dt = float(args.dt) * int(args.n_substeps)
    print(
        f"Spindle speed range: {omega_to_rpm(probe_plant.omega_min):.0f} - "
        f"{omega_to_rpm(probe_plant.omega_max):.0f} rpm "
        f"({probe_plant.omega_min:.2f} - {probe_plant.omega_max:.2f} rad/s)"
    )
    print(
        f"Integrator: RK4 dt={args.dt:g} s, n_substeps={args.n_substeps}, "
        f"control step={control_dt:g} s"
    )
    print(
        f"Displacement limit/scale: w_limit={probe_plant.w_limit:g} m, "
        f"w_obs_scale={probe_plant.w_obs_scale:g} m"
    )
    print(
        f"Action bounds: ap={probe_plant.ap_min:g}-{probe_plant.ap_max:g} mm, "
        f"ae_default={probe_plant.ae_default:g} mm"
    )
    tau_min = 2.0 * 3.141592653589793 / (max(int(probe_plant.N), 1) * max(float(probe_plant.omega_max), 1e-12))
    print(f"Minimum one-tooth regenerative delay at omega_max: {tau_min:.6g} s")
    if args.dt >= tau_min:
        raise ValueError(
            f"RK4 dt={args.dt:g} s is not smaller than the minimum regenerative "
            f"tooth delay {tau_min:.6g} s. Reduce --dt or lower omega_max."
        )

    safe_bias = _compute_safe_action_bias(probe_plant)
    print(f"Safe-start action bias (normalized): {safe_bias}")

    print(f"Max episode steps (cap): {_probe.unwrapped.max_episode_steps}")
    print(
        f"Note: first PPO rollout collects {PPO_N_STEPS * args.n_envs} env steps "
        "before the first PPO update/progress log; this can take several minutes on CPU."
    )
    print(
        "PPO defaults: "
        f"lr={PPO_LEARNING_RATE}, n_steps={PPO_N_STEPS}, batch={PPO_BATCH_SIZE}, "
        f"epochs={PPO_N_EPOCHS}, gamma={PPO_GAMMA}, gae_lambda={PPO_GAE_LAMBDA}, "
        f"target_kl={PPO_TARGET_KL}, net={PPO_NET_ARCH}, "
        f"log_std_init={PPO_LOG_STD_INIT}"
    )
    _probe.close()

    for seed in args.seeds:
        seed_dir = Path(args.log_dir) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        save_path = str(Path(args.save_dir) / f"best_{seed}")

        env = make_vec_env(
            env_id=make_registered_plate_env,
            n_envs=args.n_envs,
            seed=seed,
            vec_env_cls=vec_env_cls,
            monitor_dir=str(seed_dir),
            env_kwargs=env_kwargs,
        )

        eval_env = make_vec_env(
            env_id=make_registered_plate_env,
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
            learning_rate=PPO_LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            vf_coef=PPO_VF_COEF,
            max_grad_norm=PPO_MAX_GRAD_NORM,
            target_kl=PPO_TARGET_KL,
            policy_kwargs=dict(
                net_arch=dict(pi=PPO_NET_ARCH, vf=PPO_NET_ARCH),
                log_std_init=PPO_LOG_STD_INIT,
            ),
            verbose=1,
            device="cpu",
        )

        _init_action_bias(model, safe_bias)

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
