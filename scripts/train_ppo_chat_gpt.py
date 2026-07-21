"""Stronger, efficiency-aware multi-seed PPO training for the milling plate.

The defaults are intentionally stronger than the original configuration while
avoiding a proportional increase in runtime:

- more parallel environments and shorter per-env rollouts;
- the same practical rollout batch size for better throughput;
- larger minibatches and fewer PPO epochs to reduce optimizer overhead;
- a larger policy/value network;
- long-horizon gamma/GAE settings for full-pass decisions;
- non-zero final learning rate and decaying policy clip range;
- tighter KL protection and value-function clipping;
- periodic model + VecNormalize checkpoints;
- VecNormalize statistics saved together with every best model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from custom_rl import DEFAULT_LOG_DIR, DEFAULT_MODEL_DIR, register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from custom_rl.plants.plate import omega_to_rpm


ENV_ID = "CustomODEPlate-v0"


def _default_n_envs() -> int:
    """Choose a conservative CPU-parallel default without over-spawning."""
    cpu_count = os.cpu_count() or 2
    return min(4, max(2, cpu_count // 2))


# ---------------------------------------------------------------------------
# Stronger but efficiency-aware PPO defaults
# ---------------------------------------------------------------------------
# Old effective rollout: 2 envs * 2048 steps = 4096 transitions/update.
# New typical rollout: 4 envs * 1024 steps = 4096 transitions/update.
# Therefore the policy still sees a well-sized rollout, but simulation can be
# collected faster. Batch 512 and 8 epochs produce 64 optimizer minibatches per
# update instead of 160 in the old 256-batch/10-epoch setup.
PPO_LEARNING_RATE = 2.5e-4
PPO_FINAL_LEARNING_RATE = 2.5e-5
PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 512
PPO_N_EPOCHS = 8

# A ~1000-control-step pass retains exp(-0.5) ~= 0.61 terminal influence with
# gamma=0.9995, compared with ~0.37 for gamma=0.999.
PPO_GAMMA = 0.9995
PPO_GAE_LAMBDA = 0.97

PPO_CLIP_RANGE = 0.20
PPO_FINAL_CLIP_RANGE = 0.08
PPO_CLIP_RANGE_VF = 0.25
PPO_ENT_COEF = 0.002
PPO_VF_COEF = 0.75
PPO_MAX_GRAD_NORM = 0.5
PPO_TARGET_KL = 0.015

# Larger than [256, 256], but the reduction in optimizer minibatches keeps the
# total learning-side cost controlled. Actor and critic remain separate.
PPO_NET_ARCH = [384, 384, 256]
PPO_LOG_STD_INIT = -1.0
PPO_ADAM_EPS = 1e-5

# gSDE can be useful for continuous actions, but is optional because this plant
# has a narrow safe operating band. Enable it only after validating the standard
# Gaussian policy baseline.
PPO_USE_SDE = False
PPO_SDE_SAMPLE_FREQ = 16

SAFE_AP_FRACTION = 0.05

VECNORM_CLIP_OBS = 10.0
VECNORM_CLIP_REWARD = 10.0
VECNORM_EPSILON = 1e-8

DEFAULT_TOTAL_TIMESTEPS = 2_400_000
DEFAULT_EVAL_FREQ = 50_000
DEFAULT_N_EVAL_EPISODES = 3
DEFAULT_CHECKPOINT_FREQ = 200_000


Schedule = Callable[[float], float]


def _linear_schedule(initial_value: float, final_value: float = 0.0) -> Schedule:
    """Linearly interpolate from initial to final as training progresses.

    Stable-Baselines3 passes ``progress_remaining`` from 1.0 at the beginning
    to 0.0 at the end. A non-zero final learning rate avoids completely freezing
    the policy during the final part of a long run.
    """
    if initial_value < 0.0 or final_value < 0.0:
        raise ValueError("Schedule values must be non-negative.")

    def schedule(progress_remaining: float) -> float:
        progress = float(np.clip(progress_remaining, 0.0, 1.0))
        return float(final_value + progress * (initial_value - final_value))

    return schedule


class SaveVecNormalizeOnBestCallback(BaseCallback):
    """Save normalization statistics whenever EvalCallback finds a new best.

    EvalCallback saves ``best_model.zip`` but does not automatically save the
    exact training VecNormalize state corresponding to that model. Keeping the
    pair together prevents evaluating the best policy with incompatible final
    normalization statistics.
    """

    def __init__(self, save_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.save_path = Path(save_path)

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            raise RuntimeError(
                "Best-model VecNormalize save requested, but the training "
                "environment is not wrapped with VecNormalize."
            )

        output_path = self.save_path / "vecnormalize.pkl"
        vec_normalize.save(str(output_path))
        if self.verbose:
            print(f"Saved best-model VecNormalize statistics to {output_path}")
        return True


def _compute_safe_action_bias(plant) -> list[float]:
    """Compute a normalized initial action bias from the physical plant bounds."""
    action_dim = int(getattr(plant, "action_dim", 2))

    # Finishing mode controls omega only, so neutral normalized omega is enough.
    if action_dim <= 1:
        return [0.0]

    ap_range = float(plant.ap_max - plant.ap_min)
    if ap_range > 0.0:
        safe_ap = float(plant.ap_min) + SAFE_AP_FRACTION * ap_range
        normalized_ap = 2.0 * (safe_ap - float(plant.ap_min)) / ap_range - 1.0
    else:
        normalized_ap = 0.0

    bias = [0.0, float(np.clip(normalized_ap, -1.0, 1.0))]
    if action_dim > 2:
        bias.extend([0.0] * (action_dim - 2))
    return bias


def _init_action_bias(model: PPO, bias: list[float]) -> None:
    """Initialize the policy mean so the untrained agent starts safely."""
    import torch

    action_net = model.policy.action_net
    if hasattr(action_net, "bias") and action_net.bias is not None:
        n_values = min(len(bias), action_net.bias.numel())
        with torch.no_grad():
            bias_tensor = torch.as_tensor(
                bias[:n_values],
                dtype=action_net.bias.dtype,
                device=action_net.bias.device,
            )
            action_net.bias[:n_values].copy_(bias_tensor)


def make_registered_plate_env(_env_id: str = ENV_ID, **env_kwargs):
    """Register and create the custom environment inside each worker process."""
    register_envs()
    return gym.make(_env_id, **env_kwargs)


def _write_training_config(
    output_path: Path,
    args: argparse.Namespace,
    rollout_size: int,
    safe_bias: list[float],
) -> None:
    """Save the resolved run configuration for reproducibility."""
    config = {
        "arguments": vars(args),
        "resolved": {
            "rollout_size": rollout_size,
            "learning_rate_initial": args.learning_rate,
            "learning_rate_final": args.final_learning_rate,
            "clip_range_initial": args.clip_range,
            "clip_range_final": args.final_clip_range,
            "gamma": PPO_GAMMA,
            "gae_lambda": PPO_GAE_LAMBDA,
            "clip_range_vf": PPO_CLIP_RANGE_VF,
            "ent_coef": PPO_ENT_COEF,
            "vf_coef": PPO_VF_COEF,
            "max_grad_norm": PPO_MAX_GRAD_NORM,
            "target_kl": PPO_TARGET_KL,
            "net_arch": PPO_NET_ARCH,
            "log_std_init": PPO_LOG_STD_INIT,
            "use_sde": args.use_sde,
            "sde_sample_freq": PPO_SDE_SAMPLE_FREQ,
            "safe_action_bias": safe_bias,
        },
    }
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stronger efficiency-aware PPO agent on CustomODEPlate"
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--env-id",
        default=ENV_ID,
        choices=["CustomODEPlate-v0", "CustomODEPlateFinish-v0"],
        help=(
            "CustomODEPlate-v0 controls [omega, ap]; "
            "CustomODEPlateFinish-v0 controls [omega]."
        ),
    )
    parser.add_argument(
        "--reward",
        default="productive",
        choices=["dense", "productive", "quadratic", "sparse"],
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument(
        "--n-envs",
        type=int,
        default=_default_n_envs(),
        help="Parallel environments. Default is a conservative CPU-based value capped at 4.",
    )
    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default="subproc",
    )
    parser.add_argument(
        "--start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
        help=(
            "Subprocess start method. 'spawn' is the safest cross-platform option "
            "for module-level regenerative-force state."
        ),
    )

    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--n-substeps", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=None)

    parser.add_argument("--learning-rate", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=PPO_FINAL_LEARNING_RATE,
    )
    parser.add_argument("--n-steps", type=int, default=PPO_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=PPO_N_EPOCHS)
    parser.add_argument("--clip-range", type=float, default=PPO_CLIP_RANGE)
    parser.add_argument(
        "--final-clip-range",
        type=float,
        default=PPO_FINAL_CLIP_RANGE,
    )
    parser.add_argument(
        "--use-sde",
        action=argparse.BooleanOptionalAction,
        default=PPO_USE_SDE,
        help="Use generalized state-dependent exploration for continuous actions.",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=DEFAULT_N_EVAL_EPISODES,
    )
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQ,
        help="Save model and VecNormalize state every N total environment steps.",
    )

    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="CPU is usually fastest for this MLP PPO because simulation is CPU-bound.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    register_envs()

    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.n_steps <= 1:
        raise ValueError("--n-steps must be greater than 1.")
    if args.batch_size <= 1:
        raise ValueError("--batch-size must be greater than 1.")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive.")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.eval_freq <= 0 or args.checkpoint_freq <= 0:
        raise ValueError("Evaluation and checkpoint frequencies must be positive.")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("--final-learning-rate cannot exceed --learning-rate.")
    if args.final_clip_range > args.clip_range:
        raise ValueError("--final-clip-range cannot exceed --clip-range.")
    if args.vec_env == "dummy" and args.n_envs != 1:
        raise ValueError(
            "DummyVecEnv with n_envs > 1 is unsafe because the regenerative "
            "force implementation uses module-level state."
        )

    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError(
            f"batch_size={args.batch_size} exceeds rollout_size={rollout_size}."
        )
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            f"rollout_size={rollout_size} must be divisible by "
            f"batch_size={args.batch_size} to avoid truncated minibatches."
        )

    log_root = Path(args.log_dir)
    save_root = Path(args.save_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    save_root.mkdir(parents=True, exist_ok=True)

    env_kwargs = plate_env_kwargs(
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=args.n_substeps,
        max_episode_steps=args.max_episode_steps,
        dynamics_uncertainty_std=args.dynamics_uncertainty_std,
        randomize_y0=args.randomize_y0,
    )
    worker_env_kwargs = {**env_kwargs, "_env_id": args.env_id}

    probe = gym.make(args.env_id, **env_kwargs)
    plant = probe.unwrapped.plant
    control_dt = float(args.dt) * int(args.n_substeps)
    tau_min = 2.0 * np.pi / (
        max(int(plant.N), 1) * max(float(plant.omega_max), 1e-12)
    )

    if args.dt >= tau_min:
        probe.close()
        raise ValueError(
            f"RK4 dt={args.dt:g} s must be smaller than the minimum regenerative "
            f"tooth delay {tau_min:.6g} s."
        )

    if args.use_sde and not isinstance(probe.action_space, gym.spaces.Box):
        probe.close()
        raise ValueError("gSDE requires a continuous Box action space.")

    safe_bias = _compute_safe_action_bias(plant)

    print(
        f"Spindle range: {omega_to_rpm(plant.omega_min):.0f}-"
        f"{omega_to_rpm(plant.omega_max):.0f} rpm"
    )
    print(
        f"Integrator: RK4 dt={args.dt:g}, substeps={args.n_substeps}, "
        f"control_dt={control_dt:g} s, tooth_delay_min={tau_min:.6g} s"
    )
    print(
        f"Training: envs={args.n_envs}, n_steps={args.n_steps}, "
        f"rollout={rollout_size}, batch={args.batch_size}, epochs={args.n_epochs}"
    )
    print(
        f"Optimizer minibatches/update: "
        f"{(rollout_size // args.batch_size) * args.n_epochs}"
    )
    print(
        f"PPO: lr={args.learning_rate:g}->{args.final_learning_rate:g}, "
        f"clip={args.clip_range:g}->{args.final_clip_range:g}, "
        f"gamma={PPO_GAMMA}, GAE={PPO_GAE_LAMBDA}, "
        f"target_kl={PPO_TARGET_KL}, network={PPO_NET_ARCH}"
    )
    print(
        f"Plant randomization: uncertainty_std={args.dynamics_uncertainty_std}, "
        f"randomize_y0={args.randomize_y0}; safe_bias={safe_bias}"
    )
    print(f"Episode cap: {probe.unwrapped.max_episode_steps}")
    probe.close()

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    subproc_kwargs = (
        {"start_method": args.start_method}
        if vec_env_cls is SubprocVecEnv
        else None
    )

    for seed in args.seeds:
        seed_log_dir = log_root / f"seed_{seed}"
        seed_log_dir.mkdir(parents=True, exist_ok=True)

        best_dir = save_root / f"best_{seed}"
        checkpoint_dir = save_root / f"checkpoints_{seed}"
        best_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        _write_training_config(
            seed_log_dir / "training_config.json",
            args,
            rollout_size,
            safe_bias,
        )

        env = make_vec_env(
            env_id=make_registered_plate_env,
            n_envs=args.n_envs,
            seed=seed,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=subproc_kwargs,
            monitor_dir=str(seed_log_dir),
            env_kwargs=worker_env_kwargs,
        )
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=VECNORM_CLIP_OBS,
            clip_reward=VECNORM_CLIP_REWARD,
            gamma=PPO_GAMMA,
            epsilon=VECNORM_EPSILON,
        )

        eval_env = make_vec_env(
            env_id=make_registered_plate_env,
            n_envs=1,
            seed=seed + 10_000,
            vec_env_cls=DummyVecEnv,
            env_kwargs=worker_env_kwargs,
        )
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=VECNORM_CLIP_OBS,
            gamma=PPO_GAMMA,
            epsilon=VECNORM_EPSILON,
        )

        save_best_norm = SaveVecNormalizeOnBestCallback(
            save_path=best_dir,
            verbose=1,
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=save_best_norm,
            best_model_save_path=str(best_dir),
            log_path=str(seed_log_dir),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            verbose=1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix=f"ppo_seed_{seed}",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=2,
        )

        policy_kwargs = {
            "net_arch": {"pi": PPO_NET_ARCH, "vf": PPO_NET_ARCH},
            "log_std_init": PPO_LOG_STD_INIT,
            "ortho_init": True,
            "optimizer_kwargs": {"eps": PPO_ADAM_EPS},
        }
        if args.use_sde:
            policy_kwargs.update(
                {
                    "full_std": True,
                    "use_expln": True,
                }
            )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            learning_rate=_linear_schedule(
                args.learning_rate,
                args.final_learning_rate,
            ),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=_linear_schedule(
                args.clip_range,
                args.final_clip_range,
            ),
            clip_range_vf=PPO_CLIP_RANGE_VF,
            normalize_advantage=True,
            ent_coef=PPO_ENT_COEF,
            vf_coef=PPO_VF_COEF,
            max_grad_norm=PPO_MAX_GRAD_NORM,
            target_kl=PPO_TARGET_KL,
            use_sde=args.use_sde,
            sde_sample_freq=PPO_SDE_SAMPLE_FREQ if args.use_sde else -1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(seed_log_dir / "tensorboard"),
            verbose=1,
            device=args.device,
        )

        _init_action_bias(model, safe_bias)

        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=[eval_callback, checkpoint_callback],
                progress_bar=args.progress_bar,
            )

            final_model_path = save_root / f"final_{seed}"
            final_norm_path = save_root / f"vecnormalize_{seed}.pkl"
            model.save(str(final_model_path))
            env.save(str(final_norm_path))

            print(
                f"Seed {seed} complete:\n"
                f"  best model/norm -> {best_dir}\n"
                f"  final model     -> {final_model_path}.zip\n"
                f"  final norm      -> {final_norm_path}\n"
                f"  checkpoints     -> {checkpoint_dir}"
            )
        finally:
            env.close()
            eval_env.close()

    print(f"Training done. Logs: {log_root}; models: {save_root}")


if __name__ == "__main__":
    main()
