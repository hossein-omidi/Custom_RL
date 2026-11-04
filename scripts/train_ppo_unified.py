#!/usr/bin/env python
"""Unified, evaluation-compatible PPO trainer for the flexible milling plate.

This trainer consolidates the stable environment/evaluation interface from
train_sb3_ppo.py with the reproducibility, scheduling and checkpoint ideas from
train_ppo_chat_gpt.py. It also retains the optional finishing environment used
by train_finish.py.

IMPORTANT COMPATIBILITY CHOICE
------------------------------
VecNormalize is intentionally NOT used. The current scripts/eval_policy.py
loads PPO checkpoints directly and evaluates raw environment observations; it
does not restore VecNormalize statistics. Training with VecNormalize here
would therefore make evaluation inconsistent unless eval_policy.py were also
changed. This file preserves the current evaluator's raw-observation pipeline.

Default roughing settings match eval_policy.py:
    env_id                  CustomODEPlate-v0
    reward                  productive
    RK4 dt                  1e-4 s
    n_substeps              10  -> environment/safety interval 1e-3 s
    controller hold         5 environment steps -> controller sample 5e-3 s
    max_episode_steps       250000 environment steps
    dynamics_uncertainty    0.001
    randomize_y0            True

Outputs are isolated under:
    <run-dir>/models/   best_<seed>/best_model.zip, final_<seed>.zip,
                        checkpoints_<seed>/*.zip
    <run-dir>/logs/     seed_<seed>/*.monitor.csv, evaluations.npz,
                        training_config.json, tensorboard/
    <run-dir>/evaluation/  reserved for eval_policy.py trajectory output

Run from the repository root. For three independent policies used in the
learning-curve figure, explicitly train all three seeds:
    python scripts/train_ppo_unified.py --seeds 0 1 2

The CLI default remains a single seed (0) unless --seeds is supplied.

For a quick environment/configuration check without training:
    python scripts/train_ppo_unified.py --preflight-only

The file expects the existing custom_rl package and its registered Gymnasium
plate environments to be available in the active project environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from custom_rl.plants.plate import omega_to_rpm


ENV_ID_ROUGHING = "CustomODEPlate-v0"
ENV_ID_FINISHING = "CustomODEPlateFinish-v0"
VALID_ENV_IDS = (ENV_ID_ROUGHING, ENV_ID_FINISHING)

# ---------------------------------------------------------------------------
# Balanced PPO defaults
# ---------------------------------------------------------------------------
# The 5 ms controller default keeps approximately the same physical rollout
# and optimizer batch as the former 1 ms / (2048, 256) setup:
# 3*400*5 = 6000 base environment steps per rollout and 1200/50 = 24 minibatches.
PPO_LEARNING_RATE = 2.5e-4
PPO_FINAL_LEARNING_RATE = 2.5e-5
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 8

# Reference discounts are per 1 ms environment step. They are converted to
# the controller decision interval after CLI parsing.
PPO_GAMMA_PER_ENV_STEP = 0.999995
PPO_GAE_LAMBDA_PER_ENV_STEP = 0.9995

PPO_CLIP_RANGE = 0.20
PPO_FINAL_CLIP_RANGE = 0.10
PPO_ENT_COEF = 0.002
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_TARGET_KL = 0.02
PPO_NET_ARCH = [256, 256]
PPO_LOG_STD_INIT = -1.0
PPO_ADAM_EPS = 1e-5

# Initial action mean: mid-range spindle speed and a low initial ap in roughing.
SAFE_AP_FRACTION = 0.05

# These are PPO/controller transitions. At a 5 ms hold, 200k decisions
# correspond to 1M underlying 1 ms environment steps, matching the former budget.
DEFAULT_TOTAL_TIMESTEPS = 1_000_000
DEFAULT_EVAL_FREQ = 50_000
DEFAULT_N_EVAL_EPISODES = 3
DEFAULT_CHECKPOINT_FREQ = 200_000

DEFAULT_DT = 1.0e-4
DEFAULT_N_SUBSTEPS = 10
DEFAULT_CONTROLLER_HOLD_ENV_STEPS = 1
DEFAULT_MAX_EPISODE_STEPS = 250_000
DEFAULT_DYNAMICS_UNCERTAINTY_STD = 0.001


Schedule = Callable[[float], float]


def linear_schedule(initial_value: float, final_value: float) -> Schedule:
    """SB3 schedule: interpolate using progress_remaining in [1, 0]."""
    if initial_value < 0.0 or final_value < 0.0:
        raise ValueError("Learning-rate/clip schedule values must be non-negative.")

    def schedule(progress_remaining: float) -> float:
        progress = float(np.clip(progress_remaining, 0.0, 1.0))
        return float(final_value + progress * (initial_value - final_value))

    return schedule


class ControllerSampleTimeWrapper(gym.Wrapper):
    """Hold each PPO action across several fine environment/safety steps.

    Rewards are discounted and accumulated within the held-action interval.
    The PPO gamma is therefore specified per controller decision, while the
    wrapper applies its per-environment-step root to the intermediate rewards.
    """

    def __init__(
        self,
        env: gym.Env,
        hold_env_steps: int,
        gamma_per_env_step: float,
    ) -> None:
        super().__init__(env)
        self.hold_env_steps = int(hold_env_steps)
        self.gamma_per_env_step = float(gamma_per_env_step)
        if self.hold_env_steps <= 0:
            raise ValueError("hold_env_steps must be positive.")
        if not np.isfinite(self.gamma_per_env_step) or not (0.0 < self.gamma_per_env_step <= 1.0):
            raise ValueError("gamma_per_env_step must be finite and in (0, 1].")
        base_env = env.unwrapped
        self.environment_step_s = float(
            getattr(base_env, "_step_dt", float(base_env.dt) * int(base_env.n_substeps))
        )
        self.controller_sample_time_s = self.environment_step_s * self.hold_env_steps

    def step(self, action):
        total_reward = 0.0
        reward_discount = 1.0
        observation = None
        terminated = truncated = False
        info: dict[str, Any] = {}

        steps_taken = 0
        for steps_taken in range(1, self.hold_env_steps + 1):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward_discount * float(reward)
            reward_discount *= self.gamma_per_env_step
            if terminated or truncated:
                break

        info = dict(info)
        info["controller_hold_env_steps"] = int(steps_taken)
        info["controller_sample_time_s"] = float(
            self.environment_step_s * steps_taken
        )
        info["controller_reward_discount_gamma_per_env_step"] = float(
            self.gamma_per_env_step
        )
        return observation, float(total_reward), bool(terminated), bool(truncated), info


def make_registered_plate_env(
    _env_id: str = ENV_ID_ROUGHING,
    _controller_hold_env_steps: int = 1,
    _gamma_per_env_step: float = 1.0,
    **env_kwargs: Any,
) -> gym.Env:
    """Build the same fine-step plant and controller wrapper in every worker.

    The plant still integrates and checks termination at its configured
    environment interval (1 ms by default). PPO receives one transition only
    after the held action has advanced the plant by 5 ms.
    """
    register_envs()
    env = gym.make(_env_id, **env_kwargs)
    if int(_controller_hold_env_steps) > 1:
        env = ControllerSampleTimeWrapper(
            env,
            hold_env_steps=int(_controller_hold_env_steps),
            gamma_per_env_step=float(_gamma_per_env_step),
        )
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an evaluation-compatible PPO policy for the milling plate."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--env-id",
        choices=VALID_ENV_IDS,
        default=ENV_ID_ROUGHING,
        help="Roughing controls [omega, ap]; finishing controls [omega].",
    )
    parser.add_argument(
        "--reward",
        choices=["dense", "productive", "quadratic", "sparse"],
        default=None,
        help="Default: productive for roughing, dense for finishing.",
    )
    parser.add_argument(
        "--ap-max",
        type=float,
        default=None,
        help="Finishing only: upper bound [mm] for episode-randomized axial depth; default 1.0 mm.",
    )

    parser.add_argument("--output-root", default="runs/ppo")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional unique run folder name. If omitted, a timestamped name is generated.",
    )

    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help="Total PPO/controller transitions across all workers (default 200k at 5 ms each).")
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument(
        "--start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
        help="Subprocess start method. 'spawn' is the safest cross-platform default.",
    )

    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="RK4 integration substep [s].")
    parser.add_argument("--n-substeps", type=int, default=DEFAULT_N_SUBSTEPS,
                        help="RK4 substeps per 1 ms environment/safety step.")
    parser.add_argument("--controller-hold-env-steps", type=int,
                        default=DEFAULT_CONTROLLER_HOLD_ENV_STEPS,
                        help="Hold each PPO action for this many environment steps (default: 5 -> 5 ms).")
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS,
                        help="Episode cap in fine environment steps, not PPO decisions.")
    parser.add_argument(
        "--dynamics-uncertainty-std",
        type=float,
        default=DEFAULT_DYNAMICS_UNCERTAINTY_STD,
    )
    parser.add_argument(
        "--randomize-y0",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Expose only the principal PPO controls; the remaining architecture and
    # optimizer settings are deliberately fixed in the versioned script.
    parser.add_argument("--learning-rate", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument("--final-learning-rate", type=float, default=PPO_FINAL_LEARNING_RATE)
    parser.add_argument("--n-steps", type=int, default=PPO_N_STEPS,
                        help="Rollout controller transitions per environment.")
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE,
                        help="PPO minibatch size in controller transitions.")
    parser.add_argument("--n-epochs", type=int, default=PPO_N_EPOCHS)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount per PPO/controller transition; default preserves the 1 ms physical discount horizon.")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="GAE lambda per PPO/controller transition; default preserves the 1 ms trace horizon.")
    parser.add_argument("--clip-range", type=float, default=PPO_CLIP_RANGE)
    parser.add_argument("--final-clip-range", type=float, default=PPO_FINAL_CLIP_RANGE)

    parser.add_argument("--n-eval-episodes", type=int, default=DEFAULT_N_EVAL_EPISODES)
    parser.add_argument(
        "--callback-eval-seed", type=int, default=None,
        help=("Optional shared EvalCallback RNG seed for every independently trained policy. "
              "Default None preserves the historical seed+10000 protocol. "
              "A shared seed improves comparability but does not guarantee identical "
              "disturbance paths if policies consume RNG differently."),
    )
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ,
                        help="Evaluate every this many aggregate controller transitions.")
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ,
                        help="Save periodic checkpoints every this many aggregate controller transitions.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Construct and validate the environment/configuration, then exit without training.",
    )
    return parser.parse_args()


def _resolve_mode_defaults(args: argparse.Namespace) -> None:
    if args.reward is None:
        args.reward = "dense" if args.env_id == ENV_ID_FINISHING else "productive"

    if args.controller_hold_env_steps <= 0:
        raise ValueError("--controller-hold-env-steps must be positive.")
    if args.gamma is None:
        args.gamma = PPO_GAMMA_PER_ENV_STEP ** args.controller_hold_env_steps
    if args.gae_lambda is None:
        args.gae_lambda = PPO_GAE_LAMBDA_PER_ENV_STEP ** args.controller_hold_env_steps

    if args.env_id == ENV_ID_FINISHING:
        if args.ap_max is None:
            args.ap_max = 1.0
        if args.ap_max <= 0.0:
            raise ValueError("For finishing, --ap-max must be positive.")
    elif args.ap_max is not None:
        raise ValueError("--ap-max is only valid with --env-id CustomODEPlateFinish-v0.")


def _validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative integers.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed in one run.")

    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.vec_env == "dummy" and args.n_envs != 1:
        raise ValueError(
            "DummyVecEnv with n_envs > 1 is unsafe for this plant's module-level "
            "regenerative-force state. Use --vec-env subproc."
        )
    if os.name == "nt" and args.start_method != "spawn":
        raise ValueError("On Windows, use --start-method spawn.")
    if args.start_method not in mp.get_all_start_methods():
        raise ValueError(
            f"Start method {args.start_method!r} is not available on this platform; "
            f"available methods: {mp.get_all_start_methods()}."
        )

    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.n_steps <= 1 or args.batch_size <= 1 or args.n_epochs <= 0:
        raise ValueError("n_steps must be >1, batch_size >1 and n_epochs positive.")
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError(
            f"batch_size={args.batch_size} exceeds rollout_size={rollout_size}."
        )
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            f"rollout_size={rollout_size} must be divisible by batch_size={args.batch_size}."
        )

    if args.dt <= 0.0 or args.n_substeps <= 0 or args.max_episode_steps <= 0:
        raise ValueError("dt, n_substeps and max_episode_steps must all be positive.")
    if args.controller_hold_env_steps <= 0:
        raise ValueError("--controller-hold-env-steps must be positive.")
    if args.max_episode_steps % args.controller_hold_env_steps != 0:
        raise ValueError(
            "--max-episode-steps must be divisible by --controller-hold-env-steps "
            "so time-limit truncations occur at a controller boundary."
        )
    if args.dynamics_uncertainty_std < 0.0:
        raise ValueError("dynamics_uncertainty_std cannot be negative.")
    if args.callback_eval_seed is not None and args.callback_eval_seed < 0:
        raise ValueError("--callback-eval-seed must be non-negative.")
    if args.eval_freq <= 0 or args.checkpoint_freq <= 0 or args.n_eval_episodes <= 0:
        raise ValueError("Evaluation/checkpoint frequencies and evaluation episodes must be positive.")

    if args.learning_rate <= 0.0 or args.final_learning_rate < 0.0:
        raise ValueError("Learning rate must be positive and final learning rate non-negative.")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("final_learning_rate cannot exceed learning_rate.")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    if not (0.0 < args.gae_lambda <= 1.0):
        raise ValueError("gae_lambda must be in (0, 1].")
    if args.clip_range <= 0.0 or args.final_clip_range <= 0.0:
        raise ValueError("Clip ranges must be positive.")
    if args.final_clip_range > args.clip_range:
        raise ValueError("final_clip_range cannot exceed clip_range.")


def _build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = plate_env_kwargs(
        reward_id=args.reward,
        dt=args.dt,
        n_substeps=args.n_substeps,
        max_episode_steps=args.max_episode_steps,
        dynamics_uncertainty_std=args.dynamics_uncertainty_std,
        randomize_y0=args.randomize_y0,
        omega_cost_weight=3.0,
        pass_completion_bonus=1000.0,
    )
    if args.env_id == ENV_ID_FINISHING:
        kwargs["ap_max"] = float(args.ap_max)
    return kwargs


def _compute_safe_action_bias(plant: Any, action_dim: int) -> list[float]:
    """Return a normalized initial action mean consistent with the plant mode."""
    if action_dim <= 1:
        return [0.0] * action_dim

    ap_min = float(getattr(plant, "ap_min", 0.0))
    ap_max = float(getattr(plant, "ap_max", ap_min))
    ap_range = ap_max - ap_min
    if ap_range > 0.0:
        safe_ap = ap_min + SAFE_AP_FRACTION * ap_range
        normalized_ap = 2.0 * (safe_ap - ap_min) / ap_range - 1.0
    else:
        normalized_ap = 0.0

    bias = [0.0, float(np.clip(normalized_ap, -1.0, 1.0))]
    if action_dim > 2:
        bias.extend([0.0] * (action_dim - 2))
    return bias[:action_dim]


def _init_action_bias(model: PPO, bias: list[float]) -> None:
    """Initialize the policy mean; no effect on action bounds or plant scaling."""
    action_net = model.policy.action_net
    if not hasattr(action_net, "bias") or action_net.bias is None:
        return

    count = min(len(bias), int(action_net.bias.numel()))
    with torch.no_grad():
        bias_tensor = torch.as_tensor(
            bias[:count],
            dtype=action_net.bias.dtype,
            device=action_net.bias.device,
        )
        action_net.bias[:count].copy_(bias_tensor)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("stable-baselines3", "gymnasium", "numpy", "torch", "tensorboard"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _script_sha256() -> str | None:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return None


def _space_snapshot(space: gym.Space) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": type(space).__name__,
        "shape": list(getattr(space, "shape", ()) or ()),
        "dtype": str(getattr(space, "dtype", "unknown")),
    }
    if isinstance(space, gym.spaces.Box):
        result["low"] = np.asarray(space.low).tolist()
        result["high"] = np.asarray(space.high).tolist()
    return result


def _plant_snapshot(plant: Any, env: gym.Env) -> dict[str, Any]:
    keys = (
        "N", "omega_min", "omega_max", "ap_min", "ap_max", "ae_default",
        "w_limit", "w_obs_scale", "wdot_obs_scale", "L1", "L2",
        "feed_rate", "max_episode_steps", "action_dim",
    )
    values: dict[str, Any] = {}
    for key in keys:
        if hasattr(plant, key):
            values[key] = _jsonable(getattr(plant, key))
    values["observation_space"] = _space_snapshot(env.observation_space)
    values["action_space"] = _space_snapshot(env.action_space)
    values["max_episode_steps_env"] = int(getattr(env.unwrapped, "max_episode_steps", 0))
    return values


def _preflight(args: argparse.Namespace, env_kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[float], float]:
    """Instantiate the real project environment and validate key assumptions."""
    register_envs()
    probe = gym.make(args.env_id, **env_kwargs)
    try:
        if not isinstance(probe.action_space, gym.spaces.Box):
            raise TypeError(
                f"PPO MlpPolicy requires a continuous Box action space; got {type(probe.action_space).__name__}."
            )
        if not isinstance(probe.observation_space, gym.spaces.Box):
            raise TypeError(
                f"PPO MlpPolicy requires a Box observation space; got {type(probe.observation_space).__name__}."
            )

        observation, _ = probe.reset(seed=int(args.seeds[0]))
        observation = np.asarray(observation)
        if observation.shape != probe.observation_space.shape:
            raise ValueError(
                f"Reset observation shape {observation.shape} does not match "
                f"observation_space {probe.observation_space.shape}."
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("Environment reset produced non-finite observation values.")

        low = np.asarray(probe.action_space.low, dtype=np.float64)
        high = np.asarray(probe.action_space.high, dtype=np.float64)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Action-space bounds must be finite for this continuous-control setup.")
        if np.any(high <= low):
            raise ValueError("Every action-space upper bound must exceed its lower bound.")

        plant = probe.unwrapped.plant
        tooth_count = max(int(getattr(plant, "N", 1)), 1)
        omega_max = max(float(getattr(plant, "omega_max", 0.0)), 1e-12)
        minimum_tooth_delay = 2.0 * np.pi / (tooth_count * omega_max)
        if args.dt >= minimum_tooth_delay:
            raise ValueError(
                f"RK4 dt={args.dt:g} s must be smaller than the minimum one-tooth "
                f"regenerative delay {minimum_tooth_delay:.6g} s."
            )

        action_dim = int(np.prod(probe.action_space.shape))
        safe_bias = _compute_safe_action_bias(plant, action_dim)
        # This bias is defined for the environment's documented normalized action.
        # If a future plant version changes the action bounds, skip it rather than
        # silently applying a normalized bias to a differently scaled action.
        normalized_action_bounds = (
            np.allclose(low, -1.0, atol=1e-6) and np.allclose(high, 1.0, atol=1e-6)
        )
        if not normalized_action_bounds:
            print("WARNING: action bounds are not [-1, 1]; safe action-bias initialization is disabled.")
            safe_bias = [0.0] * action_dim

        snapshot = _plant_snapshot(plant, probe)
        snapshot["minimum_tooth_delay_s"] = float(minimum_tooth_delay)
        snapshot["environment_step_dt_s"] = float(args.dt * args.n_substeps)
        snapshot["controller_hold_env_steps"] = int(args.controller_hold_env_steps)
        snapshot["controller_sample_time_s"] = float(
            args.dt * args.n_substeps * args.controller_hold_env_steps
        )
        snapshot["control_dt_s"] = snapshot["controller_sample_time_s"]
        snapshot["reset_observation_finite"] = True
        snapshot["safe_action_bias"] = safe_bias
        snapshot["normalized_action_bounds"] = bool(normalized_action_bounds)

        if args.env_id == ENV_ID_FINISHING:
            ap_min = float(getattr(plant, "ap_min", 0.0))
            ap_max = float(getattr(plant, "ap_max", args.ap_max))
            if not (ap_max > ap_min):
                raise ValueError(f"Finishing ap range is invalid: ap_min={ap_min}, ap_max={ap_max}.")
            if args.ap_max is not None and ap_max > float(args.ap_max) + 1e-9:
                raise ValueError(
                    f"Environment ap_max={ap_max} exceeds requested --ap-max={args.ap_max}."
                )

        return snapshot, safe_bias, float(minimum_tooth_delay)
    finally:
        probe.close()


def _make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    mode = "finishing" if args.env_id == ENV_ID_FINISHING else "roughing"
    run_name = args.run_name or f"{mode}_{timestamp}"
    run_dir = (Path(args.output_root).expanduser() / run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}\n"
            "Choose a new --run-name. Existing results are never overwritten."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "evaluation").mkdir()
    return run_dir


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_sb3_zip(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected model archive was not created: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Model archive is empty: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt member {bad_member!r} in model archive {path}")
        members = [item.filename for item in archive.infolist()]
    return {"path": str(path), "bytes": path.stat().st_size, "zip_integrity": "PASS", "members": members}


def _print_evaluation_commands(
    args: argparse.Namespace, run_dir: Path, plant_snapshot: dict[str, Any]
) -> None:
    model_dir = run_dir / "models"
    trajectory_dir = run_dir / "evaluation"
    common = (
        f'python scripts/eval_policy.py --model-dir "{model_dir}" '
        f'--out-dir "{trajectory_dir}" --seeds ' + " ".join(str(s) for s in args.seeds)
        + f' --env-id {args.env_id} --reward {args.reward}'
        + f' --dt {args.dt:g} --n-substeps {args.n_substeps}'
        + f' --policy-hold-env-steps {args.controller_hold_env_steps}'
        + f' --max-episode-steps {args.max_episode_steps}'
        + f' --dynamics-uncertainty-std {args.dynamics_uncertainty_std:g}'
        + f' --n-episodes 5 --n-mc 5'
    )
    common += " --randomize-y0" if args.randomize_y0 else " --no-randomize-y0"

    print("\nEvaluation command (loads the saved best checkpoint by default):")
    if args.env_id == ENV_ID_FINISHING:
        ap_min = float(plant_snapshot.get("ap_min", 0.0))
        ap_max = float(plant_snapshot.get("ap_max", args.ap_max))
        test_ap = 0.5 * (ap_min + ap_max)
        print(common)
        print(
            "Finishing evaluation: with no --ap argument, one ap value is sampled "
            "per outer episode from the configured range and held fixed across that "
            "episode's MC replicas."
        )
        print(
            f"For a separate fixed-depth sensitivity case, append --ap {test_ap:g}."
        )
    else:
        print(common)

    print("\nCombined learning-curve, training-diagnostic, and trajectory plots:")
    print(
        f'python scripts/plot_results.py --log-dir "{run_dir / "logs"}" '
        f'--traj-dir "{trajectory_dir}" --out-dir "{run_dir / "plots"}" '
        f'--seeds ' + " ".join(str(s) for s in args.seeds)
    )


def _train_one_seed(
    args: argparse.Namespace,
    seed: int,
    env_kwargs: dict[str, Any],
    run_dir: Path,
    plant_snapshot: dict[str, Any],
    safe_bias: list[float],
    minimum_tooth_delay: float,
) -> dict[str, Any]:
    model_root = run_dir / "models"
    log_root = run_dir / "logs"
    seed_log_dir = log_root / f"seed_{seed}"
    seed_log_dir.mkdir(parents=True, exist_ok=True)

    best_dir = model_root / f"best_{seed}"
    checkpoint_dir = model_root / f"checkpoints_{seed}"
    best_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rollout_size = args.n_steps * args.n_envs
    callback_eval_seed = int(args.callback_eval_seed if args.callback_eval_seed is not None else seed + 10_000)
    minibatches_per_update = (rollout_size // args.batch_size) * args.n_epochs
    eval_freq_calls = max(1, min(args.eval_freq, args.total_timesteps) // args.n_envs)
    checkpoint_freq_calls = max(1, args.checkpoint_freq // args.n_envs)
    gamma_per_env_step = float(args.gamma ** (1.0 / args.controller_hold_env_steps))
    environment_step_dt = float(args.dt * args.n_substeps)
    controller_sample_time = environment_step_dt * args.controller_hold_env_steps

    seed_config = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": int(seed),
        "command_line": sys.argv,
        "arguments": vars(args),
        "environment": {
            "env_id": args.env_id,
            "reward_id": args.reward,
            "kwargs": env_kwargs,
            "probe": plant_snapshot,
            "minimum_tooth_delay_s": minimum_tooth_delay,
            "dt_s": args.dt,
            "n_substeps": args.n_substeps,
            "environment_step_dt_s": environment_step_dt,
            "controller_hold_env_steps": args.controller_hold_env_steps,
            "controller_sample_time_s": controller_sample_time,
            "control_dt_s": controller_sample_time,
            "max_episode_steps": args.max_episode_steps,
            "dynamics_uncertainty_std": args.dynamics_uncertainty_std,
            "randomize_y0": args.randomize_y0,
        },
        "ppo": {
            "learning_rate_initial": args.learning_rate,
            "learning_rate_final": args.final_learning_rate,
            "n_steps": args.n_steps,
            "n_envs": args.n_envs,
            "rollout_size": rollout_size,
            "rollout_environment_steps": rollout_size * args.controller_hold_env_steps,
            "rollout_duration_s": args.n_steps * controller_sample_time,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "optimizer_minibatches_per_update": minibatches_per_update,
            "gamma": args.gamma,
            "gamma_per_environment_step": gamma_per_env_step,
            "gae_lambda": args.gae_lambda,
            "gae_lambda_per_environment_step": float(
                args.gae_lambda ** (1.0 / args.controller_hold_env_steps)
            ),
            "controller_hold_env_steps": args.controller_hold_env_steps,
            "clip_range_initial": args.clip_range,
            "clip_range_final": args.final_clip_range,
            "ent_coef": PPO_ENT_COEF,
            "vf_coef": PPO_VF_COEF,
            "max_grad_norm": PPO_MAX_GRAD_NORM,
            "target_kl": PPO_TARGET_KL,
            "net_arch": {"pi": PPO_NET_ARCH, "vf": PPO_NET_ARCH},
            "log_std_init": PPO_LOG_STD_INIT,
            "adam_epsilon": PPO_ADAM_EPS,
            "normalize_observations": False,
            "normalize_rewards": False,
            "normalization_reason": "Compatibility with current eval_policy.py raw-observation PPO.load pipeline.",
            "safe_action_bias": safe_bias,
        },
        "evaluation_callback": {
            "frequency_controller_transitions_requested": args.eval_freq,
            "frequency_vecenv_calls_actual": eval_freq_calls,
            "frequency_controller_transitions_actual": eval_freq_calls * args.n_envs,
            "frequency_environment_transitions_actual": (
                eval_freq_calls * args.n_envs * args.controller_hold_env_steps
            ),
            "episodes_per_evaluation": args.n_eval_episodes,
            "deterministic_policy": True,
            "evaluation_env_seed": callback_eval_seed,
            "evaluation_seed_shared_across_training_seeds": args.callback_eval_seed is not None,
            "checkpoint_selection_metric": "mean evaluation episode reward",
            "cross_seed_note": (
                "The plotted between-seed SD describes observed seed-level means, "
                "including stochastic evaluation-condition variability. A shared "
                "initial EvalCallback seed does not guarantee identical disturbance "
                "paths or eliminate test noise."
            ),
        },
        "tensorboard_diagnostics": {
            "enabled": True,
            "log_dir": str(seed_log_dir / "tensorboard"),
            "scalar_tags_expected": [
                "rollout/ep_rew_mean",
                "train/explained_variance",
                "train/approx_kl",
                "train/clip_fraction",
                "train/std",
            ],
        },
        "periodic_checkpoint": {
            "frequency_controller_transitions_requested": args.checkpoint_freq,
            "frequency_vecenv_calls_actual": checkpoint_freq_calls,
            "frequency_controller_transitions_actual": checkpoint_freq_calls * args.n_envs,
            "frequency_environment_transitions_actual": (
                checkpoint_freq_calls * args.n_envs * args.controller_hold_env_steps
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git_commit": _git_commit(),
            "script_sha256": _script_sha256(),
        },
    }
    _write_json(seed_log_dir / "training_config.json", seed_config)

    worker_env_kwargs = {
        **env_kwargs,
        "_env_id": args.env_id,
        "_controller_hold_env_steps": args.controller_hold_env_steps,
        "_gamma_per_env_step": gamma_per_env_step,
    }
    start_kwargs = {"start_method": args.start_method}

    train_vec_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    train_vec_kwargs = start_kwargs if train_vec_cls is SubprocVecEnv else None

    # Use a separate subprocess for evaluation even when training uses
    # DummyVecEnv. This isolates evaluation from the plant's module-level
    # regenerative-force state and prevents evaluation rollouts contaminating
    # a single-process training environment.
    env = make_vec_env(
        env_id=make_registered_plate_env,
        n_envs=args.n_envs,
        seed=seed,
        vec_env_cls=train_vec_cls,
        vec_env_kwargs=train_vec_kwargs,
        monitor_dir=str(seed_log_dir),
        env_kwargs=worker_env_kwargs,
    )
    eval_env = make_vec_env(
        env_id=make_registered_plate_env,
        n_envs=1,
        seed=callback_eval_seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=start_kwargs,
        env_kwargs=worker_env_kwargs,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(seed_log_dir),
        eval_freq=eval_freq_calls,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq_calls,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_seed_{seed}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    policy_kwargs = {
        "net_arch": {"pi": PPO_NET_ARCH, "vf": PPO_NET_ARCH},
        "log_std_init": PPO_LOG_STD_INIT,
        "ortho_init": True,
        "optimizer_kwargs": {"eps": PPO_ADAM_EPS},
    }

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=linear_schedule(args.learning_rate, args.final_learning_rate),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=linear_schedule(args.clip_range, args.final_clip_range),
        normalize_advantage=True,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        target_kl=PPO_TARGET_KL,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(seed_log_dir / "tensorboard"),
        verbose=1,
        device=args.device,
    )
    _init_action_bias(model, safe_bias)

    print(f"\n========== TRAINING SEED {seed} ==========")
    print(f"Training envs: {args.n_envs} ({args.vec_env}); rollout={rollout_size}; batch={args.batch_size}")
    print(f"Optimizer minibatches/update: {minibatches_per_update}")
    print(
        f"PPO: lr={args.learning_rate:g}->{args.final_learning_rate:g}, "
        f"gamma={args.gamma}, GAE={args.gae_lambda}, "
        f"clip={args.clip_range:g}->{args.final_clip_range:g}, target_kl={PPO_TARGET_KL}"
    )
    print(f"Output models: {model_root}; logs: {seed_log_dir}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=args.progress_bar,
            tb_log_name=f"ppo_seed_{seed}",
        )

        final_prefix = model_root / f"final_{seed}"
        model.save(str(final_prefix))
        final_archive = Path(str(final_prefix) + ".zip")
        final_validation = _validate_sb3_zip(final_archive)

        best_archive = best_dir / "best_model.zip"
        best_validation = None
        if best_archive.exists():
            best_validation = _validate_sb3_zip(best_archive)
        else:
            raise RuntimeError(
                f"EvalCallback did not create {best_archive}. Check evaluations.npz "
                "and the evaluation episode rewards; no best checkpoint can be claimed."
            )

        eval_npz = seed_log_dir / "evaluations.npz"
        if not eval_npz.is_file():
            raise RuntimeError(f"Expected EvalCallback history is missing: {eval_npz}")
        with np.load(eval_npz) as evaluation_data:
            eval_history = {
                key: {
                    "shape": list(evaluation_data[key].shape),
                    "dtype": str(evaluation_data[key].dtype),
                }
                for key in evaluation_data.files
            }

        monitor_files = sorted(seed_log_dir.glob("*.monitor.csv"))
        summary = {
            "seed": int(seed),
            "status": "completed",
            "timesteps_reported_by_model": int(model.num_timesteps),
            "controller_transitions_reported": int(model.num_timesteps),
            "environment_transitions_reported": int(
                model.num_timesteps * args.controller_hold_env_steps
            ),
            "controller_sample_time_s": float(controller_sample_time),
            "best_mean_eval_reward": (
                float(eval_callback.best_mean_reward)
                if np.isfinite(eval_callback.best_mean_reward)
                else None
            ),
            "best_checkpoint": best_validation,
            "final_checkpoint": final_validation,
            "evaluation_history": {
                "path": str(eval_npz),
                "arrays": eval_history,
            },
            "monitor_files": [str(path) for path in monitor_files],
            "monitor_file_count": len(monitor_files),
            "normalization_used": False,
        }
        _write_json(seed_log_dir / "training_summary.json", summary)
        print(f"Seed {seed} completed. Best mean evaluation reward: {summary['best_mean_eval_reward']}")
        return summary
    finally:
        env.close()
        eval_env.close()


def main() -> int:
    args = _parse_args()
    _resolve_mode_defaults(args)
    _validate_args(args)

    env_kwargs = _build_env_kwargs(args)
    plant_snapshot, safe_bias, minimum_tooth_delay = _preflight(args, env_kwargs)

    rollout_size = args.n_steps * args.n_envs
    control_dt = args.dt * args.n_substeps
    print("\n========== PPO PREFLIGHT: PASS ==========")
    print(f"Environment: {args.env_id}; reward: {args.reward}")
    print(
        f"RK4: dt={args.dt:g} s x {args.n_substeps}; "
        f"environment/safety step={control_dt:g} s"
    )
    print(
        f"Controller: hold={args.controller_hold_env_steps} environment steps; "
        f"sample_time={control_dt * args.controller_hold_env_steps:g} s"
    )
    print(
        f"Minimum tooth delay: {minimum_tooth_delay:.6g} s; "
        f"RK4 substep / tooth delay = {args.dt / minimum_tooth_delay:.6g}"
    )
    if "omega_min" in plant_snapshot and "omega_max" in plant_snapshot:
        print(
            f"Spindle range: {omega_to_rpm(float(plant_snapshot['omega_min'])):.0f}-"
            f"{omega_to_rpm(float(plant_snapshot['omega_max'])):.0f} rpm"
        )
    print(
        f"Rollout size: {rollout_size}; batch size: {args.batch_size}; "
        f"minibatches/update: {(rollout_size // args.batch_size) * args.n_epochs}"
    )
    print(f"Safe initial action bias: {safe_bias}")
    print("Observation/reward normalization: OFF (required for current eval_policy.py compatibility)")
    print(
        f"Training budget: {args.total_timesteps} controller transitions "
        f"(approximately {args.total_timesteps * args.controller_hold_env_steps} "
        "fine environment steps across all workers)."
    )

    if args.preflight_only:
        print("Preflight-only requested; no run directory was created and no training was started.")
        return 0

    run_dir = _make_run_dir(args)
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "command_line": sys.argv,
        "arguments": vars(args),
        "environment_probe": plant_snapshot,
        "resolved_env_kwargs": env_kwargs,
        "controller": {
            "environment_step_dt_s": float(args.dt * args.n_substeps),
            "hold_environment_steps": int(args.controller_hold_env_steps),
            "sample_time_s": float(args.dt * args.n_substeps * args.controller_hold_env_steps),
        },
        "ppo_defaults": {
            "learning_rate_initial": args.learning_rate,
            "learning_rate_final": args.final_learning_rate,
            "n_steps": args.n_steps,
            "n_envs": args.n_envs,
            "rollout_size": rollout_size,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range_initial": args.clip_range,
            "clip_range_final": args.final_clip_range,
            "ent_coef": PPO_ENT_COEF,
            "vf_coef": PPO_VF_COEF,
            "target_kl": PPO_TARGET_KL,
            "net_arch": PPO_NET_ARCH,
            "safe_action_bias": safe_bias,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git_commit": _git_commit(),
            "script_sha256": _script_sha256(),
        },
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)

    print(f"\nRun directory (new, isolated): {run_dir}")
    print(f"  models     : {run_dir / 'models'}")
    print(f"  logs       : {run_dir / 'logs'}")
    print(f"  evaluation : {run_dir / 'evaluation'}")

    summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        summaries.append(
            _train_one_seed(
                args,
                seed,
                env_kwargs,
                run_dir,
                plant_snapshot,
                safe_bias,
                minimum_tooth_delay,
            )
        )

    run_manifest["status"] = "completed"
    run_manifest["completed_at"] = datetime.now().astimezone().isoformat()
    run_manifest["seed_summaries"] = summaries
    _write_json(run_dir / "run_manifest.json", run_manifest)

    print("\n========== TRAINING COMPLETE ==========")
    print(f"All artifacts saved under: {run_dir}")
    _print_evaluation_commands(args, run_dir, plant_snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
