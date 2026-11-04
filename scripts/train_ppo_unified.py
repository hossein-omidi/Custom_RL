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
    controller hold         1 environment step -> controller sample 1e-3 s
    max_episode_steps       250000 environment steps
    dynamics_uncertainty    0.001
    randomize_y0            True

Outputs are isolated under:
    <run-dir>/models/   best_<seed>/best_model.zip, final_<seed>.zip,
                        checkpoints_<seed>/*.zip
    <run-dir>/logs/     seed_<seed>/*.monitor.csv, evaluations.npz,
                        training_config.json, tensorboard/
    <run-dir>/evaluation/  reserved for eval_policy.py trajectory output
    <run-dir>/plots/       reserved for plot_results.py

Preferred workflow (config → train → eval → plot, automatic paths):
    python scripts/experiment.py run --config configs/roughing.yaml

Or call this trainer directly. For three independent policies used in the
learning-curve figure, explicitly train all three seeds:
    python scripts/train_ppo_unified.py --seeds 0 1 2

The CLI default remains a single seed (0) unless --seeds is supplied.
A LATEST pointer is written under --output-root so later stages can use
``python scripts/experiment.py eval --run latest``.

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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_rl import register_envs
from custom_rl.eval.pipeline import plate_env_kwargs
from custom_rl.plants.plate import omega_to_rpm
from ppo_reliability import reliability_checkpoint_rank, summarize_reliability_episodes

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from run_paths import write_latest_pointer  # noqa: E402


ENV_ID_ROUGHING = "CustomODEPlate-v0"
ENV_ID_FINISHING = "CustomODEPlateFinish-v0"
VALID_ENV_IDS = (ENV_ID_ROUGHING, ENV_ID_FINISHING)

# ---------------------------------------------------------------------------
# Balanced PPO defaults
# ---------------------------------------------------------------------------
# 1 ms controller baseline. With 3 workers, rollout_size=3*2048=6144;
# batch_size=256 divides this exactly (24 minibatches per epoch).
PPO_LEARNING_RATE = 2.5e-4
PPO_FINAL_LEARNING_RATE = 2.5e-5
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 8

# Long-horizon settings per 1 ms environment/controller transition.
# gamma horizon ~= 200 s; GAE trace horizon ~= 2 s.
PPO_GAMMA_PER_ENV_STEP = 0.999995
PPO_GAE_LAMBDA_PER_ENV_STEP = 0.9995

PPO_CLIP_RANGE = 0.20
PPO_FINAL_CLIP_RANGE = 0.10
PPO_ENT_COEF = 0.002
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_TARGET_KL = 0.02
PPO_NET_ARCH = [256, 256]
PPO_LOG_STD_INIT = -2.5
PPO_ADAM_EPS = 1e-5

# Initial action mean: mid-range spindle speed and a low initial ap in roughing.
SAFE_AP_FRACTION = 0.05

# PPO transitions equal fine environment transitions because controller hold=1.
DEFAULT_TOTAL_TIMESTEPS = 1_000_000
DEFAULT_EVAL_FREQ = 50_000
DEFAULT_N_EVAL_EPISODES = 10  # paired screening set for reliability-based checkpoint selection
DEFAULT_CHECKPOINT_FREQ = 200_000

DEFAULT_DT = 1.0e-4
DEFAULT_N_SUBSTEPS = 10
DEFAULT_CONTROLLER_HOLD_ENV_STEPS = 1
DEFAULT_MAX_EPISODE_STEPS = 250_000
DEFAULT_DYNAMICS_UNCERTAINTY_STD = 0.001

# Single-stage completion-priority reward. The existing physical reward terms
# and their relative weights are preserved; only the accumulated running part
# is scaled, while Terminal completion/failure adjustments remain unscaled.
# The 1/24 factor is a first calibration based on the previous run's successful
# episode return (~4.75e5 including a 1e3 completion bonus): it brings dense
# running return to O(2e4) per pass, so a 1e3 completion bonus and 5e2--1e3
# failure penalty are no longer numerically negligible. It is a tunable scale,
# not a claimed universal optimum; verify with the pilot evaluation.
DEFAULT_RUNNING_REWARD_SCALE = 1.0 / 24.0
REWARD_KWARGS = {
    "w_weight": 0.6,
    "wdot_weight": 0.02,
    "w_scale": 1.5e-3,
    "wdot_scale": 1.0,
    "action_weight": 0.0,
    "negative_ap_weight": 2.0,
    "ap_action_weight": 0.0,
    "alive_bonus": 0.0,
    "require_physical_info": True,
    "w_clip": None,
    "wdot_clip": None,
    "ap_productive_target": 0.5,
    "include_ae_in_productivity": True,
    "productivity_weight": 240.0,
    "omega_cost_weight": 3.0,
    "termination_penalty": 500.0,
    "truncation_penalty": 500.0,
    "pass_completion_bonus": 1000.0,
    "failure_progress_penalty_weight": 1.0,
    "productivity_gate_enabled": True,
    "productivity_w_gate": 1.0e-3,
    "productivity_wdot_gate": 1.0,
    "productivity_wdot_gate_weight": 0.05,
    "productivity_gate_power": 2.0,
    "productivity_gate_min": 0.0,
    "running_reward_scale": DEFAULT_RUNNING_REWARD_SCALE,
}


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



class ReliabilityEvalCallback(BaseCallback):
    """Deterministic policy evaluation with reliability-first checkpoint selection.

    The evaluation environment remains stochastic (random milling line and
    configured dynamics uncertainty), while the policy action is deterministic.
    `evaluations.npz` retains SB3-compatible `timesteps`, `results`, and
    `ep_lengths` arrays and adds reliability metrics for plotting.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_path: Path,
        log_dir: Path,
        displacement_limit_m: float,
        evaluation_seed: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_model_path = Path(best_model_path)
        self.log_dir = Path(log_dir)
        self.displacement_limit_m = float(displacement_limit_m)
        self.evaluation_seed = int(evaluation_seed)
        self.best_rank = None
        self.best_metrics = None
        self.history = {
            "timesteps": [], "results": [], "ep_lengths": [],
            "pass_completion_rate": [], "safety_termination_rate": [],
            "truncation_rate": [], "median_completed_return": [],
            "median_completed_productivity": [], "median_progress": [], "mean_max_displacement_m": [],
        }
        self.episode_csv = self.log_dir / "evaluation_episodes.csv"
        self.history_npz = self.log_dir / "evaluations.npz"

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        episodes = self._evaluate_episodes()
        metrics = summarize_reliability_episodes(episodes)
        rank = reliability_checkpoint_rank(metrics)

        self.history["timesteps"].append(int(self.num_timesteps))
        self.history["results"].append([float(x["return"]) for x in episodes])
        self.history["ep_lengths"].append([int(x["length"]) for x in episodes])
        self.history["pass_completion_rate"].append(metrics["pass_completion_rate"])
        self.history["safety_termination_rate"].append(metrics["safety_termination_rate"])
        self.history["truncation_rate"].append(metrics["truncation_rate"])
        self.history["median_completed_return"].append(metrics["median_completed_return"])
        self.history["median_completed_productivity"].append(metrics["median_completed_productivity"])
        self.history["median_progress"].append(metrics["median_progress"])
        self.history["mean_max_displacement_m"].append(metrics["mean_max_displacement_m"])
        np.savez_compressed(self.history_npz, **{
            key: np.asarray(value, dtype=np.float64) for key, value in self.history.items()
        })
        self._append_episode_csv(episodes, metrics)

        for key in (
            "pass_completion_rate", "safety_termination_rate", "truncation_rate",
            "median_completed_return", "median_completed_productivity", "median_progress", "mean_max_displacement_m",
        ):
            self.logger.record(f"eval/{key}", float(metrics[key]))

        improved = self.best_rank is None or rank > self.best_rank
        if improved:
            self.best_rank = rank
            self.best_metrics = dict(metrics)
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_path))
            with (self.best_model_path.parent / "best_model_metrics.json").open("w", encoding="utf-8") as f:
                json.dump({"rank": list(rank), "metrics": metrics, "timesteps": int(self.num_timesteps)}, f, indent=2)

        if self.verbose:
            print(
                f"Reliability evaluation at {self.num_timesteps}: "
                f"completion={metrics['pass_completion_rate']:.1%}, "
                f"safety_termination={metrics['safety_termination_rate']:.1%}, "
                f"median_completed_productivity={metrics['median_completed_productivity']:.3g}, "
                f"median_progress={metrics['median_progress']:.3f}; "
                f"{'new best checkpoint' if improved else 'checkpoint unchanged'}"
            )
        return True

    def _evaluate_episodes(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        n_sensors = max(1, (int(self.eval_env.observation_space.shape[0]) - 2) // 2)
        progress_index = 2 * n_sensors

        for episode_index in range(self.n_eval_episodes):
            # Replay the same seeded evaluation scenarios at every checkpoint
            # and across training seeds. This makes checkpoint comparisons paired.
            episode_seed = self.evaluation_seed + episode_index
            self.eval_env.seed(episode_seed)
            obs = self.eval_env.reset()
            episode_return = 0.0
            episode_length = 0
            max_displacement = 0.0
            final_progress = 0.0
            productivity_sum = 0.0
            vibration_cost_sum = 0.0
            omega_cost_sum = 0.0
            terminated_final = False
            truncated_final = False
            final_info: dict[str, Any] = {}
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                info = dict(infos[0])
                episode_return += float(rewards[0])
                episode_length += 1

                if "w_sensor" in info:
                    w = np.asarray(info["w_sensor"], dtype=np.float64).reshape(-1)
                    if w.size and np.all(np.isfinite(w)):
                        max_displacement = max(max_displacement, float(np.max(np.abs(w))))
                if "max_displacement_m" in info:
                    max_displacement = max(max_displacement, float(info["max_displacement_m"]))
                if "feed_progress" in info:
                    final_progress = float(np.clip(info["feed_progress"], 0.0, 1.0))
                else:
                    # VecEnv auto-resets on done; use terminal_observation when
                    # available instead of accidentally measuring the next reset.
                    progress_obs = info.get("terminal_observation", obs)
                    progress_obs = np.asarray(progress_obs)
                    if progress_obs.ndim == 2 and progress_obs.shape[0] == 1:
                        progress_obs = progress_obs[0]
                    if progress_obs.ndim == 1 and progress_obs.size > progress_index:
                        final_progress = float(np.clip(1.0 - float(progress_obs[progress_index]), 0.0, 1.0))

                terms = info.get("reward_terms", {})
                if isinstance(terms, dict):
                    productivity_sum += float(terms.get("productivity_scaled", terms.get("productivity", 0.0)) or 0.0)
                    vibration_cost_sum += float(terms.get("vibration_w_cost_scaled", terms.get("vibration_w_cost", 0.0)) or 0.0)
                    vibration_cost_sum += float(terms.get("vibration_wdot_cost_scaled", terms.get("vibration_wdot_cost", 0.0)) or 0.0)
                    omega_cost_sum += float(terms.get("omega_cost_scaled", terms.get("omega_cost", 0.0)) or 0.0)

                if bool(dones[0]):
                    final_info = info
                    terminated_final = not bool(info.get("TimeLimit.truncated", False))
                    truncated_final = bool(info.get("TimeLimit.truncated", False))
                    break

            reason = str(final_info.get("termination_reason") or (
                "time_limit_truncated" if truncated_final else "terminated_failure" if terminated_final else "unknown"
            ))
            completed = bool(final_info.get("pass_completed", False)) or reason in {
                "pass_completed", "pass_completed_90percent"
            }
            safety_failure = reason in {"excessive_sensor_displacement", "invalid_state"}
            episodes.append({
                "episode": episode_index + 1,
                "evaluation_seed": int(episode_seed),
                "return": float(episode_return),
                "length": int(episode_length),
                "pass_completed": bool(completed),
                "termination_reason": reason,
                "safety_termination": bool(safety_failure),
                "truncated": bool(truncated_final),
                "final_progress": float(final_progress),
                "max_displacement_m": float(max_displacement),
                "max_displacement_ratio": float(max_displacement / max(self.displacement_limit_m, 1e-12)),
                "productivity_reward_sum": float(productivity_sum),
                "vibration_cost_sum": float(vibration_cost_sum),
                "omega_cost_sum": float(omega_cost_sum),
            })
        return episodes

    def _append_episode_csv(self, episodes: list[dict[str, Any]], metrics: dict[str, float]) -> None:
        import csv
        fields = [
            "timesteps", "episode", "evaluation_seed", "return", "length", "pass_completed",
            "termination_reason", "safety_termination", "truncated", "final_progress",
            "max_displacement_m", "max_displacement_ratio", "productivity_reward_sum",
            "vibration_cost_sum", "omega_cost_sum", "checkpoint_completion_rate",
            "checkpoint_safety_termination_rate",
        ]
        write_header = not self.episode_csv.exists()
        with self.episode_csv.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for episode in episodes:
                row = dict(episode)
                row["timesteps"] = int(self.num_timesteps)
                row["checkpoint_completion_rate"] = metrics["pass_completion_rate"]
                row["checkpoint_safety_termination_rate"] = metrics["safety_termination_rate"]
                writer.writerow(row)

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
                        help="Total PPO/controller transitions across all workers (default 1M at 1 ms each).")
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
                        help="Hold each PPO action for this many environment steps (default: 1 -> 1 ms).")
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
        help=("Optional shared ReliabilityEvalCallback RNG seed for every independently trained policy. "
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
    if args.reward in {"productive", "dense"}:
        kwargs.update(REWARD_KWARGS)
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
    print("\nNext steps (paths are automatic via --run-dir / --run latest):")
    print(f'  python scripts/experiment.py eval --run-dir "{run_dir}"')
    print(f'  python scripts/experiment.py plot --run-dir "{run_dir}"')
    print("  # or against the latest run pointer:")
    print("  python scripts/experiment.py eval --run latest")
    print("  python scripts/experiment.py plot --run latest")
    print(f'  tensorboard --logdir "{run_dir / "logs"}"')
    if args.env_id == ENV_ID_FINISHING:
        ap_min = float(plant_snapshot.get("ap_min", 0.0))
        ap_max = float(plant_snapshot.get("ap_max", args.ap_max))
        test_ap = 0.5 * (ap_min + ap_max)
        print(
            "Finishing evaluation: with no --ap argument, one ap value is sampled "
            "per outer episode from the configured range and held fixed across that "
            "episode's MC replicas."
        )
        print(
            f'For a fixed-depth sensitivity case: '
            f'python scripts/experiment.py eval --run-dir "{run_dir}" --ap {test_ap:g}'
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
    callback_eval_seed = int(args.callback_eval_seed if args.callback_eval_seed is not None else 10_000)
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
            "reliability_selection": True,
            "running_reward_scale": env_kwargs.get("running_reward_scale", 1.0),
            "evaluation_env_seed": callback_eval_seed,
            "evaluation_seed_shared_across_training_seeds": True,
            "checkpoint_selection_metric": (
                "lexicographic: pass_completion_rate desc, safety_termination_rate asc, "
                "median_completed_productivity desc, median_progress desc, mean_return desc"
            ),
            "cross_seed_note": (
                "Checkpoint selection replays the same fixed evaluation seed set "
                "for every checkpoint and every training seed. This paired test set "
                "reduces scenario-sampling noise in checkpoint comparisons; final "
                "reliability claims still require a larger independent test set."
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

    eval_callback = ReliabilityEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq_calls,
        n_eval_episodes=args.n_eval_episodes,
        best_model_path=best_dir / "best_model",
        log_dir=seed_log_dir,
        displacement_limit_m=float(env_kwargs.get("w_limit", 3.0e-3)),
        evaluation_seed=callback_eval_seed,
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
                f"ReliabilityEvalCallback did not create {best_archive}. Check evaluations.npz "
                "and the evaluation episode rewards; no best checkpoint can be claimed."
            )

        eval_npz = seed_log_dir / "evaluations.npz"
        if not eval_npz.is_file():
            raise RuntimeError(f"Expected ReliabilityEvalCallback history is missing: {eval_npz}")
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
                float(eval_callback.best_metrics["mean_return"])
                if eval_callback.best_metrics is not None else None
            ),
            "best_evaluation_metrics": eval_callback.best_metrics,
            "best_checkpoint_selection_rank": list(eval_callback.best_rank) if eval_callback.best_rank is not None else None,
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
        print(
            f"Seed {seed} completed. Best completion rate: "
            f"{(eval_callback.best_metrics or {}).get('pass_completion_rate', float('nan')):.3f}; "
            f"median completed productivity reward: "
            f"{(eval_callback.best_metrics or {}).get('median_completed_productivity', float('nan')):.3g}"
        )
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
    print(f"Axial-depth action range: {float(plant_snapshot.get('ap_min', 0.0)):.3g}-"
          f"{float(plant_snapshot.get('ap_max', 12.0)):.3g} mm")
    print(f"PPO gamma={args.gamma:.8f}; GAE lambda={args.gae_lambda:.8f}; log_std_init={PPO_LOG_STD_INIT}")
    print("Observation/reward normalization: OFF (raw observations; no VecNormalize)")
    if "running_reward_scale" in env_kwargs:
        print(
            f"Reward: running terms x {env_kwargs['running_reward_scale']:.6g}; "
            f"omega cost={env_kwargs['omega_cost_weight']}; "
            f"completion bonus={env_kwargs['pass_completion_bonus']} (terminal terms unscaled)"
        )
    print(
        f"Training budget: {args.total_timesteps} controller transitions "
        f"(approximately {args.total_timesteps * args.controller_hold_env_steps} "
        "fine environment steps across all workers)."
    )

    if args.preflight_only:
        print("Preflight-only requested; no run directory was created and no training was started.")
        return 0

    run_dir = _make_run_dir(args)
    write_latest_pointer(run_dir, Path(args.output_root).expanduser().resolve())
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
    print(f"  LATEST     : {Path(args.output_root).expanduser().resolve() / 'LATEST'}")

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
