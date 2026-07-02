"""Smoke validation for omega bounds, stochastic dynamics, rewards, and Monte Carlo."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from custom_rl import register_envs
from custom_rl.plants.plate import (
    OMEGA_MAX_RAD_S,
    OMEGA_MIN_RAD_S,
    RPM_MAX,
    RPM_MIN,
)
from custom_rl.rewards.plate_rewards import DenseProductivePlateReward

ENV_ID = "CustomODEPlate-v0"


def check_omega_bounds() -> None:
    register_envs()
    env = gym.make(
        ENV_ID,
        reward_id="productive",
        randomize_y0=False,
        dynamics_uncertainty_std=0.0,
        max_episode_steps=50,
    )
    plant = env.unwrapped.plant
    assert plant.omega_min == OMEGA_MIN_RAD_S
    assert plant.omega_max == OMEGA_MAX_RAD_S
    assert plant.u_phys_low[0] == OMEGA_MIN_RAD_S
    assert plant.u_phys_high[0] == OMEGA_MAX_RAD_S
    env.close()
    print(f"[ok] spindle speed range {RPM_MIN:.0f} - {RPM_MAX:.0f} rpm "
          f"({OMEGA_MIN_RAD_S:.3f} - {OMEGA_MAX_RAD_S:.3f} rad/s)")


def check_deterministic_without_uncertainty() -> None:
    register_envs()
    env = gym.make(
        ENV_ID,
        reward_id="sparse",
        randomize_y0=False,
        dynamics_uncertainty_std=0.0,
        max_episode_steps=20,
    )
    action = np.array([0.0, 0.0], dtype=np.float64)
    obs1, _ = env.reset(seed=7)
    total1 = 0.0
    for _ in range(10):
        obs1, r, term, trunc, _ = env.step(action)
        total1 += r
        if term or trunc:
            break

    obs2, _ = env.reset(seed=7)
    total2 = 0.0
    for _ in range(10):
        obs2, r, term, trunc, _ = env.step(action)
        total2 += r
        if term or trunc:
            break

    assert np.allclose(obs1, obs2)
    assert np.isclose(total1, total2)
    env.close()
    print("[ok] deterministic rollout when uncertainty disabled")


def check_stochastic_differs_with_uncertainty() -> None:
    register_envs()
    kwargs = dict(
        reward_id="sparse",
        randomize_y0=False,
        dynamics_uncertainty_std=0.1,
        max_episode_steps=30,
    )
    env1 = gym.make(ENV_ID, **kwargs)
    env2 = gym.make(ENV_ID, **kwargs)
    action = np.array([0.5, 0.8], dtype=np.float64)

    env1.reset(seed=1)
    env2.reset(seed=2)
    finals = []
    for env in (env1, env2):
        last_modal = None
        for _ in range(15):
            _, _, term, trunc, info = env.step(action)
            last_modal = np.asarray(info["x_modal"], dtype=np.float64)
            if term or trunc:
                break
        finals.append(last_modal)

    assert finals[0] is not None and finals[1] is not None
    assert not np.allclose(finals[0], finals[1], atol=1e-8)
    env1.close()
    env2.close()
    print("[ok] stochastic dynamics produce different rollouts")


def check_reward_terms() -> None:
    reward = DenseProductivePlateReward()
    info = {"w_sensor": np.array([0.001, 0.0005]), "wdot_sensor": np.array([0.01, 0.005])}
    x = np.zeros(12)
    u = np.array([1.0, 0.5], dtype=np.float64)
    val = reward(0.0, x, u, x, False, False, info)
    terms = reward.last_reward_terms
    assert np.isfinite(val)
    for key in (
        "vibration_w_cost",
        "vibration_wdot_cost",
        "productivity",
        "omega_cost",
        "ac_action_cost",
    ):
        assert key in terms, f"missing {key}"
    assert terms["omega_cost"] > 0.0
    assert terms["ac_action_cost"] == 0.0
    print("[ok] reward decomposition includes vibration, productivity, omega cost")


def check_y0_randomization() -> None:
    register_envs()
    env = gym.make(
        ENV_ID,
        reward_id="sparse",
        randomize_y0=True,
        y0_min=0.1,
        y0_max=0.9,
        dynamics_uncertainty_std=0.0,
        max_episode_steps=5,
    )
    y0_values = []
    for seed in range(20):
        _, info = env.reset(seed=seed)
        y0_values.append(info["y0"])
    assert len(set(round(y, 4) for y in y0_values)) > 1
    env.close()
    print("[ok] y0 randomization explores different start lines")


def check_pipeline_env_kwargs() -> None:
    from custom_rl.eval.pipeline import (
        discover_model_seeds,
        plate_env_kwargs,
        plant_plot_metadata,
        resolve_ppo_model_path,
    )

    register_envs()
    kwargs = plate_env_kwargs(
        reward_id="productive",
        dynamics_uncertainty_std=0.0,
        randomize_y0=True,
        max_episode_steps=20,
    )
    env = gym.make(ENV_ID, **kwargs)
    meta = plant_plot_metadata(env)
    assert np.isclose(meta["rpm_min"], RPM_MIN)
    assert np.isclose(meta["rpm_max"], RPM_MAX)
    assert meta["physical_actions_are_rad_s_mm"] is True
    assert kwargs["dynamics_uncertainty_std"] == 0.0
    assert resolve_ppo_model_path(Path("models/ppo_plate"), 99) is None
    assert discover_model_seeds(Path("models/ppo_plate_missing")) == []
    env.close()
    print("[ok] shared train/eval env kwargs and plot metadata")


def main() -> int:
    check_omega_bounds()
    check_pipeline_env_kwargs()
    check_deterministic_without_uncertainty()
    check_stochastic_differs_with_uncertainty()
    check_reward_terms()
    check_y0_randomization()
    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
