"""End-to-end system audit: config, env, reward, regenerative history, training dims."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from configs import load_config
from custom_rl import register_envs
from custom_rl.plants import f_nonlinear2 as fmod
from custom_rl.rewards.plate_rewards import SensorProductivePlateReward


def test_config_env_matches_training_interface() -> None:
    """Config-built env must expose 6-dim sensor obs and productive reward."""
    register_envs()
    cfg = load_config("conf1")
    env_kwargs = {**cfg["env"], **cfg["reward"]}
    env = gym.make("CustomODEPlate-v0", **env_kwargs)
    assert env.observation_space.shape == (6,)
    assert env.action_space.shape == (2,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (6,)
    assert "pass_line_index" in info
    env.close()


def test_reward_uses_sensor_not_modal() -> None:
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        reward_id="productive",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        max_episode_steps=5,
    )
    obs, _ = env.reset(seed=1)
    obs2, reward, _, _, info = env.step(env.action_space.sample())
    assert obs2.shape == (6,)
    assert "reward_components" in info
    rc = info["reward_components"]
    assert "vibration_cost" in rc
    assert "productivity_score" in rc
    assert np.isfinite(reward)
    # Reward must not depend on raw modal state exposed to agent
    assert "sensor_obs_norm" in info
    assert len(info["sensor_obs_norm"]) == 4
    env.close()


def test_regenerative_history_one_entry_per_env_step() -> None:
    """RK4 internals must not flood delay history."""
    register_envs()
    env = gym.make(
        "CustomODEPlate-v0",
        enable_geometry_uncertainty=False,
        enable_sensor_uncertainty=False,
        enable_process_noise=False,
        dt=0.001,
        n_substeps=1,
        max_episode_steps=10,
    )
    env.reset(seed=0)
    plant = env.unwrapped.plant
    n0 = len(plant._modal_state_history)
    assert n0 == 1  # recorded at reset t=0

    for _ in range(5):
        env.step(env.action_space.sample())

    n1 = len(plant._modal_state_history)
    assert n1 == n0 + 5, f"expected 6 history entries, got {n1}"
    env.close()


def test_chatter_suppression_reward_penalizes_high_vibration() -> None:
    """Productive reward decreases when normalized sensor vibration increases."""
    fn = SensorProductivePlateReward()
    base_info = {
        "sensor_obs_norm": np.array([0.1, 0.1, 0.05, 0.05], dtype=np.float64),
        "action_phys": np.array([500.0, 5.0], dtype=np.float64),
        "action_phys_low": np.array([50.0, 0.0], dtype=np.float64),
        "action_phys_high": np.array([2000.0, 10.0], dtype=np.float64),
        "action_norm": np.zeros(2),
        "prev_action_norm": np.zeros(2),
    }
    x = np.zeros(8)
    u = np.zeros(2)
    r_low = fn(0.0, x, u, x, False, False, dict(base_info))
    high = dict(base_info)
    high["sensor_obs_norm"] = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    r_high = fn(0.0, x, u, x, False, False, high)
    assert r_high < r_low


def test_config_paths_isolated() -> None:
    for name in ("conf1", "conf2", "conf3"):
        cfg = load_config(name)
        run = cfg["run_dir"].replace("\\", "/")
        assert run.endswith(f"runs/{name}")
        assert name in cfg["model_dir"]
        assert name in cfg["plot_dir"]
