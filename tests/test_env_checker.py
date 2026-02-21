"""Automated env compliance tests using Gymnasium check_env."""

from __future__ import annotations

import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from custom_rl import register_envs


@pytest.fixture(scope="module")
def registered():
    register_envs()


def test_check_env_dense(registered) -> None:
    env = gym.make("CustomODECartPole-v0", reward_id="dense")
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_check_env_sparse(registered) -> None:
    env = gym.make("CustomODECartPole-v0", reward_id="sparse")
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_rollout_finite(registered) -> None:
    register_envs()
    env = gym.make("CustomODECartPole-v0", reward_id="dense")
    obs, _ = env.reset(seed=42)
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.dtype.kind == "f"
        assert (obs == obs).all()
        assert (abs(obs) != float("inf")).all()
        assert (reward == reward) and abs(reward) != float("inf")
        if terminated or truncated:
            obs, _ = env.reset(seed=43)
    env.close()


def test_deterministic_reset(registered) -> None:
    env = gym.make("CustomODECartPole-v0", reward_id="dense")
    o1, _ = env.reset(seed=99)
    o2, _ = env.reset(seed=99)
    assert (o1 == o2).all()
    env.close()
