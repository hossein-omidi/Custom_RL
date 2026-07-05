import gymnasium as gym
import numpy as np

from custom_rl import register_envs

register_envs()

env = gym.make(
    "CustomODEPlate-v0",
    reward_id="productive",
    max_episode_steps=10,
)

obs, info = env.reset(seed=0)

print("obs shape:", obs.shape)
print("obs:", obs)
print("info keys:", sorted(info.keys()))

assert obs.shape == env.observation_space.shape
assert np.all(np.isfinite(obs))
assert "x_modal" in info
assert "w_sensor" in info
assert "wdot_sensor" in info

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
    assert "x_modal" in info
    assert "w_sensor" in info
    assert "wdot_sensor" in info

    print("reward:", reward)
    print("w_sensor:", info["w_sensor"])
    print("wdot_sensor:", info["wdot_sensor"])

    if terminated or truncated:
        break

env.close()