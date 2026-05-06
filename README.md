# Custom RL

Gymnasium-compatible ODE/RK4 control framework for the milling/plate vibration process with PPO-based reinforcement learning.

The current environment uses a nonlinear modal plate vibration plant for milling-like excitation:

```text
Environment ID: CustomODEPlate-v0
Plant: PlatePlant
State: [eta1, eta1_dot, eta2, eta2_dot, ..., etaK, etaK_dot]
Action: normalized PPO action [u_omega, u_ac] in [-1, 1]^2
Physical action: [omega, ac]
Integrator: fixed-step RK4
RL algorithm: PPO actor-critic from Stable-Baselines3
```

## Quickstart

```bash
pip install -e .
```

## Register and create environment

```python
import gymnasium as gym
from custom_rl import register_envs

register_envs()

env = gym.make(
    "CustomODEPlate-v0",
    reward_id="dense",
    max_episode_steps=1000,
)

obs, info = env.reset(seed=42)
```

## Verify environment with random policy

```bash
python scripts/check_plate_random_policy.py
```

This checks environment registration, reset, step, observation/action dimensions, reward calculation, and RK4 integration.

## Fast PPO training check

Use this command to quickly verify that PPO training runs correctly:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 2048 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 1
```

## Full PPO training

Recommended single-seed training:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

Recommended multi-seed training:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 1 2 \
  --reward dense \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

For the productive-cutting reward:

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward productive \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

## Notes on parallel environments

Use `--n-envs 1` for the current milling/plate model because the nonlinear dynamics module uses module-level cached variables. Parallel training should only be used after confirming that the plant/dynamics cache is safe for multiple simultaneous environments.

Options:

```text
--n-envs N              number of vectorized environments
--vec-env dummy|subproc vectorized environment type
```

## Evaluate trained policy

Short evaluation:

```bash
python scripts/eval_policy.py \
  --seeds 0 \
  --reward dense \
  --n-episodes 5 \
  --max-episode-steps 1000
```

Longer evaluation for clearer trajectory analysis:

```bash
python scripts/eval_policy.py \
  --seeds 0 \
  --reward dense \
  --n-episodes 10 \
  --max-episode-steps 3000
```

Multi-seed evaluation:

```bash
python scripts/eval_policy.py \
  --seeds 0 1 2 \
  --reward dense \
  --n-episodes 10 \
  --max-episode-steps 3000
```

The evaluation script saves states, normalized actions, physical actions `[omega, ac]`, rewards, and time values.

## Plot training and evaluation results

```bash
python scripts/plot_results.py \
  --seeds 0
```

For multi-seed plotting:

```bash
python scripts/plot_results.py \
  --seeds 0 1 2
```

To specify directories manually:

```bash
python scripts/plot_results.py \
  --log-dir logs/ppo_plate \
  --traj-dir eval_trajectories \
  --out-dir plots \
  --seeds 0 1 2
```

Generated plots include the learning curve, modal state trajectories, and physical control actions.

## Procedure: Train, Evaluate, Plot

### 1. Verify the environment

```bash
python scripts/check_plate_random_policy.py
```

### 2. Run fast PPO check

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 2048 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 1
```

### 3. Run full training

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 \
  --reward dense \
  --total-timesteps 300000 \
  --n-envs 1 \
  --max-episode-steps 1000 \
  --n-eval-episodes 3
```

### 4. Evaluate the trained policy

```bash
python scripts/eval_policy.py \
  --seeds 0 \
  --reward dense \
  --n-episodes 10 \
  --max-episode-steps 3000
```

### 5. Plot results

```bash
python scripts/plot_results.py \
  --seeds 0
```

## Main files

```text
custom_rl/envs/
  ode_control_env.py       Generic Gymnasium ODE environment
  registration.py          Registers CustomODEPlate-v0

custom_rl/plants/
  base.py                  ODEPlant interface
  plate.py                 Milling/plate vibration plant
  f_nonlinear2.py          Nonlinear modal-force dynamics
  compute_mode_shapes.py
  compute_natural_frequencies.py
  compute_nonlinear_stiffness.py

custom_rl/rewards/
  base.py                  Reward protocol
  plate_rewards.py         Dense/productive/quadratic/sparse rewards

custom_rl/integration/
  rk4.py                   Fixed-step RK4 integrator

scripts/
  check_plate_random_policy.py
  train_sb3_ppo.py
  eval_policy.py
  plot_results.py
```

## Default output directories

```text
logs/ppo_plate/                 training monitor logs
models/ppo_plate/               saved PPO models
eval_trajectories/              evaluation trajectory JSON files
plots/                          generated plots
```
