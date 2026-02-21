# Custom RL

Gymnasium-compatible ODE control environment framework with pluggable plants and reward functions.

## Quickstart

```bash
pip install -e .
```

### Register and create environment

```python
import gymnasium as gym
from custom_rl import register_envs

register_envs()
env = gym.make("CustomODECartPole-v0", reward_id="dense")
obs, info = env.reset(seed=42)
```

### Verify environment

```bash
python scripts/check_env.py
```

### Train with PPO (multi-seed)

```bash
python scripts/train_sb3_ppo.py --seeds 0 1 2 --total-timesteps 100000
```

### Plot results

```bash
python scripts/plot_results.py --log-dir logs/ppo_cartpole --out-dir plots
```

## Procedure: Train, Evaluate, Plot

### 1. Verify environment

```bash
python scripts/check_env.py [--reward dense|sparse]
```

### 2. Train

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 1 2 \
  --reward dense \
  --total-timesteps 100000 \
  --log-dir logs/ppo_cartpole \
  --save-dir models/ppo_cartpole
```

**Parallel environments** (faster training via vectorization):

```bash
python scripts/train_sb3_ppo.py \
  --seeds 0 1 2 \
  --n-envs 4 \
  --vec-env dummy \
  --total-timesteps 100000
```

Options:
- `--n-envs N`: Number of parallel environments per training run (default: 1)
- `--vec-env dummy|subproc`: `dummy` for sequential (low overhead), `subproc` for multiprocess (true parallelism)

Output: `models/ppo_cartpole/best_{seed}/best_model.zip` and `logs/ppo_cartpole/seed_{seed}/*.monitor.csv`.

### 3. Evaluate (save trajectories)

```bash
python scripts/eval_policy.py \
  --model-dir models/ppo_cartpole \
  --seeds 0 1 2 \
  --n-episodes 10 \
  --out-dir eval_trajectories \
  --reward dense
```

Output: `eval_trajectories/trajectories_seed{seed}.json`.

### 4. Plot

```bash
python scripts/plot_results.py \
  --log-dir logs/ppo_cartpole \
  --traj-dir eval_trajectories \
  --out-dir plots \
  --seeds 0 1 2
```

Output: `plots/learning_curve.png` and `plots/trajectory_states.png`.

## Structure

- `custom_rl/envs/` – Generic ODE control env and Gymnasium registration
- `custom_rl/plants/` – ODE plant dynamics (e.g. CartPole)
- `custom_rl/rewards/` – Pluggable reward functions (dense, sparse)
- `custom_rl/integration/` – RK4 integrator
- `scripts/` – Check env, train, eval, plot
